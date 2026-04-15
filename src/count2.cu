#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"


__device__ __constant__ MeshAttrs device_mattrs;
__device__ __constant__ SelectionAttrs device_sattrs;
__device__ __constant__ SplitAttrs device_spattrs;
//__device__ __constant__ BinAttrs device_battrs;


__device__ inline FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize) {
    FLOAT half = 0.5 * boxsize;
    FLOAT x = dxyz + half;
    x = fmod(x, boxsize);  // negative if x is negative
    if (x < 0) x += boxsize;
    return x - half;
}


__device__ inline int wrap_periodic_int(int idx, int meshsize) {
    int r = idx % meshsize;
    return (r < 0) ? r + meshsize : r;
}


__device__ void set_angular_bounds(FLOAT *sposition, int *bounds) {
    FLOAT cth, phi;
    int icth, iphi;
    FLOAT theta, th_hi, th_lo;
    FLOAT phi_hi, phi_lo;
    FLOAT cth_max, cth_min;

    cth = sposition[2];
    phi = atan2(sposition[1], sposition[0]);
    if (phi < 0) phi += 2 * M_PI; // Wrap phi into [0, 2π]

    // Compute pixel indices
    icth = (cth >= 1) ? (device_mattrs.meshsize[0] - 1) : (int)(0.5 * (1 + cth) * device_mattrs.meshsize[0]);
    iphi = (int)(0.5 * phi / M_PI * device_mattrs.meshsize[1]);

    // Compute angular bounds
    theta = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.5)) / device_mattrs.meshsize[0]);
    th_hi = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.0)) / device_mattrs.meshsize[0]);
    th_lo = acos(-1.0 + 2.0 * ((FLOAT)(icth + 1.0)) / device_mattrs.meshsize[0]);
    phi_hi = 2 * M_PI * ((FLOAT)(iphi + 1.0) / device_mattrs.meshsize[1]);
    phi_lo = 2 * M_PI * ((FLOAT)(iphi + 0.0) / device_mattrs.meshsize[1]);
    FLOAT smax = acos(device_mattrs.smax);

    // Handle edge cases for angular bounds
    if (th_hi > M_PI - smax) {
        cth_min = -1;
        cth_max = cos(th_lo - smax);
        bounds[2] = 0;
        bounds[3] = device_mattrs.meshsize[1] - 1;
    } else if (th_lo < smax) {
        cth_min = cos(th_hi + smax);
        cth_max = 1;
        bounds[2] = 0;
        bounds[3] = device_mattrs.meshsize[1] - 1;
    } else {  // theta-stripe
        FLOAT dphi, calpha = cos(smax);
        cth_min = cos(th_hi + smax);
        cth_max = cos(th_lo - smax);

        if (theta < 0.5 * M_PI) {
            FLOAT cth_lo = cos(th_lo);
            dphi = acos(sqrt((calpha * calpha - cth_lo * cth_lo) / (1 - cth_lo * cth_lo)));
        } else {
            FLOAT cth_hi = cos(th_hi);
            dphi = acos(sqrt((calpha * calpha - cth_hi * cth_hi) / (1 - cth_hi * cth_hi)));
        }

        if (dphi < M_PI) {
            FLOAT phi_min = phi_lo - dphi;
            FLOAT phi_max = phi_hi + dphi;
            bounds[2] = (int)(floor(0.5 * phi_min / M_PI * device_mattrs.meshsize[1]));
            bounds[3] = (int)(floor(0.5 * phi_max / M_PI * device_mattrs.meshsize[1]));
        } else {
            bounds[2] = 0;
            bounds[3] = device_mattrs.meshsize[1] - 1;
        }
    }

    // Apply mask bounds
    cth_min = MAX(cth_min, device_mattrs.boxcenter[0] - device_mattrs.boxsize[0] / 2.);
    cth_max = MIN(cth_max, device_mattrs.boxcenter[0] + device_mattrs.boxsize[0] / 2.);

    bounds[0] = (int)(0.5 * (1 + cth_min) * device_mattrs.meshsize[0]);
    bounds[1] = (int)(0.5 * (1 + cth_max) * device_mattrs.meshsize[0]);
    if (bounds[0] < 0) bounds[0] = 0;
    if (bounds[1] >= device_mattrs.meshsize[0]) bounds[1] = device_mattrs.meshsize[0] - 1;
}


__device__ void set_cartesian_bounds(FLOAT *position, int *bounds) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        int meshsize = (int) device_mattrs.meshsize[axis];
        FLOAT offset = device_mattrs.boxcenter[axis] - device_mattrs.boxsize[axis] / 2;
        int index = (int) floor((position[axis] - offset) * meshsize / device_mattrs.boxsize[axis]);
        index = wrap_periodic_int(index, meshsize);
        int delta = (int) ceil(device_mattrs.smax / device_mattrs.boxsize[axis] * meshsize);
        bounds[2 * axis] = index - delta;
        bounds[2 * axis + 1] = index + delta;
        if (device_mattrs.periodic == 0) {
            bounds[2 * axis] = MAX(bounds[2 * axis], 0);
            bounds[2 * axis + 1] = MIN(bounds[2 * axis + 1], meshsize - 1);
        }
        else if (2 * delta + 1 >= meshsize) {  // make sure to visit each cell at most once
            bounds[2 * axis] = 0;
            bounds[2 * axis + 1] = meshsize - 1;
        }
    }
}


__device__ inline void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        add[axis] = position1[axis] + position2[axis];
        //if (device_mattrs.periodic) add[axis] = wrap_periodic_float(add[axis], device_mattrs.boxsize[axis]);
    }
}

__device__ inline void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        diff[axis] = position1[axis] - position2[axis];
        if (device_mattrs.periodic) diff[axis] = wrap_periodic_float(diff[axis], device_mattrs.boxsize[axis]);
    }
}

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2) {
    FLOAT d = 0.;
    for (size_t axis = 0; axis < NDIM; axis++) d += position1[axis] * position2[axis];
    return d;
}

__device__ inline bool is_selected_pair(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    for (size_t i = 0; i < device_sattrs.ndim; i++) {
        int var = device_sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= device_sattrs.smin[i]) && (costheta <= device_sattrs.smax[i]);
            //if (selected) printf("costheta %d %f %.4f %.4f ", i, costheta, device_sattrs.smin[i], device_sattrs.smax[i]);
        }
        if (var == VAR_S) {
            FLOAT diff[NDIM];
            difference(diff, position2, position1);
            const FLOAT s2 = dot(diff, diff);
            selected &= (s2 >= device_sattrs.smin[i] * device_sattrs.smin[i]) && (s2 <= device_sattrs.smax[i] * device_sattrs.smax[i]);
        }
    }
    return selected;
}


__device__ void set_legendre(FLOAT *legendre_cache, int ellmin, int ellmax, int ellstep, FLOAT mu, FLOAT mu2) {
    if ((ellmin % 2 == 0) && (ellstep % 2 == 0)) {
        for (int ell = ellmin; ell <= ellmax; ell+=ellstep) {
            if (ell == 0) {
                legendre_cache[ell] = 1.;
            }
            else if (ell == 2) {
                legendre_cache[ell] = (3.0 * mu2 - 1.0) / 2.0;
            }
            else if (ell == 4) {
                FLOAT mu4 = mu2 * mu2;
                legendre_cache[ell] = (35.0 * mu4 - 30.0 * mu2 + 3.0) / 8.0;
            }
            else if (ell == 6) {
                FLOAT mu4 = mu2 * mu2;
                FLOAT mu6 = mu4 * mu2;
                legendre_cache[ell] = (231.0 * mu6 - 315.0 * mu4 + 105.0 * mu2 - 5.0) / 16.0;
            }
            else if (ell == 6) {
                FLOAT mu4 = mu2 * mu2;
                FLOAT mu6 = mu4 * mu2;
                FLOAT mu8 = mu4 * mu4;
                legendre_cache[ell] = (6435.0 * mu8 - 12012.0 * mu6 + 6930.0 * mu4 - 1260.0 * mu2 + 35.0) / 128.0;
            }
            else {
                legendre_cache[ell] = 0.;
            }
        }
    }
    else {
        legendre_cache[0] = 1.0; // P_0(mu) = 1
        legendre_cache[1] = mu; // P_1(mu) = mu
        for (int ell = 2; ell <= ellmax; ell++) {
            legendre_cache[ell] = ((2.0 * ell - 1.0) * mu * legendre_cache[ell - 1] - (ell - 1.0) * legendre_cache[ell - 2]) / ell;
        }
    }
}


#define BESSEL_XMIN 0.1


__device__ FLOAT get_bessel(int ell, FLOAT x) {
    if (x < BESSEL_XMIN) {
        FLOAT x2 = x * x;
        switch (ell) {
            case 0:
                return 1. - x2 / 6. + x2 * x2 / 120. - x2 * x2 * x2 / 5040.;
            case 2:
                return x2 / 15. - x2 * x2 / 210. + x2 * x2 * x2 / 11340.;
            case 4:
                return x2 * x2 / 945.- x2 * x2 * x2 / 10395.;
            default:
                return 0.0;  // optionally handle unsupported ℓ values
        }
    } else {
        FLOAT invx  = 1.0 / x;
        FLOAT invx2 = invx * invx;
        FLOAT invx3, invx4;
        switch (ell) {
            case 0:
                return sin(x) * invx;
            case 2:
                return (3.0 * invx2 - 1.0) * sin(x) * invx - 3.0 * cos(x) * invx2;
            case 4:
                invx3 = invx2 * invx;
                invx4 = invx2 * invx2;
                //return sin(x) * (invx - 45.0 * invx3 + 105.0 * invx4) - cos(x) * (10.0 * invx - 105.0 * invx3);
                return 5 * (2 * invx2 - 21 * invx4) * cos(x) + (invx - 45 * invx3 + 105 * invx2 * invx3) * sin(x);
            default:
                return 0.0;
        }
    }
}


__global__ void reduce_add_kernel(
    const FLOAT *block_counts,
    size_t nblocks,
    FLOAT *counts,
    size_t csize)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < csize; i += stride) {
        FLOAT sum = 0;
        for (size_t iblock = 0; iblock < nblocks; iblock++) {
            sum += block_counts[iblock * csize + i];
        }
        counts[i] += sum;
    }
}


template <typename Op> __device__ inline void for_each_selected_pair_angular(size_t ii, Mesh mesh1, Mesh mesh2, Op& op)
{
    FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);

    int bounds[2 * NDIM];
    set_angular_bounds(sposition1, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * device_mattrs.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, device_mattrs.meshsize[1]);
            int icell = iphi_true + icth_n;

            int np2 = mesh2.nparticles[icell];
            size_t cum2 = mesh2.cumnparticles[icell];

            FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
            FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
            FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);

            for (int jj = 0; jj < np2; jj++) {
                FLOAT *position2 = &(positions2[NDIM * jj]);
                FLOAT *sposition2 = &(spositions2[NDIM * jj]);
                FLOAT *value2 = &(values2[mesh2.index_value.size * jj]);

                if (!is_selected_pair(sposition1, sposition2, position1, position2)) continue;

                op(ii, cum2 + (size_t)jj, position1, sposition1, position2, sposition2, value2);
            }
        }
    }
}

template <typename Op> __device__ inline void for_each_selected_pair_cartesian(size_t ii, Mesh mesh1, Mesh mesh2, Op& op)
{
    FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);

    int bounds[2 * NDIM];
    set_cartesian_bounds(position1, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int) device_mattrs.meshsize[0])
                 * device_mattrs.meshsize[2] * device_mattrs.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int) device_mattrs.meshsize[1])
                     * device_mattrs.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int) device_mattrs.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int np2 = mesh2.nparticles[icell];
                size_t cum2 = mesh2.cumnparticles[icell];

                FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
                FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
                FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);

                for (int jj = 0; jj < np2; jj++) {
                    FLOAT *position2 = &(positions2[NDIM * jj]);
                    FLOAT *sposition2 = &(spositions2[NDIM * jj]);
                    FLOAT *value2 = &(values2[mesh2.index_value.size * jj]);

                    if (!is_selected_pair(sposition1, sposition2, position1, position2)) continue;

                    op(ii, cum2 + (size_t)jj, position1, sposition1, position2, sposition2, value2);
                }
            }
        }
    }
}


__device__ inline int get_edge_bin_index(FLOAT value, const FLOAT *edges, int nbins)
{
    if (!edges || nbins <= 0) return -1;
    if ((value >= edges[nbins]) || (value < edges[0])) return -1;

    int ibin;
    for (ibin = nbins - 1; ibin >= 0; ibin--) {
        if (value >= edges[ibin]) break;
    }

    return ibin;
}


__device__ inline int get_bin_index(const BinAttrs *battrs, int index, FLOAT value)
{
    int ibin;

    if (battrs->asize[index] > 0) {
        ibin = get_edge_bin_index(value, battrs->array[index], (int)battrs->asize[index] - 1);
    }
    else {
        ibin = (int) floor((value - battrs->min[index]) / battrs->step[index]);
    }

    if (ibin < 0 || ibin >= battrs->shape[index]) return -1;
    return ibin;
}


__device__ inline int get_interp_bin_index(
    FLOAT x,
    const FLOAT *sep,
    int nsep,
    FLOAT *frac)
{
    *frac = (FLOAT)0.;

    if (!sep || nsep < 2) return -1;

    // Same convention as your original 1D code:
    // outside [sep[0], sep[nsep-1]] => no reweighting
    if (x < sep[0] || x > sep[nsep - 1]) return -1;

    for (int i = 0; i < nsep - 1; i++) {
        if (x <= sep[i + 1]) {
            FLOAT dx = sep[i + 1] - sep[i];
            *frac = (dx != (FLOAT)0.) ? (x - sep[i]) / dx : (FLOAT)0.;
            return i;
        }
    }

    return -1;
}


template <int ND> __device__ inline FLOAT lookup_angular_weight(const FLOAT (&costheta)[ND], const AngularWeight& angular)
{
    if (!angular.weight) return (FLOAT)1.;

    // ------------------------------------------------------------
    // Interpolation mode: sep[] is provided
    // ------------------------------------------------------------
    if (angular.sep[0] != NULL) {
        int i0[ND];
        FLOAT frac[ND];

        #pragma unroll
        for (int idim = 0; idim < ND; idim++) {
            const int nsep = (int)angular.shape[idim];
            if (nsep < 2) return (FLOAT)1.;
            if (!angular.sep[idim]) return (FLOAT)1.;

            i0[idim] = get_interp_bin_index(costheta[idim], angular.sep[idim], nsep, &frac[idim]);

            if (i0[idim] < 0) return (FLOAT)1.;
        }

        FLOAT result = (FLOAT)0.;
        const int ncorners = 1 << ND;

        for (int icorner = 0; icorner < ncorners; icorner++) {
            size_t idx = 0;
            FLOAT wcorner = (FLOAT)1.;

            #pragma unroll
            for (int idim = 0; idim < ND; idim++) {
                const int upper = (icorner >> idim) & 1;
                const int ibin = i0[idim] + upper;

                idx = idx * (size_t)angular.shape[idim] + (size_t)ibin;
                wcorner *= upper ? frac[idim] : ((FLOAT)1. - frac[idim]);
            }

            result += wcorner * angular.weight[idx];
        }

        return result;
    }

    // ------------------------------------------------------------
    // Binned lookup mode: edges[] is provided
    // ------------------------------------------------------------
    int ibin[ND];

    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        const int nbins = (int)angular.shape[idim];
        if (nbins <= 0) return (FLOAT)1.;
        if (!angular.edges[idim]) return (FLOAT)1.;

        ibin[idim] = get_edge_bin_index(costheta[idim], angular.edges[idim], nbins);

        if (ibin[idim] < 0) return (FLOAT)1.;
    }

    size_t idx = 0;
    #pragma unroll
    for (int idim = 0; idim < ND; idim++) {
        idx = idx * (size_t)angular.shape[idim] + (size_t)ibin[idim];
    }

    return angular.weight[idx];
}

size_t get_count2_size(IndexValue index_value1, IndexValue index_value2,
                        char names[][SIZE_NAME])
{
    // To check/modify when adding new weighting scheme
    int s1 = (index_value1.size_spin > 0);
    int s2 = (index_value2.size_spin > 0);
    size_t n = 1 + s1 + s2;

    if (names == NULL) {
        return n;
    }
    /* clear first buffers to be safe */
    size_t i;
    for (i = 0; i < MAX_NWEIGHT; ++i) {
        names[i][0] = '\0';
    }

    if (s1 && s2) {
        strncpy(names[0], "weight_plus_plus", SIZE_NAME-1);
        strncpy(names[1], "weight_plus_cross", SIZE_NAME-1);
        strncpy(names[2], "weight_cross_cross", SIZE_NAME-1);
    } else if (s1 ^ s2) {
        strncpy(names[0], "weight_plus", SIZE_NAME-1);
        strncpy(names[1], "weight_cross", SIZE_NAME-1);
    } else {
        strncpy(names[0], "weight", SIZE_NAME-1);
    }

    return n;
}



__device__ inline void compute_spin_projection_cartesian(
    const FLOAT *r1, const FLOAT *r2, FLOAT* s, int spin,
    FLOAT *splus_out, FLOAT *scross_out)
{
    // r1 and r2 are assumed on the sphere
    if (spin != 0) {

        // Reference "north pole" vector
        const FLOAT zhat[3] = {0.0, 0.0, 1.0};

        // Compute east and north basis at r1
        FLOAT east[3] = {zhat[1] * r1[2] - zhat[2] * r1[1],
                         zhat[2] * r1[0] - zhat[0] * r1[2],
                         zhat[0] * r1[1] - zhat[1] * r1[0]};
        FLOAT east_norm = rsqrtf(east[0] * east[0] + east[1] * east[1] + east[2] * east[2]);
        east[0] *= east_norm; east[1] *= east_norm; east[2] *= east_norm;

        FLOAT north[3] = {r1[1] * east[2] - r1[2] * east[1],
                          r1[2] * east[0] - r1[0] * east[2],
                          r1[0] * east[1] - r1[1] * east[0]};

        // Project r2 into tangent plane at r1
        FLOAT dot12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
        FLOAT p[3] = {r2[0] - dot12 * r1[0],
                      r2[1] - dot12 * r1[1],
                      r2[2] - dot12 * r1[2]};

        // Position angle (no need for normalization of p)
        FLOAT pe = p[0] * east[0] + p[1] * east[1] + p[2] * east[2];
        FLOAT pn = p[0] * north[0] + p[1] * north[1] + p[2] * north[2];
        FLOAT phi = atan2(pe, pn);

        FLOAT sphi = sin(spin * phi);
        FLOAT cphi = cos(spin * phi);

        *splus_out  = -(s[0] * cphi + s[1] * sphi);
        *scross_out =  (s[0] * sphi - s[1] * cphi);
    } else {
        *splus_out = - s[0];
        *scross_out = - s[1];
    }
}


template <int NSPLIT_TARGETS>
__device__ inline void accumulate_weight2(
    FLOAT *counts,
    const FLOAT *weight,
    size_t wsize,
    size_t bin_loc,
    size_t bsize,
    size_t split_size,
    const size_t *split_targets,
    FLOAT factor = 1.)
{
    if constexpr (NSPLIT_TARGETS == 0) {
        // Fast path: identical layout to the old no-split case
        for (size_t iweight = 0; iweight < wsize; iweight++) {
            atomicAdd(&(counts[bin_loc + iweight * bsize]), weight[iweight] * factor);
        }
    }
    else if constexpr (NSPLIT_TARGETS == 1) {
        const size_t weight_stride = split_size * bsize;
        const size_t offset0 = split_targets[0] * bsize + bin_loc;
        for (size_t iweight = 0; iweight < wsize; iweight++) {
            atomicAdd(&(counts[offset0 + iweight * weight_stride]), weight[iweight] * factor);
        }
    }
    else if constexpr (NSPLIT_TARGETS == 2) {
        const size_t weight_stride = split_size * bsize;
        const size_t offset0 = split_targets[0] * bsize + bin_loc;
        const size_t offset1 = split_targets[1] * bsize + bin_loc;
        for (size_t iweight = 0; iweight < wsize; iweight++) {
            const FLOAT w = weight[iweight] * factor;
            atomicAdd(&(counts[offset0 + iweight * weight_stride]), w);
            atomicAdd(&(counts[offset1 + iweight * weight_stride]), w);
        }
    }
}


__device__ inline void add_weight2(FLOAT *counts, FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2,
                                  FLOAT *value1, FLOAT *value2, IndexValue index_value1, IndexValue index_value2,
                                  BinAttrs battrs, WeightAttrs wattrs) {
    // Split targets:
    // 0 targets -> no split path
    // 1 target  -> same-split or single split slot
    // 2 targets -> cross-split, save in both locations
    int nsplit_targets = 0;
    size_t split_targets[2] = {0, 0};

    if (index_value1.size_split && index_value2.size_split) {
        if (device_spattrs.mode == SPLIT_JACKKNIFE) {
            const INT split1 = *((INT *) &(value1[index_value1.start_split]));
            const INT split2 = *((INT *) &(value2[index_value2.start_split]));
            if (split2 == split1) {
                nsplit_targets = 1;
                split_targets[0] = (size_t) split1;
            }
            else {
                nsplit_targets = 2;
                split_targets[0] = (size_t) device_spattrs.nsplits + (size_t) split1;
                split_targets[1] = (size_t) device_spattrs.nsplits * 2 + (size_t) split2;
            }
        }
    }

    FLOAT diff[NDIM];
    difference(diff, position2, position1);
    const FLOAT s2 = dot(diff, diff);
    const FLOAT DEFAULT_VALUE = -1000.;
    FLOAT s = DEFAULT_VALUE;
    FLOAT mu = DEFAULT_VALUE;
    FLOAT mu2 = DEFAULT_VALUE;
    LOS_TYPE los = LOS_NONE;
    VAR_TYPE var = VAR_NONE;
    size_t ellmin, ellmax, ellstep;

    bool REQUIRED_S = 0, REQUIRED_MU = 0, REQUIRED_MU2 = 0;
    size_t i = 0;
    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        if ((var == VAR_S) | (var == VAR_K)) REQUIRED_S = 1;
        if (var == VAR_MU) {
            los = battrs.los[i];
            REQUIRED_MU = 1;
        }
        if (var == VAR_RP) {
            los = battrs.los[i];
            REQUIRED_MU2 = 1;
        }
        if (var == VAR_PI) {
            los = battrs.los[i];
            REQUIRED_MU = 1;
        }
        if (var == VAR_POLE) {
            los = battrs.los[i];
            REQUIRED_MU2 = 1;
            ellmin = (size_t) battrs.min[i];
            ellmax = (size_t) battrs.max[i];
            if (battrs.asize[i] == 0) ellstep = (size_t) battrs.step[i];
            else ellstep = 1;
            if (!((ellmin % 2 == 0) && (ellstep % 2 == 0))) REQUIRED_MU = 1;
        }
    }
    REQUIRED_S |= REQUIRED_MU;

    if (REQUIRED_S) s = sqrt(s2);
    if (REQUIRED_MU2 || REQUIRED_MU) {
        FLOAT d;
        if (los == LOS_FIRSTPOINT) {
            d = dot(diff, sposition1);
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        else if (los == LOS_ENDPOINT) {
            d = dot(diff, sposition2);
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        else if (los == LOS_MIDPOINT) {
            FLOAT vlos[NDIM];
            addition(vlos, position1, position2);
            d = dot(diff, vlos);
            if (REQUIRED_MU) mu = d / sqrt(dot(vlos, vlos)) / s;
            else mu2 = d * d / dot(vlos, vlos) / s2;
        }
        else {
            if (los == LOS_X) d = diff[0];
            else if (los == LOS_Y) d = diff[1];
            else if (los == LOS_Z) d = diff[2];
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        if (REQUIRED_MU) mu2 = mu * mu;
        if (s2 == 0) {
            mu = 0.;
            mu2 = 0.;
        }
    }

    size_t ibin = 0;
    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        FLOAT value = 0;
        if (var == VAR_S) value = s;
        else if (var == VAR_MU) value = mu;
        else if (var == VAR_THETA) value = acos(dot(sposition1, sposition2)) / DTORAD;
        else if (var == VAR_PI) value = mu * s;
        else if (var == VAR_RP) value = (s2 <= 0.) ? 0. : sqrt(s2 - s2 * mu2);

        if ((var != VAR_POLE) && (var != VAR_K)) {
            int ibin_loc = get_bin_index(&battrs, i, value);
            if (ibin_loc < 0) return;
        }
        else {
            break;
        }
    }

    FLOAT pair_weight = 1.;
    if (index_value1.size_individual_weight) pair_weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight) pair_weight *= value2[index_value2.start_individual_weight];

    BitwiseWeight bitwise = wattrs.bitwise;
    if (index_value1.size_bitwise_weight && index_value2.size_bitwise_weight) {
        FLOAT pair_bweight = bitwise.default_value;
        int nbits = bitwise.noffset;
        int nbits1 = 0, nbits2 = 0;
        for (size_t iweight = 0; iweight < index_value1.size_bitwise_weight; iweight++) {
            INT bweight1 = *((INT *) &(value1[index_value1.start_bitwise_weight + iweight]));
            INT bweight2 = *((INT *) &(value2[index_value2.start_bitwise_weight + iweight]));
            nbits += POPCOUNT(bweight1 & bweight2);
            if (bitwise.p_nbits) {
                nbits1 += POPCOUNT(bweight1);
                nbits2 += POPCOUNT(bweight2);
            }
        }
        if (nbits != 0) {
            pair_bweight = bitwise.nrealizations / nbits;
            if (bitwise.p_nbits) {
                pair_bweight /= bitwise.p_correction_nbits[nbits1 * bitwise.p_nbits + nbits2];
            }
        }
        pair_weight *= pair_bweight;
    }

    {
        AngularWeight angular = wattrs.angular;
        if (angular.size) {
            FLOAT ct[1] = {dot(sposition1, sposition2)};
            pair_weight *= lookup_angular_weight<1>(ct, angular);
        }
    }

    if (index_value1.size_negative_weight && index_value2.size_negative_weight) {
        FLOAT pair_nweight = value1[index_value1.start_negative_weight] * value2[index_value2.start_negative_weight];
        pair_weight -= pair_nweight;
    }

    FLOAT weight[MAX_NWEIGHT];
    size_t wsize = 1;
    FLOAT splus1, scross1, splus2, scross2;
    if (index_value1.size_spin) compute_spin_projection_cartesian(sposition1, sposition2, &(value1[index_value1.start_spin]), wattrs.spin[0], &splus1, &scross1);
    if (index_value2.size_spin) compute_spin_projection_cartesian(sposition1, sposition2, &(value2[index_value2.start_spin]), wattrs.spin[1], &splus2, &scross2);

    if (index_value1.size_spin && index_value2.size_spin) {
        wsize = 3;
        weight[0] = pair_weight * splus1 * splus2;
        weight[1] = pair_weight * scross1 * splus2;
        weight[2] = pair_weight * scross1 * scross2;
    }
    else if (index_value1.size_spin) {
        wsize = 2;
        weight[0] = pair_weight * splus1;
        weight[1] = pair_weight * scross1;
    }
    else if (index_value2.size_spin) {
        wsize = 2;
        weight[0] = pair_weight * splus2;
        weight[1] = pair_weight * scross2;
    }
    else {
        wsize = 1;
        weight[0] = pair_weight;
    }

    if (i == battrs.ndim) {
        if (nsplit_targets == 0) {
            accumulate_weight2<0>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
        }
        else if (nsplit_targets == 1) {
            accumulate_weight2<1>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
        }
        else {
            accumulate_weight2<2>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
        }
    }

    else if ((i == battrs.ndim - 1) && (var == VAR_POLE)) {
        FLOAT legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
        for (int ill = 0; ill < battrs.shape[i]; ill++) {
            size_t ell;
            if (battrs.asize[i] > 0) ell = (size_t) battrs.array[i][ill];
            else ell = ill * ellstep + ellmin;
            const size_t bin_loc = ibin * battrs.shape[i] + ill;
            const FLOAT leg = (2 * ell + 1) * legendre_cache[ell];

            if (nsplit_targets == 0) {
                accumulate_weight2<0>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
            }
            else if (nsplit_targets == 1) {
                accumulate_weight2<1>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
            }
            else {
                accumulate_weight2<2>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
            }
        }
    }

    else if ((i == battrs.ndim - 2) && (battrs.var[i] == VAR_K) && (battrs.var[i + 1] == VAR_POLE)) {
        size_t ik_dim = i;
        size_t ip_dim = i + 1;
        FLOAT legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
        size_t nk = battrs.shape[ik_dim];
        size_t npole = battrs.shape[ip_dim];
        for (size_t ill = 0; ill < npole; ill++) {
            int ell;
            if (battrs.asize[ip_dim] > 0) ell = (int) battrs.array[ip_dim][ill];
            else ell = ill * ((int) ellstep) + ((int) ellmin);

            FLOAT leg = (((ell / 2) & 1) ? -1.0 : 1.0) * (2 * ell + 1) * legendre_cache[ell];
            for (size_t ik = 0; ik < nk; ik++) {
                FLOAT k = 0.;
                if (battrs.asize[ik_dim] > 0) k = battrs.array[ik_dim][ik];
                else k = ik * battrs.step[ik_dim] + battrs.min[ik_dim];

                const size_t bin_loc = (ibin * nk + ik) * npole + ill;
                const FLOAT leg_bessel = leg * get_bessel(ell, k * s);

                if (nsplit_targets == 0) {
                    accumulate_weight2<0>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
                }
                else if (nsplit_targets == 1) {
                    accumulate_weight2<1>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
                }
                else {
                    accumulate_weight2<2>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
                }
            }
        }
    }
}



struct Count2Op {
    FLOAT *local_counts;
    FLOAT *value1;
    IndexValue index_value1;
    IndexValue index_value2;
    BinAttrs battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t ii,
        size_t jj,
        FLOAT *position1,
        FLOAT *sposition1,
        FLOAT *position2,
        FLOAT *sposition2,
        FLOAT *value2)
    {
        (void)ii;
        (void)jj;
        add_weight2(
            local_counts,
            sposition1, sposition2,
            position1, position2,
            value1, value2,
            index_value1, index_value2,
            battrs, wattrs
        );
    }
};


__global__ void count2_angular_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh1,
    Mesh mesh2,
    BinAttrs battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;

    FLOAT *local_counts = &block_counts[blockIdx.x * csize];
    for (int i = tid; i < csize; i += blockDim.x) local_counts[i] = 0;

    __syncthreads();

    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * ii]);

        Count2Op op{
            local_counts,
            value1,
            mesh1.index_value,
            mesh2.index_value,
            battrs,
            wattrs
        };

        for_each_selected_pair_angular(ii, mesh1, mesh2, op);
    }
}


__global__ void count2_cartesian_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh1,
    Mesh mesh2,
    BinAttrs battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;

    FLOAT *local_counts = &block_counts[blockIdx.x * csize];
    for (int i = tid; i < csize; i += blockDim.x) local_counts[i] = 0;

    __syncthreads();

    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * ii]);

        Count2Op op{
            local_counts,
            value1,
            mesh1.index_value,
            mesh2.index_value,
            battrs,
            wattrs
        };

        for_each_selected_pair_cartesian(ii, mesh1, mesh2, op);
    }
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream) {

    // counts expected on the device already
    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count2_cartesian_kernel, nblocks, nthreads_per_block, buffer);

    // CUDA timing eventsis
    cudaEvent_t start, stop;
    float elapsed_time;

    // Determine output array size based on spin parameters
    size_t csize = get_count2_size(list_mesh[0].index_value, list_mesh[1].index_value, NULL) * battrs.size * spattrs.size;

    // Initialize histograms
    CUDA_CHECK(cudaMemset(counts, 0, csize * sizeof(FLOAT)));

    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_sattrs, &sattrs, sizeof(SelectionAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_spattrs, &spattrs, sizeof(SplitAttrs)));
    //CUDA_CHECK(cudaMemcpyToSymbol(device_battrs, &battrs, sizeof(BinAttrs)));
    BinAttrs device_battrs;
    copy_bin_attrs_to_device(&device_battrs, &battrs, buffer);

    WeightAttrs device_wattrs = wattrs;
    copy_weight_attrs_to_device(&device_wattrs, &wattrs, buffer);

    // allocate histogram arrays
    // printf("ALLOCATING histogram\n");
    FLOAT *block_counts = (FLOAT*) my_device_malloc(nblocks * csize * sizeof(FLOAT), buffer);
    //CUDA_CHECK(cudaMemset(block_counts, 0, nblocks * battrs.size * sizeof(FLOAT)));  // set to 0 in the kernel

    // Create CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    CUDA_CHECK(cudaDeviceSynchronize());
    if (mattrs.type == MESH_ANGULAR) count2_angular_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts, csize, list_mesh[0], list_mesh[1], device_battrs, device_wattrs);
    else count2_cartesian_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts, csize, list_mesh[0], list_mesh[1], device_battrs, device_wattrs);

    CUDA_CHECK(cudaDeviceSynchronize());
    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts, nblocks, counts, csize);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    log_message(LOG_LEVEL_DEBUG, "Time elapsed: %3.1f ms.\n", elapsed_time);

    // Free GPU memory
    my_device_free(block_counts, buffer);

    free_device_bin_attrs(&device_battrs, buffer);
    free_device_weight_attrs(&device_wattrs, buffer);

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
