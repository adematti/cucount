#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"

__device__ __constant__ MeshAttrs device_mattrs;
__device__ __constant__ SelectionAttrs device_sattrs;
//__device__ __constant__ BinAttrs device_battrs;


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
        FLOAT offset = device_mattrs.boxcenter[axis] - device_mattrs.boxsize[axis] / 2;
        int index = (int) ((position[axis] - offset) * device_mattrs.meshsize[axis] / device_mattrs.boxsize[axis]);
        int delta = (int) (device_mattrs.smax / device_mattrs.boxsize[axis] * device_mattrs.meshsize[axis]) + 1;
        bounds[2 * axis] = index - delta;
        bounds[2 * axis + 1] = index + delta;
        if (device_mattrs.periodic) {
            bounds[2 * axis] = MAX(bounds[2 * axis], 0);
            bounds[2 * axis + 1] = MIN(bounds[2 * axis + 1], device_mattrs.meshsize[axis] - 1);
        };
        //bounds[2 * axis] = 0;
        //bounds[2 * axis + 1] = device_mattrs.meshsize[axis] - 1;
    }
}

__device__ inline void addition(FLOAT *add, const FLOAT *position1, const FLOAT *position2, const FLOAT *boxsize) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        add[axis] = position1[axis] + position2[axis];
        if (boxsize != NULL) add[axis] %= boxsize[axis];
    }
}

__device__ inline void difference(FLOAT *diff, const FLOAT *position1, const FLOAT *position2, const FLOAT *boxsize) {
    for (size_t axis = 0; axis < NDIM; axis++) {
        diff[axis] = position1[axis] - position2[axis];
        if (boxsize != NULL) diff[axis] %= boxsize[axis];
    }
}

__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2) {
    FLOAT d = 0.;
    for (size_t axis = 0; axis < NDIM; axis++) d += position1[axis] * position2[axis];
    return d;
}

__device__ inline bool is_selected(FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2) {
    bool selected = 1;
    for (size_t i = 0; i < device_sattrs.ndim; i++) {
        int var = device_sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= device_sattrs.smin[i]) && (costheta <= device_sattrs.smax[i]);
            //if (selected) printf("costheta %d %f %.4f %.4f ", i, costheta, device_sattrs.smin[i], device_sattrs.smax[i]);
        }
    }
    return selected;
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


#define BESSEL_XMIN 0.01


__device__ FLOAT get_bessel(int ell, FLOAT x) {
    if (x < BESSEL_XMIN) {
        switch (ell) {
            case 0:
                return 1. - x * x / 6. + x * x * x * x / 120.;
            case 2:
                return x * x / 15. - x * x * x * x / 210.;
            case 4:
                return x * x * x * x / 945.;
            default:
                return 0.0;  // optionally handle unsupported ℓ values
        }
    } else {
        FLOAT x2 = x * x;
        switch (ell) {
            case 0:
                return sin(x) / x;
            case 2:
                return (3.0 / (x2) - 1.0) * sin(x) / x - 3.0 * cos(x) / (x2);
            case 4:
                return (105.0 / (x2 * x2) - 45.0 / x2 + 1.0) * sin(x) / x
                     - (105.0 / (x2 * x) - 10.0 / x) * cos(x);
            default:
                return 0.0;
        }
    }
}


__device__ inline void add_weight(FLOAT *counts, FLOAT *sposition1, FLOAT *sposition2, FLOAT *position1, FLOAT *position2,
                                  FLOAT *value1, FLOAT *value2, IndexValue index_value1, IndexValue index_value2,
                                  BinAttrs battrs, WeightAttrs wattrs) {
    int ibin = 0;
    FLOAT diff[NDIM];
    FLOAT *boxsize = NULL;
    if (device_mattrs.periodic) boxsize = device_mattrs.boxisze;
    difference(diff, position2, position1, boxsize);
    const FLOAT s2 = dot(diff, diff);
    const FLOAT DEFAULT_VALUE = -1000.;
    FLOAT s = DEFAULT_VALUE;
    FLOAT mu = DEFAULT_VALUE;
    FLOAT mu2 = DEFAULT_VALUE;
    LOS_TYPE los = LOS_NONE;
    VAR_TYPE var = VAR_NONE;
    size_t ellmin, ellmax, ellstep;
    // First find bins!
    bool REQUIRED_S = 0, REQUIRED_MU = 0, REQUIRED_MU2 = 0;
    size_t i = 0;
    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        if ((var == VAR_S) | (var == VAR_K)) {
            REQUIRED_S = 1;
        }
        if (var == VAR_MU) {
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
            mu = dot(diff, sposition2);
            if (REQUIRED_MU) mu = d / s;
            else mu2 = (d * d) / s2;
        }
        else if (los == LOS_MIDPOINT) {
            FLOAT vlos[NDIM];
            addition(vlos, position1, position2, boxsize);
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
        if (s == 0) {
            mu = 0.;
            mu2 = 0.;
        };
    }

    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];
        FLOAT value = 0;
        if (var == VAR_S) {
            value = s;
        }
        if (var == VAR_THETA) {
            value = acos(dot(sposition1, sposition2));
        }
        if (var == VAR_MU) {
            value = mu;
        }
        if ((var == VAR_S) || (var == VAR_THETA) || (var == VAR_MU)) {
            int ibin_loc = 0;  // int and not size_t as can go negative
            if (battrs.asize[i] > 0) {  // custom binning
                if ((value >= battrs.array[i][battrs.asize[i] - 1]) || (value < battrs.array[i][0])) return;
                for (ibin_loc = battrs.asize[i] - 1; ibin_loc >= 0; ibin_loc--) {
                    if (value >= battrs.array[i][ibin_loc]) break;
                }
            }
            else {  // linear binning
                ibin_loc = (int) (floor((value - battrs.min[i]) / battrs.step[i]));
            }
            if ((ibin_loc >= 0) && (ibin_loc < battrs.shape[i])) ibin = ibin * battrs.shape[i] + ibin_loc;
            else return;
        }
        else {
            break;
        }
    }
    // Then find weight!
    FLOAT pair_weight = 1.;
    if (index_value1.size_individual_weight) pair_weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight) pair_weight *= value2[index_value2.start_individual_weight];

    // TODO: add bitwise and angular upweights here
    BitwiseWeight bitwise = wattrs.bitwise;
    if (bitwise.size) {
        FLOAT pair_bweight = 1.;
        int nbits = bitwise.noffset;
        int nbits1 = 0, nbits2 = 0;
        for (size_t iweight = 0; iweight < bitwise.size; iweight++) {
            INT bweight1 = *((INT *) &(value1[index_value1.start_bitwise_weight + iweight]));  // reinterpret float as int
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
                pair_bweight /= bitwise.p_correction_bits[nbits1 * bitwise.p_nbits + nbits2];
            }
        }
        pair_weight *= pair_bweight;
    }
    AngularWeight angular = wattrs.angular;
    if (angular.size) {
        FLOAT pair_aweight = 1.;
        FLOAT costheta = dot(sposition1, sposition2);
        if (costheta > angular.sep[angular.size - 1] || (costheta <= angular.sep[0])) {
            ;
        }
        else {
            for (size_t ibin_loc = 0; ibin_loc < angular.size - 1; ibin_loc++) {
                if(costheta <= angular.sep[ibin_loc + 1]) { // ]min, max], as costheta instead of theta
                    FLOAT frac = (costheta - angular.sep[ibin_loc]) / (angular.sep[ibin_loc + 1] - angular.sep[ibin_loc]);
                    pair_aweight *= (1 - frac) * angular.weight[ibin_loc] + frac * angular.weight[ibin_loc + 1];
                    break;
                }
            }
        }
        pair_weight *= pair_aweight;
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
    else { // no spin
        wsize = 1;
        weight[0] = pair_weight;
    }
    if (i == battrs.ndim) {
        for (size_t iweight = 0; iweight < wsize; iweight++) {
            atomicAdd(&(counts[ibin + iweight * battrs.size]), weight[iweight]);
        }
    }

    else if ((i == battrs.ndim - 1) && (var == VAR_POLE)) {
        FLOAT legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
        for (int ill = 0; ill < battrs.shape[i]; ill++) {
            size_t ell;
            if (battrs.asize[i] > 0) ell = (size_t) battrs.array[i][ill];
            else ell = ill * ellstep + ellmin;
            size_t ibin_loc = ibin * battrs.shape[i] + ill;
            FLOAT leg = (2 * ell + 1) * legendre_cache[ell];
            for (size_t iweight = 0; iweight < wsize; iweight++) {
                atomicAdd(&(counts[ibin_loc + iweight * battrs.size]), weight[iweight] * leg);
            }
        }
    }
    else if ((i == battrs.ndim - 2) && (battrs.var[i] == VAR_K) && (battrs.var[i + 1] == VAR_POLE)) {
        FLOAT legendre_cache[MAX_POLE + 1];
        set_legendre(legendre_cache, ellmin, ellmax, ellstep, mu, mu2);
        for (int ill = 0; ill < battrs.shape[i]; ill++) {
            size_t ell;
            if (battrs.asize[i] > 0) ell = (size_t) battrs.array[i][ill];
            else ell = ill * ellstep + ellmin;
            FLOAT leg = pow(-1, ell / 2) * (2 * ell + 1) * legendre_cache[ell];
            for (int ik = 0; ik < battrs.shape[i]; ik++) {
                FLOAT k = 0.;
                if (battrs.asize[i] > 0) k = battrs.array[i][ik];
                else k = ik * battrs.step[i] + battrs.min[i];
                size_t ibin_loc = (ibin * battrs.shape[i] + ik) * battrs.shape[i + 1] + ill;
                FLOAT leg_bessel = leg * get_bessel(ell, k * s);
                for (size_t iweight = 0; iweight < wsize; iweight++) {
                    atomicAdd(&(counts[ibin_loc + iweight * battrs.size]), weight[iweight] * leg_bessel);
                }
            }
        }
    }
}

__global__ void count2_angular_kernel(FLOAT *block_counts, size_t csize, Mesh mesh1, Mesh mesh2, BinAttrs battrs, WeightAttrs wattrs) {

    size_t tid = threadIdx.x;

    // Initialize local histogram
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];
    // Zero initialize histogram for this block
    for (int i = tid; i < csize; i += blockDim.x) local_counts[i] = 0;

    __syncthreads();
    // Global thread index
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    // Process particles
    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * ii]);
        int bounds[2 * NDIM];
        set_angular_bounds(sposition1, bounds);
        for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
            int icth_n = icth * device_mattrs.meshsize[1];
            for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
                int iphi_true = (iphi + device_mattrs.meshsize[1]) % device_mattrs.meshsize[1];
                int icell = iphi_true + icth_n;
                int np2 = mesh2.nparticles[icell];
                size_t cum2 = mesh2.cumnparticles[icell];
                FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
                FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
                FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);
                for (size_t jj = 0; jj < np2; jj++) {
                    if (!is_selected(sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]))) {
                        continue;
                    }
                    add_weight(local_counts, sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]),
                               value1, &(values2[mesh2.index_value.size * jj]), mesh1.index_value, mesh2.index_value, battrs, wattrs);
                }
            }
        }
    }
}

__global__ void count2_cartesian_kernel(FLOAT *block_counts, size_t csize, Mesh mesh1, Mesh mesh2, BinAttrs battrs, WeightAttrs wattrs) {

    size_t tid = threadIdx.x;

    // Initialize local histogram
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];
    // Zero initialize histogram for this block
    for (int i = tid; i < csize; i += blockDim.x) local_counts[i] = 0;

    __syncthreads();
    // Global thread index
    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    // Process particles
    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *position1 = &(mesh1.positions[NDIM * ii]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * ii]);
        int bounds[2 * NDIM];
        set_cartesian_bounds(position1, bounds);
        //printf("%d %d %d %d %d %d\n", bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
        for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
            int ix_n = (ix % device_mattrs.meshsize[0]) * device_mattrs.meshsize[2] * device_mattrs.meshsize[1];
            for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
                int iy_n = (iy % device_mattrs.meshsize[1]) *  device_mattrs.meshsize[2];
                for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                    int iz_n = iz % device_mattrs.meshsize[2];
                    int icell = ix_n + iy_n + iz_n;
                    int np2 = mesh2.nparticles[icell];
                    size_t cum2 = mesh2.cumnparticles[icell];
                    FLOAT *positions2 = &(mesh2.positions[NDIM * cum2]);
                    FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
                    FLOAT *values2 = &(mesh2.values[mesh2.index_value.size * cum2]);
                    for (size_t jj = 0; jj < np2; jj++) {
                        if (!is_selected(sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]))) {
                            continue;
                        }
                        add_weight(local_counts, sposition1, &(spositions2[NDIM * jj]), position1, &(positions2[NDIM * jj]),
                                   value1, &(values2[mesh2.index_value.size * jj]), mesh1.index_value, mesh2.index_value, battrs, wattrs);
                    }
                }
            }
        }
    }
}


__global__ void reduce_kernel(const FLOAT *block_counts,
                              int nblocks,
                              FLOAT *final_counts,
                              size_t size)
{
    // Flatten thread index across entire grid
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Loop over all i values this thread should handle
    for (size_t i = tid; i < size; i += stride) {
        FLOAT sum = 0;
        for (int b = 0; b < nblocks; ++b) {
            sum += block_counts[(size_t)b * size + i];
        }
        final_counts[i] = sum;
    }
}


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream) {

    // counts expected on the device already
    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count2_cartesian_kernel, nblocks, nthreads_per_block, buffer);

    // CUDA timing eventsis
    cudaEvent_t start, stop;
    float elapsed_time;

    // Determine output array size based on spin parameters
    size_t csize = get_count2_size(list_mesh[0].index_value, list_mesh[1].index_value, NULL) * battrs.size;

    // Initialize histograms
    CUDA_CHECK(cudaMemset(counts, 0, csize * sizeof(FLOAT)));

    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_sattrs, &sattrs, sizeof(SelectionAttrs)));
    //CUDA_CHECK(cudaMemcpyToSymbol(device_battrs, &battrs, sizeof(BinAttrs)));
    BinAttrs device_battrs = battrs;
    for (size_t i = 0; i < battrs.ndim; i++) {
        if (battrs.asize[i] > 0) {
            // printf("ALLOCATING bin %d with size %d\n", i, battrs.asize[i]);
            FLOAT *array = (FLOAT*) my_device_malloc(battrs.asize[i] * sizeof(FLOAT), buffer);
            CUDA_CHECK(cudaMemcpy(array, battrs.array[i], battrs.asize[i] * sizeof(FLOAT), cudaMemcpyHostToDevice));
            device_battrs.array[i] = array;
        }
    }
    WeightAttrs device_wattrs = wattrs;

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
    reduce_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts, nblocks, counts, csize);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    log_message(LOG_LEVEL_INFO, "Time elapsed: %3.1f ms.\n", elapsed_time);

    // Free GPU memory
    my_device_free(block_counts, buffer);

    for (size_t i = 0; i < battrs.ndim; i++) {
        if (battrs.asize[i] > 0) my_device_free(device_battrs.array[i], buffer);
    }

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
