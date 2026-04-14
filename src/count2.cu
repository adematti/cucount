#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"


__device__ __constant__ SplitAttrs device_spattrs;
//__device__ __constant__ BinAttrs device_battrs;


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
            int ibin_loc = get_bin_index(battrs, i, value);
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
        WeightAngular angular = wattrs.angular;
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
            accumulate_weight<0>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
        }
        else if (nsplit_targets == 1) {
            accumulate_weight<1>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
        }
        else {
            accumulate_weight<2>(counts, weight, wsize, ibin, battrs.size, device_spattrs.size, split_targets);
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
                accumulate_weight<0>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
            }
            else if (nsplit_targets == 1) {
                accumulate_weight<1>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
            }
            else {
                accumulate_weight<2>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg);
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
                    accumulate_weight<0>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
                }
                else if (nsplit_targets == 1) {
                    accumulate_weight<1>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
                }
                else {
                    accumulate_weight<2>(counts, weight, wsize, bin_loc, battrs.size, device_spattrs.size, split_targets, leg_bessel);
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
