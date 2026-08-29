#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "count2.h"
// set_legendre, get_bessel, compute_spin_projection_cartesian: shared with the
// CPU backend. Note the spin projection now normalizes in FLOAT precision
// (the pre-extraction code used rsqrtf, truncating the double path to float).
#include "pair_math.h"

using cucount::pairmath::set_legendre;
using cucount::pairmath::get_bessel;
using cucount::pairmath::compute_spin_projection_cartesian;
using cucount::pairmath::compute_pair_mu;
using cucount::pairmath::pair_bitwise_weight;


__device__ __constant__ SplitAttrs device_spattrs;
static __device__ __constant__ DeviceCount2Layout device_layout;


using cucount::pairmath::search_bin_index;
using cucount::pairmath::get_sep_bin_index;
using cucount::pairmath::get_interp_sep_index;
using cucount::pairmath::lookup_angular_weight;
DEFINE_COMPUTE_UTILS


// ============================================================================
// Layout helpers
// ============================================================================



size_t fill_ells(const BinAttrs *battrs, int index, size_t *ells)
{
    size_t ellmin = (size_t)battrs->min[index];
    size_t ellmax = (size_t)battrs->max[index];
    size_t ellstep = (battrs->bin[index] == BIN_LIN) ? (size_t)battrs->step[index] : (size_t)1;

    if (ellstep == 0) return 0;
    size_t nells = 0;

    for (size_t ell = ellmin; ell <= ellmax; ell += ellstep) {
        ells[nells++] = ell;
    }

    return nells;
}


static inline DeviceCount2Layout make_device_count2_layout(
    const IndexValue index_value1,
    const IndexValue index_value2,
    const BinAttrs battrs,
    const SplitAttrs spattrs)
{
    DeviceCount2Layout layout;
    memset(&layout, 0, sizeof(DeviceCount2Layout));

    layout.nbins = battrs.size;

    const size_t nweights = get_count2_weight_names(
        index_value1,
        index_value2,
        NULL);

    layout.csize = nweights * spattrs.size * battrs.size;

    if (battrs.ndim > 0 && battrs.var[battrs.ndim - 1] == VAR_POLE) {
        const int ipole = (int)battrs.ndim - 1;

        layout.nells = fill_ells(&battrs, ipole, layout.ells);

        if (layout.nells > 0) {
            if (layout.nells == 1) {
                layout.ells_even = (layout.ells[0] % 2 == 0);
            }
            else {
                size_t ellstep;

                if (battrs.asize[ipole] == 0) {
                    ellstep = (size_t)battrs.step[ipole];
                }
                else {
                    ellstep = 1;
                }

                layout.ells_even = ((layout.ells[0] % 2 == 0) && (ellstep % 2 == 0));
            }
        }
    }

    return layout;
}


// ============================================================================
// Accumulation
// ============================================================================

template <int NSPLIT_TARGETS>
__device__ inline void accumulate_weight2(
    FLOAT *counts,
    const FLOAT *weight,
    size_t wsize,
    size_t bin_loc,
    const size_t *split_targets,
    FLOAT factor = 1.)
{
    if constexpr (NSPLIT_TARGETS == 0) {
        for (size_t iweight = 0; iweight < wsize; iweight++) {
            atomicAdd(
                &(counts[bin_loc + iweight * device_layout.nbins]),
                weight[iweight] * factor);
        }
    }
    else if constexpr (NSPLIT_TARGETS == 1) {
        const size_t weight_stride =
            device_spattrs.size * device_layout.nbins;

        const size_t offset0 =
            split_targets[0] * device_layout.nbins + bin_loc;

        for (size_t iweight = 0; iweight < wsize; iweight++) {
            atomicAdd(
                &(counts[offset0 + iweight * weight_stride]),
                weight[iweight] * factor);
        }
    }
    else if constexpr (NSPLIT_TARGETS == 2) {
        const size_t weight_stride =
            device_spattrs.size * device_layout.nbins;

        const size_t offset0 =
            split_targets[0] * device_layout.nbins + bin_loc;

        const size_t offset1 =
            split_targets[1] * device_layout.nbins + bin_loc;

        for (size_t iweight = 0; iweight < wsize; iweight++) {
            const FLOAT w = weight[iweight] * factor;

            atomicAdd(&(counts[offset0 + iweight * weight_stride]), w);
            atomicAdd(&(counts[offset1 + iweight * weight_stride]), w);
        }
    }
}


__device__ inline void add_weight2(
    FLOAT *counts,
    const FLOAT *sposition1,
    const FLOAT *sposition2,
    const FLOAT *position1,
    const FLOAT *position2,
    const FLOAT *value1,
    const FLOAT *value2,
    const IndexValue index_value1,
    const IndexValue index_value2,
    const BinAttrs &battrs,
    const WeightAttrs &wattrs,
    const MeshAttrs &mattrs)
{
    int nsplit_targets = 0;
    size_t split_targets[2] = {0, 0};

    if (index_value1.size_split && index_value2.size_split) {
        if (device_spattrs.mode == SPLIT_JACKKNIFE) {
            const INT split1 =
                *((INT *) &(value1[index_value1.start_split]));

            const INT split2 =
                *((INT *) &(value2[index_value2.start_split]));

            if (split2 == split1) {
                nsplit_targets = 1;
                split_targets[0] = (size_t)split1;
            }
            else {
                nsplit_targets = 2;
                split_targets[0] =
                    (size_t)device_spattrs.nsplits + (size_t)split1;

                split_targets[1] =
                    (size_t)device_spattrs.nsplits * 2 + (size_t)split2;
            }
        }
    }

    FLOAT diff[NDIM];
    difference(diff, position2, position1, mattrs);

    const FLOAT s2 = dot(diff, diff);
    const FLOAT DEFAULT_VALUE = -1000.;

    FLOAT s   = DEFAULT_VALUE;
    FLOAT mu  = DEFAULT_VALUE;
    FLOAT mu2 = DEFAULT_VALUE;

    LOS_TYPE los = LOS_NONE;
    VAR_TYPE var = VAR_NONE;

    bool REQUIRED_S = 0;
    bool REQUIRED_MU = 0;
    bool REQUIRED_MU2 = 0;

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

            if (!device_layout.ells_even) {
                REQUIRED_MU = 1;
            }
        }
    }

    REQUIRED_S |= REQUIRED_MU;

    if (REQUIRED_S) {
        s = sqrt(s2);
    }

    if (REQUIRED_MU2 || REQUIRED_MU) {
        // Shared with the CPU backend (pair_math.h); lifted verbatim.
        compute_pair_mu(
            diff,
            sposition1,
            sposition2,
            position1,
            position2,
            los,
            s,
            s2,
            (bool)REQUIRED_MU,
            &mu,
            &mu2);
    }

    size_t ibin = 0;

    for (i = 0; i < battrs.ndim; i++) {
        var = battrs.var[i];

        FLOAT value = 0.;

        if (var == VAR_S) {
            value = s;
        }
        else if (var == VAR_MU) {
            value = mu;
        }
        else if (var == VAR_THETA) {
            value = acos(dot(sposition1, sposition2)) / DTORAD;
        }
        else if (var == VAR_PI) {
            value = mu * s;
        }
        else if (var == VAR_RP) {
            value = (s2 <= 0.) ? 0. : sqrt(s2 - s2 * mu2);
        }

        if ((var != VAR_POLE) && (var != VAR_K)) {
            int ibin_loc = get_bin_index(&battrs, i, value);

            if (ibin_loc < 0) {
                return;
            }

            ibin = ibin * (size_t)battrs.shape[i] + (size_t)ibin_loc;
        }
        else {
            break;
        }
    }

    FLOAT pair_weight = 1.;

    if (index_value1.size_individual_weight) {
        pair_weight *= value1[index_value1.start_individual_weight];
    }

    if (index_value2.size_individual_weight) {
        pair_weight *= value2[index_value2.start_individual_weight];
    }

    if (index_value1.size_bitwise_weight &&
        index_value2.size_bitwise_weight) {
        // Shared with the CPU backend (pair_math.h); lifted verbatim.
        pair_weight *= pair_bitwise_weight(
            value1,
            value2,
            index_value1,
            index_value2,
            wattrs.bitwise);
    }

    {
        AngularWeight angular = wattrs.angular;

        if (angular.size) {
            FLOAT ct[1] = {dot(sposition1, sposition2)};
            pair_weight *= lookup_angular_weight<1>(ct, angular);
        }
    }

    if (index_value1.size_negative_weight &&
        index_value2.size_negative_weight) {
        FLOAT pair_nweight =
            value1[index_value1.start_negative_weight] *
            value2[index_value2.start_negative_weight];

        pair_weight -= pair_nweight;
    }

    FLOAT weight[MAX_NWEIGHT];
    size_t wsize = 1;

    FLOAT splus1, scross1;
    FLOAT splus2, scross2;

    if (index_value1.size_spin) {
        compute_spin_projection_cartesian(
            sposition1,
            sposition2,
            &(value1[index_value1.start_spin]),
            wattrs.spin[0],
            &splus1,
            &scross1);
    }

    if (index_value2.size_spin) {
        compute_spin_projection_cartesian(
            sposition1,
            sposition2,
            &(value2[index_value2.start_spin]),
            wattrs.spin[1],
            &splus2,
            &scross2);
    }

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

    const int ellstep_legendre = device_layout.ells_even ? 2 : 1;

    if (i == battrs.ndim) {
        if (nsplit_targets == 0) {
            accumulate_weight2<0>(counts, weight, wsize, ibin, split_targets);
        }
        else if (nsplit_targets == 1) {
            accumulate_weight2<1>(counts, weight, wsize, ibin, split_targets);
        }
        else {
            accumulate_weight2<2>(counts, weight, wsize, ibin, split_targets);
        }
    }
    else if ((i == battrs.ndim - 1) && (var == VAR_POLE)) {
        FLOAT legendre_cache[MAX_POLE + 1];

        set_legendre(
            legendre_cache,
            device_layout.ells[0],
            device_layout.ells[device_layout.nells - 1],
            ellstep_legendre,
            mu,
            mu2);

        for (size_t ill = 0; ill < device_layout.nells; ++ill) {
            const size_t ell = device_layout.ells[ill];
            const size_t bin_loc = ibin * device_layout.nells + ill;
            const FLOAT leg = (2 * ell + 1) * legendre_cache[ell];

            if (nsplit_targets == 0) {
                accumulate_weight2<0>(
                    counts, weight, wsize, bin_loc, split_targets, leg);
            }
            else if (nsplit_targets == 1) {
                accumulate_weight2<1>(
                    counts, weight, wsize, bin_loc, split_targets, leg);
            }
            else {
                accumulate_weight2<2>(
                    counts, weight, wsize, bin_loc, split_targets, leg);
            }
        }
    }
    else if ((i == battrs.ndim - 2) &&
             (battrs.var[i] == VAR_K) &&
             (battrs.var[i + 1] == VAR_POLE)) {
        size_t ik_dim = i;

        FLOAT legendre_cache[MAX_POLE + 1];

        set_legendre(
            legendre_cache,
            device_layout.ells[0],
            device_layout.ells[device_layout.nells - 1],
            ellstep_legendre,
            mu,
            mu2);

        size_t nk = battrs.shape[ik_dim];
        size_t npole = device_layout.nells;

        for (size_t ill = 0; ill < npole; ++ill) {
            const int ell = (int)device_layout.ells[ill];

            FLOAT leg =
                (((ell / 2) & 1) ? -1.0 : 1.0) *
                (2 * ell + 1) *
                legendre_cache[ell];

            for (size_t ik = 0; ik < nk; ik++) {
                FLOAT k = 0.;

                if (battrs.asize[ik_dim] > 0) {
                    k = battrs.array[ik_dim][ik];
                }
                else {
                    k = ik * battrs.step[ik_dim] + battrs.min[ik_dim];
                }

                const size_t bin_loc =
                    (ibin * nk + ik) * npole + ill;

                const FLOAT leg_bessel =
                    leg * get_bessel(ell, k * s);

                if (nsplit_targets == 0) {
                    accumulate_weight2<0>(
                        counts,
                        weight,
                        wsize,
                        bin_loc,
                        split_targets,
                        leg_bessel);
                }
                else if (nsplit_targets == 1) {
                    accumulate_weight2<1>(
                        counts,
                        weight,
                        wsize,
                        bin_loc,
                        split_targets,
                        leg_bessel);
                }
                else {
                    accumulate_weight2<2>(
                        counts,
                        weight,
                        wsize,
                        bin_loc,
                        split_targets,
                        leg_bessel);
                }
            }
        }
    }
}


// ============================================================================
// Generic candidate traversal
// ============================================================================

DEFINE_FOR_EACH_CANDIDATE_ANGULAR
DEFINE_FOR_EACH_CANDIDATE_CARTESIAN
DEFINE_FOR_EACH_CANDIDATE


// ============================================================================
// Pair counting op
// ============================================================================

struct Count2Op {
    FLOAT *local_counts;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    IndexValue index_value1;
    IndexValue index_value2;

    BinAttrs battrs;
    WeightAttrs wattrs;
    SelectionAttrs sattrs;
    MeshAttrs mattrs;

    __device__ inline void operator()(
        size_t jj,
        FLOAT *position2,
        FLOAT *sposition2,
        FLOAT *value2)
    {
        (void)jj;

        if (!is_selected_pair(
                sposition1,
                sposition2,
                position1,
                position2,
                sattrs,
                mattrs)) {
            return;
        }

        add_weight2(
            local_counts,
            sposition1,
            sposition2,
            position1,
            position2,
            value1,
            value2,
            index_value1,
            index_value2,
            battrs,
            wattrs,
            mattrs);
    }
};


// ============================================================================
// Kernels
// ============================================================================


template <MESH_TYPE TARGET_MESH_TYPE>
__global__ void count2_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh1,
    Mesh mesh2,
    MeshAttrs mattrs,
    SelectionAttrs sattrs,
    BinAttrs battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;

    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (int i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }

    __syncthreads();

    size_t stride = gridDim.x * blockDim.x;
    size_t gid = tid + blockIdx.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FLOAT *position1  = &(mesh1.positions[NDIM * ii]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * ii]);
        FLOAT *value1     = &(mesh1.values[mesh1.index_value.size * ii]);

        Count2Op op{
            local_counts,
            position1,
            sposition1,
            value1,
            mesh1.index_value,
            mesh2.index_value,
            battrs,
            wattrs,
            sattrs,
            mattrs
        };

        for_each_candidate<TARGET_MESH_TYPE>(
            position1,
            sposition1,
            mesh2,
            mattrs,
            op);
    }
}


// ============================================================================
// Host entry point
// ============================================================================

void count2(
    FLOAT *counts,
    const Mesh *list_mesh,
    const MeshAttrs mattrs,
    const SelectionAttrs sattrs,
    BinAttrs battrs,
    WeightAttrs wattrs,
    SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream)
{
    int nblocks, nthreads_per_block;

    if (mattrs.type == MESH_ANGULAR) {
        CONFIGURE_KERNEL_LAUNCH(
            count2_kernel<MESH_ANGULAR>,
            nblocks,
            nthreads_per_block,
            buffer);
    }
    else {
        CONFIGURE_KERNEL_LAUNCH(
            count2_kernel<MESH_CARTESIAN>,
            nblocks,
            nthreads_per_block,
            buffer);
    }

    cudaEvent_t start, stop;
    float elapsed_time;

    DeviceCount2Layout layout = make_device_count2_layout(
        list_mesh[0].index_value,
        list_mesh[1].index_value,
        battrs,
        spattrs);

    const size_t csize = layout.csize;

    CUDA_CHECK(cudaMemset(counts, 0, csize * sizeof(FLOAT)));

    CUDA_CHECK(cudaMemcpyToSymbol(
        device_spattrs,
        &spattrs,
        sizeof(SplitAttrs)));

    CUDA_CHECK(cudaMemcpyToSymbol(
        device_layout,
        &layout,
        sizeof(DeviceCount2Layout)));

    BinAttrs device_battrs;
    copy_bin_attrs_to_device(&device_battrs, &battrs, buffer);

    WeightAttrs device_wattrs = wattrs;
    copy_weight_attrs_to_device(&device_wattrs, &wattrs, buffer);

    FLOAT *block_counts = (FLOAT *)my_device_malloc(
        nblocks * csize * sizeof(FLOAT),
        buffer);

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mattrs.type == MESH_ANGULAR) {
        count2_kernel<MESH_ANGULAR><<<
            nblocks,
            nthreads_per_block,
            0,
            stream>>>(
                block_counts,
                csize,
                list_mesh[0],
                list_mesh[1],
                mattrs,
                sattrs,
                device_battrs,
                device_wattrs);
    }
    else {
        count2_kernel<MESH_CARTESIAN><<<
            nblocks,
            nthreads_per_block,
            0,
            stream>>>(
                block_counts,
                csize,
                list_mesh[0],
                list_mesh[1],
                mattrs,
                sattrs,
                device_battrs,
                device_wattrs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts,
        nblocks,
        counts,
        csize);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    log_message(LOG_LEVEL_DEBUG, "Time elapsed: %3.1f ms.\n", elapsed_time);

    my_device_free(block_counts, buffer);

    free_device_bin_attrs(&device_battrs, buffer);
    free_device_weight_attrs(&device_wattrs, buffer);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}