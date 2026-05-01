#include "common.h"
#include "count2.h"
#include "count3close.h"
#include "count3.h"

/*
 * count3.cu
 *
 * Triplet counts around each primary D1.
 */

static __device__ __constant__ DeviceCount3Layout device_layout;


__device__ inline FLOAT clamp1(FLOAT x)
{
    return MIN((FLOAT)1., MAX((FLOAT)-1., x));
}


enum Count3Leg {
    COUNT3_LEG_12 = 0,
    COUNT3_LEG_13 = 1
};


__device__ inline void add_pair_weight(
    FLOAT *hist,
    const FLOAT local_frame[3][NDIM],
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *position1,
    FLOAT *position2,
    FLOAT *value2,
    IndexValue index_value2,
    BinAttrs battrs,
    Count3Leg leg,
    const MeshAttrs &mattrs)
{
    if (battrs.ndim == 0) return;

    const size_t nprojs = (leg == COUNT3_LEG_12) ? device_layout.nprojs1 : device_layout.nprojs2;
    const bool need_pole = (bool)(nprojs > 0);

    FLOAT costheta = clamp1(dot(sposition1, sposition2));

    FLOAT diff[NDIM];
    FLOAT r = (FLOAT)0.;
    FLOAT value;

    const VAR_TYPE var = battrs.var[0];

    if (var == VAR_S || var == VAR_POLE) {
        difference(diff, position2, position1, mattrs);
        r = sqrt(dot(diff, diff));
        value = r;
    }
    else if (var == VAR_THETA) {
        value = acos(costheta) / DTORAD;
    }
    else {
        return;
    }

    int ibin = get_bin_index(&battrs, 0, value);
    if (ibin < 0) return;

    FLOAT weight = (FLOAT)1.;
    if (index_value2.size_individual_weight) {
        weight *= value2[index_value2.start_individual_weight];
    }
    if (index_value2.size_negative_weight) {
        weight -= value2[index_value2.start_negative_weight];
    }

    if (!need_pole) {
        atomicAdd(&hist[ibin], weight);
        return;
    }

    FLOAT rhat[NDIM];

    if (r == (FLOAT)0.) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            rhat[icoord] = local_frame[0][icoord];
        }
    }
    else {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            rhat[icoord] = diff[icoord] / r;
        }
    }

    const FLOAT *ez = local_frame[0];
    const FLOAT *ex = local_frame[1];
    const FLOAT *ey = local_frame[2];

    FLOAT mu = (FLOAT)0.;
    FLOAT x = (FLOAT)0.;
    FLOAT y = (FLOAT)0.;

    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        mu += rhat[icoord] * ez[icoord];
        x  += rhat[icoord] * ex[icoord];
        y  += rhat[icoord] * ey[icoord];
    }

    mu = clamp1(mu);

    FLOAT rho = sqrt(MAX((FLOAT)0., x * x + y * y));
    FLOAT cphi = (FLOAT)1.;
    FLOAT sphi = (FLOAT)0.;

    if (rho > (FLOAT)1e-12) {
        cphi = x / rho;
        sphi = y / rho;
    }

    const size_t nells = (leg == COUNT3_LEG_12) ? device_layout.nells1 : device_layout.nells2;
    const size_t *ells = (leg == COUNT3_LEG_12) ? device_layout.ells1 : device_layout.ells2;

    int global_mmax = 0;
    for (size_t iell = 0; iell < nells; iell++) {
        global_mmax = MAX(global_mmax, (int)ells[iell]);
    }

    FLOAT cm[MMAX_SIZE], sm[MMAX_SIZE];
    compute_trig_up_to_m(global_mmax, cphi, sphi, cm, sm);

    FLOAT *hist_bin = hist + (size_t)ibin * nprojs;

    size_t iproj = 0;

    for (size_t iell = 0; iell < nells; iell++) {
        int ell = (int)ells[iell];
        int mmax = ell;

        FLOAT P[MMAX_SIZE];
        compute_pbar_row_lmax5(ell, mmax, mu, P);

        for (int m = 0; m <= mmax; m++) {
            atomicAdd(&hist_bin[iproj + (size_t)m], weight * P[m] * cm[m]);

            if (m > 0) {
                atomicAdd(
                    &hist_bin[iproj + (size_t)(mmax + m)],
                    weight * P[m] * sm[m]);
            }
        }

        iproj += (size_t)(2 * mmax + 1);
    }
}


struct Count3PairOp {
    FLOAT *hist;
    const FLOAT (*local_frame)[NDIM];

    FLOAT *sposition1;
    FLOAT *position1;

    MeshAttrs mattrs;
    SelectionAttrs sattrs;
    SelectionAttrs veto;
    BinAttrs battrs;

    IndexValue index_value;
    Count3Leg leg;

    __device__ inline void operator()(
        size_t i,
        FLOAT *position,
        FLOAT *sposition,
        FLOAT *value)
    {
        (void)i;

        if (!is_selected_pair(
                sposition1, sposition,
                position1, position,
                sattrs,
                mattrs)) return;

        if (veto.ndim && is_selected_pair(
                sposition1, sposition,
                position1, position,
                veto,
                mattrs)) return;

        add_pair_weight(
            hist,
            local_frame,
            sposition1,
            sposition,
            position1,
            position,
            value,
            index_value,
            battrs,
            leg,
            mattrs);
    }
};


template <MESH_TYPE MESH_TYPE_2, MESH_TYPE MESH_TYPE_3>
__global__ void count3_kernel(
    FLOAT *block_counts,
    FLOAT *hist2_all,
    FLOAT *hist3_all,
    size_t csize,
    size_t hsize2,
    size_t hsize3,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs)
{
    (void)battrs23;
    (void)wattrs;

    size_t tid = threadIdx.x;
    size_t gtid = blockIdx.x * blockDim.x + tid;
    size_t nthreads_total = gridDim.x * blockDim.x;

    FLOAT *local_counts = &block_counts[blockIdx.x * csize];
    FLOAT *hist2 = hist2_all + gtid * hsize2;
    FLOAT *hist3 = hist3_all + gtid * hsize3;

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = (FLOAT)0.;
    }
    __syncthreads();

    for (size_t i1 = gtid; i1 < mesh1.total_nparticles; i1 += nthreads_total) {
        FLOAT *position1 = &(mesh1.positions[NDIM * i1]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * i1]);

        for (size_t i = 0; i < hsize2; i++) {
            hist2[i] = (FLOAT)0.;
        }
        for (size_t i = 0; i < hsize3; i++) {
            hist3[i] = (FLOAT)0.;
        }

        FLOAT local_frame[3][NDIM];
        LOS_TYPE los = LOS_FIRSTPOINT;

        if (battrs12.ndim > 0 && battrs12.var[0] == VAR_POLE) {
            los = battrs12.los[0];
        }
        else if (battrs13.ndim > 0 && battrs13.var[0] == VAR_POLE) {
            los = battrs13.los[0];
        }

        build_los_frame(sposition1, los, local_frame);

        Count3PairOp op2{
            hist2,
            local_frame,
            sposition1,
            position1,
            mattrs2,
            sattrs12,
            veto12,
            battrs12,
            mesh2.index_value,
            COUNT3_LEG_12
        };

        for_each_candidate<MESH_TYPE_2>(
            position1,
            sposition1,
            mesh2,
            mattrs2,
            op2
        );

        Count3PairOp op3{
            hist3,
            local_frame,
            sposition1,
            position1,
            mattrs3,
            sattrs13,
            veto13,
            battrs13,
            mesh3.index_value,
            COUNT3_LEG_13
        };

        for_each_candidate<MESH_TYPE_3>(
            position1,
            sposition1,
            mesh3,
            mattrs3,
            op3
        );

        FLOAT w1 = (FLOAT)1.;
        if (mesh1.index_value.size_individual_weight) {
            w1 *= value1[mesh1.index_value.start_individual_weight];
        }
        if (mesh1.index_value.size_negative_weight) {
            w1 -= value1[mesh1.index_value.start_negative_weight];
        }

        if (device_layout.nprojs == 0) {
            for (size_t ibin12 = 0; ibin12 < (size_t)battrs12.shape[0]; ibin12++) {
                FLOAT w2 = hist2[ibin12];
                if (w2 == (FLOAT)0.) continue;

                for (size_t ibin13 = 0; ibin13 < (size_t)battrs13.shape[0]; ibin13++) {
                    FLOAT w3 = hist3[ibin13];
                    if (w3 == (FLOAT)0.) continue;

                    size_t ibin = ibin12 * (size_t)battrs13.shape[0] + ibin13;
                    atomicAdd(&local_counts[ibin], w1 * w2 * w3);
                }
            }
        }
        else {
            size_t iproj = 0;
            size_t iproj1 = 0;

            for (size_t iell1 = 0; iell1 < device_layout.nells1; iell1++) {
                int ell1 = (int)device_layout.ells1[iell1];
                size_t iproj2 = 0;

                for (size_t iell2 = 0; iell2 < device_layout.nells2; iell2++) {
                    int ell2 = (int)device_layout.ells2[iell2];
                    int mmax = MIN(ell1, ell2);

                    FLOAT ell_norm = sqrt((FLOAT)((2 * ell1 + 1) * (2 * ell2 + 1)));

                    for (size_t ibin12 = 0; ibin12 < (size_t)battrs12.shape[0]; ibin12++) {
                        FLOAT *hist2_bin = hist2 + ibin12 * device_layout.nprojs1;

                        for (size_t ibin13 = 0; ibin13 < (size_t)battrs13.shape[0]; ibin13++) {
                            FLOAT *hist3_bin = hist3 + ibin13 * device_layout.nprojs2;

                            size_t ibin = ibin12 * (size_t)battrs13.shape[0] + ibin13;
                            FLOAT *counts_bin = local_counts + ibin * device_layout.nprojs;

                            FLOAT c2 = hist2_bin[iproj1];
                            FLOAT c3 = hist3_bin[iproj2];

                            atomicAdd(&counts_bin[iproj], ell_norm * w1 * c2 * c3);

                            for (int m = 1; m <= mmax; m++) {
                                size_t ireal  = iproj + (size_t)m;
                                size_t iimag  = iproj + (size_t)(mmax + m);

                                size_t ireal1 = iproj1 + (size_t)m;
                                size_t iimag1 = iproj1 + (size_t)(ell1 + m);

                                size_t ireal2 = iproj2 + (size_t)m;
                                size_t iimag2 = iproj2 + (size_t)(ell2 + m);

                                FLOAT c2 = hist2_bin[ireal1];
                                FLOAT s2 = hist2_bin[iimag1];

                                FLOAT c3 = hist3_bin[ireal2];
                                FLOAT s3 = hist3_bin[iimag2];

                                atomicAdd(&counts_bin[ireal], ell_norm * w1 * (c2 * c3 + s2 * s3));
                                atomicAdd(&counts_bin[iimag], ell_norm * w1 * (s2 * c3 - c2 * s3));
                            }
                        }
                    }

                    iproj += (size_t)(2 * mmax + 1);
                    iproj2 += (size_t)(2 * ell2 + 1);
                }

                iproj1 += (size_t)(2 * ell1 + 1);
            }
        }
    }
}


#define LAUNCH_COUNT3_KERNEL(MESH_TYPE_2, MESH_TYPE_3)                         \
    count3_kernel<MESH_TYPE_2, MESH_TYPE_3>                                    \
        <<<nblocks, nthreads_per_block, 0, stream>>>(                          \
            block_counts,                                                      \
            hist2_all,                                                         \
            hist3_all,                                                         \
            csize,                                                             \
            hsize2,                                                            \
            hsize3,                                                            \
            mesh1, mesh2, mesh3,                                               \
            mattrs2, mattrs3,                                                  \
            sattrs12, sattrs13,                                                \
            veto12, veto13,                                                    \
            device_battrs12, device_battrs13, device_battrs23,                 \
            wattrs)


void count3(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    BinAttrs battrs12,
    BinAttrs battrs13,
    WeightAttrs wattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream)
{
    BinAttrs battrs23;
    memset(&battrs23, 0, sizeof(BinAttrs));

    DeviceCount3Layout layout = make_device_count3_layout(battrs12, battrs13, battrs23);

    size_t csize = layout.csize;

    BinAttrs device_battrs12 = battrs12;
    BinAttrs device_battrs13 = battrs13;
    BinAttrs device_battrs23 = battrs23;

    copy_bin_attrs_to_device(&device_battrs12, &battrs12, buffer);
    copy_bin_attrs_to_device(&device_battrs13, &battrs13, buffer);

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(
        (count3_kernel<MESH_CARTESIAN, MESH_CARTESIAN>),
        nblocks,
        nthreads_per_block,
        buffer);

    const size_t hsize2 = (size_t)battrs12.shape[0] * ((layout.nprojs1 > 0) ? layout.nprojs1 : 1);
    const size_t hsize3 = (size_t)battrs13.shape[0] * ((layout.nprojs2 > 0) ? layout.nprojs2 : 1);

    const size_t nthreads = (size_t)nblocks * (size_t)nthreads_per_block;

    FLOAT *block_counts = (FLOAT *)my_device_malloc(nblocks * csize * sizeof(FLOAT), buffer);
    FLOAT *hist2_all = (FLOAT *)my_device_malloc(nthreads * hsize2 * sizeof(FLOAT), buffer);
    FLOAT *hist3_all = (FLOAT *)my_device_malloc(nthreads * hsize3 * sizeof(FLOAT), buffer);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, csize * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemcpyToSymbol(device_layout, &layout, sizeof(DeviceCount3Layout)));

    if (mattrs2.type == MESH_ANGULAR && mattrs3.type == MESH_ANGULAR) {
        LAUNCH_COUNT3_KERNEL(MESH_ANGULAR, MESH_ANGULAR);
    }
    else if (mattrs2.type == MESH_ANGULAR && mattrs3.type == MESH_CARTESIAN) {
        LAUNCH_COUNT3_KERNEL(MESH_ANGULAR, MESH_CARTESIAN);
    }
    else if (mattrs2.type == MESH_CARTESIAN && mattrs3.type == MESH_ANGULAR) {
        LAUNCH_COUNT3_KERNEL(MESH_CARTESIAN, MESH_ANGULAR);
    }
    else if (mattrs2.type == MESH_CARTESIAN && mattrs3.type == MESH_CARTESIAN) {
        LAUNCH_COUNT3_KERNEL(MESH_CARTESIAN, MESH_CARTESIAN);
    }
    else {
        log_message(LOG_LEVEL_ERROR, "count3: unsupported mesh type.\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaGetLastError());

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts,
        nblocks,
        counts,
        csize);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    my_device_free(hist3_all, buffer);
    my_device_free(hist2_all, buffer);
    my_device_free(block_counts, buffer);

    free_device_bin_attrs(&device_battrs12, buffer);
    free_device_bin_attrs(&device_battrs13, buffer);
}


#undef LAUNCH_COUNT3_KERNEL