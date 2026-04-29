#include "common.h"
#include "count2.h"
#include "count3close.h"

/*
 * count3.cu
 *
 * Triplet counts around each primary D1:
 *
 *   for each x1 in D1:
 *     for each selected x2 in D2, binned by r12/theta12:
 *       for each selected x3 in D3, binned by r13/theta13:
 *         counts += w1 w2 w3
 *
 */

static __device__ __constant__ DeviceCount3Layout device_layout;


template <typename Op>
__device__ inline void for_each_candidate_angular(
    FLOAT *center_sposition,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    int bounds[2 * NDIM];
    set_angular_bounds_from_attrs(center_sposition, target_mattrs, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)target_mattrs.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)target_mattrs.meshsize[1]);
            int icell = iphi_true + icth_n;

            int np = target_mesh.nparticles[icell];
            size_t cum = target_mesh.cumnparticles[icell];

            FLOAT *positions = &(target_mesh.positions[NDIM * cum]);
            FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);
            FLOAT *values = &(target_mesh.values[target_mesh.index_value.size * cum]);

            for (int j = 0; j < np; j++) {
                op(cum + (size_t)j,
                   &(positions[NDIM * j]),
                   &(spositions[NDIM * j]),
                   &(values[target_mesh.index_value.size * j]));
            }
        }
    }
}


template <typename Op>
__device__ inline void for_each_candidate_cartesian(
    FLOAT *center_position,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    int bounds[2 * NDIM];
    set_cartesian_bounds_from_attrs(center_position, target_mattrs, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int)target_mattrs.meshsize[0])
                 * (int)target_mattrs.meshsize[2]
                 * (int)target_mattrs.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int)target_mattrs.meshsize[1])
                     * (int)target_mattrs.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int)target_mattrs.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int np = target_mesh.nparticles[icell];
                size_t cum = target_mesh.cumnparticles[icell];

                FLOAT *positions = &(target_mesh.positions[NDIM * cum]);
                FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);
                FLOAT *values = &(target_mesh.values[target_mesh.index_value.size * cum]);

                for (int j = 0; j < np; j++) {
                    op(cum + (size_t)j,
                       &(positions[NDIM * j]),
                       &(spositions[NDIM * j]),
                       &(values[target_mesh.index_value.size * j]));
                }
            }
        }
    }
}


template <MESH_TYPE TARGET_MESH_TYPE, typename Op>
__device__ inline void for_each_candidate(
    FLOAT *center_position,
    FLOAT *center_sposition,
    Mesh target_mesh,
    const MeshAttrs &target_mattrs,
    Op &op)
{
    if constexpr (TARGET_MESH_TYPE == MESH_ANGULAR) {
        for_each_candidate_angular(center_sposition, target_mesh, target_mattrs, op);
    } else if constexpr (TARGET_MESH_TYPE == MESH_CARTESIAN) {
        for_each_candidate_cartesian(center_position, target_mesh, target_mattrs, op);
    }
}


__device__ inline void add_pair_weight(
    FLOAT *hist,
    const FLOAT local_frame[3][NDIM],
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *position1,
    FLOAT *position2,
    FLOAT *value2,
    IndexValue index_value2,
    BinAttrs battrs)
{
    if (battrs.ndim == 0) return;

    const bool need_pole = (bool)(device_layout.nprojs > 0);

    FLOAT costheta = clamp1(dot(sposition1, sposition2));

    FLOAT diff[NDIM];
    FLOAT r = (FLOAT)0.;
    FLOAT value;

    const VAR_TYPE var = battrs.var[0];

    if (var == VAR_S || var == VAR_POLE) {
        difference(diff, position2, position1);
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
        x += rhat[icoord] * ex[icoord];
        y += rhat[icoord] * ey[icoord];
    }

    mu = clamp1(mu);

    FLOAT rho = sqrt(MAX((FLOAT)0., x * x + y * y));
    FLOAT cphi = (FLOAT)1.;
    FLOAT sphi = (FLOAT)0.;

    if (rho > (FLOAT)1e-12) {
        cphi = x / rho;
        sphi = y / rho;
    }

    int global_mmax = 0;
    for (size_t iell = 0; iell < device_layout.nells1; iell++) {
        global_mmax = MAX(global_mmax, (int)device_layout.ells1[iell]);
    }

    FLOAT cm[5], sm[5];
    compute_trig_up_to_m(global_mmax, cphi, sphi, cm, sm);

    FLOAT *hist_bin = hist + (size_t)ibin * device_layout.nprojs;

    size_t iproj = 0;
    for (size_t iell = 0; iell < device_layout.nells1; iell++) {
        int ell = (int)device_layout.ells1[iell];

        FLOAT P[5];
        compute_pbar_row_lmax4(ell, ell, mu, P);

        for (int m = 0; m <= ell; m++) {
            atomicAdd(&hist_bin[iproj + (size_t)m], weight * P[m] * cm[m]);

            if (m > 0) {
                atomicAdd(
                    &hist_bin[iproj + (size_t)(ell + m)],
                    weight * P[m] * sm[m]
                );
            }
        }

        iproj += (size_t)(2 * ell + 1);
    }
}


struct Count3PairOp {
    FLOAT *hist;
    const FLOAT (*local_frame)[NDIM];

    FLOAT *sposition1;
    FLOAT *position1;

    SelectionAttrs sattrs;
    SelectionAttrs veto;
    BinAttrs battrs;

    IndexValue index_value;

    __device__ inline void operator()(
        size_t i,
        FLOAT *position,
        FLOAT *sposition,
        FLOAT *value)
    {
        (void)i;

        if (!is_selected_pair_with_sattrs(
                sposition1, sposition, position1, position, sattrs)) return;

        if (veto.ndim && is_selected_pair_with_sattrs(
                sposition1, sposition, position1, position, veto)) return;

        add_pair_weight(
            hist,
            local_frame,
            sposition1,
            sposition,
            position1,
            position,
            value,
            index_value,
            battrs
        );
    }
};

template <MESH_TYPE MESH_TYPE_2, MESH_TYPE MESH_TYPE_3>
__global__ void count3_kernel(
    FLOAT *block_counts,
    size_t csize,
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
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = (FLOAT)0.;
    }
    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i1 = gid; i1 < mesh1.total_nparticles; i1 += stride) {
        FLOAT *position1 = &(mesh1.positions[NDIM * i1]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);
        FLOAT *value1 = &(mesh1.values[mesh1.index_value.size * i1]);

        for (size_t i = 0; i < csize; i++) {
            hist2[i] = (FLOAT)0.;
            hist3[i] = (FLOAT)0.;
        }

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition1, local_frame);

        Count3PairOp op2{
            hist2,
            local_frame,
            sposition1,
            position1,
            sattrs12,
            veto12,
            battrs12,
            mesh2.index_value
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
            sattrs13,
            veto13,
            battrs13,
            mesh3.index_value
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

        for (size_t i = 0; i < csize; i++) {
            atomicAdd(&local_counts[i], w1 * hist2[i] * hist3[i]);
        }
    }
}


#define LAUNCH_COUNT3_KERNEL(MESH_TYPE_2, MESH_TYPE_3)                         \
    count3_kernel<MESH_TYPE_2, MESH_TYPE_3>                                    \
        <<<nblocks, nthreads_per_block, 0, stream>>>(                          \
            block_counts,                                                      \
            csize,                                                             \
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

    DeviceCount3Layout layout = make_device_count3_layout(
        battrs12, battrs13, battrs23);
    size_t csize = layout.csize;

    BinAttrs device_battrs12 = battrs12;
    BinAttrs device_battrs13 = battrs13;
    BinAttrs device_battrs23 = battrs23;

    copy_bin_attrs_to_device(&device_battrs12, &battrs12, buffer);
    copy_bin_attrs_to_device(&device_battrs13, &battrs13, buffer);

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH((count3_kernel), nblocks, nthreads_per_block, buffer);

    FLOAT *block_counts = (FLOAT *)my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, csize * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemcpyToSymbol(device_layout, &layout, sizeof(DeviceCount3Layout)));

    if (mattrs2.type == MESH_ANGULAR && mattrs3.type == MESH_ANGULAR) {
        LAUNCH_COUNT3_KERNEL(MESH_ANGULAR, MESH_ANGULAR);
    } else if (mattrs2.type == MESH_ANGULAR && mattrs3.type == MESH_CARTESIAN) {
        LAUNCH_COUNT3_KERNEL(MESH_ANGULAR, MESH_CARTESIAN);
    } else if (mattrs2.type == MESH_CARTESIAN && mattrs3.type == MESH_ANGULAR) {
        LAUNCH_COUNT3_KERNEL(MESH_CARTESIAN, MESH_ANGULAR);
    } else if (mattrs2.type == MESH_CARTESIAN && mattrs3.type == MESH_CARTESIAN) {
        LAUNCH_COUNT3_KERNEL(MESH_CARTESIAN, MESH_CARTESIAN);
    } else {
        log_message(LOG_LEVEL_ERROR, "count3: unsupported mesh type.\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaGetLastError());

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts, nblocks, counts, csize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    my_device_free(block_counts, buffer);
    free_device_bin_attrs(&device_battrs12, buffer);
    free_device_bin_attrs(&device_battrs13, buffer);
}

#undef LAUNCH_COUNT3_KERNEL