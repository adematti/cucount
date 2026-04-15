// close_pairs.cu

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "common.h"
#include "utils.h"
#include "count2.h"
#include "count3close.h"

__device__ inline FLOAT clamp1(FLOAT x)
{
    return MIN((FLOAT)1., MAX((FLOAT)-1., x));
}


__device__ inline void normalize(FLOAT *out, const FLOAT *x)
{
    FLOAT norm = 0.;
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        norm += x[icoord] * x[icoord];
    }
    norm = sqrt(norm);

    if (norm > 0.) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            out[icoord] = x[icoord] / norm;
        }
    }
    else {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            out[icoord] = 0.;
        }
    }
}

__device__ inline void cross3(FLOAT *out, const FLOAT *a, const FLOAT *b)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void build_local_frame(const FLOAT *ez_in, FLOAT local_frame[3][NDIM])
{
    FLOAT ref[NDIM];
    if (fabs((double)ez_in[2]) < 0.9) {
        ref[0] = (FLOAT)0.;
        ref[1] = (FLOAT)0.;
        ref[2] = (FLOAT)1.;
    }
    else {
        ref[0] = (FLOAT)1.;
        ref[1] = (FLOAT)0.;
        ref[2] = (FLOAT)0.;
    }

    // ez
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        local_frame[0][icoord] = ez_in[icoord];
    }

    FLOAT proj = 0.;
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        proj += ref[icoord] * local_frame[0][icoord];
    }

    FLOAT tmp[NDIM];
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) {
        tmp[icoord] = ref[icoord] - proj * local_frame[0][icoord];
    }

    // ex
    normalize(local_frame[1], tmp);
    // ey
    cross3(local_frame[2], local_frame[0], local_frame[1]);
}


__device__ inline void compute_pbar_lmax4(FLOAT mu, FLOAT P[5][5])
{
    FLOAT x  = clamp1(mu);
    FLOAT x2 = x * x;
    FLOAT x3 = x2 * x;
    FLOAT x4 = x2 * x2;

    FLOAT s2 = MAX((FLOAT)0., (FLOAT)1. - x2);
    FLOAT s  = sqrt(s2);
    FLOAT s3 = s2 * s;
    FLOAT s4 = s2 * s2;

    #pragma unroll
    for (int l = 0; l <= 4; l++) {
        #pragma unroll
        for (int m = 0; m <= 4; m++) {
            P[l][m] = (FLOAT)0.;
        }
    }

    P[0][0] = (FLOAT)1.;

    P[1][0] = x;
    P[1][1] = -(FLOAT)0.70710678118654752440 * s;

    P[2][0] = ((FLOAT)0.5) * (((FLOAT)3.) * x2 - (FLOAT)1.);
    P[2][1] = -(FLOAT)1.22474487139158904910 * x * s;
    P[2][2] =  (FLOAT)0.61237243569579452455 * s2;

    P[3][0] = ((FLOAT)0.5) * (((FLOAT)5.) * x3 - ((FLOAT)3.) * x);
    P[3][1] = -(FLOAT)0.43301270189221932338 * ((((FLOAT)5.) * x2) - (FLOAT)1.) * s;
    P[3][2] =  (FLOAT)1.36930639376291527536 * x * s2;
    P[3][3] = -(FLOAT)0.55901699437494742410 * s3;

    P[4][0] = ((FLOAT)0.125) * (((FLOAT)35.) * x4 - ((FLOAT)30.) * x2 + (FLOAT)3.);
    P[4][1] = -(FLOAT)0.55901699437494742410 * x * ((((FLOAT)7.) * x2) - (FLOAT)3.) * s;
    P[4][2] =  (FLOAT)0.39528470752104741743 * ((((FLOAT)7.) * x2) - (FLOAT)1.) * s2;
    P[4][3] = -(FLOAT)0.93541434669348534640 * x * s3;
    P[4][4] =  (FLOAT)0.52291251658379721705 * s4;
}

__device__ inline void compute_trig_mmax4(FLOAT c1, FLOAT s1, FLOAT cm[5], FLOAT sm[5])
{
    FLOAT c2 = c1 * c1;
    FLOAT s2 = s1 * s1;

    cm[0] = (FLOAT)1.;
    sm[0] = (FLOAT)0.;

    cm[1] = c1;
    sm[1] = s1;

    cm[2] = ((FLOAT)2.) * c2 - (FLOAT)1.;
    sm[2] = ((FLOAT)2.) * c1 * s1;

    cm[3] = c1 * ((((FLOAT)4.) * c2) - (FLOAT)3.);
    sm[3] = s1 * (((FLOAT)3.) - ((FLOAT)4.) * s2);

    cm[4] = ((FLOAT)8.) * c2 * c2 - ((FLOAT)8.) * c2 + (FLOAT)1.;
    sm[4] = ((FLOAT)4.) * c1 * s1 * ((((FLOAT)2.) * c2) - (FLOAT)1.);
}


void free_device_close_pairs(ClosePairs *cpairs, DeviceMemoryBuffer *buffer)
{
    if (cpairs->offsets) {
        my_device_free(cpairs->offsets, buffer);
        cpairs->offsets = NULL;
    }
    if (cpairs->secondary) {
        my_device_free(cpairs->secondary, buffer);
        cpairs->secondary = NULL;
    }
    cpairs->nprimaries = 0;
    cpairs->npairs = 0;
}


// Per-match operations

struct CountOp {
    size_t count;

    template <class... Args>
    __device__ inline void operator()(size_t, size_t, Args&&...) {
        count++;
    }
};

struct FillOp {
    size_t *secondary;
    size_t out;

    template <class... Args>
    __device__ inline void operator()(size_t, size_t jpacked, Args&&...) {
        secondary[out++] = jpacked;
    }
};

// Kernels


__global__ void count_close_pairs_angular_kernel(
    size_t *pair_counts,
    Mesh mesh1,
    Mesh mesh2)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        CountOp op{0};
        for_each_selected_pair_angular(ii, mesh1, mesh2, op);
        pair_counts[ii] = op.count;
    }
}

__global__ void fill_close_pairs_angular_kernel(
    size_t *secondary,
    const size_t *offsets,
    Mesh mesh1,
    Mesh mesh2)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FillOp op{secondary, offsets[ii]};
        for_each_selected_pair_angular(ii, mesh1, mesh2, op);
    }
}

__global__ void count_close_pairs_cartesian_kernel(
    size_t *pair_counts,
    Mesh mesh1,
    Mesh mesh2)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        CountOp op{0};
        for_each_selected_pair_cartesian(ii, mesh1, mesh2, op);
        pair_counts[ii] = op.count;
    }
}

__global__ void fill_close_pairs_cartesian_kernel(
    size_t *secondary,
    const size_t *offsets,
    Mesh mesh1,
    Mesh mesh2)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ii = gid; ii < mesh1.total_nparticles; ii += stride) {
        FillOp op{secondary, offsets[ii]};
        for_each_selected_pair_cartesian(ii, mesh1, mesh2, op);
    }
}


// Host entry point

void fill_close_pairs(ClosePairs *cpairs, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream)
{
    const Mesh mesh1 = list_mesh[0];
    const Mesh mesh2 = list_mesh[1];

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count_close_pairs_angular_kernel, nblocks, nthreads_per_block, buffer);

    cpairs->nprimaries = mesh1.total_nparticles;
    cpairs->npairs = 0;
    cpairs->offsets = NULL;
    cpairs->secondary = NULL;

    if (cpairs->nprimaries == 0) return;

    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_sattrs, &sattrs, sizeof(SelectionAttrs)));

    size_t *pair_counts = (size_t *)my_device_malloc(cpairs->nprimaries * sizeof(size_t), buffer);
    cpairs->offsets = (size_t *)my_device_malloc((cpairs->nprimaries + 1) * sizeof(size_t), buffer);

    if (mattrs.type == MESH_ANGULAR) {
        count_close_pairs_angular_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            pair_counts, mesh1, mesh2);
    }
    else if (mattrs.type == MESH_CARTESIAN) {
        count_close_pairs_cartesian_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            pair_counts, mesh1, mesh2);
    }
    else {
        log_message(LOG_LEVEL_ERROR, "fill_close_pairs: unsupported mesh type.\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(cpairs->offsets, 0, sizeof(size_t), stream));

    thrust::inclusive_scan(
        thrust::cuda::par.on(stream),
        pair_counts,
        pair_counts + cpairs->nprimaries,
        cpairs->offsets + 1
    );

    CUDA_CHECK(cudaMemcpyAsync(
        &(cpairs->npairs),
        cpairs->offsets + cpairs->nprimaries,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (cpairs->npairs > 0) {
        cpairs->secondary = (size_t *)my_device_malloc(cpairs->npairs * sizeof(size_t), buffer);

        if (mattrs.type == MESH_ANGULAR) {
            fill_close_pairs_angular_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
                cpairs->secondary, cpairs->offsets, mesh1, mesh2);
        }
        else {
            fill_close_pairs_cartesian_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
                cpairs->secondary, cpairs->offsets, mesh1, mesh2);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    my_device_free(pair_counts, buffer);
}


// ============================================================================
// Count3 layout + constant device state
// ============================================================================

typedef struct Count3Layout {
    size_t nbins;      // number of geometric bins
    size_t nprojs;     // number of projections per geometric bin
    size_t csize;      // nbins * nprojs
    size_t nells1;
    size_t nells2;
    size_t ells1[4];   // precomputed ell list for attr 0
    size_t ells2[4];   // precomputed ell list for attr 1
} Count3Layout;

__device__ __constant__ Count3Layout device_layout;


// ============================================================================
// Host helpers
// ============================================================================

static inline size_t fill_ells_host(const BinAttrs *battrs, int index, size_t *ells)
{
    size_t ellmin = (size_t)battrs->min[index];
    size_t ellmax = (size_t)battrs->max[index];
    size_t ellstep = (battrs->asize[index] == 0) ? (size_t)battrs->step[index] : (size_t)1;

    if (ellstep == 0) return 0;

    size_t nells = 0;
    for (size_t ell = ellmin; ell <= ellmax; ell += ellstep) {
        if (ell <= 4 && nells < 4) {
            ells[nells++] = ell;
        }
        if (ell + ellstep < ell) break;  // overflow guard
    }
    return nells;
}

static inline size_t count3_nprojs_from_ells(
    size_t nells1, const size_t *ells1,
    size_t nells2, const size_t *ells2)
{
    size_t nprojs = 0;
    for (size_t i1 = 0; i1 < nells1; i1++) {
        for (size_t i2 = 0; i2 < nells2; i2++) {
            nprojs += (size_t)(2 * MIN(ells1[i1], ells2[i2]) + 1);
        }
    }
    return nprojs;
}

static inline Count3Layout make_count3_layout(BinAttrs *battrs[3])
{
    Count3Layout layout;
    memset(&layout, 0, sizeof(Count3Layout));

    if (battrs[0] == NULL || battrs[1] == NULL) {
        return layout;
    }

    layout.nbins =
        (size_t)battrs[0]->shape[0] *
        (size_t)battrs[1]->shape[0];

    if (battrs[2] != NULL) {
        layout.nbins *= (size_t)battrs[2]->shape[0];
        layout.nprojs = 1;
        layout.csize = layout.nbins;
        return layout;
    }

    if (battrs[0]->var[0] == VAR_POLE && battrs[1]->var[0] == VAR_POLE) {
        layout.nells1 = fill_ells_host(battrs[0], 1, layout.ells1);
        layout.nells2 = fill_ells_host(battrs[1], 1, layout.ells2);
        layout.nprojs = count3_nprojs_from_ells(
            layout.nells1, layout.ells1,
            layout.nells2, layout.ells2
        );
    }
    else {
        layout.nprojs = 1;
    }

    layout.csize = layout.nbins * layout.nprojs;
    return layout;
}


// ============================================================================
// Device helpers
// ============================================================================


__device__ inline int pair_block_size(int ell1, int ell2)
{
    return 2 * MIN(ell1, ell2) + 1;
}


// ============================================================================
// add_weight3
// ============================================================================


__device__ inline void add_weight3(
    FLOAT *counts,
    const FLOAT local_frame[3][NDIM],
    FLOAT *sposition_a,
    FLOAT *sposition_b,
    FLOAT *sposition_c,
    FLOAT *position_a,
    FLOAT *position_b,
    FLOAT *position_c,
    FLOAT *value_a,
    FLOAT *value_b,
    FLOAT *value_c,
    IndexValue index_value_a,
    IndexValue index_value_b,
    IndexValue index_value_c,
    const BinAttrs *battrs,
    WeightAttrs wattrs)
{
    if (battrs[0].ndim == 0 || battrs[1].ndim == 0) return;

    const bool has_third = (battrs[2].ndim > 0);
    const int ncoords = has_third ? 3 : 2;

    const VAR_TYPE var0 = battrs[0].var[0];
    const VAR_TYPE var1 = battrs[1].var[0];
    const bool need_pole = (var0 == VAR_POLE);

    if (need_pole != (var1 == VAR_POLE)) return;

    if (has_third) {
        const VAR_TYPE var2 = battrs[2].var[0];
        if (var0 != var1 || var0 != var2) return;
        if (var0 != VAR_S && var0 != VAR_THETA) return;
    }

    // Pair ordering:
    // 0 -> ab
    // 1 -> ac
    // 2 -> bc
    const FLOAT *spos1[3] = {sposition_a, sposition_a, sposition_b};
    const FLOAT *spos2[3] = {sposition_b, sposition_c, sposition_c};

    const FLOAT *pos1[3] = {position_a, position_a, position_b};
    const FLOAT *pos2[3] = {position_b, position_c, position_c};

    FLOAT costheta[3];
    FLOAT diff[3][NDIM];
    FLOAT r[3] = {(FLOAT)0., (FLOAT)0., (FLOAT)0.};

    #pragma unroll
    for (int icoord = 0; icoord < 3; icoord++) {
        costheta[icoord] = clamp1(dot((FLOAT*)spos1[icoord], (FLOAT*)spos2[icoord]));
    }

    size_t ibin = 0;

    #pragma unroll
    for (int icoord = 0; icoord < 3; icoord++) {
        if (icoord >= ncoords) continue;

        const BinAttrs *battr = &battrs[icoord];
        const VAR_TYPE var = battr->var[0];
        FLOAT value;

        if (var == VAR_S || var == VAR_POLE) {
            difference(diff[icoord], (FLOAT*)pos2[icoord], (FLOAT*)pos1[icoord]);
            r[icoord] = sqrt(dot(diff[icoord], diff[icoord]));
            value = r[icoord];
        }
        else if (var == VAR_THETA) {
            value = acos(costheta[icoord]) / DTORAD;
        }
        else {
            return;
        }

        int ib = get_bin_index(battr, 0, value);
        if (ib < 0) return;

        ibin = ibin * (size_t)battr->shape[0] + (size_t)ib;
    }

    if (!has_third && need_pole) {
        if (device_layout.nells1 == 0 || device_layout.nells2 == 0) return;
        if (r[0] <= (FLOAT)0. || r[1] <= (FLOAT)0.) return;
    }

    FLOAT triplet_weight = (FLOAT)1.;

    if (index_value_a.size_individual_weight) {
        triplet_weight *= value_a[index_value_a.start_individual_weight];
    }
    if (index_value_b.size_individual_weight) {
        triplet_weight *= value_b[index_value_b.start_individual_weight];
    }
    if (index_value_c.size_individual_weight) {
        triplet_weight *= value_c[index_value_c.start_individual_weight];
    }

    {
        AngularWeight angular = wattrs.angular;
        if (angular.size) {
            triplet_weight *= lookup_angular_weight<3>(costheta, angular);
        }
    }

    if (has_third || !need_pole) {
        atomicAdd(&counts[ibin * device_layout.nprojs], triplet_weight);
        return;
    }

    FLOAT rhat[2][NDIM];
    #pragma unroll
    for (int ivec = 0; ivec < 2; ivec++) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            rhat[ivec][icoord] = diff[ivec][icoord] / r[ivec];
        }
    }

    const FLOAT *ez = local_frame[0];
    const FLOAT *ex = local_frame[1];
    const FLOAT *ey = local_frame[2];

    FLOAT mu[2] = {0., 0.};
    #pragma unroll
    for (int ivec = 0; ivec < 2; ivec++) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            mu[ivec] += rhat[ivec][icoord] * ez[icoord];
        }
        mu[ivec] = clamp1(mu[ivec]);
    }

    FLOAT xy[2][2] = {{0., 0.}, {0., 0.}};
    #pragma unroll
    for (int ivec = 0; ivec < 2; ivec++) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) {
            xy[ivec][0] += rhat[ivec][icoord] * ex[icoord];
            xy[ivec][1] += rhat[ivec][icoord] * ey[icoord];
        }
    }

    FLOAT rho[2];
    #pragma unroll
    for (int ivec = 0; ivec < 2; ivec++) {
        rho[ivec] = sqrt(MAX((FLOAT)0., (FLOAT)1. - mu[ivec] * mu[ivec]));
    }

    FLOAT cdphi = (FLOAT)1.;
    FLOAT sdphi = (FLOAT)0.;
    if (rho[0] > (FLOAT)1e-12 && rho[1] > (FLOAT)1e-12) {
        FLOAT inv = (FLOAT)1. / (rho[0] * rho[1]);
        cdphi = clamp1((xy[0][0] * xy[1][0] + xy[0][1] * xy[1][1]) * inv);
        sdphi = MIN((FLOAT)1., MAX((FLOAT)-1.,
            (xy[0][0] * xy[1][1] - xy[0][1] * xy[1][0]) * inv));
    }

    FLOAT P[2][5][5], cm[5], sm[5];
    compute_pbar_lmax4(mu[0], P[0]);
    compute_pbar_lmax4(mu[1], P[1]);
    compute_trig_mmax4(cdphi, sdphi, cm, sm);

    FLOAT *counts_bin = counts + ibin * device_layout.nprojs;

    size_t iproj = 0;
    for (size_t i1 = 0; i1 < device_layout.nells1; i1++) {
        int ell1 = (int)device_layout.ells1[i1];
        for (size_t i2 = 0; i2 < device_layout.nells2; i2++) {
            int ell2 = (int)device_layout.ells2[i2];
            int mmax = MIN(ell1, ell2);

            for (int m = 0; m <= 4; m++) {
                if (m > mmax) continue;
                FLOAT amp = triplet_weight * P[0][ell1][m] * P[1][ell2][m];
                atomicAdd(&counts_bin[iproj + (size_t)m], amp * cm[m]);
            }

            for (int m = 1; m <= 4; m++) {
                if (m > mmax) continue;
                FLOAT amp = triplet_weight * P[0][ell1][m] * P[1][ell2][m];
                atomicAdd(&counts_bin[iproj + (size_t)(mmax + m)], amp * sm[m]);
            }

            iproj += (size_t)pair_block_size(ell1, ell2);
        }
    }
}

// ============================================================================
// Third-point traversal helpers
// ============================================================================

template <typename Op>
__device__ inline void for_each_candidate_from_a_angular(
    size_t ia,
    Mesh mesh_a,
    Mesh mesh_x,
    Op& op)
{
    FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);

    int bounds[2 * NDIM];
    set_angular_bounds(sposition_a, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * device_mattrs.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)device_mattrs.meshsize[1]);
            int icell = iphi_true + icth_n;

            int npx = mesh_x.nparticles[icell];
            size_t cumx = mesh_x.cumnparticles[icell];

            FLOAT *positions_x  = &(mesh_x.positions[NDIM * cumx]);
            FLOAT *spositions_x = &(mesh_x.spositions[NDIM * cumx]);
            FLOAT *values_x     = &(mesh_x.values[mesh_x.index_value.size * cumx]);

            for (int jx = 0; jx < npx; jx++) {
                size_t ix = cumx + (size_t)jx;
                FLOAT *position_x  = &(positions_x[NDIM * jx]);
                FLOAT *sposition_x = &(spositions_x[NDIM * jx]);
                FLOAT *value_x     = &(values_x[mesh_x.index_value.size * jx]);

                op(ix, position_x, sposition_x, value_x);
            }
        }
    }
}

template <typename Op>
__device__ inline void for_each_candidate_from_a_cartesian(
    size_t ia,
    Mesh mesh_a,
    Mesh mesh_x,
    Op& op)
{
    FLOAT *position_a = &(mesh_a.positions[NDIM * ia]);

    int bounds[2 * NDIM];
    set_cartesian_bounds(position_a, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int)device_mattrs.meshsize[0])
                 * device_mattrs.meshsize[2] * device_mattrs.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int)device_mattrs.meshsize[1])
                     * device_mattrs.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int)device_mattrs.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int npx = mesh_x.nparticles[icell];
                size_t cumx = mesh_x.cumnparticles[icell];

                FLOAT *positions_x  = &(mesh_x.positions[NDIM * cumx]);
                FLOAT *spositions_x = &(mesh_x.spositions[NDIM * cumx]);
                FLOAT *values_x     = &(mesh_x.values[mesh_x.index_value.size * cumx]);

                for (int jx = 0; jx < npx; jx++) {
                    size_t ixp = cumx + (size_t)jx;
                    FLOAT *position_x  = &(positions_x[NDIM * jx]);
                    FLOAT *sposition_x = &(spositions_x[NDIM * jx]);
                    FLOAT *value_x     = &(values_x[mesh_x.index_value.size * jx]);

                    op(ixp, position_x, sposition_x, value_x);
                }
            }
        }
    }
}


// ============================================================================
// AB-close pass
// ============================================================================

struct Count3ABOp {
    FLOAT *local_counts;

    FLOAT *position_a;
    FLOAT *sposition_a;
    FLOAT *position_b;
    FLOAT *sposition_b;
    FLOAT *value_a;
    FLOAT *value_b;

    FLOAT local_frame[3][NDIM];

    Mesh mesh_a;
    Mesh mesh_b;
    Mesh mesh_c;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t ic,
        FLOAT *position_c,
        FLOAT *sposition_c,
        FLOAT *value_c)
    {
        (void) ic;

        add_weight3(
            local_counts,
            local_frame,
            sposition_a, sposition_b, sposition_c,
            position_a, position_b, position_c,
            value_a, value_b, value_c,
            mesh_a.index_value, mesh_b.index_value, mesh_c.index_value,
            battrs, wattrs);
    }
};


__global__ void count3_ab_close_angular_kernel(
    FLOAT *block_counts,
    size_t csize,
    ClosePairs close_ab,
    Mesh mesh_a,
    Mesh mesh_b,
    Mesh mesh_c,
    const BinAttrs *battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }
    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ia = gid; ia < close_ab.nprimaries; ia += stride) {
        FLOAT *position_a  = &(mesh_a.positions[NDIM * ia]);
        FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);
        FLOAT *value_a     = &(mesh_a.values[mesh_a.index_value.size * ia]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition_a, local_frame);

        for (size_t p = close_ab.offsets[ia]; p < close_ab.offsets[ia + 1]; p++) {
            size_t ib = close_ab.secondary[p];

            FLOAT *position_b  = &(mesh_b.positions[NDIM * ib]);
            FLOAT *sposition_b = &(mesh_b.spositions[NDIM * ib]);
            FLOAT *value_b     = &(mesh_b.values[mesh_b.index_value.size * ib]);

            Count3ABOp op{
                local_counts,
                position_a, sposition_a,
                position_b, sposition_b,
                value_a, value_b,
                {
                    {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                    {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                    {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                },
                mesh_a, mesh_b, mesh_c,
                battrs, wattrs
            };

            for_each_candidate_from_a_angular(ia, mesh_a, mesh_c, op);
        }
    }
}

__global__ void count3_ab_close_cartesian_kernel(
    FLOAT *block_counts,
    size_t csize,
    ClosePairs close_ab,
    Mesh mesh_a,
    Mesh mesh_b,
    Mesh mesh_c,
    const BinAttrs *battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }
    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ia = gid; ia < close_ab.nprimaries; ia += stride) {
        FLOAT *position_a  = &(mesh_a.positions[NDIM * ia]);
        FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);
        FLOAT *value_a     = &(mesh_a.values[mesh_a.index_value.size * ia]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition_a, local_frame);

        for (size_t p = close_ab.offsets[ia]; p < close_ab.offsets[ia + 1]; p++) {
            size_t ib = close_ab.secondary[p];

            FLOAT *position_b  = &(mesh_b.positions[NDIM * ib]);
            FLOAT *sposition_b = &(mesh_b.spositions[NDIM * ib]);
            FLOAT *value_b     = &(mesh_b.values[mesh_b.index_value.size * ib]);

            Count3ABOp op{
                local_counts,
                position_a, sposition_a,
                position_b, sposition_b,
                value_a, value_b,
                {
                    {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                    {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                    {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                },
                mesh_a, mesh_b, mesh_c,
                battrs, wattrs
            };

            for_each_candidate_from_a_cartesian(ia, mesh_a, mesh_c, op);
        }
    }
}


// ============================================================================
// AC-close pass
// De-dup rule: if AB is also close, skip here because AB wins.
// ============================================================================

struct Count3ACOp {
    FLOAT *local_counts;

    FLOAT *position_a;
    FLOAT *sposition_a;
    FLOAT *position_c;
    FLOAT *sposition_c;
    FLOAT *value_a;
    FLOAT *value_c;

    FLOAT local_frame[3][NDIM];

    Mesh mesh_a;
    Mesh mesh_b;
    Mesh mesh_c;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t ib,
        FLOAT *position_b,
        FLOAT *sposition_b,
        FLOAT *value_b)
    {
        (void) ib;

        // AB wins over AC when both are close
        if (is_selected_pair(sposition_a, sposition_b, position_a, position_b)) return;

        add_weight3(
            local_counts,
            local_frame,
            sposition_a, sposition_b, sposition_c,
            position_a, position_b, position_c,
            value_a, value_b, value_c,
            mesh_a.index_value, mesh_b.index_value, mesh_c.index_value,
            battrs, wattrs);
    }
};

__global__ void count3_ac_close_angular_kernel(
    FLOAT *block_counts,
    size_t csize,
    ClosePairs close_ac,
    Mesh mesh_a,
    Mesh mesh_b,
    Mesh mesh_c,
    const BinAttrs *battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }
    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ia = gid; ia < close_ac.nprimaries; ia += stride) {
        FLOAT *position_a  = &(mesh_a.positions[NDIM * ia]);
        FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);
        FLOAT *value_a     = &(mesh_a.values[mesh_a.index_value.size * ia]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition_a, local_frame);

        for (size_t p = close_ac.offsets[ia]; p < close_ac.offsets[ia + 1]; p++) {
            size_t ic = close_ac.secondary[p];

            FLOAT *position_c  = &(mesh_c.positions[NDIM * ic]);
            FLOAT *sposition_c = &(mesh_c.spositions[NDIM * ic]);
            FLOAT *value_c     = &(mesh_c.values[mesh_c.index_value.size * ic]);

            Count3ACOp op{
                local_counts,
                position_a, sposition_a,
                position_c, sposition_c,
                value_a, value_c,
                {
                    {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                    {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                    {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                },
                mesh_a, mesh_b, mesh_c,
                battrs, wattrs
            };

            for_each_candidate_from_a_angular(ia, mesh_a, mesh_b, op);
        }
    }
}

__global__ void count3_ac_close_cartesian_kernel(
    FLOAT *block_counts,
    size_t csize,
    ClosePairs close_ac,
    Mesh mesh_a,
    Mesh mesh_b,
    Mesh mesh_c,
    const BinAttrs *battrs,
    WeightAttrs wattrs)
{
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }
    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t ia = gid; ia < close_ac.nprimaries; ia += stride) {
        FLOAT *position_a  = &(mesh_a.positions[NDIM * ia]);
        FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);
        FLOAT *value_a     = &(mesh_a.values[mesh_a.index_value.size * ia]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition_a, local_frame);

        for (size_t p = close_ac.offsets[ia]; p < close_ac.offsets[ia + 1]; p++) {
            size_t ic = close_ac.secondary[p];

            FLOAT *position_c  = &(mesh_c.positions[NDIM * ic]);
            FLOAT *sposition_c = &(mesh_c.spositions[NDIM * ic]);
            FLOAT *value_c     = &(mesh_c.values[mesh_c.index_value.size * ic]);

            Count3ACOp op{
                local_counts,
                position_a, sposition_a,
                position_c, sposition_c,
                value_a, value_c,
                {
                    {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                    {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                    {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                },
                mesh_a, mesh_b, mesh_c,
                battrs, wattrs
            };

            for_each_candidate_from_a_cartesian(ia, mesh_a, mesh_b, op);
        }
    }
}


// ============================================================================
// Higher-level wrapper
// ============================================================================

void count3_close(FLOAT *counts, ClosePairs close_ab, ClosePairs close_ac, const Mesh *list_mesh,
    MeshAttrs mattrs, BinAttrs *battrs[3], WeightAttrs wattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream)
{
    const Mesh mesh_a = list_mesh[0];
    const Mesh mesh_b = list_mesh[1];
    const Mesh mesh_c = list_mesh[2];

    Count3Layout layout = make_count3_layout(battrs);
    size_t csize = layout.csize;

    BinAttrs device_battrs[3];

    for (int i = 0; i < 3; i++) {
        if (battrs[i] != NULL) copy_bin_attrs_to_device(&device_battrs[i], battrs[i], buffer);
        else memset(&device_battrs[i], 0, sizeof(BinAttrs));
    }

    WeightAttrs device_wattrs = wattrs;
    copy_weight_attrs_to_device(&device_wattrs, &wattrs, buffer);

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(count3_ab_close_cartesian_kernel,
                            nblocks, nthreads_per_block, buffer);

    FLOAT *block_counts_ab = (FLOAT*) my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);
    FLOAT *block_counts_ac = (FLOAT*) my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);

    CUDA_CHECK(cudaMemset(counts, 0, csize * sizeof(FLOAT)));

    CUDA_CHECK(cudaMemcpyToSymbol(device_mattrs, &mattrs, sizeof(MeshAttrs)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_layout, &layout, sizeof(Count3Layout)));

    if (mattrs.type == MESH_ANGULAR) {
        count3_ab_close_angular_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ab, csize, close_ab, mesh_a, mesh_b, mesh_c,
            device_battrs, device_wattrs);

        count3_ac_close_angular_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ac, csize, close_ac, mesh_a, mesh_b, mesh_c,
            device_battrs, device_wattrs);
    }
    else {
        count3_ab_close_cartesian_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ab, csize, close_ab, mesh_a, mesh_b, mesh_c,
            device_battrs, device_wattrs);

        count3_ac_close_cartesian_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ac, csize, close_ac, mesh_a, mesh_b, mesh_c,
            device_battrs, device_wattrs);
    }

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts_ab, nblocks, counts, csize);
    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(block_counts_ac, nblocks, counts, csize);

    CUDA_CHECK(cudaDeviceSynchronize());

    my_device_free(block_counts_ab, buffer);
    my_device_free(block_counts_ac, buffer);

    for (int i = 0; i < 3; i++) {
        free_device_bin_attrs(&device_battrs[i], buffer);
    }

    free_device_weight_attrs(&device_wattrs, buffer);

}