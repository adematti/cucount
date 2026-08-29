#ifndef _CUCOUNT_COUNT2_
#define _CUCOUNT_COUNT2_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"


// count2.h helpers
// code can be 100% faster (e.g. count3close) if e.g. for_each_candidate functions are defined at the same place at the kernels (not sure why)
// so strategy is to define inline functions in macros


typedef struct DeviceCount2Layout {
    size_t nbins;
    size_t csize;
    size_t nells;
    size_t ells[10];
    bool ells_even;
} DeviceCount2Layout;


size_t fill_ells(const BinAttrs *battrs, int index, size_t *ells);


size_t get_count2_weight_names(IndexValue index_value1, IndexValue index_value2,
                        char names[][SIZE_NAME]);


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs,
    const SelectionAttrs sattrs, BinAttrs battrs, WeightAttrs wattrs, SplitAttrs spattrs,
    DeviceMemoryBuffer *buffer, cudaStream_t stream);



#define DEFINE_COMPUTE_UTILS                                                       \
__device__ inline int wrap_periodic_int(int idx, int meshsize)                     \
{                                                                                  \
    int r = idx % meshsize;                                                        \
    return (r < 0) ? r + meshsize : r;                                             \
}                                                                                  \
                                                                                   \
__device__ inline FLOAT wrap_periodic_float(FLOAT dxyz, FLOAT boxsize)             \
{                                                                                  \
    FLOAT half = (FLOAT)0.5 * boxsize;                                             \
    FLOAT x = dxyz + half;                                                         \
    x = fmod(x, boxsize);                                                          \
    if (x < 0) x += boxsize;                                                       \
    return x - half;                                                               \
}                                                                                  \
                                                                                   \
__device__ inline FLOAT dot(const FLOAT *position1, const FLOAT *position2)        \
{                                                                                  \
    FLOAT d = (FLOAT)0.;                                                           \
    for (size_t axis = 0; axis < NDIM; axis++) {                                   \
        d += position1[axis] * position2[axis];                                    \
    }                                                                              \
    return d;                                                                      \
}                                                                                  \
                                                                                   \
__device__ inline void addition(                                                   \
    FLOAT *add,                                                                    \
    const FLOAT *position1,                                                        \
    const FLOAT *position2)                                                        \
{                                                                                  \
    for (size_t axis = 0; axis < NDIM; axis++) {                                   \
        add[axis] = position1[axis] + position2[axis];                             \
    }                                                                              \
}                                                                                  \
                                                                                   \
__device__ inline FLOAT clamp1(FLOAT x)                                            \
{                                                                                  \
    return MIN((FLOAT)1., MAX((FLOAT)-1., x));                                     \
}                                                                                  \
                                                                                   \
__device__ inline void difference(                                                 \
    FLOAT *diff,                                                                   \
    const FLOAT *position1,                                                        \
    const FLOAT *position2,                                                        \
    const MeshAttrs &mattrs)                                                       \
{                                                                                  \
    _Pragma("unroll")                                                              \
    for (int axis = 0; axis < NDIM; axis++) {                                      \
        diff[axis] = position1[axis] - position2[axis];                            \
        if (mattrs.periodic) {                                                     \
            diff[axis] = wrap_periodic_float(diff[axis], mattrs.boxsize[axis]);    \
        }                                                                          \
    }                                                                              \
}                                                                                  \
                                                                                   \
__device__ inline bool is_selected_pair(                                           \
    FLOAT *sposition1,                                                             \
    FLOAT *sposition2,                                                             \
    FLOAT *position1,                                                              \
    FLOAT *position2,                                                              \
    const SelectionAttrs &sattrs,                                                  \
    const MeshAttrs &mattrs)                                                       \
{                                                                                  \
    bool selected = true;                                                          \
    for (size_t i = 0; i < sattrs.ndim; i++) {                                     \
        int var = sattrs.var[i];                                                   \
        if (var == VAR_THETA) {                                                    \
            FLOAT costheta = dot(sposition1, sposition2);                          \
            selected &= (costheta >= sattrs.smin[i]) &&                            \
                        (costheta <= sattrs.smax[i]);                              \
        }                                                                          \
        if (var == VAR_S) {                                                        \
            FLOAT diff[NDIM];                                                      \
            difference(diff, position2, position1, mattrs);                        \
            const FLOAT s2 = dot(diff, diff);                                      \
            selected &= (s2 >= sattrs.smin[i] * sattrs.smin[i]) &&                 \
                        (s2 <= sattrs.smax[i] * sattrs.smax[i]);                   \
        }                                                                          \
    }                                                                              \
    return selected;                                                               \
}                                                                                  \
                                                                                   \
__global__ static void reduce_add_kernel(                                          \
    const FLOAT *block_counts,                                                     \
    size_t nblocks,                                                                \
    FLOAT *counts,                                                                 \
    size_t csize)                                                                  \
{                                                                                  \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    size_t stride = gridDim.x * blockDim.x;                                        \
                                                                                   \
    for (; i < csize; i += stride) {                                               \
        FLOAT sum = (FLOAT)0.;                                                     \
        for (size_t iblock = 0; iblock < nblocks; iblock++) {                      \
            sum += block_counts[iblock * csize + i];                               \
        }                                                                          \
        counts[i] += sum;                                                          \
    }                                                                              \
}                                                                                  \
                                                                                   \
__device__ inline int search_bin_index(                                            \
    FLOAT value,                                                                   \
    const FLOAT *edges,                                                            \
    int nbins)                                                                     \
{                                                                                  \
    if (!edges || nbins <= 0) return -1;                                           \
    if (value < edges[0] || value >= edges[nbins]) return -1;                      \
                                                                                   \
    int lo = 0;                                                                    \
    int hi = nbins;                                                                \
                                                                                   \
    while (lo + 1 < hi) {                                                          \
        int mid = lo + (hi - lo) / 2;                                              \
                                                                                   \
        if (value >= edges[mid]) {                                                 \
            lo = mid;                                                              \
        }                                                                          \
        else {                                                                     \
            hi = mid;                                                              \
        }                                                                          \
    }                                                                              \
                                                                                   \
    return lo;                                                                     \
}                                                                                  \
                                                                                   \
__device__ inline int get_sep_bin_index(                                           \
    FLOAT value,                                                                   \
    const FLOAT *sep,                                                              \
    int shape,                                                                     \
    BIN_TYPE bin,                                                                  \
    bool sep_is_edges)                                                             \
{                                                                                  \
    const int nbins = sep_is_edges ? shape : shape - 1;                            \
                                                                                   \
    if (bin == BIN_CUSTOM) {                                                       \
        return search_bin_index(value, sep, nbins);                                \
    }                                                                              \
                                                                                   \
    const FLOAT min = sep[0];                                                      \
    const FLOAT max = sep[nbins];                                                  \
                                                                                   \
    if (value < min || value >= max) return -1;                                    \
                                                                                   \
    if (bin == BIN_LIN) {                                                          \
        const FLOAT step = sep[1] - sep[0];                                        \
        int ibin = (int)floor((value - min) / step);                               \
        return (ibin >= 0 && ibin < nbins) ? ibin : -1;                            \
    }                                                                              \
                                                                                   \
    if (bin == BIN_LOG) {                                                          \
        if (value <= (FLOAT)0.) return -1;                                         \
        const FLOAT logstep = log(sep[1] / sep[0]);                                \
        int ibin = (int)floor(log(value / min) / logstep);                         \
        return (ibin >= 0 && ibin < nbins) ? ibin : -1;                            \
    }                                                                              \
                                                                                   \
    return -1;                                                                     \
}                                                                                  \
                                                                                   \
__device__ inline int get_bin_index(                                               \
    const BinAttrs *battrs,                                                        \
    int idim,                                                                      \
    FLOAT value)                                                                   \
{                                                                                  \
    return get_sep_bin_index(                                                      \
        value,                                                                     \
        battrs->array[idim],                                                       \
        (int)battrs->shape[idim],                                                  \
        battrs->bin[idim],                                                         \
        true);                                                                     \
}


#define DEFINE_FOR_EACH_CANDIDATE_ANGULAR                                          \
__device__ inline void set_angular_bounds(                                         \
    const FLOAT *sposition,                                                        \
    const MeshAttrs &mattrs,                                                       \
    int *bounds)                                                                   \
{                                                                                  \
    FLOAT cth = sposition[2];                                                      \
    FLOAT phi = atan2(sposition[1], sposition[0]);                                 \
    if (phi < 0) phi += 2 * M_PI;                                                  \
                                                                                   \
    int icth = (cth >= 1)                                                          \
        ? ((int)mattrs.meshsize[0] - 1)                                            \
        : (int)(0.5 * (1 + cth) * mattrs.meshsize[0]);                             \
    int iphi = (int)(0.5 * phi / M_PI * mattrs.meshsize[1]);                       \
                                                                                   \
    FLOAT theta  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.5)) / mattrs.meshsize[0]);  \
    FLOAT th_hi  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.0)) / mattrs.meshsize[0]);  \
    FLOAT th_lo  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 1.0)) / mattrs.meshsize[0]);  \
    FLOAT phi_hi = 2 * M_PI * ((FLOAT)(iphi + 1.0) / mattrs.meshsize[1]);          \
    FLOAT phi_lo = 2 * M_PI * ((FLOAT)(iphi + 0.0) / mattrs.meshsize[1]);          \
    FLOAT smax   = acos(mattrs.smax);                                              \
                                                                                   \
    FLOAT cth_max, cth_min;                                                        \
                                                                                   \
    if (th_hi > M_PI - smax) {                                                     \
        cth_min = -1;                                                              \
        /* window may also wrap the north pole: cos(th_lo - smax) is even in its  \
           argument and would describe a spurious southern cap instead */          \
        cth_max = (th_lo < smax) ? (FLOAT)1. : cos(th_lo - smax);                  \
        bounds[2] = 0;                                                             \
        bounds[3] = (int)mattrs.meshsize[1] - 1;                                   \
    }                                                                              \
    else if (th_lo < smax) {                                                       \
        cth_min = (th_hi + smax > M_PI) ? (FLOAT)-1. : cos(th_hi + smax);          \
        cth_max = 1;                                                               \
        bounds[2] = 0;                                                             \
        bounds[3] = (int)mattrs.meshsize[1] - 1;                                   \
    }                                                                              \
    else {                                                                         \
        FLOAT dphi;                                                                \
        FLOAT calpha = cos(smax);                                                  \
        cth_min = cos(th_hi + smax);                                               \
        cth_max = cos(th_lo - smax);                                               \
                                                                                   \
        if (theta < 0.5 * M_PI) {                                                  \
            FLOAT cth_lo = cos(th_lo);                                             \
            dphi = acos(sqrt((calpha * calpha - cth_lo * cth_lo) /                 \
                             (1 - cth_lo * cth_lo)));                              \
        }                                                                          \
        else {                                                                     \
            FLOAT cth_hi2 = cos(th_hi);                                            \
            dphi = acos(sqrt((calpha * calpha - cth_hi2 * cth_hi2) /               \
                             (1 - cth_hi2 * cth_hi2)));                            \
        }                                                                          \
                                                                                   \
        if (dphi < M_PI) {                                                         \
            FLOAT phi_min = phi_lo - dphi;                                         \
            FLOAT phi_max = phi_hi + dphi;                                         \
            bounds[2] = (int)floor(0.5 * phi_min / M_PI * mattrs.meshsize[1]);     \
            bounds[3] = (int)floor(0.5 * phi_max / M_PI * mattrs.meshsize[1]);     \
        }                                                                          \
        else {                                                                     \
            bounds[2] = 0;                                                         \
            bounds[3] = (int)mattrs.meshsize[1] - 1;                               \
        }                                                                          \
    }                                                                              \
                                                                                   \
    cth_min = MAX(cth_min, mattrs.boxcenter[0] - mattrs.boxsize[0] / 2.);          \
    cth_max = MIN(cth_max, mattrs.boxcenter[0] + mattrs.boxsize[0] / 2.);          \
                                                                                   \
    bounds[0] = (int)(0.5 * (1 + cth_min) * mattrs.meshsize[0]);                   \
    bounds[1] = (int)(0.5 * (1 + cth_max) * mattrs.meshsize[0]);                   \
                                                                                   \
    if (bounds[0] < 0) bounds[0] = 0;                                               \
    if (bounds[1] >= (int)mattrs.meshsize[0]) {                                     \
        bounds[1] = (int)mattrs.meshsize[0] - 1;                                   \
    }                                                                              \
}                                                                                   \
template <typename Op>                                                               \
__device__ inline void for_each_candidate_angular(                                   \
    FLOAT *center_sposition,                                                         \
    Mesh target_mesh,                                                                \
    const MeshAttrs &target_mattrs,                                                  \
    Op &op)                                                                          \
{                                                                                   \
    int bounds[2 * NDIM];                                                            \
    set_angular_bounds(center_sposition, target_mattrs, bounds);                     \
                                                                                    \
    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {                          \
        int icth_n = icth * (int)target_mattrs.meshsize[1];                           \
                                                                                    \
        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {                      \
            int iphi_true = wrap_periodic_int(iphi, (int)target_mattrs.meshsize[1]); \
            int icell = iphi_true + icth_n;                                          \
                                                                                    \
            size_t np  = (size_t)target_mesh.nparticles[icell];                      \
            size_t cum = target_mesh.cumnparticles[icell];                           \
                                                                                    \
            FLOAT *positions  = &(target_mesh.positions[NDIM * cum]);                \
            FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);               \
            FLOAT *values     = &(target_mesh.values[                                \
                target_mesh.index_value.size * cum]);                                \
                                                                                    \
            for (size_t j = 0; j < np; j++) {                                        \
                op(                                                                  \
                    cum + j,                                                         \
                    &(positions[NDIM * j]),                                          \
                    &(spositions[NDIM * j]),                                         \
                    &(values[target_mesh.index_value.size * j])                      \
                );                                                                   \
            }                                                                        \
        }                                                                            \
    }                                                                                \
}                                                                                   \


#define DEFINE_FOR_EACH_CANDIDATE_CARTESIAN                                        \
__device__ inline void set_cartesian_bounds(                                       \
    const FLOAT *position,                                                         \
    const MeshAttrs &mattrs,                                                       \
    int *bounds)                                                                   \
{                                                                                  \
    for (int axis = 0; axis < NDIM; axis++) {                                      \
        int meshsize = (int)mattrs.meshsize[axis];                                 \
        FLOAT offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;          \
        int index = (int)floor(                                                    \
            (position[axis] - offset) * meshsize / mattrs.boxsize[axis]);          \
        index = wrap_periodic_int(index, meshsize);                                \
        int delta = (int)ceil(mattrs.smax / mattrs.boxsize[axis] * meshsize);      \
                                                                                   \
        bounds[2 * axis]     = index - delta;                                      \
        bounds[2 * axis + 1] = index + delta;                                      \
                                                                                   \
        if (mattrs.periodic == 0) {                                                \
            bounds[2 * axis]     = MAX(bounds[2 * axis], 0);                      \
            bounds[2 * axis + 1] = MIN(bounds[2 * axis + 1], meshsize - 1);        \
        }                                                                          \
        else if (2 * delta + 1 >= meshsize) {                                      \
            bounds[2 * axis]     = 0;                                              \
            bounds[2 * axis + 1] = meshsize - 1;                                   \
        }                                                                          \
    }                                                                              \
}                                                                                  \
template <typename Op>                                                               \
__device__ inline void for_each_candidate_cartesian(                                 \
    FLOAT *center_position,                                                          \
    Mesh target_mesh,                                                                \
    const MeshAttrs &target_mattrs,                                                  \
    Op &op)                                                                          \
{                                                                                   \
    int bounds[2 * NDIM];                                                            \
    set_cartesian_bounds(center_position, target_mattrs, bounds);                    \
                                                                                    \
    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {                                \
        int ix_n = wrap_periodic_int(ix, (int)target_mattrs.meshsize[0])             \
                 * (int)target_mattrs.meshsize[2]                                    \
                 * (int)target_mattrs.meshsize[1];                                   \
                                                                                    \
        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {                            \
            int iy_n = wrap_periodic_int(iy, (int)target_mattrs.meshsize[1])          \
                     * (int)target_mattrs.meshsize[2];                               \
                                                                                    \
            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {                        \
                int iz_n = wrap_periodic_int(iz, (int)target_mattrs.meshsize[2]);     \
                int icell = ix_n + iy_n + iz_n;                                      \
                                                                                    \
                size_t np  = (size_t)target_mesh.nparticles[icell];                  \
                size_t cum = target_mesh.cumnparticles[icell];                       \
                                                                                    \
                FLOAT *positions  = &(target_mesh.positions[NDIM * cum]);            \
                FLOAT *spositions = &(target_mesh.spositions[NDIM * cum]);           \
                FLOAT *values     = &(target_mesh.values[                            \
                    target_mesh.index_value.size * cum]);                            \
                                                                                    \
                for (size_t j = 0; j < np; j++) {                                    \
                    op(                                                              \
                        cum + j,                                                     \
                        &(positions[NDIM * j]),                                      \
                        &(spositions[NDIM * j]),                                     \
                        &(values[target_mesh.index_value.size * j])                  \
                    );                                                               \
                }                                                                    \
            }                                                                        \
        }                                                                            \
    }                                                                                \
}                                                                                   \

#define DEFINE_FOR_EACH_CANDIDATE                                                   \
template <MESH_TYPE TARGET_MESH_TYPE, typename Op>                                   \
__device__ inline void for_each_candidate(                                           \
    FLOAT *center_position,                                                          \
    FLOAT *center_sposition,                                                         \
    Mesh target_mesh,                                                                \
    const MeshAttrs &target_mattrs,                                                  \
    Op &op)                                                                          \
{                                                                                   \
    if constexpr (TARGET_MESH_TYPE == MESH_ANGULAR) {                                \
        for_each_candidate_angular(center_sposition, target_mesh, target_mattrs, op);\
    } else if constexpr (TARGET_MESH_TYPE == MESH_CARTESIAN) {                       \
        for_each_candidate_cartesian(center_position, target_mesh, target_mattrs, op);\
    }                                                                                \
}


#define DEFINE_ANGULAR_WEIGHT                                                 \
__device__ inline int get_interp_sep_index(                                    \
    FLOAT x,                                                                  \
    const FLOAT *sep,                                                         \
    int nsep,                                                                 \
    BIN_TYPE bin,                                                             \
    FLOAT *frac)                                                              \
{                                                                             \
    *frac = (FLOAT)0.;                                                        \
    if (!sep || nsep < 2) return -1;                                          \
                                                                              \
    if (x < sep[0] || x > sep[nsep - 1]) return -1;                           \
                                                                              \
    if (x == sep[nsep - 1]) {                                                 \
        *frac = (FLOAT)1.;                                                    \
        return nsep - 2;                                                      \
    }                                                                         \
                                                                              \
    int ibin = get_sep_bin_index(x, sep, nsep, bin, false);                   \
    if (ibin < 0) return -1;                                                  \
                                                                              \
    FLOAT dx = sep[ibin + 1] - sep[ibin];                                     \
    *frac = (dx != (FLOAT)0.) ? (x - sep[ibin]) / dx : (FLOAT)0.;             \
    return ibin;                                                              \
}                                                                             \
                                                                              \
template <int ND>                                                             \
__device__ inline FLOAT lookup_angular_weight(                                \
    const FLOAT (&costheta)[ND],                                              \
    const AngularWeight& angular)                                             \
{                                                                             \
    if (!angular.weight) return (FLOAT)1.;                                    \
                                                                              \
    int i0[ND];                                                               \
    FLOAT frac[ND];                                                           \
                                                                              \
    _Pragma("unroll")                                                         \
    for (int idim = 0; idim < ND; idim++) {                                   \
        if (angular.sep_is_edges[idim]) {                                     \
            i0[idim] = get_sep_bin_index(                                    \
                costheta[idim],                                               \
                angular.sep[idim],                                            \
                (int) angular.shape[idim],                                    \
                angular.bin[idim],                                            \
                true);                                                        \
            frac[idim] = (FLOAT)0.;                                           \
        }                                                                     \
        else {                                                                \
            i0[idim] = get_interp_sep_index(                                  \
                costheta[idim],                                               \
                angular.sep[idim],                                            \
                (int) angular.shape[idim],                                    \
                angular.bin[idim],                                            \
                &frac[idim]);                                                 \
        }                                                                     \
                                                                              \
        if (i0[idim] < 0) return (FLOAT)1.;                                   \
    }                                                                         \
                                                                              \
    bool any_interp = false;                                                  \
                                                                              \
    _Pragma("unroll")                                                         \
    for (int idim = 0; idim < ND; idim++) {                                   \
        any_interp = any_interp || !angular.sep_is_edges[idim];               \
    }                                                                         \
                                                                              \
    if (!any_interp) {                                                        \
        size_t idx = 0;                                                       \
                                                                              \
        _Pragma("unroll")                                                     \
        for (int idim = 0; idim < ND; idim++) {                               \
            idx = idx * (size_t) angular.shape[idim] + (size_t) i0[idim];     \
        }                                                                     \
                                                                              \
        return angular.weight[idx];                                           \
    }                                                                         \
                                                                              \
    FLOAT result = (FLOAT)0.;                                                 \
    const int ncorners = 1 << ND;                                             \
                                                                              \
    for (int icorner = 0; icorner < ncorners; icorner++) {                    \
        size_t idx = 0;                                                       \
        FLOAT wcorner = (FLOAT)1.;                                            \
                                                                              \
        _Pragma("unroll")                                                     \
        for (int idim = 0; idim < ND; idim++) {                               \
            int ibin = i0[idim];                                              \
                                                                              \
            if (!angular.sep_is_edges[idim]) {                                \
                const int upper = (icorner >> idim) & 1;                      \
                ibin += upper;                                                \
                wcorner *= upper                                              \
                    ? frac[idim]                                              \
                    : ((FLOAT)1. - frac[idim]);                               \
            }                                                                 \
                                                                              \
            idx = idx * (size_t) angular.shape[idim] + (size_t) ibin;         \
        }                                                                     \
                                                                              \
        result += wcorner * angular.weight[idx];                              \
    }                                                                         \
                                                                              \
    return result;                                                            \
}


#endif