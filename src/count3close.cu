#include "common.h"
#include "count2.h"
#include "count3close.h"


// ============================================================================
// Explicit pair selection
// ============================================================================

enum Count3CloseMode {
    COUNT3_MODE_AB = 0,
    COUNT3_MODE_AC = 1
};

__device__ inline bool is_selected_pair(
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *position1,
    FLOAT *position2,
    const SelectionAttrs &sattrs)
{
    bool selected = 1;
    for (size_t i = 0; i < sattrs.ndim; i++) {
        int var = sattrs.var[i];
        if (var == VAR_THETA) {
            FLOAT costheta = dot(sposition1, sposition2);
            selected &= (costheta >= sattrs.smin[i]) && (costheta <= sattrs.smax[i]);
        }
        if (var == VAR_S) {
            FLOAT diff[NDIM];
            difference(diff, position2, position1);
            const FLOAT s2 = dot(diff, diff);
            selected &= (s2 >= sattrs.smin[i] * sattrs.smin[i]) &&
                        (s2 <= sattrs.smax[i] * sattrs.smax[i]);
        }
    }
    return selected;
}


// ============================================================================
// Search bounds helpers with explicit MeshAttrs
// ============================================================================

__device__ inline void set_angular_bounds_from_attrs(
    const FLOAT *sposition,
    const MeshAttrs &mattrs,
    int *bounds)
{
    FLOAT cth = sposition[2];
    FLOAT phi = atan2(sposition[1], sposition[0]);
    if (phi < 0) phi += 2 * M_PI;

    int icth = (cth >= 1)
        ? ((int)mattrs.meshsize[0] - 1)
        : (int)(0.5 * (1 + cth) * mattrs.meshsize[0]);
    int iphi = (int)(0.5 * phi / M_PI * mattrs.meshsize[1]);

    FLOAT theta  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.5)) / mattrs.meshsize[0]);
    FLOAT th_hi  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 0.0)) / mattrs.meshsize[0]);
    FLOAT th_lo  = acos(-1.0 + 2.0 * ((FLOAT)(icth + 1.0)) / mattrs.meshsize[0]);
    FLOAT phi_hi = 2 * M_PI * ((FLOAT)(iphi + 1.0) / mattrs.meshsize[1]);
    FLOAT phi_lo = 2 * M_PI * ((FLOAT)(iphi + 0.0) / mattrs.meshsize[1]);
    FLOAT smax   = acos(mattrs.smax);

    FLOAT cth_max, cth_min;

    if (th_hi > M_PI - smax) {
        cth_min = -1;
        cth_max = cos(th_lo - smax);
        bounds[2] = 0;
        bounds[3] = (int)mattrs.meshsize[1] - 1;
    }
    else if (th_lo < smax) {
        cth_min = cos(th_hi + smax);
        cth_max = 1;
        bounds[2] = 0;
        bounds[3] = (int)mattrs.meshsize[1] - 1;
    }
    else {
        FLOAT dphi;
        FLOAT calpha = cos(smax);
        cth_min = cos(th_hi + smax);
        cth_max = cos(th_lo - smax);

        if (theta < 0.5 * M_PI) {
            FLOAT cth_lo = cos(th_lo);
            dphi = acos(sqrt((calpha * calpha - cth_lo * cth_lo) /
                             (1 - cth_lo * cth_lo)));
        }
        else {
            FLOAT cth_hi2 = cos(th_hi);
            dphi = acos(sqrt((calpha * calpha - cth_hi2 * cth_hi2) /
                             (1 - cth_hi2 * cth_hi2)));
        }

        if (dphi < M_PI) {
            FLOAT phi_min = phi_lo - dphi;
            FLOAT phi_max = phi_hi + dphi;
            bounds[2] = (int)floor(0.5 * phi_min / M_PI * mattrs.meshsize[1]);
            bounds[3] = (int)floor(0.5 * phi_max / M_PI * mattrs.meshsize[1]);
        }
        else {
            bounds[2] = 0;
            bounds[3] = (int)mattrs.meshsize[1] - 1;
        }
    }

    cth_min = MAX(cth_min, mattrs.boxcenter[0] - mattrs.boxsize[0] / 2.);
    cth_max = MIN(cth_max, mattrs.boxcenter[0] + mattrs.boxsize[0] / 2.);

    bounds[0] = (int)(0.5 * (1 + cth_min) * mattrs.meshsize[0]);
    bounds[1] = (int)(0.5 * (1 + cth_max) * mattrs.meshsize[0]);

    if (bounds[0] < 0) bounds[0] = 0;
    if (bounds[1] >= (int)mattrs.meshsize[0]) bounds[1] = (int)mattrs.meshsize[0] - 1;
}


__device__ inline void set_cartesian_bounds_from_attrs(
    const FLOAT *position,
    const MeshAttrs &mattrs,
    int *bounds)
{
    for (int axis = 0; axis < NDIM; axis++) {
        int meshsize = (int)mattrs.meshsize[axis];
        FLOAT offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;
        int index = (int)floor((position[axis] - offset) * meshsize / mattrs.boxsize[axis]);
        index = wrap_periodic_int(index, meshsize);
        int delta = (int)ceil(mattrs.smax / mattrs.boxsize[axis] * meshsize);

        bounds[2 * axis]     = index - delta;
        bounds[2 * axis + 1] = index + delta;

        if (mattrs.periodic == 0) {
            bounds[2 * axis]     = MAX(bounds[2 * axis], 0);
            bounds[2 * axis + 1] = MIN(bounds[2 * axis + 1], meshsize - 1);
        }
        else if (2 * delta + 1 >= meshsize) {
            bounds[2 * axis]     = 0;
            bounds[2 * axis + 1] = meshsize - 1;
        }
    }
}


// ============================================================================
// Existing helpers
// ============================================================================

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

    normalize(local_frame[1], tmp);
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


// ============================================================================
// Count3 layout + constant device state
// ============================================================================

typedef struct Count3Layout {
    size_t nbins;
    size_t nprojs;
    size_t csize;
    size_t nells1;
    size_t nells2;
    size_t ells1[4];
    size_t ells2[4];
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
        if (ell + ellstep < ell) break;
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
// Candidate traversal helpers
// ============================================================================

template <typename Op>
__device__ inline void for_each_close_candidate_from_a_angular(
    size_t ia,
    Mesh mesh_a,
    Mesh mesh_x,
    const MeshAttrs &mattrs_x_close,
    Op &op)
{
    FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);

    int bounds[2 * NDIM];
    set_angular_bounds_from_attrs(sposition_a, mattrs_x_close, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)mattrs_x_close.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)mattrs_x_close.meshsize[1]);
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
__device__ inline void for_each_third_candidate_from_a_angular(
    size_t ia,
    Mesh mesh_a,
    Mesh mesh_x,
    const MeshAttrs &mattrs_x_third,
    Op &op)
{
    FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);

    int bounds[2 * NDIM];
    set_angular_bounds_from_attrs(sposition_a, mattrs_x_third, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)mattrs_x_third.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)mattrs_x_third.meshsize[1]);
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
__device__ inline void for_each_third_candidate_from_a_cartesian(
    size_t ia,
    Mesh mesh_a,
    Mesh mesh_x,
    const MeshAttrs &mattrs_x_third,
    Op &op)
{
    FLOAT *position_a = &(mesh_a.positions[NDIM * ia]);

    int bounds[2 * NDIM];
    set_cartesian_bounds_from_attrs(position_a, mattrs_x_third, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int)mattrs_x_third.meshsize[0])
                 * (int)mattrs_x_third.meshsize[2] * (int)mattrs_x_third.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int)mattrs_x_third.meshsize[1])
                     * (int)mattrs_x_third.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int)mattrs_x_third.meshsize[2]);
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
// Generic close-pass implementation
// ============================================================================

template <Count3CloseMode MODE>
struct Count3CloseOp {
    FLOAT *local_counts;

    FLOAT *position_a;
    FLOAT *sposition_a;
    FLOAT *value_a;

    FLOAT *position_x;   // close-leg partner: b in AB mode, c in AC mode
    FLOAT *sposition_x;
    FLOAT *value_x;

    FLOAT local_frame[3][NDIM];

    Mesh mesh_a;
    Mesh mesh_b;
    Mesh mesh_c;

    SelectionAttrs sattrs_ab;
    SelectionAttrs sattrs_ac;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t iy,
        FLOAT *position_y,
        FLOAT *sposition_y,
        FLOAT *value_y)
    {
        (void)iy;

        if constexpr (MODE == COUNT3_MODE_AB) {
            // x = b, y = c
            if (!is_selected_pair(sposition_a, sposition_x, position_a, position_x, sattrs_ab)) return;

            add_weight3(
                local_counts,
                local_frame,
                sposition_a, sposition_x, sposition_y,
                position_a, position_x, position_y,
                value_a, value_x, value_y,
                mesh_a.index_value, mesh_b.index_value, mesh_c.index_value,
                battrs, wattrs
            );
        }
        else {
            // x = c, y = b
            if (!is_selected_pair(sposition_a, sposition_x, position_a, position_x, sattrs_ac)) return;
            if (is_selected_pair(sposition_a, sposition_y, position_a, position_y, sattrs_ab)) return;

            add_weight3(
                local_counts,
                local_frame,
                sposition_a, sposition_y, sposition_x,
                position_a, position_y, position_x,
                value_a, value_y, value_x,
                mesh_a.index_value, mesh_b.index_value, mesh_c.index_value,
                battrs, wattrs
            );
        }
    }
};


template <Count3CloseMode MODE, bool THIRD_IS_ANGULAR>
struct Count3ProcessCloseOp {
    FLOAT *local_counts;
    size_t ia;

    FLOAT *position_a;
    FLOAT *sposition_a;
    FLOAT *value_a;

    FLOAT local_frame[3][NDIM];

    Mesh mesh_a;
    Mesh mesh_close;
    Mesh mesh_third;
    Mesh mesh_b;
    Mesh mesh_c;

    MeshAttrs mattrs_third;
    SelectionAttrs sattrs_ab;
    SelectionAttrs sattrs_ac;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t ix,
        FLOAT *position_x,
        FLOAT *sposition_x,
        FLOAT *value_x)
    {
        if constexpr (MODE == COUNT3_MODE_AB) {
            if (!is_selected_pair(sposition_a, sposition_x, position_a, position_x, sattrs_ab)) return;
        }
        else {
            if (!is_selected_pair(sposition_a, sposition_x, position_a, position_x, sattrs_ac)) return;
        }

        Count3CloseOp<MODE> op{
            local_counts,
            position_a, sposition_a, value_a,
            position_x, sposition_x, value_x,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh_a, mesh_b, mesh_c,
            sattrs_ab, sattrs_ac,
            battrs, wattrs
        };

        if constexpr (THIRD_IS_ANGULAR) {
            for_each_third_candidate_from_a_angular(ia, mesh_a, mesh_third, mattrs_third, op);
        }
        else {
            for_each_third_candidate_from_a_cartesian(ia, mesh_a, mesh_third, mattrs_third, op);
        }
    }
};


template <Count3CloseMode MODE, bool THIRD_IS_ANGULAR>
__global__ void count3_close_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh_a,
    Mesh mesh_close,
    Mesh mesh_third,
    Mesh mesh_b,
    Mesh mesh_c,
    MeshAttrs mattrs_close,
    MeshAttrs mattrs_third,
    SelectionAttrs sattrs_ab,
    SelectionAttrs sattrs_ac,
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

    for (size_t ia = gid; ia < mesh_a.total_nparticles; ia += stride) {
        FLOAT *position_a  = &(mesh_a.positions[NDIM * ia]);
        FLOAT *sposition_a = &(mesh_a.spositions[NDIM * ia]);
        FLOAT *value_a     = &(mesh_a.values[mesh_a.index_value.size * ia]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition_a, local_frame);

        Count3ProcessCloseOp<MODE, THIRD_IS_ANGULAR> op{
            local_counts,
            ia,
            position_a, sposition_a, value_a,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh_a, mesh_close, mesh_third, mesh_b, mesh_c,
            mattrs_third,
            sattrs_ab, sattrs_ac,
            battrs, wattrs
        };

        for_each_close_candidate_from_a_angular(ia, mesh_a, mesh_close, mattrs_close, op);
    }
}


// ============================================================================
// Higher-level wrapper
// ============================================================================

void count3_close(
    FLOAT *counts,
    Mesh mesh_a,
    Mesh mesh_b_close,
    Mesh mesh_c_close,
    Mesh mesh_b_third,
    Mesh mesh_c_third,
    MeshAttrs mattrs_b_close,
    MeshAttrs mattrs_c_close,
    MeshAttrs mattrs_b_third,
    MeshAttrs mattrs_c_third,
    SelectionAttrs sattrs_ab,
    SelectionAttrs sattrs_ac,
    BinAttrs *battrs[3],
    WeightAttrs wattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream)
{
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
    CONFIGURE_KERNEL_LAUNCH((count3_close_kernel<COUNT3_MODE_AB, false>),
                            nblocks, nthreads_per_block, buffer);

    FLOAT *block_counts_ab = (FLOAT*) my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);
    FLOAT *block_counts_ac = (FLOAT*) my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, csize * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemcpyToSymbol(device_layout, &layout, sizeof(Count3Layout)));

    if (mattrs_c_third.type == MESH_ANGULAR) {
        count3_close_kernel<COUNT3_MODE_AB, true><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ab,
            csize,
            mesh_a,
            mesh_b_close,
            mesh_c_third,
            mesh_b_close,
            mesh_c_third,
            mattrs_b_close,
            mattrs_c_third,
            sattrs_ab,
            sattrs_ac,
            device_battrs,
            device_wattrs
        );
    }
    else if (mattrs_c_third.type == MESH_CARTESIAN) {
        count3_close_kernel<COUNT3_MODE_AB, false><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ab,
            csize,
            mesh_a,
            mesh_b_close,
            mesh_c_third,
            mesh_b_close,
            mesh_c_third,
            mattrs_b_close,
            mattrs_c_third,
            sattrs_ab,
            sattrs_ac,
            device_battrs,
            device_wattrs
        );
    }
    else {
        log_message(LOG_LEVEL_ERROR, "count3_close: unsupported mattrs_c_third.type.\n");
        exit(EXIT_FAILURE);
    }

    if (mattrs_b_third.type == MESH_ANGULAR) {
        count3_close_kernel<COUNT3_MODE_AC, true><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ac,
            csize,
            mesh_a,
            mesh_c_close,
            mesh_b_third,
            mesh_b_third,
            mesh_c_close,
            mattrs_c_close,
            mattrs_b_third,
            sattrs_ab,
            sattrs_ac,
            device_battrs,
            device_wattrs
        );
    }
    else if (mattrs_b_third.type == MESH_CARTESIAN) {
        count3_close_kernel<COUNT3_MODE_AC, false><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts_ac,
            csize,
            mesh_a,
            mesh_c_close,
            mesh_b_third,
            mesh_b_third,
            mesh_c_close,
            mattrs_c_close,
            mattrs_b_third,
            sattrs_ab,
            sattrs_ac,
            device_battrs,
            device_wattrs
        );
    }
    else {
        log_message(LOG_LEVEL_ERROR, "count3_close: unsupported mattrs_b_third.type.\n");
        exit(EXIT_FAILURE);
    }

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts_ab, nblocks, counts, csize);
    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts_ac, nblocks, counts, csize);

    CUDA_CHECK(cudaDeviceSynchronize());

    my_device_free(block_counts_ab, buffer);
    my_device_free(block_counts_ac, buffer);

    for (int i = 0; i < 3; i++) {
        free_device_bin_attrs(&device_battrs[i], buffer);
    }

    free_device_weight_attrs(&device_wattrs, buffer);
}