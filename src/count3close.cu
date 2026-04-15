#include "common.h"
#include "count2.h"
#include "count3close.h"


// ============================================================================
// Explicit pair selection
// ============================================================================

__device__ inline bool is_selected_pair_with_sattrs(
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
    FLOAT *sposition1,
    FLOAT *sposition2,
    FLOAT *sposition3,
    FLOAT *position1,
    FLOAT *position2,
    FLOAT *position3,
    FLOAT *value1,
    FLOAT *value2,
    FLOAT *value3,
    IndexValue index_value1,
    IndexValue index_value2,
    IndexValue index_value3,
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

    const FLOAT *spos1[3] = {sposition1, sposition1, sposition2};
    const FLOAT *spos2[3] = {sposition2, sposition3, sposition3};

    const FLOAT *pos1[3] = {position1, position1, position2};
    const FLOAT *pos2[3] = {position2, position3, position3};

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

    if (index_value1.size_individual_weight) {
        triplet_weight *= value1[index_value1.start_individual_weight];
    }
    if (index_value2.size_individual_weight) {
        triplet_weight *= value2[index_value2.start_individual_weight];
    }
    if (index_value3.size_individual_weight) {
        triplet_weight *= value3[index_value3.start_individual_weight];
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
__device__ inline void for_each_close_candidate_from_1_angular(
    size_t i1,
    Mesh mesh1,
    Mesh mesh2,
    const MeshAttrs &mattrs2,
    Op &op)
{
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);

    int bounds[2 * NDIM];
    set_angular_bounds_from_attrs(sposition1, mattrs2, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)mattrs2.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)mattrs2.meshsize[1]);
            int icell = iphi_true + icth_n;

            int np2 = mesh2.nparticles[icell];
            size_t cum2 = mesh2.cumnparticles[icell];

            FLOAT *positions2  = &(mesh2.positions[NDIM * cum2]);
            FLOAT *spositions2 = &(mesh2.spositions[NDIM * cum2]);
            FLOAT *values2     = &(mesh2.values[mesh2.index_value.size * cum2]);

            for (int j2 = 0; j2 < np2; j2++) {
                size_t i2 = cum2 + (size_t)j2;
                FLOAT *position2  = &(positions2[NDIM * j2]);
                FLOAT *sposition2 = &(spositions2[NDIM * j2]);
                FLOAT *value2     = &(values2[mesh2.index_value.size * j2]);

                op(i2, position2, sposition2, value2);
            }
        }
    }
}

template <typename Op>
__device__ inline void for_each_third_candidate_from_1_angular(
    size_t i1,
    Mesh mesh1,
    Mesh mesh3,
    const MeshAttrs &mattrs3,
    Op &op)
{
    FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);

    int bounds[2 * NDIM];
    set_angular_bounds_from_attrs(sposition1, mattrs3, bounds);

    for (int icth = bounds[0]; icth <= bounds[1]; icth++) {
        int icth_n = icth * (int)mattrs3.meshsize[1];

        for (int iphi = bounds[2]; iphi <= bounds[3]; iphi++) {
            int iphi_true = wrap_periodic_int(iphi, (int)mattrs3.meshsize[1]);
            int icell = iphi_true + icth_n;

            int np3 = mesh3.nparticles[icell];
            size_t cum3 = mesh3.cumnparticles[icell];

            FLOAT *positions3  = &(mesh3.positions[NDIM * cum3]);
            FLOAT *spositions3 = &(mesh3.spositions[NDIM * cum3]);
            FLOAT *values3     = &(mesh3.values[mesh3.index_value.size * cum3]);

            for (int j3 = 0; j3 < np3; j3++) {
                size_t i3 = cum3 + (size_t)j3;
                FLOAT *position3  = &(positions3[NDIM * j3]);
                FLOAT *sposition3 = &(spositions3[NDIM * j3]);
                FLOAT *value3     = &(values3[mesh3.index_value.size * j3]);

                op(i3, position3, sposition3, value3);
            }
        }
    }
}

template <typename Op>
__device__ inline void for_each_third_candidate_from_1_cartesian(
    size_t i1,
    Mesh mesh1,
    Mesh mesh3,
    const MeshAttrs &mattrs3,
    Op &op)
{
    FLOAT *position1 = &(mesh1.positions[NDIM * i1]);

    int bounds[2 * NDIM];
    set_cartesian_bounds_from_attrs(position1, mattrs3, bounds);

    for (int ix = bounds[0]; ix <= bounds[1]; ix++) {
        int ix_n = wrap_periodic_int(ix, (int)mattrs3.meshsize[0])
                 * (int)mattrs3.meshsize[2] * (int)mattrs3.meshsize[1];

        for (int iy = bounds[2]; iy <= bounds[3]; iy++) {
            int iy_n = wrap_periodic_int(iy, (int)mattrs3.meshsize[1])
                     * (int)mattrs3.meshsize[2];

            for (int iz = bounds[4]; iz <= bounds[5]; iz++) {
                int iz_n = wrap_periodic_int(iz, (int)mattrs3.meshsize[2]);
                int icell = ix_n + iy_n + iz_n;

                int np3 = mesh3.nparticles[icell];
                size_t cum3 = mesh3.cumnparticles[icell];

                FLOAT *positions3  = &(mesh3.positions[NDIM * cum3]);
                FLOAT *spositions3 = &(mesh3.spositions[NDIM * cum3]);
                FLOAT *values3     = &(mesh3.values[mesh3.index_value.size * cum3]);

                for (int j3 = 0; j3 < np3; j3++) {
                    size_t i3 = cum3 + (size_t)j3;
                    FLOAT *position3  = &(positions3[NDIM * j3]);
                    FLOAT *sposition3 = &(spositions3[NDIM * j3]);
                    FLOAT *value3     = &(values3[mesh3.index_value.size * j3]);

                    op(i3, position3, sposition3, value3);
                }
            }
        }
    }
}


// ============================================================================
// Generic close-pass implementation
// ============================================================================

struct Count3CloseOp {
    FLOAT *local_counts;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    FLOAT *position2;
    FLOAT *sposition2;
    FLOAT *value2;

    FLOAT local_frame[3][NDIM];

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;
    bool veto13;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i3,
        FLOAT *position3,
        FLOAT *sposition3,
        FLOAT *value3)
    {
        (void)i3;

        if (!is_selected_pair_with_sattrs(sposition1, sposition3, position1, position3, sattrs13)) return;
        if (!is_selected_pair_with_sattrs(sposition2, sposition3, position2, position3, sattrs23)) return;
        if (veto13 && is_selected_pair_with_sattrs(sposition1, sposition3, position1, position3, sattrs12)) return;

        add_weight3(
            local_counts,
            local_frame,
            sposition1, sposition2, sposition3,
            position1, position2, position3,
            value1, value2, value3,
            mesh1.index_value, mesh2.index_value, mesh3.index_value,
            battrs, wattrs
        );
    }
};


template <MESH_TYPE THIRD_MESH_TYPE>
struct Count3ProcessCloseOp {
    FLOAT *local_counts;
    size_t i1;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    FLOAT local_frame[3][NDIM];

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs3;
    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;
    bool veto13;

    const BinAttrs *battrs;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i2,
        FLOAT *position2,
        FLOAT *sposition2,
        FLOAT *value2)
    {
        (void)i2;

        if (!is_selected_pair_with_sattrs(sposition1, sposition2, position1, position2, sattrs12)) return;

        Count3CloseOp op{
            local_counts,
            position1, sposition1, value1,
            position2, sposition2, value2,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh1, mesh2, mesh3,
            sattrs12, sattrs13, sattrs23, veto13,
            battrs, wattrs
        };

        if constexpr (THIRD_MESH_TYPE == MESH_ANGULAR) {
            for_each_third_candidate_from_1_angular(i1, mesh1, mesh3, mattrs3, op);
        }
        else if constexpr (THIRD_MESH_TYPE == MESH_CARTESIAN) {
            for_each_third_candidate_from_1_cartesian(i1, mesh1, mesh3, mattrs3, op);
        }
    }
};


template <MESH_TYPE THIRD_MESH_TYPE>
__global__ void count3_close_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    bool veto13,
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

    for (size_t i1 = gid; i1 < mesh1.total_nparticles; i1 += stride) {
        FLOAT *position1  = &(mesh1.positions[NDIM * i1]);
        FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);
        FLOAT *value1     = &(mesh1.values[mesh1.index_value.size * i1]);

        FLOAT local_frame[3][NDIM];
        build_local_frame(sposition1, local_frame);

        Count3ProcessCloseOp<THIRD_MESH_TYPE> op{
            local_counts,
            i1,
            position1, sposition1, value1,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh1, mesh2, mesh3,
            mattrs3,
            sattrs12, sattrs13, sattrs23, veto13,
            battrs, wattrs
        };

        for_each_close_candidate_from_1_angular(i1, mesh1, mesh2, mattrs2, op);
    }
}


// ============================================================================
// Higher-level wrapper
// ============================================================================

void count3_close(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    bool veto13,
    BinAttrs battrs12,
    BinAttrs battrs23,
    BinAttrs battrs13,
    WeightAttrs wattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream)
{
    BinAttrs host_battrs[3] = {battrs12, battrs13, battrs23};

    BinAttrs *battrs_ptrs[3] = {
        &host_battrs[0],
        &host_battrs[1],
        &host_battrs[2]
    };

    Count3Layout layout = make_count3_layout(battrs_ptrs);
    size_t csize = layout.csize;

    BinAttrs device_battrs[3];
    for (int i = 0; i < 3; i++) {
        copy_bin_attrs_to_device(&device_battrs[i], &host_battrs[i], buffer);
    }

    WeightAttrs device_wattrs = wattrs;
    copy_weight_attrs_to_device(&device_wattrs, &wattrs, buffer);

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH((count3_close_kernel<MESH_CARTESIAN>),
                            nblocks, nthreads_per_block, buffer);

    FLOAT *block_counts = (FLOAT *) my_device_malloc(
        nblocks * csize * sizeof(FLOAT), buffer);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, csize * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemcpyToSymbol(device_layout, &layout, sizeof(Count3Layout)));

    if (mattrs3.type == MESH_ANGULAR) {
        count3_close_kernel<MESH_ANGULAR><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts,
            csize,
            mesh1,
            mesh2,
            mesh3,
            mattrs2,
            mattrs3,
            sattrs12,
            sattrs13,
            sattrs23,
            veto13,
            device_battrs,
            device_wattrs
        );
    }
    else if (mattrs3.type == MESH_CARTESIAN) {
        count3_close_kernel<MESH_CARTESIAN><<<nblocks, nthreads_per_block, 0, stream>>>(
            block_counts,
            csize,
            mesh1,
            mesh2,
            mesh3,
            mattrs2,
            mattrs3,
            sattrs12,
            sattrs13,
            sattrs23,
            veto13,
            device_battrs,
            device_wattrs
        );
    }
    else {
        log_message(LOG_LEVEL_ERROR, "count3_close: unsupported mattrs3.type.\n");
        exit(EXIT_FAILURE);
    }

    reduce_add_kernel<<<nblocks, nthreads_per_block, 0, stream>>>(
        block_counts, nblocks, counts, csize);

    CUDA_CHECK(cudaDeviceSynchronize());

    my_device_free(block_counts, buffer);

    for (int i = 0; i < 3; i++) {
        free_device_bin_attrs(&device_battrs[i], buffer);
    }

    free_device_weight_attrs(&device_wattrs, buffer);
}