// Drop-in replacement: count3close.cu
// Periodic handling: pair-specific MeshAttrs
// LOS handling: build_los_frame for LOS_X / LOS_Y / LOS_Z

#include "common.h"
#include "count2.h"
#include "count3.h"
#include "count3close.h"



__device__ FLOAT clamp1(FLOAT x)
{
    return MIN((FLOAT)1., MAX((FLOAT)-1., x));
}


__device__ inline void normalize(FLOAT *out, const FLOAT *x)
{
    FLOAT norm = 0.;
    #pragma unroll
    for (int icoord = 0; icoord < NDIM; icoord++) norm += x[icoord] * x[icoord];

    norm = sqrt(norm);

    if (norm > 0.) {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) out[icoord] = x[icoord] / norm;
    }
    else {
        #pragma unroll
        for (int icoord = 0; icoord < NDIM; icoord++) out[icoord] = 0.;
    }
}


__device__ inline void cross3(FLOAT *out, const FLOAT *a, const FLOAT *b)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}


__device__ void build_local_frame(const FLOAT *ez_in, FLOAT local_frame[3][NDIM])
{
    FLOAT ref[NDIM];

    if (fabs((double)ez_in[2]) < 0.9) {
        ref[0] = 0.; ref[1] = 0.; ref[2] = 1.;
    }
    else {
        ref[0] = 1.; ref[1] = 0.; ref[2] = 0.;
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


__device__ void build_los_frame(
    FLOAT *sposition1,
    LOS_TYPE los,
    FLOAT local_frame[3][NDIM])
{
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < NDIM; j++) {
            local_frame[i][j] = (FLOAT)0.;
        }
    }

    if (los == LOS_X) {
        local_frame[0][0] = 1.;
        local_frame[1][1] = 1.;
        local_frame[2][2] = 1.;
    }
    else if (los == LOS_Y) {
        local_frame[0][1] = 1.;
        local_frame[1][2] = 1.;
        local_frame[2][0] = 1.;
    }
    else if (los == LOS_Z) {
        local_frame[0][2] = 1.;
        local_frame[1][0] = 1.;
        local_frame[2][1] = 1.;
    }
    else {
        build_local_frame(sposition1, local_frame);
    }
}


__device__ inline LOS_TYPE get_count3_los(BinAttrs battrs12, BinAttrs battrs13)
{
    LOS_TYPE los = LOS_FIRSTPOINT;

    if (battrs12.ndim > 1 && battrs12.var[1] == VAR_POLE) {
        los = battrs12.los[1];
    }
    else if (battrs13.ndim > 1 && battrs13.var[1] == VAR_POLE) {
        los = battrs13.los[1];
    }

    return los;
}


// ============================================================================
// Count3 layout + constant device state
// ============================================================================

static __device__ __constant__ DeviceCount3Layout device_layout;


// ============================================================================
// Host helpers
// ============================================================================



DeviceCount3Layout make_device_count3_layout(
    const BinAttrs battrs12,
    const BinAttrs battrs13,
    const BinAttrs battrs23)
{
    DeviceCount3Layout layout;
    memset(&layout, 0, sizeof(DeviceCount3Layout));

    if (battrs12.ndim == 0 || battrs13.ndim == 0) return layout;

    layout.nbins = (size_t)battrs12.shape[0] * (size_t)battrs13.shape[0];

    if (battrs23.ndim > 0) {
        layout.nbins *= (size_t)battrs23.shape[0];
        layout.nprojs  = 0;
        layout.nprojs1 = 0;
        layout.nprojs2 = 0;
        layout.csize   = layout.nbins;
        return layout;
    }

    if (battrs12.var[1] == VAR_POLE && battrs13.var[1] == VAR_POLE) {
        layout.nells1 = fill_ells(&battrs12, 1, layout.ells1);
        layout.nells2 = fill_ells(&battrs13, 1, layout.ells2);

        layout.nprojs  = 0;
        layout.nprojs1 = 0;
        layout.nprojs2 = 0;

        for (size_t ill1 = 0; ill1 < layout.nells1; ill1++) {
            int ell1 = (int)layout.ells1[ill1];
            layout.nprojs1 += (size_t)(2 * ell1 + 1);

            for (size_t ill2 = 0; ill2 < layout.nells2; ill2++) {
                int ell2 = (int)layout.ells2[ill2];
                layout.nprojs += (size_t)(2 * MIN(ell1, ell2) + 1);
            }
        }

        for (size_t ill2 = 0; ill2 < layout.nells2; ill2++) {
            int ell2 = (int)layout.ells2[ill2];
            layout.nprojs2 += (size_t)(2 * ell2 + 1);
        }

        layout.csize = layout.nbins * layout.nprojs;
    }
    else {
        layout.nprojs  = 0;
        layout.nprojs1 = 0;
        layout.nprojs2 = 0;
        layout.csize   = layout.nbins;
    }

    return layout;
}


// ============================================================================
// Projection helpers
// ============================================================================

__device__ void compute_trig_up_to_m(
    int mmax,
    FLOAT c1,
    FLOAT s1,
    FLOAT cm[MMAX_SIZE],
    FLOAT sm[MMAX_SIZE])
{
    #pragma unroll
    for (int m = 0; m < MMAX_SIZE; m++) {
        cm[m] = 0.;
        sm[m] = 0.;
    }

    mmax = MIN(mmax, ELLMAX);

    cm[0] = 1.;
    sm[0] = 0.;
    if (mmax <= 0) return;

    cm[1] = c1;
    sm[1] = s1;
    if (mmax <= 1) return;

    #pragma unroll
    for (int m = 2; m < MMAX_SIZE; m++) {
        if (m > mmax) break;
        cm[m] = c1 * cm[m - 1] - s1 * sm[m - 1];
        sm[m] = s1 * cm[m - 1] + c1 * sm[m - 1];
    }
}


__device__ void compute_pbar_row_lmax5(
    int ell,
    int mmax,
    FLOAT mu,
    FLOAT Prow[MMAX_SIZE])
{
    FLOAT x  = clamp1(mu);
    FLOAT x2 = x * x;
    FLOAT x3 = x2 * x;
    FLOAT x4 = x2 * x2;
    FLOAT x5 = x4 * x;

    FLOAT s2 = MAX((FLOAT)0., (FLOAT)1. - x2);
    FLOAT s  = sqrt(s2);
    FLOAT s3 = s2 * s;
    FLOAT s4 = s2 * s2;
    FLOAT s5 = s4 * s;

    #pragma unroll
    for (int m = 0; m < MMAX_SIZE; m++) Prow[m] = 0.;

    if (mmax > ell) mmax = ell;
    if (mmax > ELLMAX) mmax = ELLMAX;
    if (mmax < 0) return;

    switch (ell) {
        case 0:
            Prow[0] = 1.;
            break;

        case 1:
            if (mmax >= 0) Prow[0] = x;
            if (mmax >= 1) Prow[1] = -(FLOAT)0.70710678118654752440 * s;
            break;

        case 2:
            if (mmax >= 0) Prow[0] = ((FLOAT)0.5) * (((FLOAT)3.) * x2 - (FLOAT)1.);
            if (mmax >= 1) Prow[1] = -(FLOAT)1.22474487139158904910 * x * s;
            if (mmax >= 2) Prow[2] =  (FLOAT)0.61237243569579452455 * s2;
            break;

        case 3:
            if (mmax >= 0) Prow[0] = ((FLOAT)0.5) * (((FLOAT)5.) * x3 - ((FLOAT)3.) * x);
            if (mmax >= 1) Prow[1] = -(FLOAT)0.43301270189221932338 * ((((FLOAT)5.) * x2) - (FLOAT)1.) * s;
            if (mmax >= 2) Prow[2] =  (FLOAT)1.36930639376291527536 * x * s2;
            if (mmax >= 3) Prow[3] = -(FLOAT)0.55901699437494742410 * s3;
            break;

        case 4:
            if (mmax >= 0) Prow[0] = ((FLOAT)0.125) * (((FLOAT)35.) * x4 - ((FLOAT)30.) * x2 + (FLOAT)3.);
            if (mmax >= 1) Prow[1] = -(FLOAT)0.55901699437494742410 * x * ((((FLOAT)7.) * x2) - (FLOAT)3.) * s;
            if (mmax >= 2) Prow[2] =  (FLOAT)0.39528470752104741743 * ((((FLOAT)7.) * x2) - (FLOAT)1.) * s2;
            if (mmax >= 3) Prow[3] = -(FLOAT)0.93541434669348534640 * x * s3;
            if (mmax >= 4) Prow[4] =  (FLOAT)0.52291251658379721705 * s4;
            break;

        case 5:
            if (mmax >= 0) Prow[0] = ((FLOAT)0.125) * (((FLOAT)63.) * x5 - ((FLOAT)70.) * x3 + ((FLOAT)15.) * x);
            if (mmax >= 1) Prow[1] = -(FLOAT)0.19882122822827110675 * ((((FLOAT)21.) * x4) - ((FLOAT)14.) * x2 + (FLOAT)1.) * s;
            if (mmax >= 2) Prow[2] =  (FLOAT)0.48412291827592711065 * x * ((((FLOAT)3.) * x2) - (FLOAT)1.) * s2;
            if (mmax >= 3) Prow[3] = -(FLOAT)0.52291251658379721705 * ((((FLOAT)9.) * x2) - (FLOAT)1.) * s3;
            if (mmax >= 4) Prow[4] =  (FLOAT)1.16926793336685668103 * x * s4;
            if (mmax >= 5) Prow[5] = -(FLOAT)0.70156076002011400980 * s5;
            break;

        default:
            break;
    }
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
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs,
    const MeshAttrs &mattrs12,
    const MeshAttrs &mattrs13,
    const MeshAttrs &mattrs23)
{
    if (battrs12.ndim == 0 || battrs13.ndim == 0) return;

    const bool has_third = (bool)(battrs23.ndim > 0);
    const int ncoords = has_third ? 3 : 2;
    const bool need_pole = (bool)(device_layout.nprojs > 0);

    const FLOAT *spos1[3] = {sposition1, sposition1, sposition2};
    const FLOAT *spos2[3] = {sposition2, sposition3, sposition3};

    const FLOAT *pos1[3] = {position1, position1, position2};
    const FLOAT *pos2[3] = {position2, position3, position3};

    const MeshAttrs *mattrs[3] = {&mattrs12, &mattrs13, &mattrs23};

    BinAttrs battrs[3] = {battrs12, battrs13, battrs23};

    FLOAT costheta[3];
    FLOAT diff[3][NDIM];
    FLOAT r[3] = {0., 0., 0.};

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
            difference(diff[icoord], (FLOAT*)pos2[icoord], (FLOAT*)pos1[icoord], *mattrs[icoord]);
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

    FLOAT triplet_weight = 1.;

    if (index_value1.size_individual_weight) triplet_weight *= value1[index_value1.start_individual_weight];
    if (index_value2.size_individual_weight) triplet_weight *= value2[index_value2.start_individual_weight];
    if (index_value3.size_individual_weight) triplet_weight *= value3[index_value3.start_individual_weight];

    {
        AngularWeight angular = wattrs.angular;
        if (angular.size) triplet_weight *= lookup_angular_weight<3>(costheta, angular);
    }

    if (index_value1.size_negative_weight &&
        index_value2.size_negative_weight &&
        index_value3.size_negative_weight) {
        FLOAT triplet_nweight =
            value1[index_value1.start_negative_weight] *
            value2[index_value2.start_negative_weight] *
            value3[index_value3.start_negative_weight];

        triplet_weight -= triplet_nweight;
    }

    if (has_third || !need_pole) {
        atomicAdd(&counts[ibin], triplet_weight);
        return;
    }

    FLOAT rhat[2][NDIM];

    #pragma unroll
    for (int ivec = 0; ivec < 2; ivec++) {
        if (r[ivec] == (FLOAT)0.) {
            #pragma unroll
            for (int icoord = 0; icoord < NDIM; icoord++) {
                rhat[ivec][icoord] = local_frame[0][icoord];
            }
        }
        else {
            #pragma unroll
            for (int icoord = 0; icoord < NDIM; icoord++) {
                rhat[ivec][icoord] = diff[ivec][icoord] / r[ivec];
            }
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

    FLOAT cdphi = 1.;
    FLOAT sdphi = 0.;

    if (rho[0] > (FLOAT)1e-12 && rho[1] > (FLOAT)1e-12) {
        FLOAT inv = (FLOAT)1. / (rho[0] * rho[1]);

        cdphi = clamp1((xy[0][0] * xy[1][0] + xy[0][1] * xy[1][1]) * inv);
        sdphi = clamp1((xy[0][0] * xy[1][1] - xy[0][1] * xy[1][0]) * inv);
    }

    int global_mmax = 0;
    int row_mmax1[4] = {0, 0, 0, 0};
    int row_mmax2[4] = {0, 0, 0, 0};

    for (size_t i1 = 0; i1 < device_layout.nells1; i1++) {
        int ell1 = (int)device_layout.ells1[i1];

        for (size_t i2 = 0; i2 < device_layout.nells2; i2++) {
            int ell2 = (int)device_layout.ells2[i2];
            int mmax = MIN(ell1, ell2);

            global_mmax = MAX(global_mmax, mmax);
            row_mmax1[i1] = MAX(row_mmax1[i1], mmax);
            row_mmax2[i2] = MAX(row_mmax2[i2], mmax);
        }
    }

    FLOAT cm[MMAX_SIZE], sm[MMAX_SIZE];
    FLOAT P1[4][MMAX_SIZE];
    FLOAT P2[4][MMAX_SIZE];

    compute_trig_up_to_m(global_mmax, cdphi, sdphi, cm, sm);

    for (size_t i1 = 0; i1 < device_layout.nells1; i1++) {
        compute_pbar_row_lmax5((int)device_layout.ells1[i1], row_mmax1[i1], mu[0], P1[i1]);
    }

    for (size_t i2 = 0; i2 < device_layout.nells2; i2++) {
        compute_pbar_row_lmax5((int)device_layout.ells2[i2], row_mmax2[i2], mu[1], P2[i2]);
    }

    FLOAT *counts_bin = counts + ibin * device_layout.nprojs;

    size_t iproj = 0;

    for (size_t i1 = 0; i1 < device_layout.nells1; i1++) {
        int ell1 = (int)device_layout.ells1[i1];

        for (size_t i2 = 0; i2 < device_layout.nells2; i2++) {
            int ell2 = (int)device_layout.ells2[i2];
            int mmax = MIN(ell1, ell2);

            FLOAT ell_norm = sqrt((FLOAT)((2 * ell1 + 1) * (2 * ell2 + 1)));

            for (int m = 0; m <= mmax; m++) {
                FLOAT amp = triplet_weight * ell_norm * P1[i1][m] * P2[i2][m];

                atomicAdd(&counts_bin[iproj + (size_t)m], amp * cm[m]);

                if (m > 0) {
                    atomicAdd(&counts_bin[iproj + (size_t)(mmax + m)], amp * sm[m]);
                }
            }

            iproj += (size_t)(2 * mmax + 1);
        }
    }
}


// ============================================================================
// Emit original ordering: always add_weight3(1, 2, 3)
// ============================================================================

struct Count3EmitOp {
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

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i3,
        FLOAT *position3,
        FLOAT *sposition3,
        FLOAT *value3)
    {
        (void)i3;

        if (!is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                sattrs13,
                mattrs13)) return;

        if (!is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                sattrs23,
                mattrs23)) return;

        if (veto12.ndim && is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                veto12,
                mattrs12)) return;

        if (veto13.ndim && is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                veto13,
                mattrs13)) return;

        if (veto23.ndim && is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                veto23,
                mattrs23)) return;

        add_weight3(
            local_counts,
            local_frame,
            sposition1, sposition2, sposition3,
            position1, position2, position3,
            value1, value2, value3,
            mesh1.index_value, mesh2.index_value, mesh3.index_value,
            battrs12, battrs13, battrs23,
            wattrs,
            mattrs12, mattrs13, mattrs23);
    }
};


// ============================================================================
// close_pair == 0 : close pair is (1, 2), other is 3
// ============================================================================

template <MESH_TYPE OTHER_MESH_TYPE>
struct Count3Close12Op {
    FLOAT *local_counts;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    FLOAT local_frame[3][NDIM];

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs3;

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i2,
        FLOAT *position2,
        FLOAT *sposition2,
        FLOAT *value2)
    {
        (void)i2;

        if (!is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                sattrs12,
                mattrs12)) return;

        if (veto12.ndim && is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                veto12,
                mattrs12)) return;

        Count3EmitOp op{
            local_counts,
            position1, sposition1, value1,
            position2, sposition2, value2,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh1, mesh2, mesh3,
            mattrs12, mattrs13, mattrs23,
            sattrs12, sattrs13, sattrs23,
            veto12, veto13, veto23,
            battrs12, battrs13, battrs23,
            wattrs
        };

        for_each_candidate<OTHER_MESH_TYPE>(
            position1, sposition1,
            mesh3, mattrs3,
            op);
    }
};


// ============================================================================
// close_pair == 1 : close pair is (1, 3), other is 2
// ============================================================================

struct Count3EmitWithFixed3Op {
    FLOAT *local_counts;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    FLOAT *position3;
    FLOAT *sposition3;
    FLOAT *value3;

    FLOAT local_frame[3][NDIM];

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i2,
        FLOAT *position2,
        FLOAT *sposition2,
        FLOAT *value2)
    {
        (void)i2;

        if (!is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                sattrs12,
                mattrs12)) return;

        if (!is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                sattrs23,
                mattrs23)) return;

        if (veto12.ndim && is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                veto12,
                mattrs12)) return;

        if (veto13.ndim && is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                veto13,
                mattrs13)) return;

        if (veto23.ndim && is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                veto23,
                mattrs23)) return;

        add_weight3(
            local_counts,
            local_frame,
            sposition1, sposition2, sposition3,
            position1, position2, position3,
            value1, value2, value3,
            mesh1.index_value, mesh2.index_value, mesh3.index_value,
            battrs12, battrs13, battrs23,
            wattrs,
            mattrs12, mattrs13, mattrs23);
    }
};


template <MESH_TYPE OTHER_MESH_TYPE>
struct Count3Close13Op {
    FLOAT *local_counts;

    FLOAT *position1;
    FLOAT *sposition1;
    FLOAT *value1;

    FLOAT local_frame[3][NDIM];

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs2;

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i3,
        FLOAT *position3,
        FLOAT *sposition3,
        FLOAT *value3)
    {
        (void)i3;

        if (!is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                sattrs13,
                mattrs13)) return;

        if (veto13.ndim && is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                veto13,
                mattrs13)) return;

        Count3EmitWithFixed3Op op{
            local_counts,
            position1, sposition1, value1,
            position3, sposition3, value3,
            {
                {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
            },
            mesh1, mesh2, mesh3,
            mattrs12, mattrs13, mattrs23,
            sattrs12, sattrs13, sattrs23,
            veto12, veto13, veto23,
            battrs12, battrs13, battrs23,
            wattrs
        };

        for_each_candidate<OTHER_MESH_TYPE>(
            position1, sposition1,
            mesh2, mattrs2,
            op);
    }
};


// ============================================================================
// close_pair == 2 : close pair is (2, 3), other / LOS particle is 1
// ============================================================================

struct Count3EmitWithFixed23Op {
    FLOAT *local_counts;

    FLOAT *position2;
    FLOAT *sposition2;
    FLOAT *value2;

    FLOAT *position3;
    FLOAT *sposition3;
    FLOAT *value3;

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i1,
        FLOAT *position1,
        FLOAT *sposition1,
        FLOAT *value1)
    {
        (void)i1;

        if (!is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                sattrs12,
                mattrs12)) return;

        if (!is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                sattrs13,
                mattrs13)) return;

        if (veto12.ndim && is_selected_pair(
                sposition1, sposition2,
                position1, position2,
                veto12,
                mattrs12)) return;

        if (veto13.ndim && is_selected_pair(
                sposition1, sposition3,
                position1, position3,
                veto13,
                mattrs13)) return;

        if (veto23.ndim && is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                veto23,
                mattrs23)) return;

        FLOAT local_frame[3][NDIM];
        build_los_frame(sposition1, get_count3_los(battrs12, battrs13), local_frame);

        add_weight3(
            local_counts,
            local_frame,
            sposition1, sposition2, sposition3,
            position1, position2, position3,
            value1, value2, value3,
            mesh1.index_value, mesh2.index_value, mesh3.index_value,
            battrs12, battrs13, battrs23,
            wattrs,
            mattrs12, mattrs13, mattrs23);
    }
};


template <MESH_TYPE OTHER_MESH_TYPE>
struct Count3Close23From2Op {
    FLOAT *local_counts;

    FLOAT *position2;
    FLOAT *sposition2;
    FLOAT *value2;

    Mesh mesh1;
    Mesh mesh2;
    Mesh mesh3;

    MeshAttrs mattrs1;

    MeshAttrs mattrs12;
    MeshAttrs mattrs13;
    MeshAttrs mattrs23;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;

    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;
    WeightAttrs wattrs;

    __device__ inline void operator()(
        size_t i3,
        FLOAT *position3,
        FLOAT *sposition3,
        FLOAT *value3)
    {
        (void)i3;

        if (!is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                sattrs23,
                mattrs23)) return;

        if (veto23.ndim && is_selected_pair(
                sposition2, sposition3,
                position2, position3,
                veto23,
                mattrs23)) return;

        Count3EmitWithFixed23Op op{
            local_counts,
            position2, sposition2, value2,
            position3, sposition3, value3,
            mesh1, mesh2, mesh3,
            mattrs12, mattrs13, mattrs23,
            sattrs12, sattrs13, sattrs23,
            veto12, veto13, veto23,
            battrs12, battrs13, battrs23,
            wattrs
        };

        for_each_candidate<OTHER_MESH_TYPE>(
            position2, sposition2,
            mesh1, mattrs1,
            op);
    }
};


// ============================================================================
// Kernel
// ============================================================================

template <MESH_TYPE OTHER_MESH_TYPE>
__global__ void count3_close_kernel(
    FLOAT *block_counts,
    size_t csize,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs1,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    SelectionAttrs veto23,
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs,
    CLOSE_PAIR close_pair)
{
    size_t tid = threadIdx.x;
    FLOAT *local_counts = &block_counts[blockIdx.x * csize];

    for (size_t i = tid; i < csize; i += blockDim.x) {
        local_counts[i] = 0;
    }

    __syncthreads();

    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    if (close_pair == CLOSE_PAIR_12 || close_pair == CLOSE_PAIR_13) {
        for (size_t i1 = gid; i1 < mesh1.total_nparticles; i1 += stride) {
            FLOAT *position1  = &(mesh1.positions[NDIM * i1]);
            FLOAT *sposition1 = &(mesh1.spositions[NDIM * i1]);
            FLOAT *value1     = &(mesh1.values[mesh1.index_value.size * i1]);

            FLOAT local_frame[3][NDIM];
            build_los_frame(sposition1, get_count3_los(battrs12, battrs13), local_frame);

            if (close_pair == CLOSE_PAIR_12) {
                Count3Close12Op<OTHER_MESH_TYPE> op{
                    local_counts,
                    position1, sposition1, value1,
                    {
                        {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                        {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                        {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                    },
                    mesh1, mesh2, mesh3,
                    mattrs3,
                    mattrs1, mattrs2, mattrs3,
                    sattrs12, sattrs13, sattrs23,
                    veto12, veto13, veto23,
                    battrs12, battrs13, battrs23,
                    wattrs
                };

                for_each_candidate<MESH_ANGULAR>(
                    position1, sposition1,
                    mesh2, mattrs2,
                    op);
            }
            else {
                Count3Close13Op<OTHER_MESH_TYPE> op{
                    local_counts,
                    position1, sposition1, value1,
                    {
                        {local_frame[0][0], local_frame[0][1], local_frame[0][2]},
                        {local_frame[1][0], local_frame[1][1], local_frame[1][2]},
                        {local_frame[2][0], local_frame[2][1], local_frame[2][2]}
                    },
                    mesh1, mesh2, mesh3,
                    mattrs2,
                    mattrs1, mattrs2, mattrs3,
                    sattrs12, sattrs13, sattrs23,
                    veto12, veto13, veto23,
                    battrs12, battrs13, battrs23,
                    wattrs
                };

                for_each_candidate<MESH_ANGULAR>(
                    position1, sposition1,
                    mesh3, mattrs3,
                    op);
            }
        }
    }
    else if (close_pair == CLOSE_PAIR_23) {
        for (size_t i2 = gid; i2 < mesh2.total_nparticles; i2 += stride) {
            FLOAT *position2  = &(mesh2.positions[NDIM * i2]);
            FLOAT *sposition2 = &(mesh2.spositions[NDIM * i2]);
            FLOAT *value2     = &(mesh2.values[mesh2.index_value.size * i2]);

            Count3Close23From2Op<OTHER_MESH_TYPE> op{
                local_counts,
                position2, sposition2, value2,
                mesh1, mesh2, mesh3,
                mattrs1,
                mattrs1, mattrs2, mattrs3,
                sattrs12, sattrs13, sattrs23,
                veto12, veto13, veto23,
                battrs12, battrs13, battrs23,
                wattrs
            };

            for_each_candidate<MESH_ANGULAR>(
                position2, sposition2,
                mesh3, mattrs3,
                op);
        }
    }
}


// ============================================================================
// Compact host launch
// ============================================================================

#define LAUNCH_COUNT3_CLOSE_KERNEL(OTHER_MESH_TYPE)                              \
    count3_close_kernel<OTHER_MESH_TYPE><<<nblocks, nthreads_per_block, 0, stream>>>( \
        block_counts,                                                            \
        csize,                                                                   \
        mesh1, mesh2, mesh3,                                                     \
        mattrs1, mattrs2, mattrs3,                                               \
        sattrs12, sattrs13, sattrs23,                                            \
        veto12, veto13, veto23,                                                  \
        device_battrs12, device_battrs13, device_battrs23,                       \
        device_wattrs,                                                           \
        close_pair                                                               \
    )


// ============================================================================
// Higher-level wrapper
// ============================================================================

void count3_close(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs1,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    SelectionAttrs veto23,
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs,
    CLOSE_PAIR close_pair,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream)
{
    const MeshAttrs *other_mattrs = nullptr;

    if (close_pair == CLOSE_PAIR_12) {
        if (mattrs1.type != MESH_ANGULAR || mattrs2.type != MESH_ANGULAR) {
            log_message(LOG_LEVEL_ERROR, "count3_close: close pair (1, 2) must be angular.\n");
            exit(EXIT_FAILURE);
        }
        other_mattrs = &mattrs3;
    }
    else if (close_pair == CLOSE_PAIR_13) {
        if (mattrs1.type != MESH_ANGULAR || mattrs3.type != MESH_ANGULAR) {
            log_message(LOG_LEVEL_ERROR, "count3_close: close pair (1, 3) must be angular.\n");
            exit(EXIT_FAILURE);
        }
        other_mattrs = &mattrs2;
    }
    else {
        if (mattrs2.type != MESH_ANGULAR || mattrs3.type != MESH_ANGULAR) {
            log_message(LOG_LEVEL_ERROR, "count3_close: close pair (2, 3) must be angular.\n");
            exit(EXIT_FAILURE);
        }
        other_mattrs = &mattrs1;
    }

    DeviceCount3Layout layout = make_device_count3_layout(
        battrs12, battrs13, battrs23);

    size_t csize = layout.csize;

    BinAttrs device_battrs12 = battrs12;
    BinAttrs device_battrs13 = battrs13;
    BinAttrs device_battrs23 = battrs23;

    copy_bin_attrs_to_device(&device_battrs12, &battrs12, buffer);
    copy_bin_attrs_to_device(&device_battrs13, &battrs13, buffer);
    copy_bin_attrs_to_device(&device_battrs23, &battrs23, buffer);

    WeightAttrs device_wattrs = wattrs;
    copy_weight_attrs_to_device(&device_wattrs, &wattrs, buffer);

    int nblocks, nthreads_per_block;
    CONFIGURE_KERNEL_LAUNCH(
        (count3_close_kernel<MESH_ANGULAR>),
        nblocks,
        nthreads_per_block,
        buffer);
    //printf("NBLOCKS %d, %d\n", nblocks, nthreads_per_block);

    FLOAT *block_counts = (FLOAT *)my_device_malloc(
        nblocks * csize * sizeof(FLOAT),
        buffer);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, csize * sizeof(FLOAT), stream));

    CUDA_CHECK(cudaMemcpyToSymbol(
        device_layout,
        &layout,
        sizeof(DeviceCount3Layout)));

    if (other_mattrs->type == MESH_ANGULAR) {
        LAUNCH_COUNT3_CLOSE_KERNEL(MESH_ANGULAR);
    }
    else if (other_mattrs->type == MESH_CARTESIAN) {
        LAUNCH_COUNT3_CLOSE_KERNEL(MESH_CARTESIAN);
    }
    else {
        log_message(LOG_LEVEL_ERROR, "count3_close: unsupported other mesh type.\n");
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

    my_device_free(block_counts, buffer);

    free_device_bin_attrs(&device_battrs12, buffer);
    free_device_bin_attrs(&device_battrs13, buffer);
    free_device_bin_attrs(&device_battrs23, buffer);

    free_device_weight_attrs(&device_wattrs, buffer);
}


#undef LAUNCH_COUNT3_CLOSE_KERNEL