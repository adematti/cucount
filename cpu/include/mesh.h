// Cell mesh construction and candidate traversal for the CPU backend.
//
// Two meshes, because the two paths want opposite layouts: VectorMesh is SoA
// for the Highway kernel, ScalarMesh interleaved like the CUDA Mesh for the
// scalar paths. Both bucket the same particles into the same cells, and
// for_each_candidate sweeps the window MeshAttrs describes.
//
// Plain C++, no Highway.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "count2.h"

namespace cucount {
namespace cpu {

inline int wrap_index(int i, int n) {
    int r = i % n;
    return (r < 0) ? r + n : r;
}

// spin: this catalogue's interleaved (e1, e2) components, or null.
// with_spos: fill unit-sphere positions, needed on BOTH meshes whenever either
// side of the pair count carries spin (the projection frame uses both ends).
// bw/nbit: this catalogue's interleaved bitwise (PIP) columns, or null;
// nw: its negative-weight column, or null.
template <class Float>
VectorMesh<Float> build_mesh(const double* pos, const double* w, size_t n,
                             const double boxsize[3], const double origin[3],
                             const int dims[3], const double* spin = nullptr,
                             bool with_spos = false, const double* bw = nullptr,
                             size_t nbit = 0, const double* nw = nullptr) {
    VectorMesh<Float> m;
    for (int a = 0; a < 3; ++a) {
        m.dims[a] = dims[a];
        m.cell[a] = static_cast<Float>(boxsize[a] / m.dims[a]);
        m.origin[a] = static_cast<Float>(origin[a]);
    }

    const size_t nc = m.ncells();
    std::vector<size_t> cell_of(n);
    m.start.assign(nc + 1, 0);

    for (size_t i = 0; i < n; ++i) {
        size_t idx = 0;
        for (int a = 0; a < 3; ++a) {
            const double rel = (pos[3 * i + a] - origin[a]) / boxsize[a];
            int ia = static_cast<int>(std::floor(rel * m.dims[a]));
            idx = idx * m.dims[a] + wrap_index(ia, m.dims[a]);
        }
        cell_of[i] = idx;
        ++m.start[idx + 1];
    }

    for (size_t c = 0; c < nc; ++c) m.start[c + 1] += m.start[c];

    m.x.resize(n);
    m.y.resize(n);
    m.z.resize(n);
    m.w.resize(n);
    if (with_spos) {
        m.sx.resize(n);
        m.sy.resize(n);
        m.sz.resize(n);
    }
    if (spin) {
        m.e1.resize(n);
        m.e2.resize(n);
    }
    if (bw && nbit) m.bw.resize(n * nbit);
    if (nw) m.nw.resize(n);

    std::vector<size_t> fill(m.start.begin(), m.start.end() - 1);
    for (size_t i = 0; i < n; ++i) {
        const size_t o = fill[cell_of[i]]++;
        m.x[o] = static_cast<Float>(pos[3 * i + 0]);
        m.y[o] = static_cast<Float>(pos[3 * i + 1]);
        m.z[o] = static_cast<Float>(pos[3 * i + 2]);
        m.w[o] = static_cast<Float>(w ? w[i] : 1.0);
        if (with_spos) {
            // Normalized in double before narrowing, like the CUDA mesh.
            const double px = pos[3 * i + 0], py = pos[3 * i + 1],
                         pz = pos[3 * i + 2];
            const double r = std::sqrt(px * px + py * py + pz * pz);
            m.sx[o] = static_cast<Float>(px / r);
            m.sy[o] = static_cast<Float>(py / r);
            m.sz[o] = static_cast<Float>(pz / r);
        }
        if (spin) {
            m.e1[o] = static_cast<Float>(spin[2 * i + 0]);
            m.e2[o] = static_cast<Float>(spin[2 * i + 1]);
        }
        if (bw && nbit) {
            // Bit patterns: copied verbatim, never narrowed.
            for (size_t ib = 0; ib < nbit; ++ib)
                m.bw[o * nbit + ib] = bw[i * nbit + ib];
        }
        if (nw) m.nw[o] = static_cast<Float>(nw[i]);
    }
    return m;
}


// ---------------------------------------------------------------------------
// The scalar paths' mesh
// ---------------------------------------------------------------------------
//
// A port of cuda/src/mesh.cu and the for_each_candidate macros in
// cuda/include/count2.h, kept line-for-line close: same cell index, same
// candidate window, and the same rule that a particle carrying a zero
// individual weight never enters the mesh. The pair and triplet counts then
// form exactly the candidate sets the CUDA kernels do.
//
// Interleaved like the CUDA Mesh, so the shared per-pair math reads it
// unchanged -- unlike the SoA VectorMesh above, which the Highway kernel
// needs in order to load whole vectors of one coordinate at a time.

constexpr double kPi = 3.14159265358979323846;

inline int wrap_periodic_int(int idx, int meshsize) {
    const int r = idx % meshsize;
    return (r < 0) ? r + meshsize : r;
}

// The fmod form the CUDA kernel uses, rather than the Highway kernel's
// Round-based nearest image: identical for every separation inside the box,
// and this way the port has no arithmetic of its own to get wrong.
inline double wrap_periodic_float(double dxyz, double boxsize) {
    const double half = 0.5 * boxsize;
    double x = std::fmod(dxyz + half, boxsize);
    if (x < 0) x += boxsize;
    return x - half;
}

inline double wrap_angle(double phi) {
    phi = std::fmod(phi, 2 * kPi);
    if (phi < 0) phi += 2 * kPi;
    return phi;
}

// Cell-ordered copies of the particle columns, laid out like the CUDA Mesh
// (interleaved xyz, packed values) so the same per-pair code reads both.
struct ScalarMesh {
    std::vector<double> positions;   // 3 per particle
    std::vector<double> spositions;  // 3 per particle, on the unit sphere
    std::vector<double> values;      // vsize per particle
    std::vector<size_t> start;       // ncells + 1 offsets
    size_t vsize = 0;
    size_t total = 0;
    IndexValue iv = {};

    const double* position(size_t i) const { return &positions[3 * i]; }
    const double* sposition(size_t i) const { return &spositions[3 * i]; }
    const double* value(size_t i) const { return vsize ? &values[vsize * i] : nullptr; }
};

inline size_t cartesian_cell_index(const MeshAttrs& mattrs, const double* position) {
    size_t index = 0;
    for (size_t axis = 0; axis < NDIM; axis++) {
        index *= mattrs.meshsize[axis];
        const double offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;
        const int index_axis = static_cast<int>(std::floor(
            (position[axis] - offset) * mattrs.meshsize[axis] / mattrs.boxsize[axis]));
        index += wrap_periodic_int(index_axis, static_cast<int>(mattrs.meshsize[axis]));
    }
    return index;
}

inline size_t angular_cell_index(const MeshAttrs& mattrs, double cth, double phi) {
    const int icth = (cth == 1)
        ? (static_cast<int>(mattrs.meshsize[0]) - 1)
        : static_cast<int>(0.5 * (1 + cth) * mattrs.meshsize[0]);
    const int iphi = static_cast<int>(0.5 * wrap_angle(phi) / kPi * mattrs.meshsize[1]);
    return static_cast<size_t>(iphi) + static_cast<size_t>(icth) * mattrs.meshsize[1];
}

inline size_t cell_index(const MeshAttrs& mattrs, const double* position,
                         const double* sposition) {
    if (mattrs.type == MESH_ANGULAR) {
        const double phi = (sposition[0] == 0. && sposition[1] == 0.)
            ? 0. : std::atan2(sposition[1], sposition[0]);
        return angular_cell_index(mattrs, sposition[2], phi);
    }
    return cartesian_cell_index(mattrs, position);
}

inline size_t mesh_ncells(const MeshAttrs& mattrs) {
    size_t n = 1;
    const size_t naxis = (mattrs.type == MESH_ANGULAR) ? 2 : NDIM;
    for (size_t axis = 0; axis < naxis; axis++) n *= mattrs.meshsize[axis];
    return n;
}

// Counting pass then fill, mirroring set_mesh.
inline ScalarMesh build_mesh(const Particles& p, const MeshAttrs& mattrs) {
    ScalarMesh m;
    m.iv = p.index_value;
    m.vsize = p.index_value.size;

    const size_t nc = mesh_ncells(mattrs);
    m.start.assign(nc + 1, 0);

    const size_t n = p.size;
    std::vector<size_t> cell_of(n);
    std::vector<char> keep(n, 1);
    std::vector<double> spos(3 * n);

    for (size_t i = 0; i < n; i++) {
        if (m.vsize && p.index_value.size_individual_weight &&
            p.values[i * m.vsize + p.index_value.start_individual_weight] == 0.) {
            keep[i] = 0;
            continue;
        }
        const double* position = &p.positions[NDIM * i];
        const double r = std::sqrt(pairmath::dot3(position, position));
        for (size_t axis = 0; axis < NDIM; axis++) spos[3 * i + axis] = position[axis] / r;
        const size_t c = cell_index(mattrs, position, &spos[3 * i]);
        cell_of[i] = c;
        ++m.start[c + 1];
        ++m.total;
    }

    for (size_t c = 0; c < nc; c++) m.start[c + 1] += m.start[c];

    m.positions.resize(3 * m.total);
    m.spositions.resize(3 * m.total);
    m.values.resize(m.vsize * m.total);

    std::vector<size_t> fill(m.start.begin(), m.start.end() - 1);
    for (size_t i = 0; i < n; i++) {
        if (!keep[i]) continue;
        const size_t o = fill[cell_of[i]]++;
        for (size_t axis = 0; axis < NDIM; axis++) {
            m.positions[3 * o + axis] = p.positions[NDIM * i + axis];
            m.spositions[3 * o + axis] = spos[3 * i + axis];
        }
        for (size_t iv = 0; iv < m.vsize; iv++)
            m.values[m.vsize * o + iv] = p.values[m.vsize * i + iv];
    }
    return m;
}

// ---------------------------------------------------------------------------
// Candidate windows
// ---------------------------------------------------------------------------

inline void set_cartesian_bounds(const double* position, const MeshAttrs& mattrs,
                                 int* bounds) {
    for (int axis = 0; axis < NDIM; axis++) {
        const int meshsize = static_cast<int>(mattrs.meshsize[axis]);
        const double offset = mattrs.boxcenter[axis] - mattrs.boxsize[axis] / 2;
        int index = static_cast<int>(std::floor(
            (position[axis] - offset) * meshsize / mattrs.boxsize[axis]));
        index = wrap_periodic_int(index, meshsize);
        const int delta = static_cast<int>(
            std::ceil(mattrs.smax / mattrs.boxsize[axis] * meshsize));

        bounds[2 * axis] = index - delta;
        bounds[2 * axis + 1] = index + delta;

        if (!mattrs.periodic) {
            bounds[2 * axis] = std::max(bounds[2 * axis], 0);
            bounds[2 * axis + 1] = std::min(bounds[2 * axis + 1], meshsize - 1);
        }
        else if (2 * delta + 1 >= meshsize) {
            bounds[2 * axis] = 0;
            bounds[2 * axis + 1] = meshsize - 1;
        }
    }
}

// The (cos theta, phi) window that can hold a neighbour within smax, ported
// from set_angular_bounds. smax is stored as cos(theta_max) for this mesh.
inline void set_angular_bounds(const double* sposition, const MeshAttrs& mattrs,
                               int* bounds) {
    const double cth = sposition[2];
    double phi = std::atan2(sposition[1], sposition[0]);
    if (phi < 0) phi += 2 * kPi;

    const int icth = (cth >= 1)
        ? (static_cast<int>(mattrs.meshsize[0]) - 1)
        : static_cast<int>(0.5 * (1 + cth) * mattrs.meshsize[0]);
    const int iphi = static_cast<int>(0.5 * phi / kPi * mattrs.meshsize[1]);

    const double theta = std::acos(-1.0 + 2.0 * (icth + 0.5) / mattrs.meshsize[0]);
    const double th_hi = std::acos(-1.0 + 2.0 * (icth + 0.0) / mattrs.meshsize[0]);
    const double th_lo = std::acos(-1.0 + 2.0 * (icth + 1.0) / mattrs.meshsize[0]);
    const double phi_hi = 2 * kPi * (iphi + 1.0) / mattrs.meshsize[1];
    const double phi_lo = 2 * kPi * (iphi + 0.0) / mattrs.meshsize[1];
    const double smax = std::acos(mattrs.smax);

    double cth_max, cth_min;

    if (th_hi > kPi - smax) {
        cth_min = -1;
        // The window may also wrap the north pole: cos(th_lo - smax) is even
        // in its argument and would describe a spurious southern cap instead.
        cth_max = (th_lo < smax) ? 1. : std::cos(th_lo - smax);
        bounds[2] = 0;
        bounds[3] = static_cast<int>(mattrs.meshsize[1]) - 1;
    }
    else if (th_lo < smax) {
        cth_min = (th_hi + smax > kPi) ? -1. : std::cos(th_hi + smax);
        cth_max = 1;
        bounds[2] = 0;
        bounds[3] = static_cast<int>(mattrs.meshsize[1]) - 1;
    }
    else {
        double dphi;
        const double calpha = std::cos(smax);
        cth_min = std::cos(th_hi + smax);
        cth_max = std::cos(th_lo - smax);

        if (theta < 0.5 * kPi) {
            const double cth_lo = std::cos(th_lo);
            dphi = std::acos(std::sqrt((calpha * calpha - cth_lo * cth_lo) /
                                       (1 - cth_lo * cth_lo)));
        }
        else {
            const double cth_hi2 = std::cos(th_hi);
            dphi = std::acos(std::sqrt((calpha * calpha - cth_hi2 * cth_hi2) /
                                       (1 - cth_hi2 * cth_hi2)));
        }

        if (dphi < kPi) {
            const double phi_min = phi_lo - dphi;
            const double phi_max = phi_hi + dphi;
            bounds[2] = static_cast<int>(std::floor(0.5 * phi_min / kPi * mattrs.meshsize[1]));
            bounds[3] = static_cast<int>(std::floor(0.5 * phi_max / kPi * mattrs.meshsize[1]));
        }
        else {
            bounds[2] = 0;
            bounds[3] = static_cast<int>(mattrs.meshsize[1]) - 1;
        }
    }

    cth_min = std::max(cth_min, mattrs.boxcenter[0] - mattrs.boxsize[0] / 2.);
    cth_max = std::min(cth_max, mattrs.boxcenter[0] + mattrs.boxsize[0] / 2.);

    bounds[0] = static_cast<int>(0.5 * (1 + cth_min) * mattrs.meshsize[0]);
    bounds[1] = static_cast<int>(0.5 * (1 + cth_max) * mattrs.meshsize[0]);

    if (bounds[0] < 0) bounds[0] = 0;
    if (bounds[1] >= static_cast<int>(mattrs.meshsize[0]))
        bounds[1] = static_cast<int>(mattrs.meshsize[0]) - 1;
}

// Call `op(j)` for every particle of `mesh` in a cell the candidate window
// reaches, where `j` indexes the cell-ordered mesh. One walk for both mesh
// kinds: the angular mesh is the cartesian one with a single (cos theta, phi)
// plane, so its third axis collapses to the degenerate range [0, 0].
template <class Op>
inline void for_each_candidate(const MeshAttrs& mattrs, const ScalarMesh& mesh,
                               const double* position1, const double* sposition1,
                               Op&& op) {
    const bool angular = (mattrs.type == MESH_ANGULAR);

    int bounds[2 * NDIM] = {0};
    if (angular) set_angular_bounds(sposition1, mattrs, bounds);
    else set_cartesian_bounds(position1, mattrs, bounds);

    const int nphi = angular ? static_cast<int>(mattrs.meshsize[1]) : 1;
    const int i2lo = angular ? 0 : bounds[4];
    const int i2hi = angular ? 0 : bounds[5];

    for (int i0 = bounds[0]; i0 <= bounds[1]; i0++) {
        const size_t n0 = angular
            ? static_cast<size_t>(i0) * nphi
            : static_cast<size_t>(wrap_periodic_int(i0, static_cast<int>(mattrs.meshsize[0]))) *
                  mattrs.meshsize[1] * mattrs.meshsize[2];

        for (int i1 = bounds[2]; i1 <= bounds[3]; i1++) {
            const size_t n1 = angular
                ? static_cast<size_t>(wrap_periodic_int(i1, nphi))
                : static_cast<size_t>(wrap_periodic_int(i1, static_cast<int>(mattrs.meshsize[1]))) *
                      mattrs.meshsize[2];

            for (int i2 = i2lo; i2 <= i2hi; i2++) {
                const size_t n2 = angular
                    ? 0
                    : static_cast<size_t>(wrap_periodic_int(i2, static_cast<int>(mattrs.meshsize[2])));
                const size_t icell = n0 + n1 + n2;

                for (size_t j = mesh.start[icell]; j < mesh.start[icell + 1]; j++) op(j);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-pair geometry shared by the pair and triplet counts
// ---------------------------------------------------------------------------

inline void difference(double* diff, const double* position1, const double* position2,
                       const MeshAttrs& mattrs) {
    for (int axis = 0; axis < NDIM; axis++) {
        diff[axis] = position1[axis] - position2[axis];
        if (mattrs.periodic)
            diff[axis] = wrap_periodic_float(diff[axis], mattrs.boxsize[axis]);
    }
}

inline bool is_selected_pair(const double* sposition1, const double* sposition2,
                             const double* position1, const double* position2,
                             const SelectionAttrs& sattrs, const MeshAttrs& mattrs) {
    bool selected = true;
    for (size_t i = 0; i < sattrs.ndim; i++) {
        const VAR_TYPE var = sattrs.var[i];
        if (var == VAR_THETA) {
            const double costheta = pairmath::dot3(sposition1, sposition2);
            selected &= (costheta >= sattrs.smin[i]) && (costheta <= sattrs.smax[i]);
        }
        if (var == VAR_S) {
            double diff[NDIM];
            difference(diff, position2, position1, mattrs);
            const double s2 = pairmath::dot3(diff, diff);
            selected &= (s2 >= sattrs.smin[i] * sattrs.smin[i]) &&
                        (s2 <= sattrs.smax[i] * sattrs.smax[i]);
        }
    }
    return selected;
}

}  // namespace cpu
}  // namespace cucount
