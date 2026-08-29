// Cell mesh construction. Plain C++, no Highway. Not yet parallelized.
#pragma once

#include <algorithm>
#include <cmath>

#include "cucount/cpu/types.h"

namespace cucount {
namespace cpu {

inline int wrap_index(int i, int n) {
    int r = i % n;
    return (r < 0) ? r + n : r;
}

// Cells are sized at >= smax so the candidate scan only ever needs the 27
// neighbouring cells; coarsening preserves that, so the particle cap keeps
// the cell count O(n) instead of (boxsize/smax)^3.
inline void mesh_dims(const double boxsize[3], double smax, size_t n,
                      int dims[3]) {
    const double cap = std::cbrt(0.5 * static_cast<double>(n));
    for (int a = 0; a < 3; ++a) {
        const double d = std::min(std::floor(boxsize[a] / smax), cap);
        dims[a] = std::max(1, static_cast<int>(d));
    }
}

// spin: this catalogue's interleaved (e1, e2) components, or null.
// with_spos: fill unit-sphere positions, needed on BOTH meshes whenever either
// side of the pair count carries spin (the projection frame uses both ends).
template <class Float>
Mesh<Float> build_mesh(const double* pos, const double* w, size_t n,
                       const double boxsize[3], const double origin[3],
                       const int dims[3], const double* spin = nullptr,
                       bool with_spos = false) {
    Mesh<Float> m;
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
    }
    return m;
}

}  // namespace cpu
}  // namespace cucount
