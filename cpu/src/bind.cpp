#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// First, because it pulls in cmath.h, which undefines common.h's
// convenience macros on its way out; attrs.h's own include of common.h then
// restores them for the rest of this translation unit.
#include "count2.h"
#include "count3close.h"

// The shared attrs/layout/args layer (CUDA-free flavour): the binding takes
// the same Particles/BinAttrs/MeshAttrs/... objects as the CUDA backend,
// bundles them into the shared Count2Attrs / Count3Attrs, and lowers them to
// the kernel's flattened Count2KernelArgs here, in C++.
#define CUCOUNT_NO_CUDA
#include "attrs.h"
#include "layout.h"

namespace py = pybind11;
using namespace cucount::cpu;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// LOS_NONE is the one LOS_TYPE the kernel cannot serve: an axis has to be
// named once mu is binned or projected on. Every other value goes straight
// into Config, which holds the shared enum.
LOS_TYPE checked_los(LOS_TYPE los) {
    if (los == LOS_NONE)
        throw std::invalid_argument("cpu backend: line of sight not implemented");
    return los;
}

ScatterKind parse_scatter(const std::string& s) {
    if (s == "scalar") return ScatterKind::Scalar;
    if (s == "binmajor") return ScatterKind::BinMajor;
    throw std::invalid_argument("scatter must be one of: scalar, binmajor");
}

// The Python shim declines unsupported requests by name before calling in;
// these checks are the defensive backstop, not the user-facing message.
// They bound what the backend as a whole serves, not what the vectorised
// kernel serves -- everything else falls through to the generic path.
void validate(const Particles& p1, const Particles& p2, const BinAttrs& battrs,
              const MeshAttrs& mattrs, const WeightAttrs& wattrs,
              const SelectionAttrs& sattrs, const SplitAttrs& spattrs) {
    if (mattrs.type != MESH_CARTESIAN && mattrs.type != MESH_ANGULAR)
        throw std::invalid_argument("cpu backend: mesh type not implemented");
    // Two bounds, both shared with the CUDA kernel: the ell list is held in a
    // buffer of MAX_POLE + 2, and the Legendre cache of MAX_POLE + 1 entries is
    // indexed BY ell, so the largest ell is what matters. The frontend checks
    // both and says so; this is the backstop that keeps a direct call to the
    // extension from writing past either array.
    for (size_t i = 0; i < battrs.ndim; i++) {
        if (battrs.var[i] != VAR_POLE) continue;
        if (battrs.shape[i] > MAX_POLE + 2)
            throw std::invalid_argument("cpu backend: too many multipoles requested");
        for (size_t j = 0; j < battrs.shape[i]; j++) {
            if (battrs.array[i][j] > MAX_POLE)
                throw std::invalid_argument(
                    "cpu backend: multipole ell above MAX_POLE requested");
        }
    }
    for (size_t i = 0; i < sattrs.ndim; i++) {
        if (sattrs.var[i] != VAR_S && sattrs.var[i] != VAR_THETA)
            throw std::invalid_argument(
                "cpu backend: only s and theta selections are implemented");
    }
    // Matching the CUDA count2, which looks the angular upweight up against
    // one cos(theta) axis; the N-dimensional form belongs to the triplet
    // counts, which this backend does not serve yet.
    if (wattrs.angular.size && wattrs.angular.ndim != 1)
        throw std::invalid_argument(
            "cpu backend: only 1D angular weights are implemented");
    for (const Particles* p : {&p1, &p2}) {
        const IndexValue& iv = p->index_value;
        if (iv.size_split > 1)
            throw std::invalid_argument("cpu backend: only one split label is supported");
        if (iv.size_spin && iv.size_spin != 2)
            throw std::invalid_argument("cpu backend: spin needs exactly 2 components");
        if (iv.size_negative_weight > 1)
            throw std::invalid_argument("cpu backend: only one negative weight is supported");
    }
    if (spattrs.nsplits && !(p1.index_value.size_split && p2.index_value.size_split))
        throw std::invalid_argument(
            "cpu backend: jackknife splits need a split label on both catalogues");
    if (p1.index_value.size_bitwise_weight != p2.index_value.size_bitwise_weight)
        throw std::invalid_argument(
            "cpu backend: both catalogues must carry the same number of bitwise weights");
}

// Whether the vectorised Highway kernel covers this request. Anything outside
// it is served by the scalar generic path, which is slower but complete; the
// result is the same either way, and the tests check that on the overlap.
bool vectorized(const Particles& p1, const Particles& p2, const BinAttrs& battrs,
                const MeshAttrs& mattrs, const SplitAttrs& spattrs) {
    if (mattrs.type != MESH_CARTESIAN) return false;
    if (spattrs.nsplits) return false;
    if (p1.index_value.size_split || p2.index_value.size_split) return false;
    const bool ok1 = (battrs.ndim == 1 && battrs.var[0] == VAR_S);
    const bool ok2 = (battrs.ndim == 2 && battrs.var[0] == VAR_S &&
                      battrs.var[1] == VAR_MU && battrs.bin[1] == BIN_LIN);
    const bool okp = (battrs.ndim == 2 && battrs.var[0] == VAR_S &&
                      battrs.var[1] == VAR_POLE);
    return ok1 || ok2 || okp;
}

// Contiguous copies of the packed-value columns the kernel consumes; O(n)
// doubles, trivial next to the pair loop.
void extract_columns(const Particles& p, std::vector<double>& w,
                     std::vector<double>& spin, std::vector<double>& bw,
                     std::vector<double>& nw) {
    const IndexValue& iv = p.index_value;
    if (iv.size_individual_weight) {
        w.resize(p.size);
        for (size_t i = 0; i < p.size; i++)
            w[i] = p.values[i * iv.size + iv.start_individual_weight];
    }
    if (iv.size_spin) {
        spin.resize(2 * p.size);
        for (size_t i = 0; i < p.size; i++) {
            spin[2 * i + 0] = p.values[i * iv.size + iv.start_spin + 0];
            spin[2 * i + 1] = p.values[i * iv.size + iv.start_spin + 1];
        }
    }
    if (iv.size_bitwise_weight) {
        // Bit patterns riding double storage; copied verbatim.
        bw.resize(p.size * iv.size_bitwise_weight);
        for (size_t i = 0; i < p.size; i++)
            for (size_t ib = 0; ib < iv.size_bitwise_weight; ib++)
                bw[i * iv.size_bitwise_weight + ib] =
                    p.values[i * iv.size + iv.start_bitwise_weight + ib];
    }
    if (iv.size_negative_weight) {
        nw.resize(p.size);
        for (size_t i = 0; i < p.size; i++)
            nw[i] = p.values[i * iv.size + iv.start_negative_weight];
    }
}

// Reshape the flat accumulator into the named per-channel arrays. Both the
// vectorised and the generic path land here, so the two cannot present their
// results differently.
py::object finish(py::array_t<double>& counts_py, const Count2Layout& layout,
                  const double* timings, bool return_timings) {
    py::dict result;
    for (size_t iweight = 0; iweight < layout.nweights; ++iweight) {
        py::array_t<double> array_py(
            {(ssize_t)layout.size},
            {(ssize_t)sizeof(double)},
            counts_py.data() + iweight * layout.size,
            counts_py);
        result[layout.names[iweight].c_str()] =
            array_py.attr("reshape")(layout.shape).cast<py::array_t<double>>();
    }
    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(timings[0], timings[1]));
    }
    return std::move(result);
}

// The attrs-based entry point, mirroring cucountlib.cucount.count2.
py::object count2_py(Particles_py& particles1, Particles_py& particles2,
                     MeshAttrs_py mattrs_py, BinAttrs_py battrs_py,
                     WeightAttrs_py wattrs_py,
                     const SelectionAttrs_py sattrs_py,
                     const SplitAttrs_py spattrs_py,
                     const int nthreads, const bool float32,
                     const std::string& scatter, const bool return_timings) {
    BinAttrs battrs = battrs_py.data();
    MeshAttrs mattrs = mattrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    SplitAttrs spattrs = spattrs_py.data();
    Particles p1 = particles1.data();
    Particles p2 = particles2.data();

    validate(p1, p2, battrs, mattrs, wattrs, sattrs, spattrs);

    // Output through the shared layout, so names, ordering and shape cannot
    // diverge from the CUDA binding. Allocated before the path split, because
    // both paths fill this one buffer.
    Count2Layout layout = get_count2_layout(p1.index_value, p2.index_value,
                                            battrs, spattrs);
    const size_t csize = layout.nweights * layout.size;
    py::array_t<double> counts_py(csize);
    std::memset(counts_py.mutable_data(), 0, csize * sizeof(double));

    double timings[2] = {0.0, 0.0};
    double* timings_ptr = return_timings ? timings : nullptr;

    // Zero requested bins is served as the empty result, like the CUDA backend.
    if (layout.size == 0)
        return finish(counts_py, layout, timings, return_timings);

    // Anything the vectorised kernel does not cover -- theta / rp / pi / k
    // axes, the angular mesh, jackknife splits -- goes to the scalar generic
    // path, which walks the same mesh one pair at a time. float32 and the
    // scatter strategy are kernel tuning and have no effect there.
    if (!vectorized(p1, p2, battrs, mattrs, spattrs)) {
        Count2Args g;
        g.p1 = p1;
        g.p2 = p2;
        g.attrs.mattrs = mattrs;
        g.attrs.battrs = battrs;
        g.attrs.wattrs = wattrs;
        g.attrs.sattrs = sattrs;
        g.attrs.spattrs = spattrs;
        g.nthreads = nthreads;
        g.out = counts_py.mutable_data();
        g.timings = timings_ptr;
        {
            py::gil_scoped_release unlock;
            Count2(g);
        }
        return finish(counts_py, layout, timings, return_timings);
    }

    // Which second axis: mu bins, or the multipole channel axis. (ndim == 2
    // alone is ambiguous -- (s, pole) has ndim 2 as well.)
    const bool ok2 = (battrs.ndim == 2 && battrs.var[1] == VAR_MU);
    const bool okp = (battrs.ndim == 2 && battrs.var[1] == VAR_POLE);

    Count2KernelArgs a;
    a.pos1 = p1.positions;
    a.n1 = p1.size;
    a.pos2 = p2.positions;
    a.n2 = p2.size;

    std::vector<double> w1, w2, spin1, spin2, bw1, bw2, nw1, nw2;
    extract_columns(p1, w1, spin1, bw1, nw1);
    extract_columns(p2, w2, spin2, bw2, nw2);
    a.w1 = w1.empty() ? nullptr : w1.data();
    a.w2 = w2.empty() ? nullptr : w2.data();
    if (!spin1.empty()) {
        a.spin1 = spin1.data();
        a.spin_order1 = static_cast<int>(wattrs.spin[0]);
    }
    if (!spin2.empty()) {
        a.spin2 = spin2.data();
        a.spin_order2 = static_cast<int>(wattrs.spin[1]);
    }
    if (!bw1.empty() && !bw2.empty()) {
        a.bw1 = bw1.data();
        a.bw2 = bw2.data();
        a.nbitwise = p1.index_value.size_bitwise_weight;
        a.bitwise_default = wattrs.bitwise.default_value;
        a.bitwise_nrealizations = wattrs.bitwise.nrealizations;
        a.bitwise_noffset = wattrs.bitwise.noffset;
        a.bitwise_p_nbits = wattrs.bitwise.p_nbits;
        // Points into arrays wattrs_py keeps alive for the call.
        a.bitwise_p_correction = wattrs.bitwise.p_correction_nbits;
    }
    if (!nw1.empty() && !nw2.empty()) {
        a.nw1 = nw1.data();
        a.nw2 = nw2.data();
    }
    for (size_t i = 0; i < sattrs.ndim; i++) {
        // smin/smax are already the comparison bounds: s for VAR_S, and
        // cos(theta) (max/min swapped) for VAR_THETA, per SelectionAttrs::data.
        if (sattrs.var[i] == VAR_S) {
            a.sel_s = true;
            a.sel_s_min = sattrs.smin[i];
            a.sel_s_max = sattrs.smax[i];
        } else {
            a.sel_theta = true;
            a.sel_ct_min = sattrs.smin[i];
            a.sel_ct_max = sattrs.smax[i];
        }
    }
    if (wattrs.angular.size) {
        // 1D only (validated above); points into arrays wattrs_py keeps
        // alive for the call, axes already ascending cos(theta).
        a.angular_sep = wattrs.angular.sep[0];
        a.angular_weight = wattrs.angular.weight;
        a.angular_shape = wattrs.angular.shape[0];
        a.angular_bin = static_cast<int>(wattrs.angular.bin[0]);
        a.angular_sep_is_edges = wattrs.angular.sep_is_edges[0];
    }

    // Edges point into the numpy buffers held by battrs_py for the call.
    a.sedges = battrs.array[0];
    a.nsbins = battrs.shape[0];
    if (ok2) {
        a.muedges = battrs.array[1];
        a.nmubins = battrs.shape[1];
    }
    // Multipoles: the array holds the ell VALUES (not edges), and they are the
    // fastest output axis. ells_even picks set_legendre's even-only closed
    // forms; anything else takes the full recursion, exactly as on the GPU.
    std::vector<int> ells;
    if (okp) {
        for (size_t i = 0; i < battrs.shape[1]; i++)
            ells.push_back(static_cast<int>(battrs.array[1][i]));
        a.ells = ells.data();
        a.nells = ells.size();
        a.ells_even = true;
        for (int ell : ells) if (ell % 2) a.ells_even = false;
    }

    // s-bin policy from the shared classification: data() marks BIN_LIN via
    // is_linear, but flags BIN_LOG only above 1000 bins; re-check smaller
    // grids here so they keep the Log fast path.
    if (battrs.bin[0] == BIN_LIN) {
        a.cfg.sbin = BIN_LIN;
    } else if (battrs.bin[0] == BIN_LOG ||
               (battrs.asize[0] > 2 && battrs.array[0][0] > 0. &&
                is_log(battrs.array[0], battrs.asize[0],
                       battrs.array[0][1] / battrs.array[0][0]))) {
        a.cfg.sbin = BIN_LOG;
    } else {
        a.cfg.sbin = BIN_CUSTOM;
    }

    a.cfg.ndim = ok2 ? 2 : 1;
    a.cfg.los = (ok2 || okp) ? checked_los(battrs.los[1]) : LOS_Z;
    a.cfg.periodic = mattrs.periodic;
    a.cfg.float32 = float32;
    a.cfg.scatter = parse_scatter(scatter);
    a.nthreads = nthreads;

    // The mesh comes from MeshAttrs, exactly as the CUDA backend receives it,
    // so meshsize= and refine= mean the same thing on both backends.
    a.smax = mattrs.smax;
    for (int axis = 0; axis < 3; ++axis) {
        a.boxsize[axis] = mattrs.boxsize[axis];
        a.origin[axis] = mattrs.boxcenter[axis] - 0.5 * mattrs.boxsize[axis];
        a.meshsize[axis] = mattrs.meshsize[axis];
    }

    a.out = counts_py.mutable_data();
    a.timings = timings_ptr;

    {
        py::gil_scoped_release unlock;
        Count2Kernel(a);
    }

    return finish(counts_py, layout, timings, return_timings);
}

// What the factorized triplet counts require of each leg, mirroring what
// add_pair_weight can actually bin on.
void validate3(const BinAttrs& battrs12, const BinAttrs& battrs13,
               const MeshAttrs& mattrs1, const MeshAttrs& mattrs2,
               const MeshAttrs& mattrs3) {
    for (const MeshAttrs* m : {&mattrs1, &mattrs2, &mattrs3}) {
        if (m->type != MESH_CARTESIAN && m->type != MESH_ANGULAR)
            throw std::invalid_argument("cpu backend: mesh type not implemented");
    }
    for (const BinAttrs* b : {&battrs12, &battrs13}) {
        if (b->ndim < 1 || b->ndim > 2)
            throw std::invalid_argument(
                "cpu backend: each triplet leg takes one separation axis and an "
                "optional multipole axis");
        if (b->var[0] != VAR_S && b->var[0] != VAR_THETA)
            throw std::invalid_argument(
                "cpu backend: triplet legs bin in s or theta only");
        if (b->ndim == 2 && b->var[1] != VAR_POLE)
            throw std::invalid_argument(
                "cpu backend: the second triplet axis must be the multipole axis");
    }
    const bool pole12 = (battrs12.ndim == 2 && battrs12.var[1] == VAR_POLE);
    const bool pole13 = (battrs13.ndim == 2 && battrs13.var[1] == VAR_POLE);
    if (pole12 != pole13)
        throw std::invalid_argument(
            "cpu backend: either both triplet legs carry a multipole axis or neither");
    if (pole12) {
        for (const BinAttrs* b : {&battrs12, &battrs13}) {
            if (b->max[1] > ELLMAX)
                throw std::invalid_argument(
                    "cpu backend: triplet multipoles are implemented up to ell = 5");
        }
    }
}

// The attrs-based triplet entry point, mirroring cucountlib.cuda.count3.
py::object count3_py(Particles_py& particles1, Particles_py& particles2,
                     Particles_py& particles3,
                     MeshAttrs_py mattrs1_py, MeshAttrs_py mattrs2_py,
                     MeshAttrs_py mattrs3_py,
                     BinAttrs_py battrs12_py, BinAttrs_py battrs13_py,
                     WeightAttrs_py wattrs_py,
                     const SelectionAttrs_py sattrs12_py,
                     const SelectionAttrs_py sattrs13_py,
                     const SelectionAttrs_py veto12_py,
                     const SelectionAttrs_py veto13_py,
                     const int nthreads, const bool return_timings) {
    Count3Args a;
    a.attrs.mattrs1 = mattrs1_py.data();
    a.attrs.mattrs2 = mattrs2_py.data();
    a.attrs.mattrs3 = mattrs3_py.data();
    a.attrs.battrs12 = battrs12_py.data();
    a.attrs.battrs13 = battrs13_py.data();
    a.attrs.wattrs = wattrs_py.data();
    a.attrs.sattrs12 = sattrs12_py.data();
    a.attrs.sattrs13 = sattrs13_py.data();
    a.attrs.veto12 = veto12_py.data();
    a.attrs.veto13 = veto13_py.data();
    a.p1 = particles1.data();
    a.p2 = particles2.data();
    a.p3 = particles3.data();
    a.nthreads = nthreads;

    validate3(a.attrs.battrs12, a.attrs.battrs13, a.attrs.mattrs1, a.attrs.mattrs2, a.attrs.mattrs3);

    BinAttrs battrs23{};
    Count3Layout layout = get_count3_layout(a.attrs.battrs12, a.attrs.battrs13, battrs23);
    const size_t csize = layout.nweights * layout.size;
    py::array_t<double> counts_py(csize);
    std::memset(counts_py.mutable_data(), 0, csize * sizeof(double));
    a.out = counts_py.mutable_data();

    double timings[2] = {0.0, 0.0};
    if (return_timings) a.timings = timings;

    if (layout.size != 0) {
        py::gil_scoped_release unlock;
        Count3(a);
    }

    py::dict result;
    result[layout.names[0].c_str()] =
        counts_py.attr("reshape")(layout.shape).cast<py::array_t<double>>();
    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(timings[0], timings[1]));
    }
    return std::move(result);
}

// The attrs-based close-triplet entry point, mirroring
// cucountlib.cuda.count3close. close_pair is accepted for signature parity and
// ignored: it selects among search strategies that all enumerate the same
// triplets, and this backend has one.
py::object count3close_py(Particles_py& particles1, Particles_py& particles2,
                          Particles_py& particles3,
                          MeshAttrs_py mattrs1_py, MeshAttrs_py mattrs2_py,
                          MeshAttrs_py mattrs3_py,
                          BinAttrs_py battrs12_py, BinAttrs_py battrs13_py,
                          py::object battrs23_py,
                          WeightAttrs_py wattrs_py,
                          const SelectionAttrs_py sattrs12_py,
                          const SelectionAttrs_py sattrs13_py,
                          const SelectionAttrs_py sattrs23_py,
                          const SelectionAttrs_py veto12_py,
                          const SelectionAttrs_py veto13_py,
                          const SelectionAttrs_py veto23_py,
                          py::tuple close_pair,
                          const int nthreads, const bool return_timings) {
    (void)close_pair;

    Count3Args a;
    a.attrs.mattrs1 = mattrs1_py.data();
    a.attrs.mattrs2 = mattrs2_py.data();
    a.attrs.mattrs3 = mattrs3_py.data();
    a.attrs.battrs12 = battrs12_py.data();
    a.attrs.battrs13 = battrs13_py.data();
    std::memset(&a.attrs.battrs23, 0, sizeof(BinAttrs));
    // Held for the duration of the call, so battrs23.array stays valid.
    BinAttrs_py battrs23_held = battrs23_py.is_none()
        ? BinAttrs_py(py::kwargs()) : battrs23_py.cast<BinAttrs_py>();
    if (!battrs23_py.is_none()) a.attrs.battrs23 = battrs23_held.data();

    a.attrs.wattrs = wattrs_py.data();
    a.attrs.sattrs12 = sattrs12_py.data();
    a.attrs.sattrs13 = sattrs13_py.data();
    a.attrs.sattrs23 = sattrs23_py.data();
    a.attrs.veto12 = veto12_py.data();
    a.attrs.veto13 = veto13_py.data();
    a.attrs.veto23 = veto23_py.data();
    a.p1 = particles1.data();
    a.p2 = particles2.data();
    a.p3 = particles3.data();
    a.nthreads = nthreads;

    validate3(a.attrs.battrs12, a.attrs.battrs13, a.attrs.mattrs1, a.attrs.mattrs2, a.attrs.mattrs3);
    if (a.attrs.battrs23.ndim > 0) {
        if (a.attrs.battrs23.ndim != 1 ||
            (a.attrs.battrs23.var[0] != VAR_S && a.attrs.battrs23.var[0] != VAR_THETA))
            throw std::invalid_argument(
                "cpu backend: the (2, 3) triplet axis bins in s or theta only");
    }

    Count3Layout layout = get_count3_layout(a.attrs.battrs12, a.attrs.battrs13, a.attrs.battrs23);
    const size_t csize = layout.nweights * layout.size;
    py::array_t<double> counts_py(csize);
    std::memset(counts_py.mutable_data(), 0, csize * sizeof(double));
    a.out = counts_py.mutable_data();

    double timings[2] = {0.0, 0.0};
    if (return_timings) a.timings = timings;

    if (layout.size != 0) {
        py::gil_scoped_release unlock;
        Count3Close(a);
    }

    py::dict result;
    result[layout.names[0].c_str()] =
        counts_py.attr("reshape")(layout.shape).cast<py::array_t<double>>();
    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(timings[0], timings[1]));
    }
    return std::move(result);
}

}  // namespace

// The kernel touches no Python objects and releases the GIL, so the module
// is safe to import into a free-threaded interpreter without re-enabling it.
PYBIND11_MODULE(cpu, m, py::mod_gil_not_used()) {
    m.doc() = "Portable-SIMD CPU pair counting";

    // Same attrs classes as the other extensions (module_local, one shared
    // registration); the frontend's cucount_attrs instances also cast in
    // through pybind's foreign module_local loading.
    register_attrs(m);

    m.def("count2", &count2_py,
          py::arg("particles1"), py::arg("particles2"),
          py::arg("mattrs"), py::arg("battrs"),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs") = SelectionAttrs_py(),
          py::arg("spattrs") = SplitAttrs_py(),
          py::arg("nthreads") = 1,
          py::arg("float32") = false,
          py::arg("scatter") = "scalar",
          py::arg("return_timings") = false,
          "Pair counts on the CPU with the same signature and named-channel\n"
          "output as cucountlib.cucount.count2; nthreads means CPU threads.\n"
          "Every ordered pair is visited, matching the CUDA backend, so an\n"
          "autocorrelation counts each pair twice and includes self-pairs.");

    m.def("count3", &count3_py,
          py::arg("particles1"), py::arg("particles2"), py::arg("particles3"),
          py::arg("mattrs1"), py::arg("mattrs2"), py::arg("mattrs3"),
          py::arg("battrs12"), py::arg("battrs13"),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs12") = SelectionAttrs_py(),
          py::arg("sattrs13") = SelectionAttrs_py(),
          py::arg("veto12") = SelectionAttrs_py(),
          py::arg("veto13") = SelectionAttrs_py(),
          py::arg("nthreads") = 1,
          py::arg("return_timings") = false,
          "Factorized triplet counts on the CPU, with the same signature and\n"
          "output as cucountlib.cuda.count3; nthreads means CPU threads.");

    m.def("count3close", &count3close_py,
          py::arg("particles1"), py::arg("particles2"), py::arg("particles3"),
          py::arg("mattrs1"), py::arg("mattrs2"), py::arg("mattrs3"),
          py::arg("battrs12"), py::arg("battrs13"),
          py::arg("battrs23") = py::none(),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs12") = SelectionAttrs_py(),
          py::arg("sattrs13") = SelectionAttrs_py(),
          py::arg("sattrs23") = SelectionAttrs_py(),
          py::arg("veto12") = SelectionAttrs_py(),
          py::arg("veto13") = SelectionAttrs_py(),
          py::arg("veto23") = SelectionAttrs_py(),
          py::arg("close_pair") = py::make_tuple(1, 2),
          py::arg("nthreads") = 1,
          py::arg("return_timings") = false,
          "Close triplet counts on the CPU, with the same signature and output\n"
          "as cucountlib.cuda.count3close; nthreads means CPU threads and\n"
          "close_pair is accepted for parity and ignored.");

    m.def("set_target", &SetTarget, py::arg("name") = "",
          "Restrict Highway to one ISA (e.g. 'AVX2'); '' restores automatic "
          "selection. Returns the target now in use, or None if unavailable.");
    m.def("current_target", &CurrentTarget);
    m.def("available_targets", &AvailableTargets);
}
