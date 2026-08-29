#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// The shared attrs/layout layer (CUDA-free flavour): the binding takes the
// same Particles/BinAttrs/MeshAttrs/... objects as the CUDA backend and
// lowers them to the kernel's Count2Args here, in C++.
#define CUCOUNT_NO_CUDA
#include "attrs.h"
#include "layout.h"

#include "cucount/cpu/types.h"

namespace py = pybind11;
using namespace cucount::cpu;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

BinKind parse_bin(const std::string& s) {
    if (s == "lin") return BinKind::Linear;
    if (s == "log") return BinKind::Log;
    if (s == "edges") return BinKind::Edges;
    throw std::invalid_argument("bin must be one of: lin, log, edges");
}

LosKind parse_los(const std::string& s) {
    if (s == "z") return LosKind::AxisZ;
    if (s == "x") return LosKind::AxisX;
    if (s == "y") return LosKind::AxisY;
    if (s == "midpoint") return LosKind::Midpoint;
    if (s == "firstpoint") return LosKind::FirstPoint;
    if (s == "endpoint") return LosKind::EndPoint;
    throw std::invalid_argument(
        "los must be one of: z, x, y, midpoint, firstpoint, endpoint");
}

LosKind los_from_type(LOS_TYPE los) {
    switch (los) {
        case LOS_Z: return LosKind::AxisZ;
        case LOS_X: return LosKind::AxisX;
        case LOS_Y: return LosKind::AxisY;
        case LOS_MIDPOINT: return LosKind::Midpoint;
        case LOS_FIRSTPOINT: return LosKind::FirstPoint;
        case LOS_ENDPOINT: return LosKind::EndPoint;
        default:
            throw std::invalid_argument("cpu backend: line of sight not implemented");
    }
}

ScatterKind parse_scatter(const std::string& s) {
    if (s == "scalar") return ScatterKind::Scalar;
    if (s == "binmajor") return ScatterKind::BinMajor;
    throw std::invalid_argument("scatter must be one of: scalar, binmajor");
}

// The Python shim declines unsupported requests by name before calling in;
// these checks are the defensive backstop, not the user-facing message.
void validate(const Particles& p1, const Particles& p2, const BinAttrs& battrs,
              const MeshAttrs& mattrs, const WeightAttrs& wattrs,
              const SelectionAttrs& sattrs, const SplitAttrs& spattrs) {
    if (mattrs.type != MESH_CARTESIAN)
        throw std::invalid_argument("cpu backend: only the cartesian mesh is implemented");
    const bool ok1 = (battrs.ndim == 1 && battrs.var[0] == VAR_S);
    const bool ok2 = (battrs.ndim == 2 && battrs.var[0] == VAR_S && battrs.var[1] == VAR_MU);
    if (!ok1 && !ok2)
        throw std::invalid_argument("cpu backend: only s and (s, mu) binning are implemented");
    if (ok2 && battrs.bin[1] != BIN_LIN)
        throw std::invalid_argument("cpu backend: non-linear mu binning not implemented");
    if (sattrs.ndim)
        throw std::invalid_argument("cpu backend: selections not implemented");
    if (spattrs.nsplits)
        throw std::invalid_argument("cpu backend: jackknife splits not implemented");
    if (wattrs.angular.size)
        throw std::invalid_argument("cpu backend: angular weights not implemented");
    for (const Particles* p : {&p1, &p2}) {
        const IndexValue& iv = p->index_value;
        if (iv.size_split)
            throw std::invalid_argument("cpu backend: weight scheme not implemented");
        if (iv.size_spin && iv.size_spin != 2)
            throw std::invalid_argument("cpu backend: spin needs exactly 2 components");
        if (iv.size_negative_weight > 1)
            throw std::invalid_argument("cpu backend: only one negative weight is supported");
    }
    if (p1.index_value.size_bitwise_weight != p2.index_value.size_bitwise_weight)
        throw std::invalid_argument(
            "cpu backend: both catalogues must carry the same number of bitwise weights");
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
    const bool ok2 = (battrs.ndim == 2);

    Count2Args a;
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

    // Edges point into the numpy buffers held by battrs_py for the call.
    a.sedges = battrs.array[0];
    a.nsbins = battrs.shape[0];
    if (ok2) {
        a.muedges = battrs.array[1];
        a.nmubins = battrs.shape[1];
    }

    // s-bin policy from the shared classification: data() marks BIN_LIN via
    // is_linear, but flags BIN_LOG only above 1000 bins; re-check smaller
    // grids here so they keep the Log fast path.
    if (battrs.bin[0] == BIN_LIN) {
        a.cfg.sbin = BinKind::Linear;
    } else if (battrs.bin[0] == BIN_LOG ||
               (battrs.asize[0] > 2 && battrs.array[0][0] > 0. &&
                is_log(battrs.array[0], battrs.asize[0],
                       battrs.array[0][1] / battrs.array[0][0]))) {
        a.cfg.sbin = BinKind::Log;
    } else {
        a.cfg.sbin = BinKind::Edges;
    }

    a.cfg.ndim = ok2 ? 2 : 1;
    a.cfg.los = ok2 ? los_from_type(battrs.los[1]) : LosKind::AxisZ;
    a.cfg.periodic = mattrs.periodic;
    a.cfg.float32 = float32;
    a.cfg.scatter = parse_scatter(scatter);
    a.nthreads = nthreads;

    for (int axis = 0; axis < 3; ++axis) {
        a.boxsize[axis] = mattrs.boxsize[axis];
        a.origin[axis] = mattrs.boxcenter[axis] - 0.5 * mattrs.boxsize[axis];
    }

    // Output through the shared layout, so names, ordering and shape cannot
    // diverge from the CUDA binding.
    Count2Layout layout = get_count2_layout(p1.index_value, p2.index_value,
                                            battrs, spattrs);
    const size_t csize = layout.nweights * layout.size;
    py::array_t<double> counts_py(csize);
    std::memset(counts_py.mutable_data(), 0, csize * sizeof(double));
    a.out = counts_py.mutable_data();

    double timings[2] = {0.0, 0.0};
    if (return_timings) a.timings = timings;

    // Zero requested bins is served as the empty result, like the CUDA backend.
    if (layout.size != 0) {
        py::gil_scoped_release unlock;
        Count2(a);
    }

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

// Low-level raw-array entry point, kept for the tests that exercise axes the
// public API hides (float32, scatter strategy, cross-ISA agreement).
py::object count2_arrays_py(Arr positions1, Arr weights1, Arr positions2,
                              Arr weights2, Arr sedges, py::object muedges,
                              std::array<double, 3> boxsize,
                              std::array<double, 3> origin,
                              const std::string& bin, const std::string& los,
                              bool periodic, bool float32,
                              const std::string& scatter, int nthreads,
                              bool return_timings,
                              py::object spin1, py::object spin2,
                              int spin_order1, int spin_order2) {
    if (positions1.ndim() != 2 || positions1.shape(1) != 3)
        throw std::invalid_argument("positions1 must have shape (n, 3)");
    if (positions2.ndim() != 2 || positions2.shape(1) != 3)
        throw std::invalid_argument("positions2 must have shape (n, 3)");

    Count2Args a;
    a.pos1 = positions1.data();
    a.w1 = weights1.size() ? weights1.data() : nullptr;
    a.n1 = static_cast<size_t>(positions1.shape(0));
    a.pos2 = positions2.data();
    a.w2 = weights2.size() ? weights2.data() : nullptr;
    a.n2 = static_cast<size_t>(positions2.shape(0));

    // Spin components stay alive through the call via these locals.
    Arr sp1, sp2;
    if (!spin1.is_none()) {
        sp1 = spin1.cast<Arr>();
        if (sp1.ndim() != 2 || sp1.shape(1) != 2 ||
            static_cast<size_t>(sp1.shape(0)) != a.n1)
            throw std::invalid_argument("spin1 must have shape (n1, 2)");
        a.spin1 = sp1.data();
        a.spin_order1 = spin_order1;
    }
    if (!spin2.is_none()) {
        sp2 = spin2.cast<Arr>();
        if (sp2.ndim() != 2 || sp2.shape(1) != 2 ||
            static_cast<size_t>(sp2.shape(0)) != a.n2)
            throw std::invalid_argument("spin2 must have shape (n2, 2)");
        a.spin2 = sp2.data();
        a.spin_order2 = spin_order2;
    }

    if (sedges.size() == 0)
        throw std::invalid_argument("sedges must not be empty");
    a.sedges = sedges.data();
    a.nsbins = static_cast<size_t>(sedges.size()) - 1;

    Arr mu;
    if (!muedges.is_none()) {
        mu = muedges.cast<Arr>();
        if (mu.size() == 0)
            throw std::invalid_argument("muedges must not be empty");
        a.muedges = mu.data();
        a.nmubins = static_cast<size_t>(mu.size()) - 1;
    }

    for (int i = 0; i < 3; ++i) {
        a.boxsize[i] = boxsize[i];
        a.origin[i] = origin[i];
    }

    a.cfg.ndim = muedges.is_none() ? 1 : 2;
    a.cfg.sbin = parse_bin(bin);
    a.cfg.los = parse_los(los);
    a.cfg.periodic = periodic;
    a.cfg.float32 = float32;
    a.cfg.scatter = parse_scatter(scatter);
    a.nthreads = nthreads;

    // Spin channels replace the plain weight (CUDA naming: weight_plus, ...);
    // they lead the shape so each channel is contiguous, like the CUDA layout.
    const size_t nw = 1 + (a.spin1 ? 1 : 0) + (a.spin2 ? 1 : 0);
    std::vector<size_t> shape;
    if (nw > 1) shape.push_back(nw);
    shape.push_back(a.nsbins);
    if (a.cfg.ndim == 2) shape.push_back(a.nmubins);
    py::array_t<double> out(shape);
    std::memset(out.mutable_data(), 0, out.size() * sizeof(double));
    a.out = out.mutable_data();

    double timings[2] = {0.0, 0.0};
    if (return_timings) a.timings = timings;

    // Zero requested bins is served as the empty result, like the CUDA backend.
    if (out.size() != 0) {
        py::gil_scoped_release unlock;
        Count2(a);
    }
    if (return_timings) {
        return py::make_tuple(out, py::make_tuple(timings[0], timings[1]));
    }
    return std::move(out);
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

    m.def("count2_arrays", &count2_arrays_py, py::arg("positions1"), py::arg("weights1"),
          py::arg("positions2"), py::arg("weights2"), py::arg("sedges"),
          py::arg("muedges") = py::none(),
          py::arg("boxsize") = std::array<double, 3>{1.0, 1.0, 1.0},
          py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          py::arg("bin") = "lin", py::arg("los") = "z",
          py::arg("periodic") = false, py::arg("float32") = false,
          py::arg("scatter") = "scalar", py::arg("nthreads") = 1,
          py::arg("return_timings") = false,
          py::arg("spin1") = py::none(), py::arg("spin2") = py::none(),
          py::arg("spin_order1") = 0, py::arg("spin_order2") = 0,
          "Low-level raw-array pair counts (single flat output; spin channels\n"
          "lead the shape when present). Prefer count2, which takes the same\n"
          "attrs objects as the CUDA backend.");

    m.def("set_target", &SetTarget, py::arg("name") = "",
          "Restrict Highway to one ISA (e.g. 'AVX2'); '' restores automatic "
          "selection. Returns the target now in use, or None if unavailable.");
    m.def("current_target", &CurrentTarget);
    m.def("available_targets", &AvailableTargets);
}
