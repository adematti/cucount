#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

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
    if (s == "midpoint") return LosKind::Midpoint;
    throw std::invalid_argument("los must be one of: z, midpoint");
}

ScatterKind parse_scatter(const std::string& s) {
    if (s == "scalar") return ScatterKind::Scalar;
    if (s == "binmajor") return ScatterKind::BinMajor;
    throw std::invalid_argument("scatter must be one of: scalar, binmajor");
}

py::object count2_py(Arr positions1, Arr weights1, Arr positions2,
                              Arr weights2, Arr sedges, py::object muedges,
                              std::array<double, 3> boxsize,
                              std::array<double, 3> origin,
                              const std::string& bin, const std::string& los,
                              bool periodic, bool float32,
                              const std::string& scatter, int nthreads,
                              bool return_timings) {
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

    std::vector<size_t> shape{a.nsbins};
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
PYBIND11_MODULE(cpucount, m, py::mod_gil_not_used()) {
    m.doc() = "Portable-SIMD CPU pair counting";

    m.def("count2", &count2_py, py::arg("positions1"), py::arg("weights1"),
          py::arg("positions2"), py::arg("weights2"), py::arg("sedges"),
          py::arg("muedges") = py::none(),
          py::arg("boxsize") = std::array<double, 3>{1.0, 1.0, 1.0},
          py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          py::arg("bin") = "lin", py::arg("los") = "z",
          py::arg("periodic") = false, py::arg("float32") = false,
          py::arg("scatter") = "scalar", py::arg("nthreads") = 1,
          py::arg("return_timings") = false,
          "Count weighted pairs between two catalogues into s or (s, mu) bins.\n"
          "Every ordered pair is visited, matching the CUDA backend, so an\n"
          "autocorrelation counts each pair twice and includes self-pairs.");

    m.def("set_target", &SetTarget, py::arg("name") = "",
          "Restrict Highway to one ISA (e.g. 'AVX2'); '' restores automatic "
          "selection. Returns the target now in use, or None if unavailable.");
    m.def("current_target", &CurrentTarget);
    m.def("available_targets", &AvailableTargets);
}
