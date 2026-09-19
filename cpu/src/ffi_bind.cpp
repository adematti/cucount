// JAX FFI entry points for the CPU backend: cucountlib.ffi_cpu.
//
// Mirrors cuda/src/ffi_bind.cu. The FFI hands the handler only buffers, so
// everything else -- the mesh, binning, weighting, selection and split attrs,
// and the particles' value layout -- is staged in module state by the
// set_*_attrs setters before the call, exactly as the CUDA module does.
//
// Two differences from the CUDA module, both because there is no device. The
// handlers take no stream, and the scratch buffer the CUDA handlers carve
// device allocations out of is accepted and ignored: the CPU paths allocate
// their own per-thread buffers. Keeping it in the signature means the jax
// frontend can call either module through one code path.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cucount/cpu/generic.h"
#include "cucount/cpu/triplet.h"

#define CUCOUNT_NO_CUDA
#include "attrs.h"
#include "layout.h"

#include "cucount/cpu/types.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;
using namespace cucount::cpu;

namespace {

// One owner per entry point, rather than the CUDA module's single list:
// staging count2's attrs must not free the arrays count3 is still pointing at.
struct OwnedArrays {
    std::vector<void*> ptrs;

    ~OwnedArrays() { clear(); }

    void clear() {
        for (void* p : ptrs) std::free(p);
        ptrs.clear();
    }

    double* copy(const double* src, size_t n) {
        double* buf = static_cast<double*>(std::malloc(n * sizeof(double)));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, src, n * sizeof(double));
        ptrs.push_back(buf);
        return buf;
    }

    void own(BinAttrs* battrs) {
        for (size_t i = 0; i < battrs->ndim; i++) {
            if (battrs->asize[i] != 0 && battrs->array[i] != nullptr)
                battrs->array[i] = copy(battrs->array[i], battrs->asize[i]);
        }
    }

    void own(WeightAttrs* wattrs) {
        if (wattrs->bitwise.p_nbits > 0 && wattrs->bitwise.p_correction_nbits != nullptr) {
            const size_t size = wattrs->bitwise.p_nbits * wattrs->bitwise.p_nbits;
            wattrs->bitwise.p_correction_nbits =
                copy(wattrs->bitwise.p_correction_nbits, size);
        }
        for (size_t idim = 0; idim < wattrs->angular.ndim; ++idim) {
            if (wattrs->angular.sep[idim] == nullptr) continue;
            const size_t n = wattrs->angular.sep_is_edges[idim]
                ? wattrs->angular.shape[idim] + 1 : wattrs->angular.shape[idim];
            wattrs->angular.sep[idim] = copy(wattrs->angular.sep[idim], n);
        }
        if (wattrs->angular.size > 0 && wattrs->angular.weight != nullptr)
            wattrs->angular.weight = copy(wattrs->angular.weight, wattrs->angular.size);
    }
};

// ---------------------------------------------------------------------------
// Staged state
// ---------------------------------------------------------------------------

int staged_nthreads = 1;

OwnedArrays owned2;
MeshAttrs mattrs2;
BinAttrs battrs2;
SelectionAttrs sattrs2;
WeightAttrs wattrs2;
SplitAttrs spattrs2;
IndexValue index_value2[2] = {};

OwnedArrays owned3;
MeshAttrs mattrs3_1, mattrs3_2, mattrs3_3;
BinAttrs battrs3_12, battrs3_13, battrs3_23;
SelectionAttrs sattrs3_12, sattrs3_13, sattrs3_23;
SelectionAttrs veto3_12, veto3_13, veto3_23;
WeightAttrs wattrs3;
IndexValue index_value3[3] = {};

Particles ffi_particles(const ffi::Buffer<ffi::F64>& positions,
                        const ffi::Buffer<ffi::F64>& values,
                        IndexValue index_value) {
    Particles particles;
    particles.positions = const_cast<double*>(positions.typed_data());
    particles.values = const_cast<double*>(values.typed_data());
    particles.size = positions.dimensions().front();
    particles.index_value = index_value;
    return particles;
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void set_nthreads_py(int nthreads) { staged_nthreads = nthreads; }

void set_count2_attrs_py(MeshAttrs_py mattrs_py, BinAttrs_py battrs_py,
                         WeightAttrs_py wattrs_py,
                         const SelectionAttrs_py sattrs_py,
                         const SplitAttrs_py spattrs_py) {
    owned2.clear();
    mattrs2 = mattrs_py.data();
    battrs2 = battrs_py.data();
    wattrs2 = wattrs_py.data();
    sattrs2 = sattrs_py.data();
    spattrs2 = spattrs_py.data();
    owned2.own(&battrs2);
    owned2.own(&wattrs2);
}

void set_count2_index_value_py(int iparticle, int size_split, int size_spin,
                               int size_individual_weight, int size_bitwise_weight,
                               int size_negative_weight) {
    index_value2[iparticle] = get_index_value(size_split, size_spin,
                                              size_individual_weight,
                                              size_bitwise_weight,
                                              size_negative_weight);
}

void set_count3_attrs_py(MeshAttrs_py mattrs1_py, MeshAttrs_py mattrs2_py,
                         MeshAttrs_py mattrs3_py, BinAttrs_py battrs12_py,
                         BinAttrs_py battrs13_py, py::object battrs23_obj,
                         WeightAttrs_py wattrs_py,
                         const SelectionAttrs_py sattrs12_py,
                         const SelectionAttrs_py sattrs13_py,
                         const SelectionAttrs_py sattrs23_py,
                         const SelectionAttrs_py veto12_py,
                         const SelectionAttrs_py veto13_py,
                         const SelectionAttrs_py veto23_py) {
    owned3.clear();
    mattrs3_1 = mattrs1_py.data();
    mattrs3_2 = mattrs2_py.data();
    mattrs3_3 = mattrs3_py.data();
    battrs3_12 = battrs12_py.data();
    battrs3_13 = battrs13_py.data();
    if (battrs23_obj.is_none()) std::memset(&battrs3_23, 0, sizeof(BinAttrs));
    else battrs3_23 = py::cast<BinAttrs_py>(battrs23_obj).data();
    wattrs3 = wattrs_py.data();
    sattrs3_12 = sattrs12_py.data();
    sattrs3_13 = sattrs13_py.data();
    sattrs3_23 = sattrs23_py.data();
    veto3_12 = veto12_py.data();
    veto3_13 = veto13_py.data();
    veto3_23 = veto23_py.data();
    owned3.own(&battrs3_12);
    owned3.own(&battrs3_13);
    if (battrs3_23.ndim) owned3.own(&battrs3_23);
    owned3.own(&wattrs3);
}

void set_count3_index_value_py(int iparticle, int size_split, int size_spin,
                               int size_individual_weight, int size_bitwise_weight,
                               int size_negative_weight) {
    index_value3[iparticle] = get_index_value(size_split, size_spin,
                                              size_individual_weight,
                                              size_bitwise_weight,
                                              size_negative_weight);
}

py::tuple shape_tuple(const std::vector<ssize_t>& shape) {
    py::tuple out(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) out[i] = py::int_(shape[i]);
    return out;
}

py::tuple get_count2_layout_py() {
    Count2Layout layout = get_count2_layout(index_value2[0], index_value2[1],
                                            battrs2, spattrs2);
    return py::make_tuple(layout.names, shape_tuple(layout.shape));
}

py::tuple get_count3_layout_py() {
    BinAttrs none{};
    Count3Layout layout = get_count3_out_layout(battrs3_12, battrs3_13, none);
    return py::make_tuple(layout.names, shape_tuple(layout.shape));
}

py::tuple get_count3close_layout_py() {
    Count3Layout layout = get_count3_out_layout(battrs3_12, battrs3_13, battrs3_23);
    return py::make_tuple(layout.names, shape_tuple(layout.shape));
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

// Whether the vectorised kernel covers this request, on the same terms the
// pybind binding uses. Kept here rather than shared because the FFI module
// stages its attrs separately.
bool count2_vectorized() {
    if (mattrs2.type != MESH_CARTESIAN) return false;
    if (spattrs2.nsplits) return false;
    if (index_value2[0].size_split || index_value2[1].size_split) return false;
    const bool ok1 = (battrs2.ndim == 1 && battrs2.var[0] == VAR_S);
    const bool ok2 = (battrs2.ndim == 2 && battrs2.var[0] == VAR_S &&
                      battrs2.var[1] == VAR_MU && battrs2.bin[1] == BIN_LIN);
    const bool okp = (battrs2.ndim == 2 && battrs2.var[0] == VAR_S &&
                      battrs2.var[1] == VAR_POLE);
    return ok1 || ok2 || okp;
}

ffi::Error count2Impl(ffi::Buffer<ffi::F64> positions1, ffi::Buffer<ffi::F64> values1,
                      ffi::Buffer<ffi::F64> positions2, ffi::Buffer<ffi::F64> values2,
                      ffi::ResultBuffer<ffi::F64> counts,
                      ffi::ResultBuffer<ffi::F64> buffer) {
    (void)buffer;  // device scratch on the CUDA side; unused here

    const Particles p1 = ffi_particles(positions1, values1, index_value2[0]);
    const Particles p2 = ffi_particles(positions2, values2, index_value2[1]);

    Count2Layout layout = get_count2_layout(index_value2[0], index_value2[1],
                                            battrs2, spattrs2);
    const size_t csize = layout.nweights * layout.size;
    double* out = counts->typed_data();
    std::memset(out, 0, csize * sizeof(double));
    if (layout.size == 0) return ffi::Error::Success();

    if (!count2_vectorized()) {
        Count2GenericArgs g;
        g.p1 = p1;
        g.p2 = p2;
        g.mattrs = mattrs2;
        g.battrs = battrs2;
        g.wattrs = wattrs2;
        g.sattrs = sattrs2;
        g.spattrs = spattrs2;
        g.nthreads = staged_nthreads;
        g.out = out;
        Count2Generic(g);
        return ffi::Error::Success();
    }

    // The vectorised kernel wants its own flattened columns; the pybind
    // binding does this too, and it is O(n) beside the pair loop.
    Count2Args a;
    a.pos1 = p1.positions;
    a.n1 = p1.size;
    a.pos2 = p2.positions;
    a.n2 = p2.size;

    std::vector<double> w1, w2, spin1, spin2, bw1, bw2, nw1, nw2;
    auto extract = [](const Particles& p, std::vector<double>& w,
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
    };
    extract(p1, w1, spin1, bw1, nw1);
    extract(p2, w2, spin2, bw2, nw2);

    a.w1 = w1.empty() ? nullptr : w1.data();
    a.w2 = w2.empty() ? nullptr : w2.data();
    if (!spin1.empty()) { a.spin1 = spin1.data(); a.spin_order1 = (int)wattrs2.spin[0]; }
    if (!spin2.empty()) { a.spin2 = spin2.data(); a.spin_order2 = (int)wattrs2.spin[1]; }
    if (!bw1.empty() && !bw2.empty()) {
        a.bw1 = bw1.data();
        a.bw2 = bw2.data();
        a.nbitwise = p1.index_value.size_bitwise_weight;
        a.bitwise_default = wattrs2.bitwise.default_value;
        a.bitwise_nrealizations = wattrs2.bitwise.nrealizations;
        a.bitwise_noffset = wattrs2.bitwise.noffset;
        a.bitwise_p_nbits = wattrs2.bitwise.p_nbits;
        a.bitwise_p_correction = wattrs2.bitwise.p_correction_nbits;
    }
    if (!nw1.empty() && !nw2.empty()) { a.nw1 = nw1.data(); a.nw2 = nw2.data(); }
    for (size_t i = 0; i < sattrs2.ndim; i++) {
        if (sattrs2.var[i] == VAR_S) {
            a.sel_s = true;
            a.sel_s_min = sattrs2.smin[i];
            a.sel_s_max = sattrs2.smax[i];
        }
        else {
            a.sel_theta = true;
            a.sel_ct_min = sattrs2.smin[i];
            a.sel_ct_max = sattrs2.smax[i];
        }
    }
    if (wattrs2.angular.size) {
        a.angular_sep = wattrs2.angular.sep[0];
        a.angular_weight = wattrs2.angular.weight;
        a.angular_shape = wattrs2.angular.shape[0];
        a.angular_bin = static_cast<int>(wattrs2.angular.bin[0]);
        a.angular_sep_is_edges = wattrs2.angular.sep_is_edges[0];
    }

    const bool ok2 = (battrs2.ndim == 2 && battrs2.var[1] == VAR_MU);
    const bool okp = (battrs2.ndim == 2 && battrs2.var[1] == VAR_POLE);

    a.sedges = battrs2.array[0];
    a.nsbins = battrs2.shape[0];
    if (ok2) {
        a.muedges = battrs2.array[1];
        a.nmubins = battrs2.shape[1];
    }
    std::vector<int> ells;
    if (okp) {
        for (size_t i = 0; i < battrs2.shape[1]; i++)
            ells.push_back(static_cast<int>(battrs2.array[1][i]));
        a.ells = ells.data();
        a.nells = ells.size();
        a.ells_even = true;
        for (int ell : ells) if (ell % 2) a.ells_even = false;
    }

    if (battrs2.bin[0] == BIN_LIN) a.cfg.sbin = BinKind::Linear;
    else if (battrs2.bin[0] == BIN_LOG ||
             (battrs2.asize[0] > 2 && battrs2.array[0][0] > 0. &&
              is_log(battrs2.array[0], battrs2.asize[0],
                     battrs2.array[0][1] / battrs2.array[0][0])))
        a.cfg.sbin = BinKind::Log;
    else a.cfg.sbin = BinKind::Edges;

    a.cfg.ndim = ok2 ? 2 : 1;
    if (ok2 || okp) {
        switch (battrs2.los[1]) {
            case LOS_Z: a.cfg.los = LosKind::AxisZ; break;
            case LOS_X: a.cfg.los = LosKind::AxisX; break;
            case LOS_Y: a.cfg.los = LosKind::AxisY; break;
            case LOS_MIDPOINT: a.cfg.los = LosKind::Midpoint; break;
            case LOS_FIRSTPOINT: a.cfg.los = LosKind::FirstPoint; break;
            case LOS_ENDPOINT: a.cfg.los = LosKind::EndPoint; break;
            default: return ffi::Error::Internal("cpu backend: line of sight not implemented");
        }
    }
    a.cfg.periodic = mattrs2.periodic;
    a.nthreads = staged_nthreads;
    for (int axis = 0; axis < 3; ++axis) {
        a.boxsize[axis] = mattrs2.boxsize[axis];
        a.origin[axis] = mattrs2.boxcenter[axis] - 0.5 * mattrs2.boxsize[axis];
    }
    a.out = out;

    Count2(a);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count2ffi, count2Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

ffi::Error count3Impl(ffi::Buffer<ffi::F64> positions1, ffi::Buffer<ffi::F64> values1,
                      ffi::Buffer<ffi::F64> positions2, ffi::Buffer<ffi::F64> values2,
                      ffi::Buffer<ffi::F64> positions3, ffi::Buffer<ffi::F64> values3,
                      ffi::ResultBuffer<ffi::F64> counts,
                      ffi::ResultBuffer<ffi::F64> buffer) {
    (void)buffer;

    Count3Args a;
    a.p1 = ffi_particles(positions1, values1, index_value3[0]);
    a.p2 = ffi_particles(positions2, values2, index_value3[1]);
    a.p3 = ffi_particles(positions3, values3, index_value3[2]);
    a.mattrs1 = mattrs3_1;
    a.mattrs2 = mattrs3_2;
    a.mattrs3 = mattrs3_3;
    a.battrs12 = battrs3_12;
    a.battrs13 = battrs3_13;
    a.wattrs = wattrs3;
    a.sattrs12 = sattrs3_12;
    a.sattrs13 = sattrs3_13;
    a.veto12 = veto3_12;
    a.veto13 = veto3_13;
    a.nthreads = staged_nthreads;

    BinAttrs none{};
    Count3Layout layout = get_count3_out_layout(battrs3_12, battrs3_13, none);
    a.out = counts->typed_data();
    std::memset(a.out, 0, layout.nweights * layout.size * sizeof(double));
    if (layout.size == 0) return ffi::Error::Success();

    Count3(a);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3ffi, count3Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

ffi::Error count3closeImpl(ffi::Buffer<ffi::F64> positions1, ffi::Buffer<ffi::F64> values1,
                           ffi::Buffer<ffi::F64> positions2, ffi::Buffer<ffi::F64> values2,
                           ffi::Buffer<ffi::F64> positions3, ffi::Buffer<ffi::F64> values3,
                           ffi::ResultBuffer<ffi::F64> counts,
                           ffi::ResultBuffer<ffi::F64> buffer) {
    (void)buffer;

    Count3CloseArgs a;
    a.p1 = ffi_particles(positions1, values1, index_value3[0]);
    a.p2 = ffi_particles(positions2, values2, index_value3[1]);
    a.p3 = ffi_particles(positions3, values3, index_value3[2]);
    a.mattrs1 = mattrs3_1;
    a.mattrs2 = mattrs3_2;
    a.mattrs3 = mattrs3_3;
    a.battrs12 = battrs3_12;
    a.battrs13 = battrs3_13;
    a.battrs23 = battrs3_23;
    a.wattrs = wattrs3;
    a.sattrs12 = sattrs3_12;
    a.sattrs13 = sattrs3_13;
    a.sattrs23 = sattrs3_23;
    a.veto12 = veto3_12;
    a.veto13 = veto3_13;
    a.veto23 = veto3_23;
    a.nthreads = staged_nthreads;

    Count3Layout layout = get_count3_out_layout(battrs3_12, battrs3_13, battrs3_23);
    a.out = counts->typed_data();
    std::memset(a.out, 0, layout.nweights * layout.size * sizeof(double));
    if (layout.size == 0) return ffi::Error::Success();

    Count3Close(a);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3closeffi, count3closeImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

template <typename T>
py::capsule EncapsulateFfiCall(T* fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "an FFI call must have the signature XLA_FFI_Error*(XLA_FFI_CallFrame*)");
    return py::capsule(reinterpret_cast<void*>(fn));
}

}  // namespace

PYBIND11_MODULE(ffi_cpu, m) {
    m.doc() = "JAX FFI entry points for the portable CPU backend";

    register_attrs(m);

    m.def("set_nthreads", &set_nthreads_py, py::arg("nthreads"),
          "CPU threads each FFI call uses (per device, so shard_map over N\n"
          "devices runs N of these).");

    m.def("set_count2_attrs", &set_count2_attrs_py,
          py::arg("mattrs"), py::arg("battrs"),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs") = SelectionAttrs_py(),
          py::arg("spattrs") = SplitAttrs_py());
    m.def("set_count2_index_value", &set_count2_index_value_py,
          py::arg("iparticle"),
          py::arg("size_split") = 0, py::arg("size_spin") = 0,
          py::arg("size_individual_weight") = 0,
          py::arg("size_bitwise_weight") = 0,
          py::arg("size_negative_weight") = 0);
    m.def("get_count2_layout", &get_count2_layout_py,
          "Return (names, shape) for count2 outputs.");
    m.def("count2", []() { return EncapsulateFfiCall(count2ffi); });

    // The unqualified names the CUDA FFI module also carries for count2, so
    // the jax frontend can hold either module behind one name.
    m.def("set_attrs", &set_count2_attrs_py,
          py::arg("mattrs"), py::arg("battrs"),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs") = SelectionAttrs_py(),
          py::arg("spattrs") = SplitAttrs_py());
    m.def("set_index_value", &set_count2_index_value_py,
          py::arg("iparticle"),
          py::arg("size_split") = 0, py::arg("size_spin") = 0,
          py::arg("size_individual_weight") = 0,
          py::arg("size_bitwise_weight") = 0,
          py::arg("size_negative_weight") = 0);

    // close_pair is accepted for signature parity with the CUDA module and
    // ignored: every search strategy enumerates the same triplets.
    m.def("set_count3close_attrs",
          [](MeshAttrs_py m1, MeshAttrs_py m2, MeshAttrs_py m3, BinAttrs_py b12,
             BinAttrs_py b13, py::object b23, WeightAttrs_py w,
             const SelectionAttrs_py s12, const SelectionAttrs_py s13,
             const SelectionAttrs_py s23, const SelectionAttrs_py v12,
             const SelectionAttrs_py v13, const SelectionAttrs_py v23,
             py::tuple close_pair) {
              (void)close_pair;
              set_count3_attrs_py(m1, m2, m3, b12, b13, b23, w, s12, s13, s23,
                                  v12, v13, v23);
          },
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
          py::arg("close_pair") = py::make_tuple(1, 2));
    m.def("set_count3close_index_value", &set_count3_index_value_py,
          py::arg("iparticle"),
          py::arg("size_split") = 0, py::arg("size_spin") = 0,
          py::arg("size_individual_weight") = 0,
          py::arg("size_bitwise_weight") = 0,
          py::arg("size_negative_weight") = 0);
    m.def("get_count3close_layout", &get_count3close_layout_py,
          "Return (names, shape) for count3close outputs.");
    m.def("count3close", []() { return EncapsulateFfiCall(count3closeffi); });

    // count3 stages the same attrs as count3close, minus the (2, 3) leg, and
    // the CUDA module names these the same way.
    m.def("set_count3_attrs",
          [](MeshAttrs_py m1, MeshAttrs_py m2, MeshAttrs_py m3, BinAttrs_py b12,
             BinAttrs_py b13, WeightAttrs_py w, const SelectionAttrs_py s12,
             const SelectionAttrs_py s13, const SelectionAttrs_py v12,
             const SelectionAttrs_py v13) {
              set_count3_attrs_py(m1, m2, m3, b12, b13, py::none(), w, s12, s13,
                                  SelectionAttrs_py(), v12, v13, SelectionAttrs_py());
          },
          py::arg("mattrs1"), py::arg("mattrs2"), py::arg("mattrs3"),
          py::arg("battrs12"), py::arg("battrs13"),
          py::arg("wattrs") = WeightAttrs_py(),
          py::arg("sattrs12") = SelectionAttrs_py(),
          py::arg("sattrs13") = SelectionAttrs_py(),
          py::arg("veto12") = SelectionAttrs_py(),
          py::arg("veto13") = SelectionAttrs_py());
    m.def("set_count3_index_value", &set_count3_index_value_py,
          py::arg("iparticle"),
          py::arg("size_split") = 0, py::arg("size_spin") = 0,
          py::arg("size_individual_weight") = 0,
          py::arg("size_bitwise_weight") = 0,
          py::arg("size_negative_weight") = 0);
    m.def("get_count3_layout", &get_count3_layout_py,
          "Return (names, shape) for count3 outputs.");
    m.def("count3", []() { return EncapsulateFfiCall(count3ffi); });
}
