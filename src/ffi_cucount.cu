#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <type_traits>
#include <memory>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "mesh.h"
#include "count2.h"
#include "count3close.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

// -----------------------------------------------------------------------------
// Static state for count2
// -----------------------------------------------------------------------------

static MeshAttrs mattrs2;
static BinAttrs battrs2;
static SelectionAttrs sattrs2;
static WeightAttrs wattrs2;
static SplitAttrs spattrs2;
static IndexValue index_value2[2] = {0};

// -----------------------------------------------------------------------------
// Static state for count3close
// -----------------------------------------------------------------------------

static MeshAttrs mattrs3_1;
static MeshAttrs mattrs3_2;
static MeshAttrs mattrs3_3;
static BinAttrs battrs3_12;
static BinAttrs battrs3_13;
static BinAttrs battrs3_23;
static SelectionAttrs sattrs3_12;
static SelectionAttrs sattrs3_13;
static SelectionAttrs sattrs3_23;
static SelectionAttrs veto3_12;
static SelectionAttrs veto3_13;
static SelectionAttrs veto3_23;
static WeightAttrs wattrs3;
static CLOSE_PAIR close_pair_3 = CLOSE_PAIR_12;
static IndexValue index_value3[3] = {0};

// keep ownership of host-side copies created here so pointers stay valid
static std::vector<void*> owned_host_ptrs;

// helper to free owned pointers
static void free_owned_ptrs() {
    for (void *p : owned_host_ptrs) {
        std::free(p);
    }
    owned_host_ptrs.clear();
}

// -----------------------------------------------------------------------------
// Owned host-copy helpers
// -----------------------------------------------------------------------------

static void own_bin_attrs_arrays(BinAttrs *battrs) {
    for (size_t i = 0; i < (size_t)battrs->ndim; i++) {
        if (battrs->asize[i] != 0 && battrs->array[i] != nullptr) {
            FLOAT *buf = (FLOAT*) std::malloc(battrs->asize[i] * sizeof(FLOAT));
            if (!buf) throw std::bad_alloc();
            std::memcpy(buf, battrs->array[i], battrs->asize[i] * sizeof(FLOAT));
            battrs->array[i] = buf;
            owned_host_ptrs.push_back(buf);
        }
    }
}

static void own_weight_attrs_arrays(WeightAttrs *wattrs) {
    // Bitwise
    if (wattrs->bitwise.p_nbits > 0 && wattrs->bitwise.p_correction_nbits != nullptr) {
        size_t size = wattrs->bitwise.p_nbits * wattrs->bitwise.p_nbits;
        FLOAT *buf = (FLOAT*) std::malloc(size * sizeof(FLOAT));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, wattrs->bitwise.p_correction_nbits, size * sizeof(FLOAT));
        wattrs->bitwise.p_correction_nbits = buf;
        owned_host_ptrs.push_back(buf);
    }

    // Angular axis arrays
    for (size_t idim = 0; idim < wattrs->angular.ndim; ++idim) {
        if (wattrs->angular.sep[idim] != nullptr) {
            const size_t n = wattrs->angular.shape[idim];
            FLOAT *buf = (FLOAT*) std::malloc(n * sizeof(FLOAT));
            if (!buf) throw std::bad_alloc();
            std::memcpy(buf, wattrs->angular.sep[idim], n * sizeof(FLOAT));
            wattrs->angular.sep[idim] = buf;
            owned_host_ptrs.push_back(buf);
        }
        if (wattrs->angular.edges[idim] != nullptr) {
            const size_t n = wattrs->angular.shape[idim] + 1;
            FLOAT *buf = (FLOAT*) std::malloc(n * sizeof(FLOAT));
            if (!buf) throw std::bad_alloc();
            std::memcpy(buf, wattrs->angular.edges[idim], n * sizeof(FLOAT));
            wattrs->angular.edges[idim] = buf;
            owned_host_ptrs.push_back(buf);
        }
    }

    // Angular weight table
    if (wattrs->angular.size > 0 && wattrs->angular.weight != nullptr) {
        FLOAT *buf = (FLOAT*) std::malloc(wattrs->angular.size * sizeof(FLOAT));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, wattrs->angular.weight, wattrs->angular.size * sizeof(FLOAT));
        wattrs->angular.weight = buf;
        owned_host_ptrs.push_back(buf);
    }
}

// -----------------------------------------------------------------------------
// Python setters for count2 attrs
// -----------------------------------------------------------------------------

void set_count2_attrs_py(
    MeshAttrs_py mattrs_py,
    BinAttrs_py battrs_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
    const SplitAttrs_py spattrs_py = SplitAttrs_py())
{
    free_owned_ptrs();

    mattrs2 = mattrs_py.data();
    battrs2 = battrs_py.data();
    wattrs2 = wattrs_py.data();
    sattrs2 = sattrs_py.data();
    spattrs2 = spattrs_py.data();

    own_bin_attrs_arrays(&battrs2);
    own_weight_attrs_arrays(&wattrs2);
}

// -----------------------------------------------------------------------------
// Python setters for count3close attrs
// -----------------------------------------------------------------------------

void set_count3close_attrs_py(
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    py::object battrs23_obj = py::none(),
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs23_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto23_py = SelectionAttrs_py(),
    py::tuple close_pair = py::make_tuple(1, 2))
{
    free_owned_ptrs();

    mattrs3_1 = mattrs1_py.data();
    mattrs3_2 = mattrs2_py.data();
    mattrs3_3 = mattrs3_py.data();

    battrs3_12 = battrs12_py.data();
    battrs3_13 = battrs13_py.data();

    if (battrs23_obj.is_none()) {
        std::memset(&battrs3_23, 0, sizeof(BinAttrs));
    }
    else {
        battrs3_23 = py::cast<BinAttrs_py>(battrs23_obj).data();
        own_bin_attrs_arrays(&battrs3_23);
    }

    wattrs3 = wattrs_py.data();

    sattrs3_12 = sattrs12_py.data();
    sattrs3_13 = sattrs13_py.data();
    sattrs3_23 = sattrs23_py.data();

    veto3_12 = veto12_py.data();
    veto3_13 = veto13_py.data();
    veto3_23 = veto23_py.data();

    close_pair_3 = parse_close_pair(close_pair);

    own_bin_attrs_arrays(&battrs3_12);
    own_bin_attrs_arrays(&battrs3_13);
    own_weight_attrs_arrays(&wattrs3);
}

// -----------------------------------------------------------------------------
// Index-value setters
// -----------------------------------------------------------------------------

void set_count2_index_value_py(
    const size_t iparticle,
    const int size_split = 0,
    const int size_spin = 0,
    const int size_individual_weight = 0,
    const int size_bitwise_weight = 0,
    const int size_negative_weight = 0)
{
    index_value2[iparticle] = get_index_value(
        size_split, size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
}

void set_count3close_index_value_py(
    const size_t iparticle,
    const int size_split = 0,
    const int size_spin = 0,
    const int size_individual_weight = 0,
    const int size_bitwise_weight = 0,
    const int size_negative_weight = 0)
{
    index_value3[iparticle] = get_index_value(
        size_split, size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
}

// -----------------------------------------------------------------------------
// Layout helpers
// -----------------------------------------------------------------------------

py::tuple get_count2_layout_py()
{
    Count2Layout layout = get_count2_layout(
        index_value2[0],
        index_value2[1],
        battrs2,
        spattrs2);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}

py::tuple get_count3close_layout_py()
{
    Count3CloseLayout layout = get_count3close_layout(
        battrs3_12,
        battrs3_13,
        battrs3_23);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}

// -----------------------------------------------------------------------------
// FFI helpers
// -----------------------------------------------------------------------------

void set_mem_buffer(DeviceMemoryBuffer *membuffer, ffi::ResultBuffer<ffi::F64> buffer) {
    membuffer->ptr = (void *) buffer->typed_data();
    membuffer->size = buffer->dimensions().front() * 8 / sizeof(char);
    membuffer->offset = 0;
}

Particles get_ffi_particles(
    ffi::Buffer<ffi::F64> positions,
    ffi::Buffer<ffi::F64> values,
    IndexValue index_value)
{
    Particles particles;
    particles.positions = positions.typed_data();
    particles.values = values.typed_data();
    particles.size = positions.dimensions().front();
    particles.index_value = index_value;
    return particles;
}

// -----------------------------------------------------------------------------
// count2 FFI impl
// -----------------------------------------------------------------------------

ffi::Error count2Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> positions1,
    ffi::Buffer<ffi::F64> values1,
    ffi::Buffer<ffi::F64> positions2,
    ffi::Buffer<ffi::F64> values2,
    ffi::ResultBuffer<ffi::F64> counts,
    ffi::ResultBuffer<ffi::F64> buffer)
{
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
        list_mesh[imesh].total_nparticles = 0;
    }

    list_particles[0] = get_ffi_particles(positions1, values1, index_value2[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value2[1]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = 256;

    set_mesh(list_particles, list_mesh, mattrs2, &membuffer, stream);
    count2(counts->typed_data(), list_mesh, mattrs2, sattrs2, battrs2, wattrs2, spattrs2, &membuffer, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count2ffi, count2Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

// -----------------------------------------------------------------------------
// count3close FFI impl
// -----------------------------------------------------------------------------

ffi::Error count3closeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> positions1,
    ffi::Buffer<ffi::F64> values1,
    ffi::Buffer<ffi::F64> positions2,
    ffi::Buffer<ffi::F64> values2,
    ffi::Buffer<ffi::F64> positions3,
    ffi::Buffer<ffi::F64> values3,
    ffi::ResultBuffer<ffi::F64> counts,
    ffi::ResultBuffer<ffi::F64> buffer)
{
    Particles list_particles[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
    }

    list_particles[0] = get_ffi_particles(positions1, values1, index_value3[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value3[1]);
    list_particles[2] = get_ffi_particles(positions3, values3, index_value3[2]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = 256;

    Mesh mesh1{};
    Mesh mesh2{};
    Mesh mesh3{};

    Particles plist[MAX_NMESH];
    Mesh mlist[MAX_NMESH];

    for (size_t i = 0; i < MAX_NMESH; ++i) {
        plist[i].size = 0;
        mlist[i].total_nparticles = 0;
    }

    plist[0] = list_particles[0];
    set_mesh(plist, mlist, mattrs3_1, &membuffer, stream);
    mesh1 = mlist[0];

    plist[0] = list_particles[1];
    set_mesh(plist, mlist, mattrs3_2, &membuffer, stream);
    mesh2 = mlist[0];

    plist[0] = list_particles[2];
    set_mesh(plist, mlist, mattrs3_3, &membuffer, stream);
    mesh3 = mlist[0];

    count3_close(counts->typed_data(), mesh1,mesh2, mesh3,
                mattrs3_1, mattrs3_2, mattrs3_3,
                sattrs3_12, sattrs3_13, sattrs3_23,
                veto3_12, veto3_13, veto3_23,
                battrs3_12, battrs3_13, battrs3_23,
                wattrs3, close_pair_3, &membuffer, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3closeffi, count3closeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

// -----------------------------------------------------------------------------
// Capsule helper
// -----------------------------------------------------------------------------

template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return py::capsule(reinterpret_cast<void *>(fn));
}

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(ffi_cucount, m) {
    py::class_<BinAttrs_py>(m, "BinAttrs", py::module_local())
        .def(py::init<py::kwargs>())
        .def_property_readonly("shape", &BinAttrs_py::shape)
        .def_property_readonly("size", &BinAttrs_py::size)
        .def_property_readonly("ndim", &BinAttrs_py::ndim)
        .def_property_readonly("varnames", &BinAttrs_py::varnames,
            "Return list of variable names in order (e.g. ['s','mu'])")
        .def_property_readonly("losnames", &BinAttrs_py::losnames,
            "Return list of line-of-sight names in order")
        .def_readonly("var", &BinAttrs_py::var)
        .def_readonly("min", &BinAttrs_py::min)
        .def_readonly("max", &BinAttrs_py::max)
        .def_readonly("step", &BinAttrs_py::step)
        .def_property_readonly("array",
            [](const BinAttrs_py &b) -> std::vector<py::array_t<FLOAT>> { return b.array; });

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs", py::module_local())
        .def(py::init<py::kwargs>())
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_property_readonly("varnames", &SelectionAttrs_py::varnames,
            "Return list of variable names in order (e.g. ['theta'])")
        .def_readonly("var", &SelectionAttrs_py::var)
        .def_readonly("min", &SelectionAttrs_py::min)
        .def_readonly("max", &SelectionAttrs_py::max);

    py::class_<WeightAttrs_py>(m, "WeightAttrs", py::module_local())
        .def(py::init<py::kwargs>());

    py::class_<MeshAttrs_py>(m, "MeshAttrs", py::module_local())
        .def(py::init<py::kwargs>());

    py::class_<SplitAttrs_py>(m, "SplitAttrs", py::module_local())
        .def(py::init<py::kwargs>())
        .def_readonly("nsplits", &SplitAttrs_py::nsplits)
        .def_readonly("size", &SplitAttrs_py::size);

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    // count2 setup
    m.def("set_count2_attrs", &set_count2_attrs_py, "Set count2 attributes",
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py());

    m.def("set_count2_index_value", &set_count2_index_value_py, "Set count2 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def("count2", []() { return EncapsulateFfiCall(count2ffi); });

    // backward-compatible aliases
    m.def("set_attrs", &set_count2_attrs_py, "Set count2 attributes",
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py());

    m.def("set_index_value", &set_count2_index_value_py, "Set count2 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count2_layout",
        &get_count2_layout_py,
        "Return (names, shape) for count2 outputs."
    );

    // count3close setup
    m.def("set_count3close_attrs", &set_count3close_attrs_py,
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("battrs23") = py::none(),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("sattrs23") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py(),
        py::arg("veto23") = SelectionAttrs_py(),
        py::arg("close_pair") = py::make_tuple(1, 2));

    m.def("set_count3close_index_value", &set_count3close_index_value_py,
        "Set count3close value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count3close_layout",
        &get_count3close_layout_py,
        "Return (names, shape) for count3close outputs."
    );

    m.def("count3close", []() { return EncapsulateFfiCall(count3closeffi); });
}