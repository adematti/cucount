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
#include "count3.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

#define MAX_NBLOCKS 256
#define MAX_NTHREADS_PER_BLOCK 512

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

// -----------------------------------------------------------------------------
// Owned host-copy helpers
// -----------------------------------------------------------------------------

// The staged attrs point into numpy buffers the caller may drop as soon as the
// setter returns, so every array is copied here and kept alive until the next
// staging of the same entry point.
//
// One owner per entry point, not one list for the module: staging count2's
// attrs must not free the arrays count3 is still pointing at. Identical to
// OwnedArrays in cpu/src/ffi_bind.cpp.
struct OwnedArrays {
    std::vector<void*> ptrs;

    ~OwnedArrays() { clear(); }

    void clear() {
        for (void* p : ptrs) std::free(p);
        ptrs.clear();
    }

    FLOAT* copy(const FLOAT* src, size_t n) {
        FLOAT* buf = (FLOAT*) std::malloc(n * sizeof(FLOAT));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, src, n * sizeof(FLOAT));
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
            // sep_is_edges is per axis: indexed, not taken whole. Read as a
            // scalar it is an array decaying to a pointer, so always true,
            // and every axis is copied one element long.
            const size_t n = wattrs->angular.sep_is_edges[idim]
                ? wattrs->angular.shape[idim] + 1 : wattrs->angular.shape[idim];
            wattrs->angular.sep[idim] = copy(wattrs->angular.sep[idim], n);
        }
        if (wattrs->angular.size > 0 && wattrs->angular.weight != nullptr)
            wattrs->angular.weight = copy(wattrs->angular.weight, wattrs->angular.size);
    }
};

static OwnedArrays owned2;
static OwnedArrays owned3;

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
    owned2.clear();

    mattrs2 = mattrs_py.data();
    battrs2 = battrs_py.data();
    wattrs2 = wattrs_py.data();
    sattrs2 = sattrs_py.data();
    spattrs2 = spattrs_py.data();

    owned2.own(&battrs2);
    owned2.own(&wattrs2);
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
    owned3.clear();

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
    }

    wattrs3 = wattrs_py.data();

    sattrs3_12 = sattrs12_py.data();
    sattrs3_13 = sattrs13_py.data();
    sattrs3_23 = sattrs23_py.data();

    veto3_12 = veto12_py.data();
    veto3_13 = veto13_py.data();
    veto3_23 = veto23_py.data();

    close_pair_3 = parse_close_pair(close_pair);

    owned3.own(&battrs3_12);
    owned3.own(&battrs3_13);
    if (battrs3_23.ndim) owned3.own(&battrs3_23);
    owned3.own(&wattrs3);
}


void set_count3_attrs_py(
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py())
{
    set_count3close_attrs_py(
        mattrs1_py,
        mattrs2_py,
        mattrs3_py,
        battrs12_py,
        battrs13_py,
        py::none(),
        wattrs_py,
        sattrs12_py,
        sattrs13_py,
        SelectionAttrs_py(),
        veto12_py,
        veto13_py,
        SelectionAttrs_py(),
        py::make_tuple(1, 2));
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
    Count3Layout layout = get_count3_layout(
        battrs3_12,
        battrs3_13,
        battrs3_23);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}


py::tuple get_count3_layout_py()
{
    BinAttrs battrs23{};
    Count3Layout layout = get_count3_layout(
        battrs3_12,
        battrs3_13,
        battrs23);

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
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;  // not used in memory allocation

    set_mesh(list_particles, list_mesh, mattrs2, &membuffer, stream);
    const Count2Attrs attrs{mattrs2, battrs2, wattrs2, sattrs2, spattrs2};
    count2(counts->typed_data(), list_mesh, attrs, &membuffer, stream);

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
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;  // not used in memory allocation

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

    const Count3Attrs attrs{
        mattrs3_1, mattrs3_2, mattrs3_3,
        battrs3_12, battrs3_13, battrs3_23,
        wattrs3,
        sattrs3_12, sattrs3_13, sattrs3_23,
        veto3_12, veto3_13, veto3_23};

    count3close(counts->typed_data(), mesh1, mesh2, mesh3,
                attrs, close_pair_3, &membuffer, stream);

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



ffi::Error count3Impl(
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
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK;

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

    // count3 has no (2, 3) axis, so those members stay default.
    const Count3Attrs attrs{
        MeshAttrs{}, mattrs3_2, mattrs3_3,
        battrs3_12, battrs3_13, BinAttrs{},
        wattrs3,
        sattrs3_12, sattrs3_13, SelectionAttrs{},
        veto3_12, veto3_13, SelectionAttrs{}};

    count3(
        counts->typed_data(),
        mesh1,
        mesh2,
        mesh3,
        attrs,
        &membuffer,
        stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }

    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3ffi, count3Impl,
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

PYBIND11_MODULE(ffi_cuda, m) {
    register_attrs(m);

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

    m.def("set_count3_attrs", &set_count3_attrs_py,
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py());

    m.def("set_count3_index_value", &set_count3close_index_value_py,
        "Set count3 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count3_layout",
        &get_count3_layout_py,
        "Return (names, shape) for count3 outputs."
    );

    m.def("count3", []() { return EncapsulateFfiCall(count3ffi); });

}