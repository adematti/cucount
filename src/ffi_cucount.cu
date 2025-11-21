#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "mesh.h"
#include "count2.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

static SelectionAttrs sattrs;
static BinAttrs battrs;
static WeightAttrs wattrs;
static IndexValue index_value[2] = {0};


// keep ownership of host-side copies created here so pointers stay valid
static std::vector<void*> owned_host_ptrs;

// helper to free owned pointers
static void free_owned_ptrs() {
    for (void *p : owned_host_ptrs) {
        std::free(p);
    }
    owned_host_ptrs.clear();
}


void set_attrs_py(BinAttrs_py battrs_py, WeightAttrs_py wattrs_py = WeightAttrs_py(), const SelectionAttrs_py sattrs_py = SelectionAttrs_py()) {
    // free previously allocated host copies (if any)
    free_owned_ptrs();

    battrs = battrs_py.data();
    wattrs = wattrs_py.data();
    sattrs = sattrs_py.data();

    // --- Make owned host copies for battrs.array entries (if provided) ---
    // Assumes battrs.asize[i] holds element count and battrs_py.array[i] is provided.
    for (size_t i = 0; i < (size_t)battrs.ndim; i++) {
        if (battrs.asize[i] != 0) {
            // compute count and allocate
            FLOAT *buf = (FLOAT*) std::malloc(battrs.asize[i] * sizeof(FLOAT));
            if (!buf) throw std::bad_alloc();
            // copy from py::array_t backing memory
            std::memcpy(buf, battrs.array[i], battrs.asize[i] * sizeof(FLOAT));
            battrs.array[i] = buf;
            owned_host_ptrs.push_back(buf);
        }
    }
    // --- Make owned host copies for wattrs.bitwise.p_correction_nbits (if provided) ---
    if (wattrs.bitwise.p_nbits > 0) {
        size_t size = wattrs.bitwise.p_nbits * wattrs.bitwise.p_nbits;
        FLOAT *buf = (FLOAT*) std::malloc(size * sizeof(FLOAT));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, wattrs.bitwise.p_correction_nbits, size * sizeof(FLOAT));
        wattrs.bitwise.p_correction_nbits = buf;
        owned_host_ptrs.push_back(buf);
    }
    // --- Make owned host copies for wattrs.angular.sep / weight (if provided) ---
    if (wattrs.angular.size > 0) {
        FLOAT *sep_buf = (FLOAT*) std::malloc(wattrs.angular.size * sizeof(FLOAT));
        FLOAT *weight_buf = (FLOAT*) std::malloc(wattrs.angular.size * sizeof(FLOAT));
        if (!sep_buf || !weight_buf) {std::free(sep_buf); std::free(weight_buf); throw std::bad_alloc();}
        std::memcpy(sep_buf, wattrs.angular.sep, wattrs.angular.size * sizeof(FLOAT));
        std::memcpy(weight_buf, wattrs.angular.weight, wattrs.angular.size * sizeof(FLOAT));
        wattrs.angular.sep = sep_buf;
        wattrs.angular.weight = weight_buf;
        owned_host_ptrs.push_back(sep_buf);
        owned_host_ptrs.push_back(weight_buf);
    }
}


void set_index_value_py(const size_t iparticle, const int size_spin = 0, const int size_individual_weight = 0, const int size_bitwise_weight = 0, const int size_negative_weight = 0) {
    index_value[iparticle] = get_index_value(size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
}


std::vector<std::string> get_count2_names_py() {
    char names[MAX_NWEIGHT][SIZE_NAME];
    size_t ncounts = get_count2_size(index_value[0], index_value[1], names);
    std::vector<std::string> toret;
    for (size_t icount = 0; icount < ncounts; icount++) toret.push_back(names[icount]);
    return toret;
}


void set_mem_buffer(DeviceMemoryBuffer *membuffer, ffi::ResultBuffer<ffi::F64> buffer) {
    membuffer->ptr = (void *) buffer->typed_data();
    membuffer->size = buffer->dimensions().front() * 8 / sizeof(char);
    //CUDA_CHECK(cudaMemset(membuffer.ptr, 0, membuffer.size));
    membuffer->offset = 0;
}


Particles get_ffi_particles(ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> values, IndexValue index_value) {
    Particles particles;
    particles.positions = positions.typed_data();
    particles.values = values.typed_data();
    particles.size = positions.dimensions().front();
    particles.index_value = index_value;
    return particles;
}



ffi::Error count2Impl(cudaStream_t stream,
                      ffi::Buffer<ffi::F64> positions1,
                      ffi::Buffer<ffi::F64> values1,
                      ffi::Buffer<ffi::F64> positions2,
                      ffi::Buffer<ffi::F64> values2,
                      ffi::ResultBuffer<ffi::F64> counts,
                      ffi::ResultBuffer<ffi::F64> buffer) {

    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    list_particles[0] = get_ffi_particles(positions1, values1, index_value[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value[1]);
    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = 256;
    membuffer.meshsize = (list_particles[0].size + list_particles[1].size) / 2;
    MeshAttrs mattrs;
    prepare_mesh_attrs(&mattrs, battrs, sattrs);
    set_mesh_attrs(list_particles, &mattrs, &membuffer, stream);
    set_mesh(list_particles, list_mesh, mattrs, &membuffer, stream);
    // Perform the computation
    count2(counts->typed_data(), list_mesh, mattrs, sattrs, battrs, wattrs, &membuffer, stream);
    free_owned_ptrs();

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
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);


template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}


// Bind the function and structs to Python
PYBIND11_MODULE(ffi_cucount, m) {
    // Register classes first!
    py::class_<BinAttrs_py>(m, "BinAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("shape", &BinAttrs_py::shape)
        .def_property_readonly("size", &BinAttrs_py::size)
        .def_property_readonly("ndim", &BinAttrs_py::ndim)
        .def_property_readonly("varnames", &BinAttrs_py::varnames, "Return list of variable names in order (e.g. ['s','mu'])")
        .def_readonly("var", &BinAttrs_py::var)
        .def_readonly("min", &BinAttrs_py::min)
        .def_readonly("max", &BinAttrs_py::max)
        .def_readonly("step", &BinAttrs_py::step) // The lambda approach is safer to avoid exposing internal mutable containers
        .def_property_readonly("array", [](const BinAttrs_py &b) -> std::vector<py::array_t<FLOAT>> {return b.array;});

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_readonly("var", &SelectionAttrs_py::var)
        .def_readonly("min", &SelectionAttrs_py::min)
        .def_readonly("max", &SelectionAttrs_py::max);

    py::class_<WeightAttrs_py>(m, "WeightAttrs", py::module_local())
        .def(py::init<py::kwargs>()); // Accept Python kwargs

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    m.def("set_attrs", &set_attrs_py, "Set attributes",
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(), // Default value
        py::arg("sattrs") = SelectionAttrs_py()); // Default value

    m.def("set_index_value", &set_index_value_py, "Set value indices",
        py::arg("iparticle"),
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    // Expose helper to get output names
    m.def("get_count2_names",
          &get_count2_names_py,
          "Return list of output names for count2.");

    m.def("count2", []() { return EncapsulateFfiCall(count2ffi); });
}