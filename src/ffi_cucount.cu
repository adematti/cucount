#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "mesh.h"
#include "count2.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

static BinAttrs battrs;
static SelectionAttrs sattrs;


void set_attrs_py(BinAttrs_py& battrs_py, const SelectionAttrs_py& sattrs_py = SelectionAttrs_py()) {
    battrs = battrs_py.data();
    sattrs = sattrs_py.data();
}


void set_mem_buffer(DeviceMemoryBuffer *membuffer, ffi::ResultBuffer<ffi::F64> buffer) {
    membuffer->ptr = (void *) buffer->typed_data();
    membuffer->size = buffer->dimensions().front() * 8 / sizeof(char);
    //CUDA_CHECK(cudaMemset(membuffer.ptr, 0, membuffer.size));
    membuffer->offset = 0;
}


Particles get_ffi_particles(ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> weights) {
    Particles particles;
    particles.positions = positions.typed_data();
    particles.weights = weights.typed_data();
    particles.size = positions.dimensions().front();
    return particles;
}



ffi::Error count2Impl(cudaStream_t stream,
                      ffi::Buffer<ffi::F64> positions1,
                      ffi::Buffer<ffi::F64> weights1,
                      ffi::Buffer<ffi::F64> positions2,
                      ffi::Buffer<ffi::F64> weights2,
                      ffi::ResultBuffer<ffi::F64> counts,
                      ffi::ResultBuffer<ffi::F64> buffer) {

    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    list_particles[0] = get_ffi_particles(positions1, weights1);
    list_particles[1] = get_ffi_particles(positions2, weights2);
    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = 256;
    membuffer.meshsize = (list_particles[0].size + list_particles[1].size) / 2;
    MeshAttrs mattrs;
    prepare_mesh_attrs(&mattrs, battrs, sattrs);
    set_mesh_attrs(list_particles, &mattrs, &membuffer, stream);
    set_mesh(list_particles, list_mesh, mattrs, &membuffer, stream);
    // Perform the computation
    count2(counts->typed_data(), list_mesh, mattrs, sattrs, battrs, &membuffer, stream);

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
        .def_readwrite("var", &BinAttrs_py::var)
        .def_readwrite("min", &BinAttrs_py::min)
        .def_readwrite("max", &BinAttrs_py::max)
        .def_readwrite("step", &BinAttrs_py::step);

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_readwrite("var", &SelectionAttrs_py::var)
        .def_readwrite("min", &SelectionAttrs_py::min)
        .def_readwrite("max", &SelectionAttrs_py::max);

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    m.def("set_attrs", &set_attrs_py, "Set attributes",
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value

    m.def("count2", []() { return EncapsulateFfiCall(count2ffi); });
}