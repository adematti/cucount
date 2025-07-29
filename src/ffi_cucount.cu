#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
static DeviceMemoryBuffer membuffer;


void set_attrs_py(BinAttrs_py& battrs_py, const SelectionAttrs_py& sattrs_py = SelectionAttrs_py(), const size_t nblocks = 256) {
    battrs = battrs_py.data();
    sattrs = sattrs_py.data();
    membuffer.nbocks = nblocks;
}


Particles get_ffi_particles(ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> weights) {
    Particles particles;
    particles.positions = positions.typed_data();
    particles.weights = weights.typed_data();
    particles.size = positions.dimensions().front();
}



ffi::Error count2Impl(cudaStream_t stream,
                      ffi::Buffer<ffi::F64> positions1,
                      ffi::Buffer<ffi::F64> weights1,
                      ffi::Buffer<ffi::F64> positions2,
                      ffi::Buffer<ffi::F64> weights2,
                      ffi::ResultBuffer<ffi::F64> counts,
                      ffi::ResultBuffer<ffi::F64> buffer) {

    Particles list_particles[MAX_NMESH];
    list_particles[0] = get_ffi_particles(positions1, weights1);
    list_particles[1] = get_ffi_particles(positions2, weights2);
    membuffer.ptr = (void *) buffer->untyped_data();
    prepare_mesh_attrs(&mattrs, battrs, sattrs);
    set_mesh_attrs(list_particles, &mattrs, membuffer, stream);
    set_mesh(list_particles, list_mesh, mattrs, membuffer, stream);
    // Perform the computation
    count2(counts->typed_data(), list_mesh, mattrs, sattrs, battrs, membuffer, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
      return ffi::Error::Internal(
          std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count2, count2Impl,
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
    m.def("count2", []() { return EncapsulateFfiCall(count2); });
    m.def("set_attrs", &set_attrs_py, "Set attributes",
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py(), // Default value
        py::arg("nblocks") = 256); // Default value
}