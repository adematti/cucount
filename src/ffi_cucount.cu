#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>

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


float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y) {
  float sm = 0.0f;
  for (int64_t n = 0; n < size; ++n) {
    sm += x[n] * x[n];
  }
  float scale = 1.0f / std::sqrt(sm / float(size) + eps);
  for (int64_t n = 0; n < size; ++n) {
    y[n] = x[n] * scale;
  }
  return scale;
}

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error RmsNormImpl(ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  std::cout << "Bin " << battrs.min[0] << " " << battrs.max[0] << std::endl;
  std::cout << "Selection " << sattrs.min[0] << " " << sattrs.max[0] << std::endl;
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeRmsNorm(1e-4, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
  }
  return ffi::Error::Success();
}

__global__ void square_kernel(const float *a, float *b, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b[i] = a[i] * a[i];
  }
}

ffi::Error RmsNormImpl_cuda(cudaStream_t stream,
                       ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
    const int block_dim = 128;
    const int grid_dim = 1;
    std::cout << "Bin " << battrs.min[0] << " " << battrs.max[0] << std::endl;
    std::cout << "Selection " << sattrs.min[0] << " " << sattrs.max[0] << std::endl;
    auto dims = x.dimensions();
  // Note how we access regular Buffer data vs Result Buffer data:
    square_kernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      x.typed_data(), y->typed_data(), dims.back());
  // Check for launch time errors. Note that this function may also
  // return error codes from previous, asynchronous launches. This
  // means that an error status returned here could have been caused
  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm, RmsNormImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm_cuda, RmsNormImpl_cuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
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
    m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
    m.def("rms_norm_cuda", []() { return EncapsulateFfiCall(RmsNorm_cuda); });
    m.def("set_attrs", &set_attrs_py, "Set attributes",
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}