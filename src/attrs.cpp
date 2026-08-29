// The CUDA-free attrs extension: the Python-facing attribute classes and
// setup_logging, compiled without nvcc. This is what lets `cucount.numpy`
// import (and serve backend='cpu') on a machine or build without CUDA; the
// CUDA extensions register the same classes module_local and accept these
// instances through pybind's foreign module_local loading.
#define CUCOUNT_NO_CUDA
#include "attrs.h"

PYBIND11_MODULE(attrs, m) {
    m.doc() = "Backend-neutral cucount attribute classes";
    register_attrs(m);
}
