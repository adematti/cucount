#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Declare the CUDA function implemented in vector_add.cu
void launch_vector_add(const float *a, const float *b, float *c, int n);


void vector_add_py(py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_c = c.request();

    if (buf_a.size != buf_b.size || buf_a.size != buf_c.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    int n = buf_a.size;
    const float *ptr_a = static_cast<const float *>(buf_a.ptr);
    const float *ptr_b = static_cast<const float *>(buf_b.ptr);
    float *ptr_c = static_cast<float *>(buf_c.ptr);

    launch_vector_add(ptr_a, ptr_b, ptr_c, n);
}

PYBIND11_MODULE(cucount, m) {
    m.def("vector_add", &vector_add_py, "Perform vector addition on the GPU");
}