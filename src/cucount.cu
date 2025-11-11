#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // for std::vector conversion
#include <cstring>  // for std::memcpy

#include "mesh.h"
#include "count2.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;


static bool is_contiguous(py::array array) {
    return array.flags() & py::array::c_style;
}



// Expose the Particles struct to Python
// NB: We could have made this more Pythonic, with arguments "spin_values", "individual_weights", "bitwise_weights", etc.
// But for simplicity, and given we also have ffi_count.cu, let's differ this to pure Python
struct Particles_py {
    py::array_t<FLOAT> positions;
    py::array_t<FLOAT> values; // Optional spin values array
    IndexValue index_value;

    // Single constructor accepting optional sky_coords and spin_values (can be None)
    Particles_py(py::array_t<FLOAT> positions_, py::array_t<FLOAT> values_ = py::none(),
        const int size_spin = 0, const int size_individual_weight = 0, const int size_bitwise_weight = 0)
        : positions(positions_), values(py::array_t<FLOAT>()) {

        this->index_value = get_index_value(size_spin, size_individual_weight, size_bitwise_weight);

        // Ensure positions are C-contiguous
        if (!is_contiguous(this->positions)) this->positions = py::array_t<FLOAT>(this->positions.attr("copy")());
        size_t npositions = this->positions.shape(0);

        if (this->index_value.size) {
            if (py::isinstance<py::none>(values_)) {
                throw std::invalid_argument(
                    "Particles_py: non-trivial values are indicated with size_*, but input values are empty")
                );
            }
            auto array = py::cast<py::array_t<FLOAT>>(weights_);
            if (!is_contiguous(array)) array = py::array_t<FLOAT>(array.attr("copy")());
            if (array.shape(0) != npositions) {
                throw std::invalid_argument(
                    "Particles_py: positions and values must have the same length, but got positions.shape(0) = " +
                    std::to_string(npositions) + " and values.shape(0) = " + std::to_string(array.shape(0))
                );
            }
            if (array.shape(1) < this->index_value.size) {
                throw std::invalid_argument(
                    "Particles_py: expected values with values.shape(1) >= " +
                    std::to_string(this->index_value.size) + " but only got values.shape(1) = " + std::to_string(array.shape(1))
                );
            }
            this->values = array;
        }

    }

    // Method to get the number of particles automatically
    size_t size() const {
        return positions.shape(0);
    }

    Particles data() {
        Particles particles;
        particles.index_value = index_value;
        particles.size = size();
        particles.positions = positions.mutable_data();
        if (values.data() != nullptr) particles.values = values;
        return particles;
    }
};


py::object count2_py(Particles_py& particles1, Particles_py& particles2,
               BinAttrs_py& battrs_py, const WeightAttrs_py& wattrs_py = WeightAttrs_py(), const SelectionAttrs_py& sattrs_py = SelectionAttrs_py()) {

    DeviceMemoryBuffer *membuffer = NULL;
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    // In this function Particles and Mesh struct live on the host (CPU)
    // but their arrays (positions, weights, etc.) live on the device (GPU)
    copy_particles_to_device(particles1.data(), &list_particles[0], 2);
    copy_particles_to_device(particles2.data(), &list_particles[1], 2);

    BinAttrs battrs = battrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    MeshAttrs mattrs;

    char names[MAX_NWEIGHT][SIZE_NAME];
    size_t ncounts = get_count2_names(list_particles[0].index_value, list_particles[1].index_value, names);
    size_t csize = ncounts * battrs.size;

    // Create a numpy array to store the results
    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data(); // Get a pointer to the array's data

    FLOAT *counts = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);

    // Create a default CUDA stream (or use 0 for the default stream)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    prepare_mesh_attrs(&mattrs, battrs, sattrs);
    set_mesh_attrs(list_particles, &mattrs, membuffer, stream);
    set_mesh(list_particles, list_mesh, mattrs, membuffer, stream);
    // Free allocated memory
    for (size_t i = 0; i < 2; i++) free_device_particles(&(list_particles[i]));
    // Perform the computation
    count2(counts, list_mesh, mattrs, sattrs, battrs, wattrs, membuffer, stream);

    CUDA_CHECK(cudaMemcpy(counts_ptr, counts, csize * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    my_device_free(counts, membuffer);

    // Free allocated memory
    for (size_t i=0; i < 2; i++) free_device_mesh(&(list_mesh[i]));

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    py::dict result;
    // Return appropriate result based on spin parameters
    for (size_t icount = 0; icount < ncounts; icount++) {
        py::array_t<FLOAT> array_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr + icount * battrs.size, counts_py);
        result[names[icount]] = array_py;
    }
    return result;
}


// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles", py::module_local())
    .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>, int, int, int>(),
         py::arg("positions"),
         py::arg("values") = py::none(),
         py::arg("size_spin") = 0,
         py::arg("size_individual_weight") = 0,
         py::arg("size_bitwise_weight") = 0)
    .def_property_readonly("size", &Particles_py::size)
    .def_property_readonly("positions", &Particles_py::positions)
    .def_property_readonly("values", &Particles_py::values)
    .def_property_readonly("index_value", &Particles_py::index_value);

    py::class_<BinAttrs_py>(m, "BinAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("shape", &BinAttrs_py::shape)
        .def_property_readonly("size", &BinAttrs_py::size)
        .def_property_readonly("ndim", &BinAttrs_py::ndim)
        .def_property_readonly("var", &BinAttrs_py::var)
        .def_property_readonly("min", &BinAttrs_py::min)
        .def_property_readonly("max", &BinAttrs_py::max)
        .def_property_readonly("step", &BinAttrs_py::step);

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_property_readonly("var", &SelectionAttrs_py::var)
        .def_property_readonly("min", &SelectionAttrs_py::min)
        .def_property_readonly("max", &SelectionAttrs_py::max);

    py::class_<WeightAttrs_py>(m, "WeightAttrs", py::module_local())
        .def(py::init<py::kwargs>()); // Accept Python kwargs

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    m.def("count2", &count2_py, "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(), // Default value
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}