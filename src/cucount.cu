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
struct Particles_py {
    py::array_t<FLOAT> positions;
    py::array_t<FLOAT> weights;
    py::array_t<FLOAT> spin_values; // Optional spin values array
    py::array_t<FLOAT> sky_coords; // Optional RA, Dec array

    // Single constructor accepting optional sky_coords and spin_values (can be None)
    Particles_py(py::array_t<FLOAT> positions_, py::array_t<FLOAT> weights_,
                 py::object sky_coords_ = py::none(), py::object spin_values_ = py::none())
        : positions(positions_), weights(weights_), spin_values(py::array_t<FLOAT>()), sky_coords(py::array_t<FLOAT>()) {

        // Ensure positions are C-contiguous
        if (!is_contiguous(this->positions)) this->positions = py::array_t<FLOAT>(this->positions.attr("copy")());

        // Ensure weights are C-contiguous
        if (!is_contiguous(this->weights)) this->weights = py::array_t<FLOAT>(this->weights.attr("copy")());

        // Check that positions and weights have the same length
        size_t npositions = this->positions.shape(0);
        size_t nweights = this->weights.shape(0);
        if (npositions != nweights) {
            throw std::invalid_argument(
                "Particles_py: positions and weights must have the same length, but got positions.shape(0) = " +
                std::to_string(npositions) + " and weights.shape(0) = " + std::to_string(nweights)
            );
        }

        // Optional: sky_coords
        if (!py::isinstance<py::none>(sky_coords_)) {
            auto arr = py::cast<py::array_t<FLOAT>>(sky_coords_);
            if (!is_contiguous(arr)) arr = py::array_t<FLOAT>(arr.attr("copy")());
            this->sky_coords = arr;
            // Validate dims/length
            if (this->sky_coords.ndim() != 2 || this->sky_coords.shape(1) != 2) {
                throw std::invalid_argument("Particles_py: sky_coords must be a 2D array with shape (N, 2) for (RA, Dec) components");
            }
            size_t nsky_coords = this->sky_coords.shape(0);
            if (nsky_coords != npositions) {
                throw std::invalid_argument(
                    "Particles_py: sky_coords and positions must have the same length, but got sky_coords.shape(0) = " +
                    std::to_string(nsky_coords) + " and positions.shape(0) = " + std::to_string(npositions)
                );
            }
        }

        // Optional: spin_values
        if (!py::isinstance<py::none>(spin_values_)) {
            auto arr = py::cast<py::array_t<FLOAT>>(spin_values_);
            if (!is_contiguous(arr)) arr = py::array_t<FLOAT>(arr.attr("copy")());
            this->spin_values = arr;
            // Validate dims/length
            if (this->spin_values.ndim() != 2 || this->spin_values.shape(1) != 2) {
                throw std::invalid_argument("Particles_py: spin_values must be a 2D array with shape (N, 2) for (s1, s2) components");
            }
            size_t nspin_values = this->spin_values.shape(0);
            if (nspin_values != npositions) {
                throw std::invalid_argument(
                    "Particles_py: spin_values and positions must have the same length, but got spin_values.shape(0) = " +
                    std::to_string(nspin_values) + " and positions.shape(0) = " + std::to_string(npositions)
                );
            }
        }
    }


    // Method to get the number of particles automatically
    size_t size() const {
        return positions.shape(0);
    }

    Particles data() {
        Particles particles;
        particles.positions = positions.mutable_data();
        particles.weights = weights.mutable_data();
        particles.size = size();
        if (spin_values.size() > 0 && spin_values.data() != nullptr) {
            particles.spin_values = spin_values.mutable_data();
        } else {
            particles.spin_values = NULL;
        }
        if (sky_coords.size() > 0 && sky_coords.data() != nullptr) {
            particles.sky_coords = sky_coords.mutable_data();
        } else {
            particles.sky_coords = NULL;
        }
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

    // Determine output size based on spin parameters
    // - if both spin parameters = 0: only need one correlation (regular pair counting)
    // - if one spin parameter != 0 and one = 0: need two correlations (plus, cross)
    // - if both spin parameters != 0: need three correlations (plus_plus, cross_plus, cross_cross)
    size_t nspins = (wattrs.spin[0] > 0) + (wattrs.spin[1] > 0);
    size_t output_size = (1 + nspins) * battrs.size;

    // Create a numpy array to store the results
    py::array_t<FLOAT> counts_py(output_size);
    auto counts_ptr = counts_py.mutable_data(); // Get a pointer to the array's data
    // Array on the GPU - allocate double size if we have spin_values
    FLOAT *counts = (FLOAT*) my_device_malloc(output_size * sizeof(FLOAT), membuffer);

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

    CUDA_CHECK(cudaMemcpy(counts_ptr, counts, output_size * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    my_device_free(counts, membuffer);

    // Free allocated memory
    for (size_t i=0; i < 2; i++) free_device_mesh(&(list_mesh[i]));

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    py::dict result;
    // Return appropriate result based on spin parameters
    if (nspins == 2) {
        // Create zero-copy numpy views into counts_py rather than memcpy'ing
        py::array_t<FLOAT> plus_plus_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr, counts_py);
        py::array_t<FLOAT> cross_plus_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr + battrs.size, counts_py);
        py::array_t<FLOAT> cross_cross_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr + 2 * battrs.size, counts_py);
        // Return dictionary with three components
        result["weights_plus_plus"] = plus_plus_py;
        result["weights_plus"] = cross_plus_py;
        result["weights_cross_cross"] = cross_cross_py;
    } else if (nspins == 1) {
        // Zero-copy views for two correlations
        py::array_t<FLOAT> plus_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr, counts_py);
        py::array_t<FLOAT> cross_py({(ssize_t)battrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr + battrs.size, counts_py);
        result["weights_plus"] = plus_py;
        result["weights_cross"] = cross_py;
        return result;
    } else {
        // Return the numpy array for regular pair counts
        result["weights"] = counts_py;
    }
    return result;
}


// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles", py::module_local())
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>, py::object, py::object>(),
             py::arg("positions"), py::arg("weights"), py::arg("sky_coords") = py::none(), py::arg("spin_values") = py::none())
        .def_property_readonly("size", &Particles_py::size)
        .def_readwrite("positions", &Particles_py::positions)
        .def_readwrite("weights", &Particles_py::weights)
        .def_readwrite("spin_values", &Particles_py::spin_values)
        .def_readwrite("sky_coords", &Particles_py::sky_coords);

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