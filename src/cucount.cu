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

    // Constructor without spin values or sky coords
    Particles_py(py::array_t<FLOAT> positions, py::array_t<FLOAT> weights)
        : positions(positions), weights(weights), spin_values(py::array_t<FLOAT>()), sky_coords(py::array_t<FLOAT>()) {
        // Ensure positions are C-contiguous
        if (!is_contiguous(positions)) {
            this->positions = py::array_t<FLOAT>(positions.attr("copy")());
        }

        // Ensure weights are C-contiguous
        if (!is_contiguous(weights)) {
            this->weights = py::array_t<FLOAT>(weights.attr("copy")());
        }

        // Check that positions and weights have the same length
        size_t npositions = this->positions.shape(0);
        size_t nweights = this->weights.shape(0);
        if (npositions != nweights) {
            throw std::invalid_argument(
                "Particles_py: positions and weights must have the same length, but got positions.shape(0) = " +
                std::to_string(npositions) + " and weights.shape(0) = " + std::to_string(nweights)
            );
        }
    }


    // Constructor with sky coordinates but no spin values
    Particles_py(py::array_t<FLOAT> positions, py::array_t<FLOAT> weights, py::array_t<FLOAT> sky_coords)
        : positions(positions), weights(weights), spin_values(py::array_t<FLOAT>()), sky_coords(sky_coords) {
        // Ensure positions are C-contiguous
        if (!is_contiguous(positions)) {
            this->positions = py::array_t<FLOAT>(positions.attr("copy")());
        }

        // Ensure weights are C-contiguous
        if (!is_contiguous(weights)) {
            this->weights = py::array_t<FLOAT>(weights.attr("copy")());
        }

        // Handle optional sky coordinates
        if (sky_coords.size() > 0 && sky_coords.data() != nullptr) {
            if (!is_contiguous(sky_coords)) {
                this->sky_coords = py::array_t<FLOAT>(sky_coords.attr("copy")());
            }
        }

        // Check that positions and weights have the same length
        size_t npositions = this->positions.shape(0);
        size_t nweights = this->weights.shape(0);
        if (npositions != nweights) {
            throw std::invalid_argument(
                "Particles_py: positions and weights must have the same length, but got positions.shape(0) = " +
                std::to_string(npositions) + " and weights.shape(0) = " + std::to_string(nweights)
            );
        }

        // Check sky coordinates dimensions if provided
        if (this->sky_coords.size() > 0 && this->sky_coords.data() != nullptr) {
            if (this->sky_coords.ndim() != 2 || this->sky_coords.shape(1) != 2) {
                throw std::invalid_argument(
                    "Particles_py: sky_coords must be a 2D array with shape (N, 2) for (RA, Dec) components"
                );
            }
            size_t nsky_coords = this->sky_coords.shape(0);
            if (nsky_coords != npositions) {
                throw std::invalid_argument(
                    "Particles_py: sky_coords and positions must have the same length, but got sky_coords.shape(0) = " +
                    std::to_string(nsky_coords) + " and positions.shape(0) = " + std::to_string(npositions)
                );
            }
        }
    }

    // Constructor with both sky coordinates and spin values
    Particles_py(py::array_t<FLOAT> positions, py::array_t<FLOAT> weights, py::array_t<FLOAT> sky_coords, py::array_t<FLOAT> spin_values)
        : positions(positions), weights(weights), spin_values(spin_values), sky_coords(sky_coords) {
        // Ensure positions are C-contiguous
        if (!is_contiguous(positions)) {
            this->positions = py::array_t<FLOAT>(positions.attr("copy")());
        }

        // Ensure weights are C-contiguous
        if (!is_contiguous(weights)) {
            this->weights = py::array_t<FLOAT>(weights.attr("copy")());
        }

        // Handle optional spin values
        if (spin_values.size() > 0 && spin_values.data() != nullptr) {
            if (!is_contiguous(spin_values)) {
                this->spin_values = py::array_t<FLOAT>(spin_values.attr("copy")());
            }
        }

        // Handle optional sky coordinates
        if (sky_coords.size() > 0 && sky_coords.data() != nullptr) {
            if (!is_contiguous(sky_coords)) {
                this->sky_coords = py::array_t<FLOAT>(sky_coords.attr("copy")());
            }
        }

        // Check that positions and weights have the same length
        size_t npositions = this->positions.shape(0);
        size_t nweights = this->weights.shape(0);
        if (npositions != nweights) {
            throw std::invalid_argument(
                "Particles_py: positions and weights must have the same length, but got positions.shape(0) = " +
                std::to_string(npositions) + " and weights.shape(0) = " + std::to_string(nweights)
            );
        }

        // Check spin values dimensions if provided
        if (this->spin_values.size() > 0 && this->spin_values.data() != nullptr) {
            if (this->spin_values.ndim() != 2 || this->spin_values.shape(1) != 2) {
                throw std::invalid_argument(
                    "Particles_py: spin_values must be a 2D array with shape (N, 2) for (s1, s2) components"
                );
            }
            size_t nspin_values = this->spin_values.shape(0);
            if (nspin_values != npositions) {
                throw std::invalid_argument(
                    "Particles_py: spin_values and positions must have the same length, but got spin_values.shape(0) = " +
                    std::to_string(nspin_values) + " and positions.shape(0) = " + std::to_string(npositions)
                );
            }
        }

        // Check sky coordinates dimensions if provided
        if (this->sky_coords.size() > 0 && this->sky_coords.data() != nullptr) {
            if (this->sky_coords.ndim() != 2 || this->sky_coords.shape(1) != 2) {
                throw std::invalid_argument(
                    "Particles_py: sky_coords must be a 2D array with shape (N, 2) for (RA, Dec) components"
                );
            }
            size_t nsky_coords = this->sky_coords.shape(0);
            if (nsky_coords != npositions) {
                throw std::invalid_argument(
                    "Particles_py: sky_coords and positions must have the same length, but got sky_coords.shape(0) = " +
                    std::to_string(nsky_coords) + " and positions.shape(0) = " + std::to_string(npositions)
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
               BinAttrs_py& battrs_py, int spin1 = 0, int spin2 = 0, const SelectionAttrs_py& sattrs_py = SelectionAttrs_py()) {

    DeviceMemoryBuffer *membuffer = NULL;
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    // In this function Particles and Mesh struct live on the host (CPU)
    // but their arrays (positions, weights, etc.) live on the device (GPU)
    copy_particles_to_device(particles1.data(), &list_particles[0], 2);
    copy_particles_to_device(particles2.data(), &list_particles[1], 2);

    SelectionAttrs sattrs = sattrs_py.data();
    BinAttrs battrs = battrs_py.data();
    MeshAttrs mattrs;

    // Determine output size based on spin parameters
    // - if both spin parameters = 0: only need one correlation (regular pair counting)
    // - if one spin parameter != 0 and one = 0: need two correlations (plus, cross)
    // - if both spin parameters != 0: need three correlations (plus_plus, cross_plus, cross_cross)
    bool both_spins = (spin1 != 0) && (spin2 != 0);
    bool one_spin = (spin1 != 0 && spin2 == 0) || (spin1 == 0 && spin2 != 0);
    size_t output_size = both_spins ? 3 * battrs.size : (one_spin ? 2 * battrs.size : battrs.size);

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
    count2(counts, list_mesh, mattrs, sattrs, battrs, spin1, spin2, membuffer, stream);

    CUDA_CHECK(cudaMemcpy(counts_ptr, counts, output_size * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    my_device_free(counts, membuffer);

    // Free allocated memory
    for (size_t i=0; i < 2; i++) free_device_mesh(&(list_mesh[i]));

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Return appropriate result based on spin parameters
    if (both_spins) {
        // Create separate arrays for three spin-spin correlations
        py::array_t<FLOAT> plus_plus_py(battrs.size);
        py::array_t<FLOAT> cross_plus_py(battrs.size);
        py::array_t<FLOAT> cross_cross_py(battrs.size);

        auto plus_plus_ptr = plus_plus_py.mutable_data();
        auto cross_plus_ptr = cross_plus_py.mutable_data();
        auto cross_cross_ptr = cross_cross_py.mutable_data();

        // Split the results: [plus_plus][cross_plus][cross_cross]
        std::memcpy(plus_plus_ptr, counts_ptr, battrs.size * sizeof(FLOAT));
        std::memcpy(cross_plus_ptr, counts_ptr + battrs.size, battrs.size * sizeof(FLOAT));
        std::memcpy(cross_cross_ptr, counts_ptr + 2 * battrs.size, battrs.size * sizeof(FLOAT));

        // Return dictionary with three components
        py::dict result;
        result["plus_plus"] = plus_plus_py;
        result["cross_plus"] = cross_plus_py;
        result["cross_cross"] = cross_cross_py;
        return result;
    } else if (one_spin) {
        // Create separate arrays for two correlations
        py::array_t<FLOAT> plus_py(battrs.size);
        py::array_t<FLOAT> cross_py(battrs.size);

        auto plus_ptr = plus_py.mutable_data();
        auto cross_ptr = cross_py.mutable_data();

        // Split the results: first half is plus, second half is cross
        std::memcpy(plus_ptr, counts_ptr, battrs.size * sizeof(FLOAT));
        std::memcpy(cross_ptr, counts_ptr + battrs.size, battrs.size * sizeof(FLOAT));

        // Return dictionary with both components
        py::dict result;
        result["plus"] = plus_py;
        result["cross"] = cross_py;
        return result;
    } else {
        // Return the numpy array for regular pair counts
        return counts_py;
    }
}


// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles", py::module_local())
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>>())                                  // positions, weights
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>, py::array_t<FLOAT>>())             // positions, weights, sky_coords
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>, py::array_t<FLOAT>, py::array_t<FLOAT>>()) // positions, weights, sky_coords, spin_values
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

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    m.def("count2", &count2_py, "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("spin1") = 0,
        py::arg("spin2") = 0,
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}