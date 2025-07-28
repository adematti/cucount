#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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

    // Constructor
    Particles_py(py::array_t<FLOAT> positions, py::array_t<FLOAT> weights)
        : positions(positions), weights(weights) {
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

    // Method to get the number of particles automatically
    size_t size() const {
        return positions.shape(0);
    }

    Particles data() {
        Particles particles;
        particles.positions = positions.mutable_data();
        particles.weights = weights.mutable_data();
        particles.size = size();
        return particles;
    }
};



py::array_t<FLOAT> count2_py(Particles_py& particles1, Particles_py& particles2,
               BinAttrs_py& battrs, const SelectionAttrs_py& sattrs = SelectionAttrs_py()) {
    // Convert Python inputs to C objects
    Particles list_particles[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    // In this function Particles and Mesh struct live on the host (CPU)
    // but their arrays (positions, weights, etc.) live on the device (GPU)
    copy_particles_to_device(particles1.data(), &list_particles[0], 2);
    copy_particles_to_device(particles2.data(), &list_particles[1], 2);
    Mesh list_mesh[MAX_NMESH];

    SelectionAttrs csattrs = sattrs.data();
    BinAttrs cbattrs = battrs.data();
    MeshAttrs cmattrs;
    for (size_t axis = 0; axis < NDIM; axis++) {
        cmattrs.meshsize[axis] = 0;
        cmattrs.boxsize[axis] = 0.;
        cmattrs.boxcenter[axis] = 0.;
    }
    // Create a numpy array to store the results
    py::array_t<FLOAT> counts(battrs.shape());

    if ((csattrs.ndim) && (csattrs.var[0] == VAR_THETA)) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(csattrs.max[0] * DTORAD);
    }
    else if ((csattrs.ndim) && (csattrs.var[0] == VAR_S)) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = csattrs.max[0];
    }
    else if (cbattrs.var[0] == VAR_THETA) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(cbattrs.max[0] * DTORAD);
    }
    else if (cbattrs.var[0] == VAR_S) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = cbattrs.max[0];
    }
    //return counts;
    set_mesh_attrs(list_particles, &cmattrs);
    set_mesh(list_particles, list_mesh, cmattrs);
    // Free allocated memory
    for (size_t i = 0; i < 2; i++) free_device_particles(&(list_particles[i]));
    // Perform the computation
    auto counts_ptr = counts.mutable_data(); // Get a pointer to the array's data
    count2(counts_ptr, list_mesh, cmattrs, csattrs, cbattrs);
    // Free allocated memory
    for (size_t i = 0; i < 2; i++) free_device_mesh(&(list_mesh[i]));
    // Return the numpy array
    return counts;
}


// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles")
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>>())
        .def_property_readonly("size", &Particles_py::size)
        .def_readwrite("positions", &Particles_py::positions)
        .def_readwrite("weights", &Particles_py::weights);

    py::class_<BinAttrs_py>(m, "BinAttrs")
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("size", &BinAttrs_py::size)
        .def_property_readonly("ndim", &BinAttrs_py::ndim)
        .def_readwrite("var", &BinAttrs_py::var)
        .def_readwrite("min", &BinAttrs_py::min)
        .def_readwrite("max", &BinAttrs_py::max)
        .def_readwrite("step", &BinAttrs_py::step);

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs")
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_readwrite("var", &SelectionAttrs_py::var)
        .def_readwrite("min", &SelectionAttrs_py::min)
        .def_readwrite("max", &SelectionAttrs_py::max);

    m.def("count2", &count2_py, "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}