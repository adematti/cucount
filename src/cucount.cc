#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mesh.h"
#include "count2.h"
#include "common.h"

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
    }

    // Method to get the number of particles automatically
    size_t nparticles() const {
        return positions.shape(0);
    }

    Particles data() {
        Particles particles;
        particles.positions = positions.mutable_data();
        particles.weights = weights.mutable_data();
        particles.nparticles = nparticles();
        return particles;
    }
};


VAR_TYPE string_to_var_type(const std::string& var_name) {
    if (var_name == "") return VAR_NONE;
    if (var_name == "s") return VAR_S;
    if (var_name == "mu") return VAR_MU;
    if (var_name == "theta") return VAR_THETA;
    throw std::invalid_argument("Invalid VAR_TYPE string: " + var_name);
}


// Expose the BinAttrs struct to Python
struct BinAttrs_py {
    FLOAT min, max, step;
    VAR_TYPE var;

    BinAttrs_py(const std::string& var_name, FLOAT min, FLOAT max, FLOAT step)
        : var(string_to_var_type(var_name)), min(min), max(max), step(step) {}

    // Method to get the number of bins automatically
    size_t nbins() const {
        size_t size = (size_t) (floor((max - min) / step));
        return size;
    }

    BinAttrs data() const {
        BinAttrs battrs;
        battrs.var = var;
        battrs.min = min;
        battrs.max = max;
        battrs.step = step;
        battrs.nbins = nbins();
        return battrs;
    }
};

// Expose the PoleAttrs struct to Python
struct PoleAttrs_py {
    size_t ellmax;

    PoleAttrs_py(size_t ellmax) : ellmax(ellmax) {}

    PoleAttrs data() const {
        PoleAttrs pattrs;
        pattrs.ellmax = ellmax;
        return pattrs;
    }
};

// Expose the SelectionAttrs struct to Python
struct SelectionAttrs_py {
    FLOAT min, max;
    VAR_TYPE var;

    SelectionAttrs_py(const std::string& var_name, FLOAT min, FLOAT max)
        : var(string_to_var_type(var_name)), min(min), max(max) {}

    SelectionAttrs data() const {
        SelectionAttrs sattrs;
        sattrs.var = var;
        sattrs.min = min;
        sattrs.max = max;
        if (var == VAR_THETA) {
            sattrs.smin = cos(max * DTORAD);
            sattrs.smax = cos(min * DTORAD);
            if (min <= 0.) sattrs.smax = 2.;
        }
        return sattrs;
    }
};


py::array_t<FLOAT> count2_py(Particles_py& particles1, Particles_py& particles2,
               const BinAttrs_py& battrs,
               const PoleAttrs_py& pattrs = PoleAttrs_py(0), // Default: 0 poles
               const SelectionAttrs_py& sattrs = SelectionAttrs_py("", 0.0, 1.0)) {
    // Convert Python inputs to C objects
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    list_particles[0] = particles1.data();
    list_particles[1] = particles2.data();
    MeshAttrs mattrs;
    for (size_t axis=0; axis<NDIM; axis++) {
        mattrs.meshsize[axis] = 0;
        mattrs.boxsize[axis] = 0.;
    }
    // Create a numpy array to store the results
    py::array_t<FLOAT> counts(battrs.nbins());
    
    if ((battrs.var == VAR_THETA) || (sattrs.var == VAR_THETA)) {
        mattrs.type = MESH_ANGULAR;
        mattrs.sepmax = sattrs.max;
    }
    set_mesh_attrs(list_particles, &mattrs);
    set_mesh(list_particles, list_mesh, mattrs);

    auto counts_ptr = counts.mutable_data(); // Get a pointer to the array's data

    // Perform the computation
    count2(counts_ptr, list_mesh, mattrs, sattrs.data(), battrs.data(), pattrs.data(), 0, 0);
    
    // Free allocated memory
    free_mesh(&(list_mesh[0]));
    free_mesh(&(list_mesh[1]));
    // Return the numpy array
    return counts;
}


// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles")
        .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>>())
        .def_property_readonly("nparticles", &Particles_py::nparticles)
        .def_readwrite("positions", &Particles_py::positions)
        .def_readwrite("weights", &Particles_py::weights);

    py::class_<BinAttrs_py>(m, "BinAttrs")
        .def(py::init<const std::string&, FLOAT, FLOAT, FLOAT>()) // Accept var as a string
        .def_property_readonly("nbins", &BinAttrs_py::nbins)
        .def_readwrite("var", &BinAttrs_py::var)
        .def_readwrite("min", &BinAttrs_py::min)
        .def_readwrite("max", &BinAttrs_py::max)
        .def_readwrite("step", &BinAttrs_py::step);

    py::class_<PoleAttrs_py>(m, "PoleAttrs")
        .def(py::init<size_t>())
        .def_readwrite("ellmax", &PoleAttrs_py::ellmax);

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs")
        .def(py::init<const std::string&, FLOAT, FLOAT>()) // Accept var as a string
        .def_readwrite("var", &SelectionAttrs_py::var)
        .def_readwrite("min", &SelectionAttrs_py::min)
        .def_readwrite("max", &SelectionAttrs_py::max);

    m.def("count2", &count2_py, "Perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("pattrs") = PoleAttrs_py(0), // Default value
        py::arg("sattrs") = SelectionAttrs_py("", 0.0, 1.0)); // Default value
}