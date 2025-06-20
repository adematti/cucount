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


VAR_TYPE string_to_var_type(const std::string& var_name) {
    if (var_name == "") return VAR_NONE;
    if (var_name == "s") return VAR_S;
    if (var_name == "mu") return VAR_MU;
    if (var_name == "theta") return VAR_THETA;
    if (var_name == "pole") return VAR_POLE;
    if (var_name == "k") return VAR_K;
    throw std::invalid_argument("Invalid VAR_TYPE string: " + var_name);
}


LOS_TYPE string_to_los_type(const std::string& los_name) {
    if (los_name == "") return LOS_NONE;
    if (los_name == "firstpoint") return LOS_FIRSTPOINT;
    if (los_name == "endpoint") return LOS_ENDPOINT;
    if (los_name == "midpoint") return LOS_MIDPOINT;
    throw std::invalid_argument("Invalid LOS_TYPE string: " + los_name);
}


// Helper function to reorder a vector based on sorted indices
template <typename T>
void reorder(std::vector<T>& vec, const std::vector<size_t>& indices) {
    std::vector<T> temp(vec.size());
    for (size_t i = 0; i < indices.size(); i++) {
        temp[i] = vec[indices[i]];
    }
    vec = std::move(temp);
}


struct BinAttrs_py {
    std::vector<FLOAT> min, max, step;
    std::vector<VAR_TYPE> var;
    std::vector<LOS_TYPE> los;

    // Constructor that takes a Python dictionary
    BinAttrs_py(const py::kwargs& kwargs) {
        for (auto item : kwargs) {
            // Extract the variable name and range tuple
            std::string var_name = py::cast<std::string>(item.first);
            // Parse the values
            VAR_TYPE v = string_to_var_type(var_name);
            var.push_back(v);
            if ((v == VAR_MU) || (v == VAR_POLE)) {
                auto range_tuple = py::cast<std::tuple<FLOAT, FLOAT, FLOAT, std::string>>(item.second);
                min.push_back(std::get<0>(range_tuple));
                max.push_back(std::get<1>(range_tuple));
                step.push_back(std::get<2>(range_tuple));
                los.push_back(string_to_los_type(std::get<3>(range_tuple)));
            }
            else {
                auto range_tuple = py::cast<std::tuple<FLOAT, FLOAT, FLOAT>>(item.second);
                min.push_back(std::get<0>(range_tuple));
                max.push_back(std::get<1>(range_tuple));
                step.push_back(std::get<2>(range_tuple));
                los.push_back(LOS_NONE);
            }
        }
        // Sort variables to ensure VAR_POLE is last
        std::vector<size_t> indices(var.size());
        std::iota(indices.begin(), indices.end(), 0); // Initialize indices: [0, 1, 2, ...]

        // Sort indices based on whether the variable is VAR_POLE
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return var[i] != VAR_POLE && var[j] == VAR_POLE;
        });

        // Reorder all attributes based on the sorted indices
        reorder(var, indices);
        reorder(min, indices);
        reorder(max, indices);
        reorder(step, indices);
        reorder(los, indices);

    }

    // Method to get the number of bins for each variable
    std::vector<size_t> shape() const {
        std::vector<size_t> sizes;
        for (size_t i = 0; i < var.size(); i++) {
            size_t s = static_cast<size_t>((max[i] - min[i]) / step[i]);
            if (var[i] == VAR_POLE) s += 1;  // poles, not edges
            sizes.push_back(s);
        }
        return sizes;
    }

    // Method to get the number of bins for each variable
    size_t size() const {
        auto sizes = shape();
        size_t size = 1;
        for (size_t i = 0; i < var.size(); i++) size *= sizes[i];
        return size;
    }

    size_t ndim() const {
        return var.size();
    }

    BinAttrs data() const {
        BinAttrs battrs;
        auto sizes = shape();
        battrs.ndim = ndim();
        battrs.size = size();
        for (size_t i = 0; i < battrs.ndim; i++) {
            battrs.var[i] = var[i];
            battrs.min[i] = min[i];
            battrs.max[i] = max[i];
            battrs.step[i] = step[i];
            battrs.los[i] = los[i];
            battrs.shape[i] = sizes[i];
        }
        return battrs;
    }
};


// Expose the SelectionAttrs struct to Python
struct SelectionAttrs_py {
    std::vector<FLOAT> min, max;
    std::vector<VAR_TYPE> var;

    // Default constructor
    SelectionAttrs_py() {}

    // Constructor that takes a Python dictionary
    SelectionAttrs_py(const py::kwargs& kwargs) {
        for (auto item : kwargs) {
            // Extract the variable name and range tuple
            std::string var_name = py::cast<std::string>(item.first);
            auto range_tuple = py::cast<std::tuple<FLOAT, FLOAT>>(item.second);

            // Parse the values
            var.push_back(string_to_var_type(var_name));
            min.push_back(std::get<0>(range_tuple));
            max.push_back(std::get<1>(range_tuple));
        }
    }

    size_t ndim() const {
        return var.size();
    }

    SelectionAttrs data() const {
        SelectionAttrs sattrs;
        sattrs.ndim = ndim();

        for (size_t i = 0; i < sattrs.ndim; i++) {
            sattrs.var[i] = var[i];
            sattrs.min[i] = min[i];
            sattrs.max[i] = max[i];
            sattrs.smin[i] = min[i];
            sattrs.smax[i] = max[i];

            // Handle special case for VAR_THETA
            if (var[i] == VAR_THETA) {
                sattrs.smin[i] = cos(max[i] * DTORAD);
                sattrs.smax[i] = cos(min[i] * DTORAD);
            }

        }
        return sattrs;
    }
};


py::array_t<FLOAT> count2_py(Particles_py& particles1, Particles_py& particles2,
               const BinAttrs_py& battrs, const SelectionAttrs_py& sattrs = SelectionAttrs_py()) {
    // Convert Python inputs to C objects
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];
    list_particles[0] = particles1.data();
    list_particles[1] = particles2.data();
    SelectionAttrs csattrs = sattrs.data();
    BinAttrs cbattrs = battrs.data();
    MeshAttrs cmattrs;
    for (size_t axis=0; axis<NDIM; axis++) {
        cmattrs.meshsize[axis] = 0;
        cmattrs.boxsize[axis] = 0.;
    }

    // Create a numpy array to store the results
    py::array_t<FLOAT> counts(battrs.shape());

    if (csattrs.var[0] == VAR_THETA) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(csattrs.max[0] * DTORAD);
    }
    else if (csattrs.var[0] == VAR_S) {
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
    set_mesh_attrs(list_particles, &cmattrs);
    set_mesh(list_particles, list_mesh, cmattrs);

    auto counts_ptr = counts.mutable_data(); // Get a pointer to the array's data

    // Perform the computation
    count2(counts_ptr, list_mesh, cmattrs, csattrs, cbattrs, 0, 0);

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

    m.def("count2", &count2_py, "Perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}