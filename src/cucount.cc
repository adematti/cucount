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



// Function to test if battrs.array[i] increases linearly
bool is_linear(const FLOAT *array, const size_t size, FLOAT step) {
    for (size_t i = 1; i < size; i++) {
        FLOAT expected_value = array[i - 1] + step;
        if (std::abs(array[i] - expected_value) > 1e-6 * step) { // Allow small numerical tolerance
            return false;
        }
    }
    return true;
}


struct BinAttrs_py {
    std::vector<FLOAT> min, max, step;
    std::vector<VAR_TYPE> var;
    std::vector<LOS_TYPE> los;
    std::vector<py::array_t<FLOAT>> array; // New member: list of arrays for each bin

    // Constructor that takes a Python dictionary
    BinAttrs_py(const py::kwargs& kwargs) {
        for (auto item : kwargs) {
            // Extract the variable name and range tuple
            std::string var_name = py::cast<std::string>(item.first);
            // Parse the values
            VAR_TYPE v = string_to_var_type(var_name);
            var.push_back(v);
            bool los_required = ((v == VAR_MU) || (v == VAR_POLE));

            // Handle different types of values
            if ((py::isinstance<py::array>(item.second)) && (!los_required)) {
                // Case: {key: numpy array}
                auto edges = py::cast<py::array_t<FLOAT>>(item.second);
                min.push_back(edges.at(0));
                max.push_back(edges.at(edges.size() - 1));
                step.push_back((max.back() - min.back()) / (edges.size() - 1));
                los.push_back(LOS_NONE);
                array.push_back(py::array_t<FLOAT>(edges.attr("copy")()));
            } else if (py::isinstance<py::tuple>(item.second)) {
                auto tuple = py::cast<py::tuple>(item.second);
                if ((tuple.size() == 3) && (!los_required)) {
                    // Case: {key: (FLOAT, FLOAT, FLOAT)}
                    min.push_back(py::cast<FLOAT>(tuple[0]));
                    max.push_back(py::cast<FLOAT>(tuple[1]));
                    step.push_back(py::cast<FLOAT>(tuple[2]));
                    los.push_back(LOS_NONE);
                    std::vector<FLOAT> barray;
                    for (FLOAT value = min.back(); value <= max.back(); value += step.back()) barray.push_back(value);
                    py::array_t<FLOAT> py_array(barray.size(), barray.data());
                    array.push_back(py_array);
                }
                else if ((tuple.size() == 4) && (los_required)) {
                    // Case: {key: (FLOAT, FLOAT, FLOAT, string)}
                    min.push_back(py::cast<FLOAT>(tuple[0]));
                    max.push_back(py::cast<FLOAT>(tuple[1]));
                    step.push_back(py::cast<FLOAT>(tuple[2]));
                    los.push_back(string_to_los_type(py::cast<std::string>(tuple[3])));
                    std::vector<FLOAT> barray;
                    for (FLOAT value = min.back(); value <= max.back(); value += step.back()) barray.push_back(value);
                    py::array_t<FLOAT> py_array(barray.size(), barray.data());
                    array.push_back(py_array);
                } else if ((tuple.size() == 2) && (los_required)) {
                    // Case: {key: (numpy array, string)}
                    auto edges = py::cast<py::array_t<FLOAT>>(tuple[0]);
                    min.push_back(edges.at(0));
                    max.push_back(edges.at(edges.size() - 1));
                    step.push_back((max.back() - min.back()) / (edges.size() - 1));
                    los.push_back(string_to_los_type(py::cast<std::string>(tuple[1])));
                    array.push_back(py::array_t<FLOAT>(edges.attr("copy")()));
                } else {
                    throw std::invalid_argument("Invalid tuple format for key: " + var_name);
                }
            } else {
                throw std::invalid_argument("Invalid value type (expected tuple or array) for key: " + var_name);
            }
            if (step.back() == 0.) throw std::invalid_argument("Invalid step = 0. for key: " + var_name);
        }
        // Sort variables to ensure VAR_POLE is last
        std::vector<size_t> indices(var.size());
        size_t current_index = 0;
        for (size_t i = 0; i < var.size(); i++) {
            if (var[i] == VAR_POLE) {
                indices[var.size() - 1] = i;
            }
            else if (var[i] == VAR_K) {
                if (var.size() < 2) throw std::invalid_argument("k must be always used with pole");
                indices[var.size() - 2] = i;
            }
            else {
                indices[current_index] = i;
                current_index += 1;
            }
        }
        // Reorder all attributes based on the sorted indices
        reorder(var, indices);
        reorder(min, indices);
        reorder(max, indices);
        reorder(step, indices);
        reorder(los, indices);
        reorder(array, indices);
    }

    // Method to get the number of bins for each variable
    std::vector<size_t> shape() const {
        std::vector<size_t> sizes;
        for (size_t i = 0; i < var.size(); i++) {
            size_t s = array[i].shape(0) - 1;
            if (var[i] == VAR_K) s += 1;  // k-values, not edges
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

    BinAttrs data() {
        BinAttrs battrs;
        auto sizes = shape();
        battrs.ndim = ndim();
        battrs.size = size();
        for (size_t i = 0; i < battrs.ndim; i++) {
            battrs.var[i] = var[i];
            battrs.min[i] = min[i];
            battrs.max[i] = max[i];
            battrs.los[i] = los[i];
            battrs.shape[i] = sizes[i];
            battrs.asize[i] = array[i].shape(0);
            battrs.array[i] = array[i].mutable_data();
            battrs.step[i] = step[i];
            if (is_linear(battrs.array[i], sizes[i], step[i])) {
                battrs.asize[i] = 0;
                battrs.array[i] = NULL;
            }
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
                if (min[i] <= 0.) sattrs.smax[i] = 2.;  // margin for numerical approximation
            }

        }
        return sattrs;
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

    m.def("count2", &count2_py, "Perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}