#ifndef _CUCOUNT_CUCOUNT_
#define _CUCOUNT_CUCOUNT_

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"

namespace py = pybind11;


// Set the global logging level from a string
void setup_logging(const std::string& level_str) {
    std::string lvl = level_str;
    std::transform(lvl.begin(), lvl.end(), lvl.begin(), ::tolower);
    if (lvl == "debug") {
        global_log_level = LOG_LEVEL_DEBUG;
    } else if (lvl == "info") {
        global_log_level = LOG_LEVEL_INFO;
    } else if (lvl == "warn" || lvl == "warning") {
        global_log_level = LOG_LEVEL_WARN;
    } else if (lvl == "error") {
        global_log_level = LOG_LEVEL_ERROR;
    } else {
        throw std::invalid_argument("Unknown log level: " + level_str);
    }
}


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
                    FLOAT diff = max.back() - min.back();
                    if (diff == 0.) step.push_back(2.); // in case ells = [0] or (0, 0, 1)
                    else step.push_back((max.back() - min.back()) / (edges.size() - 1));
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


void prepare_mesh_attrs(MeshAttrs *mattrs, BinAttrs battrs, SelectionAttrs sattrs) {

    for (size_t axis = 0; axis < NDIM; axis++) {
        mattrs->meshsize[axis] = 0;
        mattrs->boxsize[axis] = 0.;
        mattrs->boxcenter[axis] = 0.;
    }

    if ((sattrs.ndim) && (sattrs.var[0] == VAR_THETA)) {
        mattrs->type = MESH_ANGULAR;
        mattrs->smax = cos(sattrs.max[0]);
    }
    else if ((sattrs.ndim) && (sattrs.var[0] == VAR_S)) {
        mattrs->type = MESH_CARTESIAN;
        mattrs->smax = sattrs.max[0];
    }
    else if (battrs.var[0] == VAR_THETA) {
        mattrs->type = MESH_ANGULAR;
        mattrs->smax = cos(battrs.max[0]);
    }
    else if (battrs.var[0] == VAR_S) {
        mattrs->type = MESH_CARTESIAN;
        mattrs->smax = battrs.max[0];
    }

}

#endif