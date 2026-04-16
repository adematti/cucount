#ifndef _CUCOUNT_CUCOUNT_
#define _CUCOUNT_CUCOUNT_

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "count3close.h"

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
    if (var_name == "rp") return VAR_RP;
    if (var_name == "pi") return VAR_PI;
    if (var_name == "theta") return VAR_THETA;
    if (var_name == "pole") return VAR_POLE;
    if (var_name == "k") return VAR_K;
    throw std::invalid_argument("Invalid VAR_TYPE string: " + var_name);
}


// inverse mapping for VAR_TYPE -> string
std::string var_type_to_string(VAR_TYPE var_type) {
    switch (var_type) {
        case VAR_NONE: return "";
        case VAR_S: return "s";
        case VAR_MU: return "mu";
        case VAR_RP: return "rp";
        case VAR_PI: return "pi";
        case VAR_THETA: return "theta";
        case VAR_POLE: return "pole";
        case VAR_K: return "k";
        default: return "";
    }
}


LOS_TYPE string_to_los_type(const std::string& los_name) {
    if (los_name == "") return LOS_NONE;
    if (los_name == "firstpoint") return LOS_FIRSTPOINT;
    if (los_name == "endpoint") return LOS_ENDPOINT;
    if (los_name == "midpoint") return LOS_MIDPOINT;
    if (los_name == "x") return LOS_X;
    if (los_name == "y") return LOS_Y;
    if (los_name == "z") return LOS_Z;
    throw std::invalid_argument("Invalid LOS_TYPE string: " + los_name);
}


// inverse mapping for LOS_TYPE -> string
std::string los_type_to_string(LOS_TYPE los_type) {
    switch (los_type) {
        case LOS_NONE: return "";
        case LOS_FIRSTPOINT: return "firstpoint";
        case LOS_ENDPOINT: return "endpoint";
        case LOS_MIDPOINT: return "midpoint";
        case LOS_X: return "x";
        case LOS_Y: return "y";
        case LOS_Z: return "z";
        default: return "";
    }
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
            bool los_required = ((v == VAR_MU) || (v == VAR_RP) || (v == VAR_PI) || (v == VAR_POLE));

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

        // Ensure LOS consistency for all variables that require it:
        bool los_seen = false;
        LOS_TYPE common_los = LOS_NONE;
        for (size_t i = 0; i < var.size(); ++i) {
            if (los[i] != LOS_NONE) {
                if (!los_seen) {
                    common_los = los[i];
                    los_seen = true;
                } else if (los[i] != common_los) {
                    throw std::invalid_argument("All LOS specifications must be the same for variables that require LOS");
                }
            }
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

    // Return list of variable names in the same order as `var`
    std::vector<std::string> varnames() const {
        std::vector<std::string> out;
        out.reserve(var.size());
        for (auto v : var) out.push_back(var_type_to_string(v));
        return out;
    }

    // Return list of los names in the same order as `los`
    std::vector<std::string> losnames() const {
        std::vector<std::string> out;
        out.reserve(los.size());
        for (auto v : los) out.push_back(los_type_to_string(v));
        return out;
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
        // Selection support is currently limited: require VAR_THETA only.
        for (size_t i = 0; i < var.size(); ++i) {
            if (var[i] != VAR_THETA) {
                throw std::invalid_argument("SelectionAttrs currently only implemented for 'theta'");
            }
        }
    }

    size_t ndim() const {
        return var.size();
    }

    // Return list of variable names in the same order as `var`
    std::vector<std::string> varnames() const {
        std::vector<std::string> out;
        out.reserve(var.size());
        for (auto v : var) out.push_back(var_type_to_string(v));
        return out;
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


// Expose the WeightAttrs struct to Python

struct WeightAttrs_py {
    std::vector<size_t> spin;

    // Bitwise
    FLOAT bitwise_default_value = 0.0;
    FLOAT bitwise_nrealizations = 0.0;
    int bitwise_noffset = 0;
    py::array_t<FLOAT> bitwise_p_correction_nbits;

    // Angular
    std::vector<py::array_t<FLOAT>> angular_sep;
    std::vector<py::array_t<FLOAT>> angular_edges;
    py::array_t<FLOAT> angular_weight;

    WeightAttrs_py() {}

    WeightAttrs_py(const py::kwargs& kwargs) {
        for (auto item : kwargs) {
            std::string var_name = py::cast<std::string>(item.first);

            if (var_name == "spin") {
                if (!py::isinstance<py::iterable>(item.second)) {
                    throw std::invalid_argument(
                        "Invalid type for 'spin' (expected iterable)");
                }
                for (auto v : py::cast<py::iterable>(item.second)) {
                    spin.push_back(py::cast<size_t>(v));
                }
            }

            else if (var_name == "angular") {
                if (!py::isinstance<py::dict>(item.second)) {
                    throw std::invalid_argument(
                        "Invalid type for 'angular' (expected dict)");
                }

                py::dict d = py::cast<py::dict>(item.second);

                const bool has_sep = d.contains("sep");
                const bool has_edges = d.contains("edges");
                const bool has_weight = d.contains("weight");

                if (!has_weight) {
                    throw std::invalid_argument(
                        "'angular' dict must contain key 'weight'");
                }
                if (has_sep == has_edges) {
                    throw std::invalid_argument(
                        "'angular' dict must contain exactly one of 'sep' or 'edges'");
                }

                angular_weight = py::array_t<FLOAT>(d["weight"].attr("copy")());

                if (has_sep) {
                    if (!py::isinstance<py::iterable>(d["sep"])) {
                        throw std::invalid_argument(
                            "'angular.sep' must be an iterable of 1D arrays");
                    }
                    for (auto obj : py::cast<py::iterable>(d["sep"])) {
                        auto arr = py::array_t<FLOAT>(py::cast<py::object>(obj).attr("copy")());
                        if (arr.ndim() != 1) {
                            throw std::invalid_argument(
                                "Each entry of 'angular.sep' must be a 1D array");
                        }
                        angular_sep.push_back(std::move(arr));
                    }
                }

                if (has_edges) {
                    if (!py::isinstance<py::iterable>(d["edges"])) {
                        throw std::invalid_argument(
                            "'angular.edges' must be an iterable of 1D arrays");
                    }
                    for (auto obj : py::cast<py::iterable>(d["edges"])) {
                        auto arr = py::array_t<FLOAT>(py::cast<py::object>(obj).attr("copy")());
                        if (arr.ndim() != 1) {
                            throw std::invalid_argument(
                                "Each entry of 'angular.edges' must be a 1D array");
                        }
                        angular_edges.push_back(std::move(arr));
                    }
                }

                const bool use_sep = !angular_sep.empty();
                const bool use_edges = !angular_edges.empty();

                if (use_sep == use_edges) {
                    throw std::invalid_argument(
                        "'angular' must define exactly one of 'sep' or 'edges'");
                }

                if (!angular_weight.size()) {
                    throw std::invalid_argument(
                        "'angular.weight' must be provided and non-empty");
                }

                py::buffer_info winfo = angular_weight.request();
                if (winfo.ndim < 1 || winfo.ndim > (int)MAX_NBIN) {
                    throw std::invalid_argument(
                        "'angular.weight' must have ndim between 1 and MAX_NBIN");
                }

                if (use_sep) {
                    if ((ssize_t) angular_sep.size() != winfo.ndim) {
                        throw std::invalid_argument(
                            "'angular.sep' must contain one 1D array per weight dimension");
                    }
                    for (ssize_t i = 0; i < winfo.ndim; i++) {
                        if ((ssize_t) angular_sep[i].size() != winfo.shape[i]) {
                            throw std::invalid_argument(
                                "Each 'angular.sep[i]' must have length weight.shape(i)");
                        }
                    }
                }

                if (use_edges) {
                    if ((ssize_t) angular_edges.size() != winfo.ndim) {
                        throw std::invalid_argument(
                            "'angular.edges' must contain one 1D array per weight dimension");
                    }
                    for (ssize_t i = 0; i < winfo.ndim; i++) {
                        if ((ssize_t) angular_edges[i].size() != winfo.shape[i] + 1) {
                            throw std::invalid_argument(
                                "Each 'angular.edges[i]' must have length weight.shape(i) + 1");
                        }
                    }
                }
            }

            else if (var_name == "bitwise") {
                if (!py::isinstance<py::dict>(item.second)) {
                    throw std::invalid_argument(
                        "'bitwise' must be provided as a dict with keys "
                        "'default_value', 'nrealizations', 'noffset', 'p_correction_nbits'");
                }

                py::dict d = py::cast<py::dict>(item.second);

                if (d.contains("default_value")) {
                    bitwise_default_value = py::cast<FLOAT>(d["default_value"]);
                }
                if (d.contains("nrealizations")) {
                    bitwise_nrealizations = py::cast<FLOAT>(d["nrealizations"]);
                }
                if (d.contains("noffset")) {
                    bitwise_noffset = py::cast<int>(d["noffset"]);
                }
                if (d.contains("p_correction_nbits")) {
                    auto pcorr = py::cast<py::array_t<FLOAT>>(d["p_correction_nbits"]);
                    if (pcorr.ndim() != 2) {
                        throw std::invalid_argument(
                            "'p_correction_nbits' must be a 2D array");
                    }
                    if (pcorr.shape(0) != pcorr.shape(1)) {
                        throw std::invalid_argument(
                            "'p_correction_nbits' must be a square array");
                    }
                    bitwise_p_correction_nbits = py::array_t<FLOAT>(pcorr.attr("copy")());
                }
            }

            else {
                throw std::invalid_argument(
                    "Invalid argument '" + var_name + "' for WeightAttrs");
            }
        }
    }

    WeightAttrs data() {
        WeightAttrs wattrs = {0};

        for (size_t i = 0; i < MAX_NMESH; i++) {
            wattrs.spin[i] = 0;
            if (i < spin.size()) wattrs.spin[i] = spin[i];
        }

        // Angular
        for (size_t i = 0; i < MAX_NBIN; i++) {
            wattrs.angular.sep[i] = nullptr;
            wattrs.angular.edges[i] = nullptr;
            wattrs.angular.shape[i] = 0;
        }
        wattrs.angular.weight = nullptr;
        wattrs.angular.size = 0;
        wattrs.angular.ndim = 0;

        if (angular_weight.size()) {
            py::buffer_info winfo = angular_weight.request();

            wattrs.angular.ndim = static_cast<size_t>(winfo.ndim);
            wattrs.angular.size = static_cast<size_t>(angular_weight.size());
            wattrs.angular.weight = angular_weight.mutable_data();

            for (ssize_t i = 0; i < winfo.ndim; i++) {
                wattrs.angular.shape[i] = static_cast<size_t>(winfo.shape[i]);
            }

            if (!angular_sep.empty()) {
                for (size_t i = 0; i < wattrs.angular.ndim; i++) {
                    wattrs.angular.sep[i] = angular_sep[i].mutable_data();
                }
            }

            if (!angular_edges.empty()) {
                for (size_t i = 0; i < wattrs.angular.ndim; i++) {
                    wattrs.angular.edges[i] = angular_edges[i].mutable_data();
                }
            }
        }

        // Bitwise
        wattrs.bitwise.default_value = bitwise_default_value;
        wattrs.bitwise.nrealizations = bitwise_nrealizations;
        wattrs.bitwise.noffset = bitwise_noffset;
        wattrs.bitwise.p_nbits = 0;
        wattrs.bitwise.p_correction_nbits = nullptr;

        if (bitwise_p_correction_nbits.size()) {
            wattrs.bitwise.p_nbits =
                static_cast<size_t>(bitwise_p_correction_nbits.shape(0));
            wattrs.bitwise.p_correction_nbits =
                bitwise_p_correction_nbits.mutable_data();
        }

        return wattrs;
    }
};


// Expose the MeshAttrs struct to Python
struct MeshAttrs_py {
    // use std::vector for simple, copyable storage
    std::vector<FLOAT> boxsize;    // length NDIM (optional)
    std::vector<FLOAT> boxcenter;  // length NDIM (optional)
    std::vector<size_t> meshsize;  // length NDIM (optional)

    FLOAT smax = 0.0;
    bool periodic = false;
    MESH_TYPE type = MESH_CARTESIAN;

    MeshAttrs_py() {}

    // Read named kwargs; expect array-like inputs (no scalar expansion)
    MeshAttrs_py(const py::kwargs& kwargs) {
        for (auto item : kwargs) {
            std::string key = py::cast<std::string>(item.first);
            if (key == "boxsize") {
                boxsize = py::cast<std::vector<FLOAT>>(item.second);
            } else if (key == "boxcenter") {
                boxcenter = py::cast<std::vector<FLOAT>>(item.second);
            } else if (key == "meshsize") {
                meshsize = py::cast<std::vector<size_t>>(item.second);
            } else if (key == "smax") {
                smax = py::cast<FLOAT>(item.second);
            } else if (key == "periodic") {
                periodic = py::cast<bool>(item.second);
            } else if (key == "type") {
                if (py::isinstance<py::str>(item.second)) {
                    std::string t = py::cast<std::string>(item.second);
                    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
                    if (t == "angular") type = MESH_ANGULAR;
                    else if (t == "cartesian") type = MESH_CARTESIAN;
                    else throw std::invalid_argument("Invalid mesh type: " + t);
                } else {
                    type = static_cast<MESH_TYPE>(py::cast<int>(item.second));
                }
            } else {
                throw std::invalid_argument("Unknown MeshAttrs_py argument: " + key);
            }
        }
    }

    // Convert to plain C MeshAttrs. Missing entries are filled with sensible defaults.
    MeshAttrs data() const {
        MeshAttrs mattrs;
        for (size_t axis = 0; axis < NDIM; axis++) {
            mattrs.meshsize[axis] = (axis < meshsize.size() ? meshsize[axis] : 1);
            mattrs.boxsize[axis] = (axis < boxsize.size() ? boxsize[axis] : 0.0);
            mattrs.boxcenter[axis] = (axis < boxcenter.size() ? boxcenter[axis] : 0.0);
        }
        mattrs.type = type;
        mattrs.smax = smax;
        mattrs.periodic = periodic;
        return mattrs;
    }
};


// Helper functions for SPLIT_TYPE conversion
SPLIT_TYPE string_to_split_type(const std::string& split_name) {
    if (split_name == "jackknife") return SPLIT_JACKKNIFE;
    if (split_name == "none") return SPLIT_NONE;
    throw std::invalid_argument("Invalid SPLIT_TYPE string: " + split_name);
}


std::string split_type_to_string(SPLIT_TYPE split_type) {
    switch (split_type) {
        case SPLIT_JACKKNIFE: return "jackknife";
        case SPLIT_NONE: return "none";
        default: return "none";
    }
}


// Expose the SplitAttrs struct to Python
struct SplitAttrs_py {
    SPLIT_TYPE mode;
    size_t nsplits;
    size_t size;

    // Default constructor
    SplitAttrs_py() : mode(SPLIT_NONE), nsplits(0), size(1) {}

    // Constructor from Python dictionary
    // Expected format: {"mode": "jackknife", "nsplits": 100}
    SplitAttrs_py(const py::kwargs& kwargs)
        : mode(SPLIT_NONE), nsplits(0), size(1) {

        if (kwargs.size() == 0) {
            return;
        }
        // Extract mode
        if (kwargs.contains("mode")) {
            std::string mode_str = py::cast<std::string>(kwargs["mode"]);
            mode = string_to_split_type(mode_str);
        } else {
            throw std::invalid_argument("SplitAttrs requires 'mode' argument");
        }
        // Extract nsplits
        if (kwargs.contains("nsplits")) {
            nsplits = py::cast<size_t>(kwargs["nsplits"]);
            if ((mode != SPLIT_NONE) & (nsplits < 1)) {
                throw std::invalid_argument("Number of splits must be >= 1");
            }
        } else {
            throw std::invalid_argument("SplitAttrs requires 'nsplits' argument");
        }
        // Calculate size
        if (mode == SPLIT_JACKKNIFE) {
            size = 3 * nsplits;
        }
    }

    // Convert to plain C SplitAttrs
    SplitAttrs data() const {
        SplitAttrs spattrs;
        spattrs.mode = mode;
        spattrs.nsplits = nsplits;
        spattrs.size = size;
        return spattrs;
    }

};


struct Count2Layout {
    size_t nweights;
    std::vector<std::string> names;
    std::vector<ssize_t> shape;
    size_t size;
};


Count2Layout get_count2_layout(
    const IndexValue index_value1,
    const IndexValue index_value2,
    const BinAttrs& battrs,
    const SplitAttrs& spattrs)
{
    char raw_names[MAX_NWEIGHT][SIZE_NAME];
    const size_t nweights = get_count2_weight_names(
        index_value1,
        index_value2,
        raw_names);

    std::vector<std::string> names;
    names.reserve(nweights);
    for (size_t i = 0; i < nweights; ++i) {
        names.emplace_back(raw_names[i]);
    }

    std::vector<ssize_t> shape;
    if (spattrs.nsplits) {
        shape.push_back(static_cast<ssize_t>(spattrs.size));
    }

    for (size_t idim = 0; idim < battrs.ndim; ++idim) {
        shape.push_back(static_cast<ssize_t>(battrs.shape[idim]));
    }

    size_t size = 1;
    for (ssize_t s : shape) {
        size *= static_cast<size_t>(s);
    }

    return {nweights, std::move(names), std::move(shape), size};
}


struct Count3CloseLayout {
    size_t nweights;
    std::vector<std::string> names;
    std::vector<ssize_t> shape;
    size_t size;
};


static Count3CloseLayout get_count3close_layout(
    const BinAttrs& battrs12,
    const BinAttrs& battrs13,
    const BinAttrs& battrs23)
{
    std::vector<std::string> names = {"weight"};
    const size_t nweights = 1;

    std::vector<ssize_t> shape;
    size_t size = 1;

    const bool has23 = (battrs23.ndim != 0);
    BinAttrs battrs_arr[3] = {battrs12, battrs13, battrs23};
    DeviceCount3Layout layout3 = make_device_count3_layout(battrs_arr);

    // Axes from 1-2
    for (size_t idim = 0; idim < battrs12.ndim; ++idim) {
        if (battrs12.var[idim] == VAR_POLE) continue;
        shape.push_back(static_cast<ssize_t>(battrs12.shape[idim]));
    }

    // Axes from 1-3
    for (size_t idim = 0; idim < battrs13.ndim; ++idim) {
        if (battrs13.var[idim] == VAR_POLE) continue;
        shape.push_back(static_cast<ssize_t>(battrs13.shape[idim]));
    }

    // Optional axes from 2-3
    if (has23) {
        for (size_t idim = 0; idim < battrs23.ndim; ++idim) {
            if (battrs23.var[idim] == VAR_POLE) continue;
            shape.push_back(static_cast<ssize_t>(battrs23.shape[idim]));
        }
    }

    // Extra flattened projection axis
    if (layout3.nprojs > 1) {
        shape.push_back(static_cast<ssize_t>(layout3.nprojs));
    }

    for (ssize_t s : shape) {
        size *= static_cast<size_t>(s);
    }

    return {nweights, std::move(names), std::move(shape), size};
}

#endif