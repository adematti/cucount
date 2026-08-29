#ifndef _CUCOUNT_CUCOUNT_
#define _CUCOUNT_CUCOUNT_

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "count3close.h"
#include "attrs.h"

namespace py = pybind11;


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
    DeviceCount3Layout layout3 = make_device_count3_layout(battrs12, battrs13, battrs23);

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
    if (layout3.nprojs >= 1) {
        shape.push_back(static_cast<ssize_t>(layout3.nprojs));
    }

    for (ssize_t s : shape) {
        size *= static_cast<size_t>(s);
    }

    return {nweights, std::move(names), std::move(shape), size};
}


static Count3CloseLayout get_count3_layout(
    const BinAttrs& battrs12,
    const BinAttrs& battrs13)
{
    std::vector<std::string> names = {"weight"};
    const size_t nweights = 1;

    std::vector<ssize_t> shape;
    size_t size = 1;

    BinAttrs battrs23{};
    DeviceCount3Layout layout3 = make_device_count3_layout(
        battrs12,
        battrs13,
        battrs23);

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

    // Extra flattened projection axis
    if (layout3.nprojs >= 1) {
        shape.push_back(static_cast<ssize_t>(layout3.nprojs));
    }

    for (ssize_t s : shape) {
        size *= static_cast<size_t>(s);
    }

    return {nweights, std::move(names), std::move(shape), size};
}


static CLOSE_PAIR parse_close_pair(py::tuple close_pair)
{
    if (close_pair.size() != 2) {
        throw std::invalid_argument(
            "count3close: close_pair must be a tuple of length 2: "
            "(1, 2), (1, 3), or (2, 3)");
    }

    int i = py::cast<int>(close_pair[0]);
    int j = py::cast<int>(close_pair[1]);

    if (i > j) std::swap(i, j);

    if (i == 1 && j == 2) return CLOSE_PAIR_12;
    if (i == 1 && j == 3) return CLOSE_PAIR_13;
    if (i == 2 && j == 3) return CLOSE_PAIR_23;

    throw std::invalid_argument(
        "count3close: close_pair must be one of (1, 2), (1, 3), or (2, 3)");
}

#endif