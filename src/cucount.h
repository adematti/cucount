#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"

struct BinAttrs_py {
    std::vector<FLOAT> min, max, step;
    std::vector<VAR_TYPE> var;
    std::vector<LOS_TYPE> los;
    std::vector<pybind11::array_t<FLOAT>> array;

    BinAttrs_py(const pybind11::kwargs& kwargs);

    std::vector<size_t> shape() const;
    size_t size() const;
    size_t ndim() const;
    BinAttrs data();
};

struct SelectionAttrs_py {
    std::vector<FLOAT> min, max;
    std::vector<VAR_TYPE> var;

    SelectionAttrs_py();
    SelectionAttrs_py(const pybind11::kwargs& kwargs);

    size_t ndim() const;
    SelectionAttrs data() const;
};