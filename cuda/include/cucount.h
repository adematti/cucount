#ifndef _CUCOUNT_CUCOUNT_
#define _CUCOUNT_CUCOUNT_

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "count3close.h"
#include "attrs.h"
#include "layout.h"

namespace py = pybind11;



// The triplet output layout (Count3Layout, get_count3_layout) lives in the
// shared include/layout.h, alongside count2's: one definition per layout, so
// the two backends cannot present their results differently.



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