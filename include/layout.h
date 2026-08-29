#ifndef _CUCOUNT_LAYOUT_
#define _CUCOUNT_LAYOUT_

// Output layout shared by the CUDA and CPU bindings: which named weight
// channels a pair count produces (from the particles' IndexValue) and the
// shape/size of the result. One definition, so the backends cannot disagree
// on channel names or ordering.

#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>

#include "common.h"


inline size_t get_count2_weight_names(
    IndexValue index_value1,
    IndexValue index_value2,
    char names[][SIZE_NAME])
{
    int s1 = (index_value1.size_spin > 0);
    int s2 = (index_value2.size_spin > 0);
    size_t n = 1 + s1 + s2;

    if (names == NULL) {
        return n;
    }

    for (size_t i = 0; i < MAX_NWEIGHT; ++i) {
        names[i][0] = '\0';
    }

    if (s1 && s2) {
        strncpy(names[0], "weight_plus_plus", SIZE_NAME - 1);
        strncpy(names[1], "weight_plus_cross", SIZE_NAME - 1);
        strncpy(names[2], "weight_cross_cross", SIZE_NAME - 1);
    }
    else if (s1 ^ s2) {
        strncpy(names[0], "weight_plus", SIZE_NAME - 1);
        strncpy(names[1], "weight_cross", SIZE_NAME - 1);
    }
    else {
        strncpy(names[0], "weight", SIZE_NAME - 1);
    }

    return n;
}


struct Count2Layout {
    size_t nweights;
    std::vector<std::string> names;
    std::vector<ssize_t> shape;
    size_t size;
};


inline Count2Layout get_count2_layout(
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

#endif  // _CUCOUNT_LAYOUT_
