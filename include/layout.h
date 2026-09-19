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


// The ell values a multipole axis names. Equivalent to the CUDA fill_ells,
// kept here so the layout is computable without linking a CUDA object.
inline size_t fill_ells(const BinAttrs& battrs, int index, size_t* ells) {
    const size_t ellmin = static_cast<size_t>(battrs.min[index]);
    const size_t ellmax = static_cast<size_t>(battrs.max[index]);
    const size_t ellstep = (battrs.bin[index] == BIN_LIN)
        ? static_cast<size_t>(battrs.step[index]) : size_t{1};

    if (ellstep == 0) return 0;
    size_t nells = 0;
    for (size_t ell = ellmin; ell <= ellmax; ell += ellstep) ells[nells++] = ell;
    return nells;
}


// Which multipoles a count2 asks for, and where they land. The kernels want
// this resolved once on the host rather than re-derived per pair: the CUDA
// backend copies it to constant memory, the CPU backend passes it down.
struct Count2PoleLayout {
    size_t nbins = 0;
    size_t csize = 0;
    size_t split_size = 1;
    size_t nells = 0;
    size_t ells[MAX_POLE + 2] = {0};
    bool ells_even = false;
};


inline Count2PoleLayout make_count2_pole_layout(const IndexValue index_value1,
                                                const IndexValue index_value2,
                                                const BinAttrs& battrs,
                                                const SplitAttrs& spattrs) {
    Count2PoleLayout layout;
    layout.nbins = battrs.size;
    layout.split_size = spattrs.size;
    layout.csize = get_count2_weight_names(index_value1, index_value2, NULL) *
                   spattrs.size * battrs.size;

    if (battrs.ndim > 0 && battrs.var[battrs.ndim - 1] == VAR_POLE) {
        const int ipole = static_cast<int>(battrs.ndim) - 1;
        layout.nells = fill_ells(battrs, ipole, layout.ells);

        if (layout.nells > 0) {
            if (layout.nells == 1) {
                layout.ells_even = (layout.ells[0] % 2 == 0);
            }
            else {
                // A listed set of ells is stepped by one; only a (min, max,
                // step) axis can promise an even step.
                const size_t ellstep = (battrs.asize[ipole] == 0)
                    ? static_cast<size_t>(battrs.step[ipole]) : size_t{1};
                layout.ells_even = ((layout.ells[0] % 2 == 0) && (ellstep % 2 == 0));
            }
        }
    }
    return layout;
}


// ---------------------------------------------------------------------------
// Triplet counts
// ---------------------------------------------------------------------------

// Which multipoles each leg of a triplet count projects onto: how many real
// spherical-harmonic coefficients that is per leg, and how many survive the
// contraction over m. nprojs == 0 means the plain, unprojected outer product
// of the two separation histograms.
#ifndef ELLMAX
#define ELLMAX 5
#endif
#ifndef MMAX_SIZE
#define MMAX_SIZE 6
#endif

struct Count3PoleLayout {
    size_t nbins = 0;
    size_t nprojs1 = 0;
    size_t nprojs2 = 0;
    size_t nprojs = 0;
    size_t csize = 0;
    size_t nells1 = 0;
    size_t nells2 = 0;
    int ellmax1 = 0;
    int ellmax2 = 0;
    size_t ells1[MMAX_SIZE] = {0};
    size_t ells2[MMAX_SIZE] = {0};
};




inline Count3PoleLayout make_count3_pole_layout(const BinAttrs& battrs12,
                                                const BinAttrs& battrs13,
                                                const BinAttrs& battrs23) {
    Count3PoleLayout layout;

    if (battrs12.ndim == 0 || battrs13.ndim == 0) return layout;

    layout.nbins = battrs12.shape[0] * battrs13.shape[0];

    if (battrs23.ndim > 0) {
        layout.nbins *= battrs23.shape[0];
        layout.csize = layout.nbins;
        return layout;
    }

    if (battrs12.var[1] == VAR_POLE && battrs13.var[1] == VAR_POLE) {
        layout.nells1 = fill_ells(battrs12, 1, layout.ells1);
        layout.nells2 = fill_ells(battrs13, 1, layout.ells2);

        for (size_t ill1 = 0; ill1 < layout.nells1; ill1++) {
            const int ell1 = static_cast<int>(layout.ells1[ill1]);
            if (ell1 > layout.ellmax1) layout.ellmax1 = ell1;
            layout.nprojs1 += static_cast<size_t>(2 * ell1 + 1);

            for (size_t ill2 = 0; ill2 < layout.nells2; ill2++) {
                const int ell2 = static_cast<int>(layout.ells2[ill2]);
                if (ell2 > layout.ellmax2) layout.ellmax2 = ell2;
                layout.nprojs += static_cast<size_t>(2 * (ell1 < ell2 ? ell1 : ell2) + 1);
            }
        }

        for (size_t ill2 = 0; ill2 < layout.nells2; ill2++)
            layout.nprojs2 += static_cast<size_t>(2 * layout.ells2[ill2] + 1);

        layout.csize = layout.nbins * layout.nprojs;
    }
    else {
        layout.csize = layout.nbins;
    }

    return layout;
}


// The named, shaped result: one "weight" channel, the non-pole axes of each
// leg in order, then the flattened projection axis when there is one.
struct Count3Layout {
    size_t nweights;
    std::vector<std::string> names;
    std::vector<ssize_t> shape;
    size_t size;
};


inline Count3Layout get_count3_layout(const BinAttrs& battrs12,
                                          const BinAttrs& battrs13,
                                          const BinAttrs& battrs23) {
    const Count3PoleLayout proj = make_count3_pole_layout(battrs12, battrs13, battrs23);

    std::vector<ssize_t> shape;
    for (const BinAttrs* b : {&battrs12, &battrs13, &battrs23}) {
        for (size_t idim = 0; idim < b->ndim; ++idim) {
            if (b->var[idim] == VAR_POLE) continue;
            shape.push_back(static_cast<ssize_t>(b->shape[idim]));
        }
    }
    if (proj.nprojs >= 1) shape.push_back(static_cast<ssize_t>(proj.nprojs));

    size_t size = 1;
    for (ssize_t s : shape) size *= static_cast<size_t>(s);

    return {1, {"weight"}, std::move(shape), size};
}

#endif  // _CUCOUNT_LAYOUT_
