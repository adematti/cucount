#ifndef _CUCOUNT_ARGS_
#define _CUCOUNT_ARGS_

// What an entry point is asked to count, bundled: the mesh, the binning, the
// weighting, the selections and the splits. Shared by both backends, so the
// two cannot drift on what a request consists of, and so neither host
// signature has to carry seventeen parameters.
//
// Deliberately only the attrs. What each backend needs to *run* the request --
// particles or an already-built mesh, a device buffer and stream, a thread
// count, where to write -- stays in its own signature, because those have
// nothing in common.

#include "common.h"


struct Count2Attrs {
    MeshAttrs mattrs;
    BinAttrs battrs;
    WeightAttrs wattrs;
    SelectionAttrs sattrs;
    SplitAttrs spattrs;
};


// One bundle for both triplet entry points: count3 is count3close without a
// (2, 3) axis, so it simply leaves battrs23 / sattrs23 / veto23 default. The
// CUDA count3 already zeroed a local battrs23 for exactly this reason.
struct Count3Attrs {
    MeshAttrs mattrs1;
    MeshAttrs mattrs2;
    MeshAttrs mattrs3;

    BinAttrs battrs12;
    BinAttrs battrs13;
    BinAttrs battrs23;

    WeightAttrs wattrs;

    SelectionAttrs sattrs12;
    SelectionAttrs sattrs13;
    SelectionAttrs sattrs23;
    SelectionAttrs veto12;
    SelectionAttrs veto13;
    SelectionAttrs veto23;
};

#endif  // _CUCOUNT_ARGS_
