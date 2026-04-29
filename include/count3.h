#ifndef _CUCOUNT_COUNT3CLOSE_
#define _CUCOUNT_COUNT3CLOSE_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"


void count3(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs veto12,
    SelectionAttrs veto13,
    BinAttrs battrs12,
    BinAttrs battrs13,
    WeightAttrs wattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream);


#endif