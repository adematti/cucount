#ifndef _CUCOUNT_COUNT3CLOSE_
#define _CUCOUNT_COUNT3CLOSE_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"


#ifndef SQRT1_2
#define SQRT1_2 0.70710678118654752440
#endif

#ifndef SQRT3_8
#define SQRT3_8 0.61237243569579452455
#endif

#ifndef SQRT3_2
#define SQRT3_2 1.22474487139158904910
#endif

#ifndef SQRT15_8
#define SQRT15_8 1.36930639376291527536
#endif

#ifndef SQRT5_4
#define SQRT5_4 1.11803398874989484820
#endif

#ifndef SQRT10_8
#define SQRT10_8 1.11803398874989484820
#endif

#ifndef SQRT70_16
#define SQRT70_16 2.09165006633518886818
#endif


typedef struct DeviceCount3Layout {
    size_t nbins;
    size_t nprojs;
    size_t csize;
    size_t nells1;
    size_t nells2;
    size_t ells1[4];
    size_t ells2[4];
} DeviceCount3Layout;


DeviceCount3Layout make_device_count3_layout(BinAttrs battrs[3]);


void count3_close(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    MeshAttrs mattrs2,
    MeshAttrs mattrs3,
    SelectionAttrs sattrs12,
    SelectionAttrs sattrs13,
    SelectionAttrs sattrs23,
    bool veto13,
    BinAttrs battrs12,
    BinAttrs battrs13,
    BinAttrs battrs23,
    WeightAttrs wattrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream);


#endif