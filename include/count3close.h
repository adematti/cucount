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


typedef struct {
    size_t nprimaries;
    size_t npairs;
    size_t *offsets;    // device, size nprimaries + 1
    size_t *secondary;  // device, size npairs
} ClosePairs;


void free_device_close_pairs(ClosePairs *cpairs, DeviceMemoryBuffer *buffer);

void fill_close_pairs(ClosePairs *cpairs, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);

void count3_close(FLOAT *counts, ClosePairs close_ab, ClosePairs close_ac, const Mesh *list_mesh,
    MeshAttrs mattrs, BinAttrs *battrs[3], WeightAttrs wattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);



#endif