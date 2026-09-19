#ifndef _CUCOUNT_COUNT3_
#define _CUCOUNT_COUNT3_

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <sm_20_atomic_functions.h>
#include "common.h"
#include "args.h"


void count3(
    FLOAT *counts,
    Mesh mesh1,
    Mesh mesh2,
    Mesh mesh3,
    const Count3Attrs &attrs,
    DeviceMemoryBuffer *buffer,
    cudaStream_t stream);


#endif