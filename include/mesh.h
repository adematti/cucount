#include "common.h"

#ifndef _CUCOUNT_MESH_
#define _CUCOUNT_MESH_

void set_mesh_extent(const MESH_TYPE mesh_type, FLOAT *boxsize, FLOAT *boxcenter, const Particles *list_particles, DeviceMemoryBuffer *buffer, cudaStream_t stream);

void free_mesh(Mesh *mesh);

#endif