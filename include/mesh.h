#include "common.h"

#ifndef _CUCOUNT_MESH_
#define _CUCOUNT_MESH_

void set_mesh_attrs(const Particles *list_particles, MeshAttrs *mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);

void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);

void free_mesh(Mesh *mesh);

#endif