#include "common.h"

#ifndef _CUCOUNT_MESH_
#define _CUCOUNT_MESH_

// keep_original_index makes the builder record slot -> input particle index in
// mesh.original_index, for callers that write per-particle results back (the
// soft adjoint). Off by default: the forward paths must not pay for an
// allocation they never read.
void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs, DeviceMemoryBuffer *buffer, cudaStream_t stream,
              bool keep_original_index = false);

void free_mesh(Mesh *mesh);

#endif