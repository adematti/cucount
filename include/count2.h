#ifndef _CUCOUNT_COUNT2_
#define _CUCOUNT_COUNT2_


void count2(FLOAT* counts, const Mesh *list_mesh, const MeshAttrs mattrs, const SelectionAttrs sattrs, BinAttrs battrs, DeviceMemoryBuffer *buffer, cudaStream_t stream);


#endif