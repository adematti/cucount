#include <stdlib.h>
#include <cuda.h>

#ifndef _CUCOUNT_COMMON_
#define _CUCOUNT_COMMON_

#define FLOAT double
#define INT long
#define POPCOUNT __popcll
#define NDIM 3

#define MAX(a, b) (((a) > (b)) ? (a) : (b))  // maximum of two numbers
#define MIN(a, b) (((a) < (b)) ? (a) : (b))  // minimum of two numbers
#define CLIP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))  // min(max(a, low), high)
#define MAX_NMESH 3  // 3-pt correlation function at maximum
#define MAX_NBIN 3  // maximum number of binning dimension
#define MAX_POLE 8
#define DTORAD 0.017453292519943295 // x deg = x*DTORAD rad


// Logging levels
typedef enum {LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARN, LOG_LEVEL_ERROR} LogLevel;

// Declare the global logging level as extern
extern LogLevel global_log_level;

void log_message(LogLevel level, const char *format, ...);

typedef enum {MESH_NONE, MESH_CARTESIAN, MESH_ANGULAR} MESH_TYPE;
typedef enum {VAR_NONE, VAR_S, VAR_MU, VAR_THETA, VAR_POLE, VAR_K} VAR_TYPE;
typedef enum {LOS_NONE, LOS_FIRSTPOINT, LOS_ENDPOINT, LOS_MIDPOINT, LOS_X, LOS_Y, LOS_Z} LOS_TYPE;


#define atomicAddSizet(address, val)                            \
    (sizeof(size_t) == 4 ?                                      \
        atomicAdd((unsigned int*)(address), (unsigned int)(val)) \
      : atomicAdd((unsigned long long*)(address), (unsigned long long)(val)))


// Helper function for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            log_message(LOG_LEVEL_ERROR, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CONFIGURE_KERNEL_LAUNCH(kernel, nblocks_var, nthreads_var, buffer) \
    do { \
        cudaOccupancyMaxPotentialBlockSize(&(nblocks_var), &(nthreads_var), kernel, 0, 0); \
        if (buffer) nblocks_var = MIN(buffer->nblocks, nblocks_var); \
        log_message(LOG_LEVEL_INFO, "Configured kernel with %d blocks and %d threads per block.\n", (nblocks_var), (nthreads_var)); \
    } while (0)


typedef struct {
    // To check/modify when adding new weighting scheme
    size_t start_spin, size_spin, start_individual_weight, size_individual_weight, start_bitwise_weight, size_bitwise_weight,
    start_negative_weight, size_negative_weight, size;
} IndexValue;


#define MAX_NWEIGHT 4
#define SIZE_NAME 32

typedef struct {
    size_t size;
    size_t total_nparticles;
    size_t *nparticles;
    size_t *cumnparticles;
    FLOAT *spositions;
    FLOAT *positions;
    FLOAT *values;  // contains spin components, individual weights, bitwise weights in this order
    IndexValue index_value;
} Mesh;


// Particles
typedef struct {
    size_t size;
    FLOAT *spositions;  // positions on the sphere
    FLOAT *positions;
    FLOAT *values;  // contains shear values, individual weights, bitwise weights in this order
    IndexValue index_value;
} Particles; // Particles


typedef struct {
    VAR_TYPE var[MAX_NBIN];
    LOS_TYPE los[MAX_NBIN];
    FLOAT min[MAX_NBIN], max[MAX_NBIN], step[MAX_NBIN];
    FLOAT *array[MAX_NBIN];
    size_t shape[MAX_NBIN], asize[MAX_NBIN];
    size_t size;
    size_t ndim;  // binning dimensionality (e.g. 1D or 2D)
} BinAttrs;


typedef struct {
    VAR_TYPE var[MAX_NBIN];
    FLOAT min[MAX_NBIN], max[MAX_NBIN];
    FLOAT smin[MAX_NBIN], smax[MAX_NBIN];
    size_t ndim;
} SelectionAttrs;


typedef struct {
    size_t meshsize[NDIM];
    FLOAT boxsize[NDIM];
    FLOAT boxcenter[NDIM];
    FLOAT smax;
    MESH_TYPE type;
    bool periodic;
} MeshAttrs;


typedef struct {
    FLOAT default_value;
    FLOAT nrealizations;
    int noffset;
    size_t p_nbits;
    FLOAT *p_correction_nbits;
} BitwiseWeight;


typedef struct {
    FLOAT *sep;
    FLOAT *weight;
    size_t size;
} AngularWeight;


// To check/modify when adding new weighting scheme
typedef struct {
    size_t spin[MAX_NMESH];
    BitwiseWeight bitwise;
    AngularWeight angular;
} WeightAttrs;


// Device memory buffer struct
struct DeviceMemoryBuffer {
    void* ptr;
    size_t size;
    size_t offset;
    size_t nblocks;
};


IndexValue get_index_value(int size_spin, int size_individual_weight, int size_bitwise_weight, int size_negative_weight);

size_t get_count2_size(IndexValue index_value1, IndexValue index_value2, char names[][SIZE_NAME]);

void* my_device_malloc(size_t nbytes, DeviceMemoryBuffer* buffer);

void my_device_free(void* ptr, DeviceMemoryBuffer* buffer);

void* my_calloc(size_t num, size_t size);

void* my_malloc(size_t size);

void copy_particles_to_device(Particles particles, Particles *device_particles, int mode);

void copy_mesh_to_device(Mesh mesh, Mesh *device_mesh, int mode);

void free_device_particles(Particles *particles);

void free_device_mesh(Mesh *mesh);

void copy_particles_to_host(Particles particles, Particles *host_particles, int mode);

void copy_mesh_to_host(Mesh mesh, Mesh *host_mesh, int mode);

void free_host_particles(Particles *particles);

void free_host_mesh(Mesh *mesh);


#endif //_CUCOUNT_COMMON_