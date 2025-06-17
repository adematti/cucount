#include <stdlib.h>

#ifndef _CUCOUNT_COMMON_
#define _CUCOUNT_COMMON_

#define FLOAT double
#define NDIM 3

#define MAX(a, b) (((a) > (b)) ? (a) : (b))  // maximum of two numbers
#define MIN(a, b) (((a) < (b)) ? (a) : (b))  // minimum of two numbers
#define CLIP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))  // min(max(a, low), high)
#define MAX_NMESH 3  // 3-pt correlation function at maximum
#define MAX_NBIN 3  // maximum number of binning dimension
#define DTORAD 0.017453292519943295 // x deg = x*DTORAD rad


// Logging levels
typedef enum {LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARN, LOG_LEVEL_ERROR} LogLevel;

// Declare the global logging level as extern
extern LogLevel global_log_level;

void log_message(LogLevel level, const char *format, ...);


typedef enum {MESH_CARTESIAN, MESH_ANGULAR} MESH_TYPE;
typedef enum {VAR_NONE, VAR_S, VAR_MU, VAR_THETA, VAR_POLE, VAR_K} VAR_TYPE;
typedef enum {LOS_NONE, LOS_FIRSTPOINT, LOS_ENDPOINT, LOS_MIDPOINT} LOS_TYPE;


typedef struct {
    size_t size;
    size_t total_nparticles;
    size_t *nparticles;
    size_t *cumnparticles;
    FLOAT *spositions;
    FLOAT *positions;
    FLOAT *weights;
} Mesh;


// Particles
typedef struct {
    size_t size=0;
    FLOAT *spositions;  // positions on the sphere
    FLOAT *positions;
    FLOAT *weights;
} Particles; // Particles


typedef struct {
    VAR_TYPE var[MAX_NBIN];
    LOS_TYPE los[MAX_NBIN];
    FLOAT min[MAX_NBIN], max[MAX_NBIN], step[MAX_NBIN], los[MAX_NBIN];
    size_t shape[MAX_NBIN];
    size_t size;
    size_t ndim;
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
} MeshAttrs;

void* my_calloc(size_t num, size_t size);
void* my_malloc(size_t size);

#endif //_CUCOUNT_COMMON_