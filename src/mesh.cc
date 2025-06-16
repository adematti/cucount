#include <math.h>
#include <stdio.h>
#include "common.h"

#define LARGE_VALUE 1e4 // Replace magic numbers with constants
#define DTORAD 0.017453292519943295 // x deg = x*DTORAD rad


// Function to calculate the Cartesian distance
FLOAT cartesian_distance(const FLOAT* positions) {
    FLOAT rr = 0.0;
    for (size_t i = 0; i < NDIM; i++) {
        rr += positions[i] * positions[i];
    }
    return sqrt(rr); // Added sqrt to compute actual distance
}

// Function to convert Cartesian coordinates to spherical coordinates
void cartesian_to_sphere(const FLOAT* positions, FLOAT *r, FLOAT *cth, FLOAT *phi) {
    *r = cartesian_distance(positions);

    if (*r == 0) {
        *cth = 1.0;
        *phi = 0.0;
    } else {
        FLOAT x_norm = positions[0] / *r;
        FLOAT y_norm = positions[1] / *r;
        FLOAT z_norm = positions[2] / *r;

        *cth = z_norm;
        if (x_norm == 0 && y_norm == 0) {
            *phi = 0.0;
        } else {
            *phi = atan2(y_norm, x_norm);
            if (*phi < 0) {
                *phi += 2 * M_PI;
            }
        }
    }
}


static FLOAT wrap_angle(FLOAT phi) {
    // Wrap phi into the range [0, 2 * M_PI]
    phi = fmod(phi, 2 * M_PI); // Use modulo to handle wrapping
    if (phi < 0) {
        phi += 2 * M_PI; // Ensure phi is positive
    }
    return phi;
}


static size_t angle_to_cell(const size_t *meshsize, const FLOAT cth, const FLOAT phi) {
    // Returns the pixel index for coordinates cth (cos(theta)) and phi (azimuthal angle)

    // Validate input
    if (cth < -1 || cth > 1) {
        log_message(LOG_LEVEL_ERROR, "Invalid cos(theta) value: %lf. Must be in range [-1, 1].", cth);
        exit(EXIT_FAILURE); // Exit on invalid input
    }

    // Compute pixel indices
    int icth = (cth == 1) ? (meshsize[0] - 1) : (int)(0.5 * (1 + cth) * meshsize[1]);
    int iphi = (int)(0.5 * wrap_angle(phi) / M_PI * meshsize[1]);

    // Return combined pixel index
    return iphi + icth * meshsize[1];
}


void set_mesh_attrs(const Particles *list_particles, MeshAttrs mattrs) {
    if ((mattrs.type == MESH_ANGULAR) && (mattrs.meshsize[0] == 0 || mattrs.meshsize[1] == 0)) {
        FLOAT cth_min = LARGE_VALUE, cth_max = -LARGE_VALUE;
        FLOAT phi_min = LARGE_VALUE, phi_max = -LARGE_VALUE;

        // Loop through particle positions
        size_t sum_nparticles = 0, n_nparticles = 0;
        for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
            const Particles particles = list_particles[imesh];
            if (particles.nparticles == 0) continue;
            sum_nparticles += particles.nparticles;
            n_nparticles += 1;
            for (size_t i = 0; i < particles.nparticles; i++) {
                const FLOAT *position = &(particles.positions[NDIM * i]);
                FLOAT cth, phi, r;
                cartesian_to_sphere(position, &r, &cth, &phi);

                if (i == 0) {
                    cth_min = cth_max = cth;
                    phi_min = phi_max = phi;
                }

                if (cth < cth_min) cth_min = cth;
                if (cth > cth_max) cth_max = cth;
                if (phi < phi_min) phi_min = phi;
                if (phi > phi_max) phi_max = phi;
            }
        }

        FLOAT fsky = (cth_max - cth_min) * (phi_max - phi_min) / (4 * M_PI);
        FLOAT theta_max = mattrs.sepmax * DTORAD;
        int nside1 = 5 * (int)(M_PI / theta_max);
        size_t nparticles = sum_nparticles / n_nparticles;
        int nside2 = (int)(sqrt(0.25 * nparticles / fsky));
        mattrs.meshsize[0] = (size_t) MIN(nside1, nside2);
        mattrs.meshsize[1] = 2 * mattrs.meshsize[0];
        size_t meshsize = mattrs.meshsize[0] * mattrs.meshsize[1];
        FLOAT pixel_resolution = sqrt(4 * M_PI / meshsize) / DTORAD;
        log_message(LOG_LEVEL_INFO, "There will be %d pixels in total\n", meshsize);
        log_message(LOG_LEVEL_INFO, "Pixel resolution is %.4lf deg\n", pixel_resolution);
    }
}


// Function to set the mesh
void set_mesh(const Particles *list_particles, Mesh *list_mesh, MeshAttrs mattrs) {

    for (size_t imesh=0; imesh<MAX_NMESH; imesh++) {
        const Particles particles = list_particles[imesh];
        if (particles.nparticles == 0) continue;
        Mesh mesh = list_mesh[imesh];
        mesh.size = 0;
        size_t meshsize[NDIM];
        size_t* index = (size_t*)my_calloc(particles.nparticles, sizeof(size_t));
        FLOAT* spositions = (FLOAT*)my_calloc(particles.nparticles, NDIM * sizeof(FLOAT));

        if (mattrs.type == MESH_ANGULAR) {
            mesh.size = mattrs.meshsize[0] * mattrs.meshsize[1];
            for (size_t i = 0; i < particles.nparticles; i++) {
                const FLOAT *position = &(particles.positions[NDIM * i]);
                FLOAT cth, phi, r;
                cartesian_to_sphere(position, &r, &cth, &phi);
                index[i] = angle_to_cell(mattrs.meshsize, cth, phi);
            }
        }

        for (size_t i = 0; i < particles.nparticles; i++) {
            const FLOAT *position = &(particles.positions[NDIM * i]);
            FLOAT r = cartesian_distance(position);
            for (size_t axis; axis < NDIM; axis++) spositions[NDIM * i + axis] = position[axis] / r;
        }

        // Allocate memory for box variables
        mesh.nparticles = (size_t*)my_calloc(mesh.size, sizeof(size_t));
        mesh.cumnparticles = (size_t*)my_malloc(mesh.size * sizeof(size_t));
        mesh.positions = (FLOAT*)my_malloc(NDIM * particles.nparticles * sizeof(FLOAT));
        mesh.spositions = (FLOAT*)my_malloc(NDIM * particles.nparticles * sizeof(FLOAT));
        mesh.weights = (FLOAT*)my_malloc(particles.nparticles * sizeof(FLOAT));

        // Initialize box variables
        size_t n_full_boxes = 0;
        for (size_t i = 0; i < particles.nparticles; i++) {
            size_t idx = index[i];
            if (mesh.nparticles[idx] == 0) n_full_boxes++;
            mesh.nparticles[idx]++;
        }

        log_message(LOG_LEVEL_INFO, "There are objects in %d out of %zu boxes.", n_full_boxes, meshsize);

        // Compute number of particles up to this cell
        size_t total_particles = 0;
        for (size_t i = 0; i < mesh.size; i++) {
            mesh.cumnparticles[i] = total_particles;
            total_particles += mesh.nparticles[i];
            mesh.nparticles[i] = 0; // Reset for reuse
        }

        // Assign particle positions to boxes
        for (size_t i = 0; i < particles.nparticles; i++) {
            size_t idx = index[i];
            FLOAT* position = &(particles.positions[NDIM * i]);
            FLOAT* sposition = &(spositions[NDIM * i]);
            size_t offset = NDIM * (mesh.cumnparticles[idx] + mesh.nparticles[idx]);
            for (size_t axis; axis < NDIM; axis++) {
                mesh.positions[offset + axis] = position[axis];
                mesh.spositions[offset + axis] = sposition[axis];
            }
            offset = (mesh.cumnparticles[idx] + mesh.nparticles[idx]);
            mesh.weights[offset] = particles.weights[i];
            mesh.nparticles[idx]++;
        }
        mesh.total_nparticles = particles.nparticles;

        log_message(LOG_LEVEL_INFO, "Mesh variables successfully set.");
    }
}


void free_mesh(Mesh mesh) {
    free(mesh.spositions);
    free(mesh.positions);
    free(mesh.weights);
    free(mesh.nparticles);
    free(mesh.cumnparticles);
}