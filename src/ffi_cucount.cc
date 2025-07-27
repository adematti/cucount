#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mesh.h"
#include "count2.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;


void set_mesh_attrs_py(MeshAttrs *cmattrs, BinAttrs_py& battrs, const SelectionAttrs_py& sattrs = SelectionAttrs_py()) {
    SelectionAttrs csattrs = sattrs.data();
    BinAttrs cbattrs = battrs.data();
    MeshAttrs cmattrs;
    for (size_t axis = 0; axis < NDIM; axis++) {
        cmattrs.meshsize[axis] = 0;
        cmattrs.boxsize[axis] = 0.;
        cmattrs.boxcenter[axis] = 0.;
    }
    if ((csattrs.ndim) && (csattrs.var[0] == VAR_THETA)) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(csattrs.max[0] * DTORAD);
    }
    else if ((csattrs.ndim) && (csattrs.var[0] == VAR_S)) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = csattrs.max[0];
    }
    else if (cbattrs.var[0] == VAR_THETA) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(cbattrs.max[0] * DTORAD);
    }
    else if (cbattrs.var[0] == VAR_S) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = cbattrs.max[0];
    }

}


py::array_t<FLOAT> count2_py(Particles_py& particles1, Particles_py& particles2,
               BinAttrs_py& battrs, const SelectionAttrs_py& sattrs = SelectionAttrs_py()) {
    // Convert Python inputs to C objects
    Particles list_particles[MAX_NMESH];
    for (size_t imesh=0; imesh < MAX_NMESH; imesh++) list_particles[imesh].size = 0;
    // In this function Particles and Mesh struct live on the host (CPU)
    // but their arrays (positions, weights, etc.) live on the device (GPU)
    copy_particles_to_device(particles1.data(), &list_particles[0], 2);
    copy_particles_to_device(particles2.data(), &list_particles[1], 2);
    Mesh list_mesh[MAX_NMESH];

    SelectionAttrs csattrs = sattrs.data();
    BinAttrs cbattrs = battrs.data();
    MeshAttrs cmattrs;
    for (size_t axis = 0; axis < NDIM; axis++) {
        cmattrs.meshsize[axis] = 0;
        cmattrs.boxsize[axis] = 0.;
        cmattrs.boxcenter[axis] = 0.;
    }
    // Create a numpy array to store the results
    py::array_t<FLOAT> counts(battrs.shape());

    if ((csattrs.ndim) && (csattrs.var[0] == VAR_THETA)) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(csattrs.max[0] * DTORAD);
    }
    else if ((csattrs.ndim) && (csattrs.var[0] == VAR_S)) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = csattrs.max[0];
    }
    else if (cbattrs.var[0] == VAR_THETA) {
        cmattrs.type = MESH_ANGULAR;
        cmattrs.smax = cos(cbattrs.max[0] * DTORAD);
    }
    else if (cbattrs.var[0] == VAR_S) {
        cmattrs.type = MESH_CARTESIAN;
        cmattrs.smax = cbattrs.max[0];
    }
    //return counts;
    set_mesh_attrs(list_particles, &cmattrs);
    set_mesh(list_particles, list_mesh, cmattrs);
    // Free allocated memory
    for (size_t i = 0; i < 2; i++) free_device_particles(&(list_particles[i]));
    // Perform the computation
    auto counts_ptr = counts.mutable_data(); // Get a pointer to the array's data
    count2(counts_ptr, list_mesh, cmattrs, csattrs, cbattrs);
    // Free allocated memory
    for (size_t i = 0; i < 2; i++) free_device_mesh(&(list_mesh[i]));
    // Return the numpy array
    return counts;
}


// Bind the function and structs to Python
PYBIND11_MODULE(ffi_cucount, m) {

    m.def("count2", &count2_py, "Perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("battrs"),
        py::arg("sattrs") = SelectionAttrs_py()); // Default value
}