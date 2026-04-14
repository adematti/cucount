#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // for std::vector conversion
#include <cstring>  // for std::memcpy
#include <thread>
#include <vector>
#include <memory>

#include "mesh.h"
#include "count2.h"
#include "count3close.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;


static bool is_contiguous(py::array array) {
    return array.flags() & py::array::c_style;
}



// Expose the Particles struct to Python
// NB: We could have made this more Pythonic, with arguments "spin_values", "individual_weights", "bitwise_weights", etc.
// But for simplicity, and given we also have ffi_count.cu, let's differ this to pure Python
struct Particles_py {
    py::array_t<FLOAT> positions;
    py::array_t<FLOAT> values; // Optional spin values array
    IndexValue index_value;

    // Single constructor accepting optional sky_coords and spin_values (can be None)
    Particles_py(py::array_t<FLOAT> positions_, py::array_t<FLOAT> values_ = py::none(),
        const int size_split = 0, const int size_spin = 0, const int size_individual_weight = 0, const int size_bitwise_weight = 0, const int size_negative_weight = 0)
        : positions(positions_), values(py::array_t<FLOAT>()) {

        this->index_value = get_index_value(size_split, size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
        //printf("%d %d %d %d\n", this->index_value.start_spin, this->index_value.size_spin, this->index_value.start_individual_weight, this->index_value.size_individual_weight);

        // Ensure positions are C-contiguous
        if (!is_contiguous(this->positions)) this->positions = py::array_t<FLOAT>(this->positions.attr("copy")());
        size_t npositions = this->positions.shape(0);

        if (this->index_value.size) {
            if (py::isinstance<py::none>(values_)) {
                throw std::invalid_argument(
                    "Particles_py: non-trivial values are indicated with size_*, but input values are empty");
            }
            auto array = py::cast<py::array_t<FLOAT>>(values_);
            if (!is_contiguous(array)) array = py::array_t<FLOAT>(array.attr("copy")());
            if (array.shape(0) != npositions) {
                throw std::invalid_argument(
                    "Particles_py: positions and values must have the same length, but got positions.shape(0) = " +
                    std::to_string(npositions) + " and values.shape(0) = " + std::to_string(array.shape(0))
                );
            }
            if (array.shape(1) < this->index_value.size) {
                throw std::invalid_argument(
                    "Particles_py: expected values with values.shape(1) >= " +
                    std::to_string(this->index_value.size) + " but only got values.shape(1) = " + std::to_string(array.shape(1))
                );
            }
            this->values = array;
        }

    }

    // Method to get the number of particles automatically
    size_t size() const {
        return positions.shape(0);
    }

    Particles data() {
        Particles particles;
        particles.index_value = index_value;
        particles.size = size();
        particles.positions = positions.mutable_data();
        if (values.data() != nullptr) particles.values = values.mutable_data();
        return particles;
    }
};


py::object count2_py(Particles_py& particles1, Particles_py& particles2,
                    MeshAttrs_py mattrs_py, BinAttrs_py battrs_py, WeightAttrs_py wattrs_py = WeightAttrs_py(),
                    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
                    const SplitAttrs_py spattrs_py = SplitAttrs_py(),
                    const int nthreads = 1) {

    BinAttrs battrs = battrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    MeshAttrs mattrs = mattrs_py.data();
    SplitAttrs spattrs = spattrs_py.data();
    //for (size_t axis = 0; axis < NDIM; axis++) printf("boxsize %.4f %.4f\n", mattrs.boxsize[axis], mattrs.boxcenter[axis]);

    // prepare host-side particle descriptors (point into numpy buffers)
    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();

    // number of GPUs available
    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    // number of counts and total size per-count
    char names[MAX_NWEIGHT][SIZE_NAME];
    size_t ncounts = get_count2_size(p1_host.index_value, p2_host.index_value, names);
    size_t csize = ncounts * battrs.size * spattrs.size;

    // Host output (accumulated across GPUs)
    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data();
    // initialize to zero
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    // Partition particles1 across GPUs (split by particle index)
    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    // Prepare container for per-GPU results
    std::vector<std::vector<FLOAT>> dev_results(ngpus);

    // Launch one std::thread per GPU so each thread can set its own current device
    std::vector<std::thread> workers;
    workers.reserve(ngpus);

    for (int dev = 0; dev < ngpus; ++dev) {
        const size_t start = starts[dev];
        const size_t end = ends[dev];
        const size_t nchunk = (end > start) ? (end - start) : 0;
        // if no particles in this chunk, skip launching heavy work (leave zero contribution)
        if (nchunk == 0) continue;

        workers.emplace_back([dev, start, nchunk, &p1_host, &p2_host, &battrs, &mattrs, &sattrs, &wattrs, &spattrs, ncounts, csize, &dev_results]() {
            // set device for this thread
            CUDA_CHECK(cudaSetDevice(dev));
            // create CUDA stream
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // prepare per-thread DeviceMemoryBuffer (nullptr means use internal allocator)
            DeviceMemoryBuffer *membuffer = NULL;

            // create host-side Particles describing the chunk (point into original host arrays)
            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            // positions are stored as contiguous rows of NDIM floats
            chunk_p1.positions = p1_host.positions + (start * NDIM);
            if (p1_host.values != nullptr) {
                // values are row-major with width = index_value.size
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            // copy chunk and full second catalogue to device
            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_particles_dev[i].size = 0;
            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 2);
            copy_particles_to_device(p2_host, &list_particles_dev[1], 2);

            // build meshes on this device for the two catalogs
            Mesh list_mesh_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_dev[i].total_nparticles = 0;
            set_mesh(list_particles_dev, list_mesh_dev, mattrs, membuffer, stream);
            // free host->device particle structures (device-side meshes remain)
            for (size_t i = 0; i < 2; ++i) free_device_particles(&(list_particles_dev[i]));

            // allocate device histogram for this GPU
            FLOAT *device_counts = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);
            CUDA_CHECK(cudaMemsetAsync(device_counts, 0, csize * sizeof(FLOAT), stream));

            // run count2 on this device (asynchronous on stream)
            count2(device_counts, list_mesh_dev, mattrs, sattrs, battrs, wattrs, spattrs, membuffer, stream);

            // synchronize device stream to ensure kernel completion before copyback
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // copy device counts back to host
            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));
            CUDA_CHECK(cudaMemcpy(dev_results[dev].data(), device_counts, csize * sizeof(FLOAT), cudaMemcpyDeviceToHost));

            // free device memory and device meshes
            my_device_free(device_counts, membuffer);
            for (size_t i = 0; i < 2; ++i) free_device_mesh(&(list_mesh_dev[i]));

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    // wait for all workers to finish
    for (auto &t : workers) t.join();

    // accumulate per-GPU results into final counts_py
    for (int dev=0; dev<ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i=0; i<csize; ++i) counts_ptr[i] += dev_results[dev][i];
    }

    // Return named arrays reshaped to bin shape
    py::dict result;
    std::vector<ssize_t> total_shape;
    if (spattrs.nsplits) total_shape.push_back(static_cast<ssize_t>(spattrs.size));
    auto bshape = battrs_py.shape();
    total_shape.insert(total_shape.end(), bshape.begin(), bshape.end());
    // Return appropriate result based on spin parameters
    for (size_t icount=0; icount<ncounts; icount++) {
        py::array_t<FLOAT> array_py({(ssize_t)battrs.size * spattrs.size}, {(ssize_t)sizeof(FLOAT)}, counts_ptr + icount * battrs.size * spattrs.size, counts_py);
        result[names[icount]] = array_py.attr("reshape")(total_shape).cast<py::array_t<FLOAT>>();
    }
    return result;
}


py::object count3close_py(
    Particles_py& particles1,
    Particles_py& particles2,
    Particles_py& particles3,
    MeshAttrs_py mattrs_py,
    BinAttrs_py battrs_ab_py,
    BinAttrs_py battrs_ac_py,
    py::object battrs_bc_obj = py::none(),
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_ab_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs_ac_py = SelectionAttrs_py(),
    const int nthreads = 1)
{
    MeshAttrs mattrs = mattrs_py.data();
    BinAttrs battrs_ab = battrs_ab_py.data();
    BinAttrs battrs_ac = battrs_ac_py.data();

    const bool has_bc = !battrs_bc_obj.is_none();
    std::unique_ptr<BinAttrs_py> battrs_bc_py;
    BinAttrs battrs_bc{};
    if (has_bc) {
        battrs_bc_py = std::make_unique<BinAttrs_py>(py::cast<BinAttrs_py>(battrs_bc_obj));
        battrs_bc = battrs_bc_py->data();
    }

    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs_ab = sattrs_ab_py.data();
    SelectionAttrs sattrs_ac = sattrs_ac_py.data();

    // prepare host-side particle descriptors (point into numpy buffers)
    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();
    Particles p3_host = particles3.data();

    // number of GPUs available
    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    // number of counts and total size per-count
    char names[1][SIZE_NAME] = {"weight"};
    const size_t ncounts = 1;

    size_t bsize = battrs_ab.size * battrs_ac.size;
    if (has_bc) bsize *= battrs_bc.size;
    size_t csize = ncounts * bsize;

    // Host output (accumulated across GPUs)
    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data();
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    // Partition particles1 across GPUs (split by primary particle index)
    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    // Prepare container for per-GPU results
    std::vector<std::vector<FLOAT>> dev_results(ngpus);

    // Launch one std::thread per GPU so each thread can set its own current device
    std::vector<std::thread> workers;
    workers.reserve(ngpus);

    for (int dev = 0; dev < ngpus; ++dev) {
        const size_t start = starts[dev];
        const size_t end = ends[dev];
        const size_t nchunk = (end > start) ? (end - start) : 0;
        if (nchunk == 0) continue;

        workers.emplace_back(
            [dev, start, nchunk,
             &p1_host, &p2_host, &p3_host,
             &mattrs, &sattrs_ab, &sattrs_ac, &wattrs,
             &battrs_ab, &battrs_ac, &battrs_bc,
             has_bc, bsize, csize, &dev_results]()
        {
            CUDA_CHECK(cudaSetDevice(dev));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            DeviceMemoryBuffer *membuffer = NULL;

            // create host-side Particles describing the chunk
            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            chunk_p1.positions = p1_host.positions + (start * NDIM);
            if (p1_host.values != nullptr) {
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            // copy chunk and full second/third catalogues to device
            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_particles_dev[i].size = 0;
            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 3);
            copy_particles_to_device(p2_host,   &list_particles_dev[1], 3);
            copy_particles_to_device(p3_host,   &list_particles_dev[2], 3);

            // build meshes on this device
            Mesh list_mesh_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_dev[i].total_nparticles = 0;
            set_mesh(list_particles_dev, list_mesh_dev, mattrs, membuffer, stream);

            // free host->device particle structures (device-side meshes remain)
            for (size_t i = 0; i < 3; ++i) free_device_particles(&(list_particles_dev[i]));

            // Build close pairs A-B
            Mesh list_mesh_ab[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_ab[i].total_nparticles = 0;
            list_mesh_ab[0] = list_mesh_dev[0];
            list_mesh_ab[1] = list_mesh_dev[1];

            ClosePairs close_ab = {0};
            fill_close_pairs(&close_ab, list_mesh_ab, mattrs, sattrs_ab, membuffer, stream);

            // Build close pairs A-C
            Mesh list_mesh_ac[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_ac[i].total_nparticles = 0;
            list_mesh_ac[0] = list_mesh_dev[0];
            list_mesh_ac[1] = list_mesh_dev[2];

            ClosePairs close_ac = {0};
            fill_close_pairs(&close_ac, list_mesh_ac, mattrs, sattrs_ac, membuffer, stream);

            // allocate device histogram for this GPU
            FLOAT *device_counts = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);
            CUDA_CHECK(cudaMemsetAsync(device_counts, 0, csize * sizeof(FLOAT), stream));

            BinAttrs *battrs_local[3] = {
                &battrs_ab,
                &battrs_ac,
                has_bc ? &battrs_bc : nullptr
            };

            // run count3_close on this device
            count3_close(
                device_counts,
                close_ab,
                close_ac,
                list_mesh_dev,
                mattrs,
                battrs_local,
                wattrs,
                membuffer,
                stream);

            // synchronize device stream to ensure completion before copyback
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // copy device counts back to host
            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));
            CUDA_CHECK(cudaMemcpy(
                dev_results[dev].data(),
                device_counts,
                csize * sizeof(FLOAT),
                cudaMemcpyDeviceToHost));

            // free device memory
            my_device_free(device_counts, membuffer);
            free_device_close_pairs(&close_ab, membuffer);
            free_device_close_pairs(&close_ac, membuffer);

            for (size_t i = 0; i < 3; ++i) free_device_mesh(&(list_mesh_dev[i]));

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    // wait for all workers to finish
    for (auto &t : workers) t.join();

    // accumulate per-GPU results into final counts_py
    for (int dev = 0; dev < ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i = 0; i < csize; ++i) counts_ptr[i] += dev_results[dev][i];
    }

    // Return named arrays reshaped to bin shape
    py::dict result;
    std::vector<ssize_t> total_shape;

    auto shape_ab = battrs_ab_py.shape();
    auto shape_ac = battrs_ac_py.shape();
    total_shape.insert(total_shape.end(), shape_ab.begin(), shape_ab.end());
    total_shape.insert(total_shape.end(), shape_ac.begin(), shape_ac.end());

    if (has_bc) {
        auto shape_bc = battrs_bc_py->shape();
        total_shape.insert(total_shape.end(), shape_bc.begin(), shape_bc.end());
    }

    for (size_t icount = 0; icount < ncounts; ++icount) {
        py::array_t<FLOAT> array_py(
            {(ssize_t)bsize},
            {(ssize_t)sizeof(FLOAT)},
            counts_ptr + icount * bsize,
            counts_py);
        result[names[icount]] =
            array_py.attr("reshape")(total_shape).cast<py::array_t<FLOAT>>();
    }

    return result;
}



// Bind the function and structs to Python
PYBIND11_MODULE(cucount, m) {
    py::class_<Particles_py>(m, "Particles", py::module_local())
    .def(py::init<py::array_t<FLOAT>, py::array_t<FLOAT>, int, int, int, int, int>(),
         py::arg("positions"),
         py::arg("values") = py::none(),
         py::arg("size_split") = 0,
         py::arg("size_spin") = 0,
         py::arg("size_individual_weight") = 0,
         py::arg("size_bitwise_weight") = 0,
         py::arg("size_negative_weight") = 0)
    .def_property_readonly("size", &Particles_py::size)
    .def_readonly("positions", &Particles_py::positions)
    .def_readonly("values", &Particles_py::values)
    .def_readonly("index_value", &Particles_py::index_value);

    py::class_<BinAttrs_py>(m, "BinAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("shape", &BinAttrs_py::shape)
        .def_property_readonly("size", &BinAttrs_py::size)
        .def_property_readonly("ndim", &BinAttrs_py::ndim)
        .def_property_readonly("varnames", &BinAttrs_py::varnames, "Return list of variable names in order (e.g. ['s','mu'])")
        .def_property_readonly("losnames", &BinAttrs_py::losnames, "Return list of line-of-sight names in order")
        .def_readonly("var", &BinAttrs_py::var)
        .def_readonly("min", &BinAttrs_py::min)
        .def_readonly("max", &BinAttrs_py::max)
        .def_readonly("step", &BinAttrs_py::step) // The lambda approach is safer to avoid exposing internal mutable containers
        .def_property_readonly("array", [](const BinAttrs_py &b) -> std::vector<py::array_t<FLOAT>> {return b.array;});

    py::class_<SelectionAttrs_py>(m, "SelectionAttrs", py::module_local())
        .def(py::init<py::kwargs>()) // Accept Python kwargs
        .def_property_readonly("ndim", &SelectionAttrs_py::ndim)
        .def_property_readonly("varnames", &SelectionAttrs_py::varnames, "Return list of variable names in order (e.g. ['theta'])")
        .def_readonly("var", &SelectionAttrs_py::var)
        .def_readonly("min", &SelectionAttrs_py::min)
        .def_readonly("max", &SelectionAttrs_py::max);

    py::class_<WeightAttrs_py>(m, "WeightAttrs", py::module_local())
        .def(py::init<py::kwargs>()); // Accept Python kwargs

    py::class_<MeshAttrs_py>(m, "MeshAttrs", py::module_local())
        .def(py::init<py::kwargs>());

    py::class_<SplitAttrs_py>(m, "SplitAttrs", py::module_local())
        .def(py::init<py::kwargs>())
        .def_readonly("nsplits", &SplitAttrs_py::nsplits)
        .def_readonly("size", &SplitAttrs_py::size);

    m.def("setup_logging", &setup_logging, "Set the global logging level (debug, info, warn, error)");

    m.def("count2", &count2_py, "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(), // Default value
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py(),
        py::arg("nthreads") = 1); // Default value

    m.def("count3close", &count3close_py,
        "Take three particle catalogs, build close-pair lists for (A,B) and (A,C), "
        "run 3-point close counts on the GPU and return numpy arrays",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("particles3"),
        py::arg("mattrs"),
        py::arg("battrs_ab"),
        py::arg("battrs_ac"),
        py::arg("battrs_bc") = py::none(),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs_ab") = SelectionAttrs_py(),
        py::arg("sattrs_ac") = SelectionAttrs_py(),
        py::arg("nthreads") = 1);
}