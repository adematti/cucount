#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // for std::vector conversion
#include <algorithm>
#include <cstring>  // for std::memcpy
#include <array>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <memory>
#include <stdexcept>

#include "mesh.h"
#include "count2.h"
#include "count2_soft.h"
#include "count3close.h"
#include "count3.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;


// The backstop the Python shim's nicer-worded checks sit in front of. It
// exists because cucountlib.cuda is an importable module: a caller reaching
// it directly bypasses every frontend check, and the two limits below are not
// preferences but bounds on fixed-size arrays. Mirrors validate() in
// cpu/src/bind.cpp, which carries the same two plus the CPU-only ones.
static void validate_count2(const BinAttrs& battrs, const WeightAttrs& wattrs,
                            const IndexValue& iv1, const IndexValue& iv2) {
    // The ell list rides a buffer of MAX_POLE + 2, and legendre_cache in
    // count2.cu is MAX_POLE + 1 entries indexed BY ell -- so the largest ell
    // is the bound, not how many were asked for. A single ell = 10 passes a
    // count-based check and writes off the end of the cache, in device code,
    // where there is nothing to catch it.
    for (size_t i = 0; i < battrs.ndim; i++) {
        if (battrs.var[i] != VAR_POLE) continue;
        if (battrs.shape[i] > MAX_POLE + 2)
            throw std::invalid_argument("cuda backend: too many multipoles requested");
        for (size_t j = 0; j < battrs.shape[i]; j++) {
            if (battrs.array[i][j] > MAX_POLE)
                throw std::invalid_argument(
                    "cuda backend: multipole ell above MAX_POLE requested");
        }
    }
    // count2 looks the angular upweight up against the one angle a pair has
    // (lookup_angular_weight<1> in count2.cu), so a table of any other
    // dimensionality is a malformed request rather than an unserved one: the
    // extra axes would be read past, or silently ignored.
    if (wattrs.angular.size && wattrs.angular.ndim != 1)
        throw std::invalid_argument(
            "cuda backend: only 1D angular weights are implemented");
    // compute_spin_projection_cartesian, shared by both backends, reads two
    // components from &values[start_spin]. One is not a smaller request but a
    // misread: the second component comes from whatever the packed row holds
    // next, or from past its end when spin is the last field.
    for (const IndexValue* iv : {&iv1, &iv2}) {
        if (iv->size_spin && iv->size_spin != 2)
            throw std::invalid_argument("cuda backend: spin needs exactly 2 components");
    }
}


py::object count2_py(Particles_py& particles1, Particles_py& particles2,
                     MeshAttrs_py mattrs_py, BinAttrs_py battrs_py, WeightAttrs_py wattrs_py = WeightAttrs_py(),
                     const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
                     const SplitAttrs_py spattrs_py = SplitAttrs_py(),
                     const int nthreads = 1,
                     const bool return_timings = false) {

    BinAttrs battrs = battrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    MeshAttrs mattrs = mattrs_py.data();
    SplitAttrs spattrs = spattrs_py.data();

    // prepare host-side particle descriptors (point into numpy buffers)
    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();

    // Before the first CUDA call, so a malformed request is refused the same
    // way on a machine with no device as on one with eight.
    validate_count2(battrs, wattrs, p1_host.index_value, p2_host.index_value);

    // number of GPUs available
    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    // output layout
    Count2Layout layout = get_count2_layout(p1_host.index_value, p2_host.index_value, battrs, spattrs);
    size_t csize = layout.nweights * layout.size;

    // Host output (accumulated across GPUs)
    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data();
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    // Partition particles1 across GPUs (split by particle index)
    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    // Prepare container for per-GPU results, and for the mesh/count split
    // each reports. The devices run concurrently, so the wall clock the
    // caller waited is the slowest of them, not the sum.
    std::vector<std::vector<FLOAT>> dev_results(ngpus);
    std::vector<std::array<double, 2>> dev_timings(ngpus, {0., 0.});

    // Launch one std::thread per GPU so each thread can set its own current device
    std::vector<std::thread> workers;
    workers.reserve(ngpus);

    for (int dev = 0; dev < ngpus; ++dev) {
        const size_t start = starts[dev];
        const size_t end = ends[dev];
        const size_t nchunk = (end > start) ? (end - start) : 0;
        if (nchunk == 0) continue;

        workers.emplace_back([dev, start, nchunk, &p1_host, &p2_host, &battrs, &mattrs, &sattrs, &wattrs, &spattrs, csize, &dev_results, &dev_timings]() {
            CUDA_CHECK(cudaSetDevice(dev));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // nullptr means use internal allocator
            DeviceMemoryBuffer *membuffer = NULL;

            // create host-side Particles describing the chunk
            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            chunk_p1.positions = p1_host.positions + (start * NDIM);
            if (p1_host.values != nullptr) {
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            // copy chunk and full second catalogue to device
            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_particles_dev[i].size = 0;
            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 2);
            copy_particles_to_device(p2_host, &list_particles_dev[1], 2);

            // build meshes. set_mesh and count2 each synchronize internally,
            // so a host clock around them needs no extra barrier of its own.
            using Clock = std::chrono::steady_clock;
            using Sec = std::chrono::duration<double>;
            const auto t_mesh0 = Clock::now();

            Mesh list_mesh_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_dev[i].total_nparticles = 0;
            set_mesh(list_particles_dev, list_mesh_dev, mattrs, membuffer, stream);

            const auto t_mesh1 = Clock::now();

            // free device particle buffers
            for (size_t i = 0; i < 2; ++i) free_device_particles(&(list_particles_dev[i]));

            // allocate device histogram
            FLOAT *device_counts = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);
            CUDA_CHECK(cudaMemsetAsync(device_counts, 0, csize * sizeof(FLOAT), stream));

            // run count2
            const Count2Attrs attrs{mattrs, battrs, wattrs, sattrs, spattrs};
            count2(device_counts, list_mesh_dev, attrs, membuffer, stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            dev_timings[dev] = {Sec(t_mesh1 - t_mesh0).count(),
                                Sec(Clock::now() - t_mesh1).count()};

            // copy back
            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));
            CUDA_CHECK(cudaMemcpy(dev_results[dev].data(), device_counts, csize * sizeof(FLOAT), cudaMemcpyDeviceToHost));

            // cleanup
            my_device_free(device_counts, membuffer);
            for (size_t i = 0; i < 2; ++i) free_device_mesh(&(list_mesh_dev[i]));

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    // wait for all workers
    for (auto &t : workers) t.join();

    // Concurrent devices, so the wall clock the caller waited is the slowest,
    // not the sum. Same two numbers, in the same order, as the CPU binding.
    double mesh_seconds = 0., count_seconds = 0.;
    for (int dev = 0; dev < ngpus; ++dev) {
        mesh_seconds = MAX(mesh_seconds, dev_timings[dev][0]);
        count_seconds = MAX(count_seconds, dev_timings[dev][1]);
    }

    // accumulate per-GPU results
    for (int dev = 0; dev < ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i = 0; i < csize; ++i) counts_ptr[i] += dev_results[dev][i];
    }

    // Return named arrays reshaped to output shape
    py::dict result;
    for (size_t iweight = 0; iweight < layout.nweights; ++iweight) {
        py::array_t<FLOAT> array_py(
            {(ssize_t) layout.size},
            {(ssize_t) sizeof(FLOAT)},
            counts_ptr + iweight * layout.size,
            counts_py
        );
        result[layout.names[iweight].c_str()] =
            array_py.attr("reshape")(layout.shape).cast<py::array_t<FLOAT>>();
    }

    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(mesh_seconds, count_seconds));
    }
    return result;
}


py::object count3close_py(
    Particles_py& particles1,
    Particles_py& particles2,
    Particles_py& particles3,
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    py::object battrs23_obj = py::none(),
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs23_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto23_py = SelectionAttrs_py(),
    py::tuple close_pair = py::make_tuple(1, 2),
    const int nthreads = 1,
    const bool return_timings = false)
{
    const CLOSE_PAIR close_pair_value = parse_close_pair(close_pair);

    MeshAttrs mattrs1 = mattrs1_py.data();
    MeshAttrs mattrs2 = mattrs2_py.data();
    MeshAttrs mattrs3 = mattrs3_py.data();

    BinAttrs battrs12 = battrs12_py.data();
    BinAttrs battrs13 = battrs13_py.data();

    const bool has23 = !battrs23_obj.is_none();
    std::unique_ptr<BinAttrs_py> battrs23_py;
    BinAttrs battrs23{};
    if (has23) {
        battrs23_py = std::make_unique<BinAttrs_py>(
            py::cast<BinAttrs_py>(battrs23_obj));
        battrs23 = battrs23_py->data();
    }

    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs12 = sattrs12_py.data();
    SelectionAttrs sattrs13 = sattrs13_py.data();
    SelectionAttrs sattrs23 = sattrs23_py.data();

    SelectionAttrs veto12 = veto12_py.data();
    SelectionAttrs veto13 = veto13_py.data();
    SelectionAttrs veto23 = veto23_py.data();

    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();
    Particles p3_host = particles3.data();

    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    Count3Layout layout = get_count3_layout(
        battrs12,
        battrs13,
        battrs23);

    size_t csize = layout.nweights * layout.size;

    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data();
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    std::vector<std::vector<FLOAT>> dev_results(ngpus);
    std::vector<std::array<double, 2>> dev_timings(ngpus, {0., 0.});
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
             &mattrs1, &mattrs2, &mattrs3,
             &wattrs, &sattrs12, &sattrs13, &sattrs23,
             &veto12, &veto13, &veto23,
             &battrs12, &battrs13, &battrs23,
             has23, close_pair_value,
             csize, &dev_results, &dev_timings]()
        {
            CUDA_CHECK(cudaSetDevice(dev));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            DeviceMemoryBuffer *membuffer = NULL;

            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            chunk_p1.positions = p1_host.positions + (start * NDIM);

            if (p1_host.spositions != nullptr) {
                chunk_p1.spositions = p1_host.spositions + (start * NDIM);
            }

            if (p1_host.values != nullptr) {
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) {
                list_particles_dev[i].size = 0;
            }

            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 3);
            copy_particles_to_device(p2_host,   &list_particles_dev[1], 3);
            copy_particles_to_device(p3_host,   &list_particles_dev[2], 3);

            // set_mesh and the count each synchronize internally, so a host
            // clock around them needs no extra barrier of its own.
            using Clock = std::chrono::steady_clock;
            using Sec = std::chrono::duration<double>;
            const auto t_mesh0 = Clock::now();

            Mesh mesh1;
            Mesh mesh2;
            Mesh mesh3;

            Particles plist[MAX_NMESH];
            Mesh mlist[MAX_NMESH];

            for (size_t i = 0; i < MAX_NMESH; ++i) {
                plist[i].size = 0;
                mlist[i].total_nparticles = 0;
            }

            plist[0] = list_particles_dev[0];
            set_mesh(plist, mlist, mattrs1, membuffer, stream);
            mesh1 = mlist[0];

            plist[0] = list_particles_dev[1];
            set_mesh(plist, mlist, mattrs2, membuffer, stream);
            mesh2 = mlist[0];

            plist[0] = list_particles_dev[2];
            set_mesh(plist, mlist, mattrs3, membuffer, stream);

            const auto t_mesh1 = Clock::now();
            mesh3 = mlist[0];

            for (size_t i = 0; i < 3; ++i) {
                free_device_particles(&(list_particles_dev[i]));
            }

            FLOAT *device_counts = (FLOAT*) my_device_malloc(
                csize * sizeof(FLOAT),
                membuffer);

            CUDA_CHECK(cudaMemsetAsync(
                device_counts,
                0,
                csize * sizeof(FLOAT),
                stream));

            const Count3Attrs attrs{
                mattrs1, mattrs2, mattrs3,
                battrs12, battrs13, has23 ? battrs23 : BinAttrs{},
                wattrs,
                sattrs12, sattrs13, sattrs23,
                veto12, veto13, veto23};

            count3close(
                device_counts,
                mesh1,
                mesh2,
                mesh3,
                attrs,
                close_pair_value,
                membuffer,
                stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            dev_timings[dev] = {Sec(t_mesh1 - t_mesh0).count(),
                                Sec(Clock::now() - t_mesh1).count()};

            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));

            CUDA_CHECK(cudaMemcpy(
                dev_results[dev].data(),
                device_counts,
                csize * sizeof(FLOAT),
                cudaMemcpyDeviceToHost));

            my_device_free(device_counts, membuffer);

            free_device_mesh(&mesh1);
            free_device_mesh(&mesh2);
            free_device_mesh(&mesh3);

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    for (auto &t : workers) {
        t.join();
    }

    // Concurrent devices, so the wall clock the caller waited is the slowest,
    // not the sum. Same two numbers, in the same order, as the CPU binding.
    double mesh_seconds = 0., count_seconds = 0.;
    for (int dev = 0; dev < ngpus; ++dev) {
        mesh_seconds = MAX(mesh_seconds, dev_timings[dev][0]);
        count_seconds = MAX(count_seconds, dev_timings[dev][1]);
    }

    for (int dev = 0; dev < ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i = 0; i < csize; ++i) {
            counts_ptr[i] += dev_results[dev][i];
        }
    }

    py::dict result;
    for (size_t iweight = 0; iweight < layout.nweights; ++iweight) {
        py::array_t<FLOAT> array_py(
            {(ssize_t) layout.size},
            {(ssize_t) sizeof(FLOAT)},
            counts_ptr + iweight * layout.size,
            counts_py);

        result[layout.names[iweight].c_str()] =
            array_py.attr("reshape")(layout.shape).cast<py::array_t<FLOAT>>();
    }

    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(mesh_seconds, count_seconds));
    }
    return result;
}


py::object count3_py(
    Particles_py& particles1,
    Particles_py& particles2,
    Particles_py& particles3,
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py(),
    const int nthreads = 1,
    const bool return_timings = false)
{
    MeshAttrs mattrs1 = mattrs1_py.data();
    MeshAttrs mattrs2 = mattrs2_py.data();
    MeshAttrs mattrs3 = mattrs3_py.data();

    BinAttrs battrs12 = battrs12_py.data();
    BinAttrs battrs13 = battrs13_py.data();

    WeightAttrs wattrs = wattrs_py.data();

    SelectionAttrs sattrs12 = sattrs12_py.data();
    SelectionAttrs sattrs13 = sattrs13_py.data();

    SelectionAttrs veto12 = veto12_py.data();
    SelectionAttrs veto13 = veto13_py.data();

    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();
    Particles p3_host = particles3.data();

    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    BinAttrs battrs23{};
    Count3Layout layout = get_count3_layout(
        battrs12,
        battrs13,
        battrs23);

    size_t csize = layout.nweights * layout.size;

    py::array_t<FLOAT> counts_py(csize);
    auto counts_ptr = counts_py.mutable_data();
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    std::vector<std::vector<FLOAT>> dev_results(ngpus);
    std::vector<std::array<double, 2>> dev_timings(ngpus, {0., 0.});
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
             &mattrs1, &mattrs2, &mattrs3,
             &wattrs, &sattrs12, &sattrs13,
             &veto12, &veto13,
             &battrs12, &battrs13,
             csize, &dev_results, &dev_timings]()
        {
            CUDA_CHECK(cudaSetDevice(dev));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            DeviceMemoryBuffer *membuffer = NULL;

            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            chunk_p1.positions = p1_host.positions + (start * NDIM);

            if (p1_host.spositions != nullptr) {
                chunk_p1.spositions = p1_host.spositions + (start * NDIM);
            }

            if (p1_host.values != nullptr) {
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) {
                list_particles_dev[i].size = 0;
            }

            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 3);
            copy_particles_to_device(p2_host,   &list_particles_dev[1], 3);
            copy_particles_to_device(p3_host,   &list_particles_dev[2], 3);

            // set_mesh and the count each synchronize internally, so a host
            // clock around them needs no extra barrier of its own.
            using Clock = std::chrono::steady_clock;
            using Sec = std::chrono::duration<double>;
            const auto t_mesh0 = Clock::now();

            Mesh mesh1;
            Mesh mesh2;
            Mesh mesh3;

            Particles plist[MAX_NMESH];
            Mesh mlist[MAX_NMESH];

            for (size_t i = 0; i < MAX_NMESH; ++i) {
                plist[i].size = 0;
                mlist[i].total_nparticles = 0;
            }

            plist[0] = list_particles_dev[0];
            set_mesh(plist, mlist, mattrs1, membuffer, stream);
            mesh1 = mlist[0];

            plist[0] = list_particles_dev[1];
            set_mesh(plist, mlist, mattrs2, membuffer, stream);
            mesh2 = mlist[0];

            plist[0] = list_particles_dev[2];
            set_mesh(plist, mlist, mattrs3, membuffer, stream);

            const auto t_mesh1 = Clock::now();
            mesh3 = mlist[0];

            for (size_t i = 0; i < 3; ++i) {
                free_device_particles(&(list_particles_dev[i]));
            }

            FLOAT *device_counts = (FLOAT*) my_device_malloc(
                csize * sizeof(FLOAT),
                membuffer);

            CUDA_CHECK(cudaMemsetAsync(
                device_counts,
                0,
                csize * sizeof(FLOAT),
                stream));

            // count3 has no (2, 3) axis, so those members stay default.
            const Count3Attrs attrs{
                MeshAttrs{}, mattrs2, mattrs3,
                battrs12, battrs13, BinAttrs{},
                wattrs,
                sattrs12, sattrs13, SelectionAttrs{},
                veto12, veto13, SelectionAttrs{}};

            count3(
                device_counts,
                mesh1,
                mesh2,
                mesh3,
                attrs,
                membuffer,
                stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            dev_timings[dev] = {Sec(t_mesh1 - t_mesh0).count(),
                                Sec(Clock::now() - t_mesh1).count()};

            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));

            CUDA_CHECK(cudaMemcpy(
                dev_results[dev].data(),
                device_counts,
                csize * sizeof(FLOAT),
                cudaMemcpyDeviceToHost));

            my_device_free(device_counts, membuffer);

            free_device_mesh(&mesh1);
            free_device_mesh(&mesh2);
            free_device_mesh(&mesh3);

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    for (auto &t : workers) {
        t.join();
    }

    // Concurrent devices, so the wall clock the caller waited is the slowest,
    // not the sum. Same two numbers, in the same order, as the CPU binding.
    double mesh_seconds = 0., count_seconds = 0.;
    for (int dev = 0; dev < ngpus; ++dev) {
        mesh_seconds = MAX(mesh_seconds, dev_timings[dev][0]);
        count_seconds = MAX(count_seconds, dev_timings[dev][1]);
    }

    for (int dev = 0; dev < ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i = 0; i < csize; ++i) {
            counts_ptr[i] += dev_results[dev][i];
        }
    }

    py::dict result;
    for (size_t iweight = 0; iweight < layout.nweights; ++iweight) {
        py::array_t<FLOAT> array_py(
            {(ssize_t) layout.size},
            {(ssize_t) sizeof(FLOAT)},
            counts_ptr + iweight * layout.size,
            counts_py);

        result[layout.names[iweight].c_str()] =
            array_py.attr("reshape")(layout.shape).cast<py::array_t<FLOAT>>();
    }

    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(mesh_seconds, count_seconds));
    }
    return result;
}


// GPUs this process can use, honouring CUDA_VISIBLE_DEVICES. No device or no
// usable driver is an answer, 0, not an error: this is how a frontend decides
// whether to run on the GPU at all, including on nodes that have none.
int device_count_py() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error state
        return 0;
    }
    return count;
}


// ---------------------------------------------------------------------------
// Soft binning: resolve the request into the kernel's per-pair parameters.
// ---------------------------------------------------------------------------
//
// Everything the v1 soft kernel does not serve is refused HERE, by name, in a
// Python exception -- before a device buffer exists or a kernel is launched. A
// kernel that quietly binned a request it cannot honour (hard-binning the mu
// axis of a soft-s request, say) would produce plausible numbers, which is the
// worst failure mode a counting code has.
// What the soft kernel actually did, returned under result["soft"] by both
// entries. One helper so the forward and the adjoint cannot drift apart.
static py::dict soft_meta(const SoftBinSpec &spec)
{
    py::dict soft;

    // A width-scaled sigma (non-arithmetic grid, sigma <= 0) is reported as the
    // multiplier of the local bin width, which is what actually ran; the
    // absolute branch covers an arithmetic grid or an explicit sigma.
    soft["sigma"] = spec.per_bin_sigma ? spec.alpha
                    : (spec.inv2sig2 > 0 ? 1. / std::sqrt(2. * spec.inv2sig2) : 0.);
    soft["sigma_units"] = "separation";
    soft["sigma_kind"] = spec.per_bin_sigma ? "width" : "absolute";
    soft["uniform"] = spec.uniform != 0;
    soft["fast_taps"] = spec.fast_taps != 0;
    soft["cutoff"] = spec.cutoff;
    soft["kmax"] = spec.kmax;
    // Only the uniform path has a fixed window length; the general path expands
    // until the cutoff stops it, so its tap count depends on the pair.
    soft["taps"] = spec.uniform ? 2 * spec.kmax + 1 : -1;
    soft["path"] = spec.path > 0 ? "shared" : "global";
    soft["copies"] = spec.path;  // private histogram copies per warp
    soft["nbins"] = spec.nbins;

    // Multipoles: the ells that were summed over, and the mu quadrature that
    // carried them. Empty/zero for a plain soft-s request.
    py::list ells;

    for (int i = 0; i < spec.nell; ++i) {
        ells.append(spec.ells[i]);
    }

    soft["multipoles"] = ells;
    soft["mu_sigma"] = spec.nell > 0 ? (double) spec.mu_sigma : 0.;
    soft["mu_bins"] = spec.nmu;
    soft["mu_kmax"] = spec.nell > 0 ? spec.mu_kmax : 0;
    soft["los"] = spec.nell > 0 ? los_type_to_string(spec.los) : std::string("");
    // Where the mu quadrature came from: an arithmetic grid (the default or a
    // caller-supplied uniform one) takes the recurrence path, anything else the
    // per-tap window; the edges are echoed so a caller can check what ran.
    soft["mu_uniform"] = spec.mu_uniform != 0;
    soft["mu_sigma_kind"] = spec.mu_per_bin_sigma ? "width" : "absolute";
    soft["mu_taps"] = spec.mu_uniform ? 2 * spec.mu_kmax + 1 : -1;

    if (spec.nell > 0 && !spec.mu_uniform && spec.host_mu_edges) {
        py::list mu_edges;

        for (int i = 0; i <= spec.nmu; ++i) {
            mu_edges.append((double) spec.host_mu_edges[i]);
        }

        soft["mu_edges"] = mu_edges;
    }

    return soft;
}


// Host copy of the optional caller-supplied mu edges. The spec points into this
// vector, so it must outlive the kernel launch; it does, because both entries
// build the spec and run the kernel in the same scope.
static std::vector<FLOAT> soft_mu_edges_from_py(const py::object &obj)
{
    std::vector<FLOAT> edges;

    if (obj.is_none()) {
        return edges;
    }

    py::array_t<FLOAT, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<FLOAT, py::array::c_style | py::array::forcecast>>(obj);

    const auto *data = arr.data();
    edges.assign(data, data + arr.size());
    return edges;
}


py::object count2_soft_py(
    Particles_py& particles1, Particles_py& particles2,
    MeshAttrs_py mattrs_py, BinAttrs_py battrs_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
    const SplitAttrs_py spattrs_py = SplitAttrs_py(),
    py::object soft_sigma_obj = py::none(),
    const double soft_cutoff = 1e-13,
    const std::string &soft_taps = "exact",
    py::object soft_mu_sigma_obj = py::none(),
    const int soft_mu_bins = 0,
    py::object soft_mu_edges_obj = py::none(),
    const int nthreads = 1,
    const bool return_timings = false) {

    BinAttrs battrs = battrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    MeshAttrs mattrs = mattrs_py.data();
    SplitAttrs spattrs = spattrs_py.data();

    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();

    // None means "the default width", half a bin, resolved in make_soft_bin_spec
    // against the binning rather than here.
    double soft_sigma = 0.;

    if (!soft_sigma_obj.is_none()) {
        soft_sigma = py::cast<double>(soft_sigma_obj);
    }

    double soft_mu_sigma = 0.;

    if (!soft_mu_sigma_obj.is_none()) {
        soft_mu_sigma = py::cast<double>(soft_mu_sigma_obj);
    }

    // Caller-supplied mu edges, if any; the spec keeps a pointer into this
    // vector, which lives until the launches below have been enqueued.
    std::vector<FLOAT> mu_edges = soft_mu_edges_from_py(soft_mu_edges_obj);

    size_t csize = 0;
    SoftBinSpec spec = make_soft_bin_spec(
        battrs, wattrs, spattrs, soft_sigma, soft_cutoff, &csize,
        soft_mu_sigma, soft_mu_bins,
        mu_edges.empty() ? nullptr : mu_edges.data(), (int) mu_edges.size());

    // The general (non-uniform) path pays one exp per tap per pass; 'fast'
    // swaps them for __expf (~1e-6 relative) when the caller prefers speed.
    if (soft_taps == "fast") {
        spec.fast_taps = 1;
    }
    else if (soft_taps != "exact") {
        throw std::invalid_argument(
            "cuda backend, soft binning: soft_taps must be 'exact' or 'fast'");
    }

    int ngpus = 1;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    ngpus = MIN(nthreads, ngpus);

    py::array_t<FLOAT> counts_py((ssize_t) csize);
    auto counts_ptr = counts_py.mutable_data();
    std::fill_n(counts_ptr, csize, static_cast<FLOAT>(0.0));

    const size_t n1 = p1_host.size;
    std::vector<size_t> starts(ngpus), ends(ngpus);
    for (int d = 0; d < ngpus; ++d) {
        starts[d] = (d * n1) / ngpus;
        ends[d] = ((d + 1) * n1) / ngpus;
    }

    std::vector<std::vector<FLOAT>> dev_results(ngpus);
    std::vector<std::array<double, 2>> dev_timings(ngpus, {0., 0.});

    std::vector<std::thread> workers;
    workers.reserve(ngpus);

    for (int dev = 0; dev < ngpus; ++dev) {
        const size_t start = starts[dev];
        const size_t end = ends[dev];
        const size_t nchunk = (end > start) ? (end - start) : 0;
        if (nchunk == 0) continue;

        workers.emplace_back([dev, start, nchunk, &p1_host, &p2_host, &battrs, &mattrs, &sattrs, &wattrs, &spattrs, &spec, csize, &dev_results, &dev_timings]() {
            CUDA_CHECK(cudaSetDevice(dev));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            DeviceMemoryBuffer *membuffer = NULL;

            Particles chunk_p1 = p1_host;
            chunk_p1.size = nchunk;
            chunk_p1.positions = p1_host.positions + (start * NDIM);
            if (p1_host.values != nullptr) {
                size_t width = p1_host.index_value.size;
                chunk_p1.values = p1_host.values + (start * width);
            }

            Particles list_particles_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_particles_dev[i].size = 0;
            copy_particles_to_device(chunk_p1, &list_particles_dev[0], 2);
            copy_particles_to_device(p2_host, &list_particles_dev[1], 2);

            using Clock = std::chrono::steady_clock;
            using Sec = std::chrono::duration<double>;
            const auto t_mesh0 = Clock::now();

            Mesh list_mesh_dev[MAX_NMESH];
            for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_dev[i].total_nparticles = 0;
            set_mesh(list_particles_dev, list_mesh_dev, mattrs, membuffer, stream);

            const auto t_mesh1 = Clock::now();

            for (size_t i = 0; i < 2; ++i) free_device_particles(&(list_particles_dev[i]));

            FLOAT *device_counts = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);
            CUDA_CHECK(cudaMemsetAsync(device_counts, 0, csize * sizeof(FLOAT), stream));

            const Count2Attrs attrs{mattrs, battrs, wattrs, sattrs, spattrs};
            count2_soft(device_counts, list_mesh_dev, attrs, spec, membuffer, stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            dev_timings[dev] = {Sec(t_mesh1 - t_mesh0).count(),
                                Sec(Clock::now() - t_mesh1).count()};

            dev_results[dev].assign(csize, static_cast<FLOAT>(0.0));
            CUDA_CHECK(cudaMemcpy(dev_results[dev].data(), device_counts, csize * sizeof(FLOAT), cudaMemcpyDeviceToHost));

            my_device_free(device_counts, membuffer);
            for (size_t i = 0; i < 2; ++i) free_device_mesh(&(list_mesh_dev[i]));

            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }

    for (auto &t : workers) t.join();

    double mesh_seconds = 0., count_seconds = 0.;
    for (int dev = 0; dev < ngpus; ++dev) {
        mesh_seconds = MAX(mesh_seconds, dev_timings[dev][0]);
        count_seconds = MAX(count_seconds, dev_timings[dev][1]);
    }

    for (int dev = 0; dev < ngpus; ++dev) {
        if (dev_results[dev].empty()) continue;
        for (size_t i = 0; i < csize; ++i) counts_ptr[i] += dev_results[dev][i];
    }

    // Same shape as count2's output -- one 'weight' array, here with shape
    // (nbins,) for a plain soft-s request and (nbins, nell) when multipoles were
    // asked for -- plus what the kernel actually did, so a caller can tell the
    // shared-tile path from the fallback and see the width it used when sigma
    // was left at its default.
    py::dict result;

    if (spec.nell > 0) {
        result["weight"] = counts_py.attr("reshape")(
            py::make_tuple((ssize_t) spec.nbins, (ssize_t) spec.nell));
    }
    else {
        result["weight"] = counts_py;
    }

    result["soft"] = soft_meta(spec);

    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(mesh_seconds, count_seconds));
    }
    return result;
}


py::object count2_soft_vjp_py(
    Particles_py& particles1, Particles_py& particles2,
    MeshAttrs_py mattrs_py, BinAttrs_py battrs_py,
    py::array_t<FLOAT, py::array::c_style | py::array::forcecast> cotangent,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
    const SplitAttrs_py spattrs_py = SplitAttrs_py(),
    py::object soft_sigma_obj = py::none(),
    const double soft_cutoff = 1e-13,
    const std::string &soft_taps = "exact",
    py::object soft_mu_sigma_obj = py::none(),
    const int soft_mu_bins = 0,
    py::object soft_mu_edges_obj = py::none(),
    const int nthreads = 1,
    const bool return_timings = false) {

    (void) nthreads;  // v1 adjoint runs on one device; see the note below

    BinAttrs battrs = battrs_py.data();
    WeightAttrs wattrs = wattrs_py.data();
    SelectionAttrs sattrs = sattrs_py.data();
    MeshAttrs mattrs = mattrs_py.data();
    SplitAttrs spattrs = spattrs_py.data();

    Particles p1_host = particles1.data();
    Particles p2_host = particles2.data();

    double soft_sigma = 0.;

    if (!soft_sigma_obj.is_none()) {
        soft_sigma = py::cast<double>(soft_sigma_obj);
    }

    double soft_mu_sigma = 0.;

    if (!soft_mu_sigma_obj.is_none()) {
        soft_mu_sigma = py::cast<double>(soft_mu_sigma_obj);
    }

    std::vector<FLOAT> mu_edges = soft_mu_edges_from_py(soft_mu_edges_obj);

    size_t csize = 0;
    SoftBinSpec spec = make_soft_bin_spec(
        battrs, wattrs, spattrs, soft_sigma, soft_cutoff, &csize,
        soft_mu_sigma, soft_mu_bins,
        mu_edges.empty() ? nullptr : mu_edges.data(), (int) mu_edges.size());

    // The general (non-uniform) path pays one exp per tap per pass; 'fast'
    // swaps them for __expf (~1e-6 relative) when the caller prefers speed.
    if (soft_taps == "fast") {
        spec.fast_taps = 1;
    }
    else if (soft_taps != "exact") {
        throw std::invalid_argument(
            "cuda backend, soft binning: soft_taps must be 'exact' or 'fast'");
    }

    const auto *pairs = cotangent.data();

    if ((size_t) cotangent.size() != csize) {
        throw std::invalid_argument(
            "cuda backend, soft binning: the cotangent must have one entry per "
            "histogram slot (" + std::to_string(csize) + "), got "
            + std::to_string(cotangent.size()));
    }

    // One device: the adjoint accumulates per input particle, so splitting the
    // first catalogue across devices would need a reduction of the second
    // catalogue's gradients that is not worth the plumbing while a single GPU
    // is the target. The forward still splits.
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DeviceMemoryBuffer *membuffer = NULL;

    Particles list_particles_dev[MAX_NMESH];
    for (size_t i = 0; i < MAX_NMESH; ++i) list_particles_dev[i].size = 0;
    copy_particles_to_device(p1_host, &list_particles_dev[0], 2);
    copy_particles_to_device(p2_host, &list_particles_dev[1], 2);

    using Clock = std::chrono::steady_clock;
    using Sec = std::chrono::duration<double>;
    const auto t_mesh0 = Clock::now();

    Mesh list_mesh_dev[MAX_NMESH];
    for (size_t i = 0; i < MAX_NMESH; ++i) list_mesh_dev[i].total_nparticles = 0;
    set_mesh(list_particles_dev, list_mesh_dev, mattrs, membuffer, stream, true);

    const auto t_mesh1 = Clock::now();

    for (size_t i = 0; i < 2; ++i) free_device_particles(&(list_particles_dev[i]));

    const size_t n1 = p1_host.size;
    const size_t n2 = p2_host.size;

    py::array_t<FLOAT> gpos1_py((ssize_t) (NDIM * n1));
    py::array_t<FLOAT> gpos2_py((ssize_t) (NDIM * n2));
    py::array_t<FLOAT> gw1_py((ssize_t) n1);
    py::array_t<FLOAT> gw2_py((ssize_t) n2);

    FLOAT *d_gpos1 = (FLOAT*) my_device_malloc(NDIM * n1 * sizeof(FLOAT), membuffer);
    FLOAT *d_gpos2 = (FLOAT*) my_device_malloc(NDIM * n2 * sizeof(FLOAT), membuffer);
    FLOAT *d_gw1 = (FLOAT*) my_device_malloc(n1 * sizeof(FLOAT), membuffer);
    FLOAT *d_gw2 = (FLOAT*) my_device_malloc(n2 * sizeof(FLOAT), membuffer);
    FLOAT *d_gsigma = (FLOAT*) my_device_malloc(sizeof(FLOAT), membuffer);
    FLOAT *d_cotangent = (FLOAT*) my_device_malloc(csize * sizeof(FLOAT), membuffer);

    CUDA_CHECK(cudaMemsetAsync(d_gpos1, 0, NDIM * n1 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(d_gpos2, 0, NDIM * n2 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(d_gw1, 0, n1 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(d_gw2, 0, n2 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(d_gsigma, 0, sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cotangent, pairs, csize * sizeof(FLOAT),
                               cudaMemcpyHostToDevice, stream));

    const Count2Attrs attrs{mattrs, battrs, wattrs, sattrs, spattrs};
    count2_soft_vjp(d_cotangent, d_gpos1, d_gpos2, d_gw1, d_gw2, d_gsigma,
                    list_mesh_dev, attrs, spec, membuffer, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double count_seconds = Sec(Clock::now() - t_mesh1).count();

    FLOAT grad_sigma = 0;

    CUDA_CHECK(cudaMemcpy(gpos1_py.mutable_data(), d_gpos1, NDIM * n1 * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpos2_py.mutable_data(), d_gpos2, NDIM * n2 * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gw1_py.mutable_data(), d_gw1, n1 * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gw2_py.mutable_data(), d_gw2, n2 * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_sigma, d_gsigma, sizeof(FLOAT), cudaMemcpyDeviceToHost));

    my_device_free(d_gpos1, membuffer);
    my_device_free(d_gpos2, membuffer);
    my_device_free(d_gw1, membuffer);
    my_device_free(d_gw2, membuffer);
    my_device_free(d_gsigma, membuffer);
    my_device_free(d_cotangent, membuffer);
    for (size_t i = 0; i < 2; ++i) free_device_mesh(&(list_mesh_dev[i]));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Shapes mirror the inputs: one gradient row per particle, and one weight
    // gradient per particle only when the catalogue carries individual weights.
    py::dict result;
    result["grad_position1"] = gpos1_py.reshape({(ssize_t) n1, (ssize_t) NDIM});
    result["grad_position2"] = gpos2_py.reshape({(ssize_t) n2, (ssize_t) NDIM});
    // The keys are always present so callers can index them unconditionally;
    // without individual weights there is no weight to differentiate.
    if (p1_host.index_value.size_individual_weight) {
        result["grad_weight1"] = gw1_py;
    }
    else {
        result["grad_weight1"] = py::none();
    }
    if (p2_host.index_value.size_individual_weight) {
        result["grad_weight2"] = gw2_py;
    }
    else {
        result["grad_weight2"] = py::none();
    }
    result["grad_sigma"] = grad_sigma;
    result["soft"] = soft_meta(spec);

    if (return_timings) {
        return py::make_tuple(result, py::make_tuple(Sec(t_mesh1 - t_mesh0).count(), count_seconds));
    }
    return result;
}


// Bind the function and structs to Python
PYBIND11_MODULE(cuda, m) {

    register_attrs(m);

    m.def("device_count", &device_count_py,
          "Number of GPUs this process can use (honours CUDA_VISIBLE_DEVICES); 0 without a device or driver");

    m.def("count2", &count2_py, "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU and return a numpy array",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(), // Default value
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py(),
        py::arg("nthreads") = 1,
        py::arg("return_timings") = false);

    m.def("count2_soft", &count2_soft_py,
        "Take particle positions and weights (numpy arrays), perform 2-pt counts on the GPU with "
        "Gaussian-softmax soft binning on the separation axis, and return a numpy array. "
        "soft_sigma is in separation units; None means half a bin. With a 'pole' axis the "
        "mu axis is softened too and the result is soft multipoles, shape (nbins, nell).",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py(),
        py::arg("soft_sigma") = py::none(),
        py::arg("soft_cutoff") = 1e-13,
        py::arg("soft_taps") = "exact",
        py::arg("soft_mu_sigma") = py::none(),
        py::arg("soft_mu_bins") = 0,
        py::arg("soft_mu_edges") = py::none(),
        py::arg("nthreads") = 1,
        py::arg("return_timings") = false);

    m.def("count2_soft_vjp", &count2_soft_vjp_py,
        "Vector-Jacobian product of count2_soft: given the cotangent of the soft "
        "histogram, return the gradients with respect to the two catalogues' "
        "positions, individual weights and soft_sigma. Gradients are per input "
        "particle (the kernel undoes the mesh's cell reordering). With soft mu "
        "binning the cotangent is the flat (nbins * nell) array and the mu "
        "weights contribute to the position gradients.",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("cotangent"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py(),
        py::arg("soft_sigma") = py::none(),
        py::arg("soft_cutoff") = 1e-13,
        py::arg("soft_taps") = "exact",
        py::arg("soft_mu_sigma") = py::none(),
        py::arg("soft_mu_bins") = 0,
        py::arg("soft_mu_edges") = py::none(),
        py::arg("nthreads") = 1,
        py::arg("return_timings") = false);

    m.def("count3close", &count3close_py,
        "Take three particle catalogs, run 3-point close counts on the GPU and return numpy arrays",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("particles3"),
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("battrs23") = py::none(),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("sattrs23") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py(),
        py::arg("veto23") = SelectionAttrs_py(),
        py::arg("close_pair") = py::make_tuple(1, 2),
        py::arg("nthreads") = 1,
        py::arg("return_timings") = false);

    m.def("count3", &count3_py,
        "Take three particle catalogs, run factorized 3-point counts on the GPU and return numpy arrays",
        py::arg("particles1"),
        py::arg("particles2"),
        py::arg("particles3"),
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py(),
        py::arg("nthreads") = 1,
        py::arg("return_timings") = false);
}