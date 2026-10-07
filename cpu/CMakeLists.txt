cmake_minimum_required(VERSION 3.20)

# Buildable standalone or as a subdirectory of the root
# project behind -DCUCOUNT_BUILD_CPU=ON.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cucount_cpu LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GCC < 9 drops the per-target SIMD options on OpenMP-outlined regions, which
# surfaces as pages of always_inline "target specific option mismatch" errors
# from Highway. Fail with the reason instead (seen with /usr/bin/g++ 7.5 when
# a newer g++ was on PATH but CMake's default compiler probe picked cc/c++).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  message(FATAL_ERROR "cucount_cpu needs GCC >= 9 (found ${CMAKE_CXX_COMPILER_VERSION} at "
          "${CMAKE_CXX_COMPILER}); pass -DCMAKE_CXX_COMPILER=g++ or a newer compiler")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(CUCOUNT_CPU_ALL_TARGETS
       "Compile every attainable SIMD target, not just Highway's default set" ON)
# For a development loop on one machine: compile only the build host's SIMD
# target. The Highway kernel is one translation unit compiled once per target,
# so this is what its build time scales with -- count2.cpp alone on a Milan
# node (gcc 14): 646 s with every attainable target, 575 s with Highway's
# default set, 75 s with the host's only. The result runs on that ISA and
# newer only; never install it.
option(CUCOUNT_CPU_NATIVE_ONLY
       "Compile only the build host's SIMD target (fast builds; not portable)" OFF)

include(FetchContent)
# contrib/math/math-inl.h (Log, for the log-binning policy) is header-only and
# comes in on hwy's interface include path. Building the contrib *library* is
# off because its vqsort sources emit AVX512-FP16 that older binutils cannot
# assemble -- nothing here needs them.
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.4.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(highway)

find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Everything but the bindings, compiled once and linked into both modules
# (cucountlib.cpu and cucountlib.ffi_cpu): the Highway kernel in count2.cpp is
# by far the most expensive translation unit to build, and the two used to
# compile it each. Objects are linked, not shared, so each module still gets
# its own copy of the module-local state below.
# ../src/logger.cpp: register_attrs binds setup_logging, which needs the
# module-local global_log_level.
# src/generic.cpp is the scalar path for everything the Highway kernel does
# not vectorise; it is a plain translation unit, compiled once, so nothing it
# adds can reach the kernel's inner loop.
add_library(cucount_cpu_kernels OBJECT src/count2.cpp src/generic.cpp src/count3.cpp src/count3close.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/logger.cpp)
# ../include for cmath.h, the scalar per-pair math shared with CUDA (the
# other headers there are CUDA-only and are not included from this target).
target_include_directories(cucount_cpu_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cucount_cpu_kernels PUBLIC hwy OpenMP::OpenMP_CXX)
target_compile_options(cucount_cpu_kernels PRIVATE -O3 -fno-math-errno)
# As pybind11_add_module sets for the modules the objects end up in
set_target_properties(cucount_cpu_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CUCOUNT_CPU_NATIVE_ONLY)
  target_compile_definitions(cucount_cpu_kernels PRIVATE HWY_COMPILE_ONLY_STATIC)
  target_compile_options(cucount_cpu_kernels PRIVATE -march=native)
elseif(CUCOUNT_CPU_ALL_TARGETS)
  target_compile_definitions(cucount_cpu_kernels PRIVATE HWY_COMPILE_ALL_ATTAINABLE)
endif()

pybind11_add_module(cucount_cpu src/bind.cpp)
target_link_libraries(cucount_cpu PRIVATE cucount_cpu_kernels)
target_compile_options(cucount_cpu PRIVATE -O3 -fno-math-errno)

# Land in a build-tree cucountlib/ (a namespace package), so a standalone
# build is importable exactly like the installed layout:
#   PYTHONPATH=<build dir> python -c "from cucountlib import cpu"
set_target_properties(cucount_cpu PROPERTIES OUTPUT_NAME cpu LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cucountlib)


# The JAX FFI module, cucountlib.ffi_cpu, built when jax provides its headers.
# Mirrors the cuda/ arm, and links the same kernel objects as cucount_cpu.
if(NOT DEFINED XLA_DIR)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
    RESULT_VARIABLE XLA_RESULT)
else()
  set(XLA_RESULT 0)
endif()

if(DEFINED ENV{BUILD_FFI_CUCOUNT})
  set(BUILD_FFI_CUCOUNT_CPU $ENV{BUILD_FFI_CUCOUNT})
elseif(XLA_RESULT EQUAL 0)
  set(BUILD_FFI_CUCOUNT_CPU ON)
else()
  set(BUILD_FFI_CUCOUNT_CPU OFF)
endif()

if(BUILD_FFI_CUCOUNT_CPU)
  message(STATUS "Compilation of the CPU JAX FFI module (XLA headers: ${XLA_DIR})")
  pybind11_add_module(cucount_ffi_cpu src/ffi_bind.cpp)
  target_include_directories(cucount_ffi_cpu PUBLIC ${XLA_DIR})
  target_link_libraries(cucount_ffi_cpu PRIVATE cucount_cpu_kernels)
  target_compile_options(cucount_ffi_cpu PRIVATE -O3 -fno-math-errno)
  set_target_properties(cucount_ffi_cpu PROPERTIES
    OUTPUT_NAME ffi_cpu LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cucountlib)
else()
  message(STATUS "XLA include directory not found: skipping the CPU JAX FFI module")
endif()
