cmake_minimum_required(VERSION 3.18)
project(cucount LANGUAGES CXX)

option(CUCOUNT_BUILD_CUDA "Build the CUDA backend (cucountlib.cucount and the JAX FFI module)" ON)
option(CUCOUNT_BUILD_CPU "Build the CPU backend" ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_STANDARD 17)  # Use C++17

# Layout: backend-neutral code lives here at the root -- include/ (attrs.h,
# pair_math.h, common.h) and src/ (attrs.cpp, logger.cpp) -- while each
# backend owns its own subdirectory, cuda/ and cpu/, with the same shape.

# Backend-neutral attrs extension: always built, no nvcc. This is what lets
# cucount.numpy import on a CUDA-less build and serve backend='cpu'.
pybind11_add_module(cucount_attrs src/attrs.cpp src/logger.cpp)
# Target names carry a cucount_ prefix (a bare `cuda` target invites
# collisions); OUTPUT_NAME is the import name: cucountlib.attrs etc.
set_target_properties(cucount_attrs PROPERTIES OUTPUT_NAME attrs)
target_include_directories(cucount_attrs PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(cucount_attrs PRIVATE pybind11::module)
# Every extension lands in a build-tree cucountlib/, so PYTHONPATH=<build dir>
# imports the same layout the installed package has.
set_target_properties(cucount_attrs PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cucountlib)
install(TARGETS cucount_attrs LIBRARY DESTINATION cucountlib)

if(CUCOUNT_BUILD_CUDA)
    add_subdirectory(cuda)
endif()

if(CUCOUNT_BUILD_CPU)
    add_subdirectory(cpu)
    install(TARGETS cucount_cpu LIBRARY DESTINATION cucountlib)
    if(TARGET cucount_ffi_cpu)
        install(TARGETS cucount_ffi_cpu LIBRARY DESTINATION cucountlib)
    endif()
endif()

if(NOT CUCOUNT_BUILD_CUDA AND NOT CUCOUNT_BUILD_CPU)
    message(FATAL_ERROR "Both CUCOUNT_BUILD_CUDA and CUCOUNT_BUILD_CPU are OFF: nothing to build")
endif()

if(NOT CUCOUNT_BUILD_CUDA)
    # The attrs layer lives in the CUDA-free cucount_attrs extension, so
    # cucount.numpy imports fine and serves backend='cpu'; only a CUDA call
    # raises (loudly, naming this flag).
    message(STATUS "CUDA backend disabled: cucount.numpy serves backend='cpu' only")
endif()
