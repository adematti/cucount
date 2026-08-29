# CUDA backend: the cucountlib.cucount extension and, when jax provides its
# FFI headers, the cucountlib.ffi_cucount module. Mirrors cpu/. The shared
# code (attrs, pair_math, common types, logger) lives at the repository root.

# Overridable: CUDA 13 dropped compute_70, so builds against a recent
# toolkit must narrow this, e.g. -DCMAKE_CUDA_ARCHITECTURES=80.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif()
enable_language(CUDA)

set(CUCOUNT_CUDA_SOURCES
    src/common.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/logger.cpp
    src/mesh.cu
    src/count2.cu
    src/count3close.cu
    src/count3.cu)

pybind11_add_module(cucount src/bind.cu ${CUCOUNT_CUDA_SOURCES})
target_include_directories(cucount PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set_target_properties(cucount PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cucount PRIVATE pybind11::module)
set_target_properties(cucount PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS cucount LIBRARY DESTINATION cucountlib PUBLIC_HEADER DESTINATION cucountlib)

# Try to get XLA_DIR and set BUILD_FFI_CUCOUNT accordingly
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
  RESULT_VARIABLE XLA_RESULT
)
if(XLA_RESULT EQUAL 0)
    message(STATUS "XLA include directory: ${XLA_DIR}")
else()
    message(STATUS "XLA include directory not found")
endif()

# Set BUILD_FFI_CUCOUNT default: ON if XLA_RESULT is 0 (success), else OFF
if(DEFINED ENV{BUILD_FFI_CUCOUNT})
    set(BUILD_FFI_CUCOUNT $ENV{BUILD_FFI_CUCOUNT})
else()
    if(XLA_RESULT EQUAL 0)
        set(BUILD_FFI_CUCOUNT ON)
    else()
        set(BUILD_FFI_CUCOUNT OFF)
    endif()
endif()

if(BUILD_FFI_CUCOUNT)
    message(STATUS "Compilation of the JAX FFI module")
    pybind11_add_module(ffi_cucount src/ffi_bind.cu ${CUCOUNT_CUDA_SOURCES})
    target_include_directories(ffi_cucount PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_include_directories(ffi_cucount PUBLIC ${XLA_DIR})
    set_target_properties(ffi_cucount PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(ffi_cucount PRIVATE pybind11::module)
    set_target_properties(ffi_cucount PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    install(TARGETS ffi_cucount LIBRARY DESTINATION cucountlib PUBLIC_HEADER DESTINATION cucountlib)
endif()
