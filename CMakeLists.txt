cmake_minimum_required(VERSION 3.18)
project(cucount LANGUAGES CUDA CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)  # or "80-real" for full PTX
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_STANDARD 17)  # Use C++17
enable_language(CUDA)

pybind11_add_module(cucount src/cucount.cc src/common.cu src/logger.cc src/mesh.cu src/count2.cu)
target_include_directories(cucount PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(cucount PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cucount PRIVATE pybind11::module)
set_target_properties(cucount PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS cucount LIBRARY DESTINATION cucountlib PUBLIC_HEADER DESTINATION cucountlib)

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

pybind11_add_module(ffi_cucount src/ffi_cucount.cu src/common.cu src/logger.cc src/mesh.cu src/count2.cu)
target_include_directories(ffi_cucount PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(ffi_cucount PUBLIC ${XLA_DIR})
set_target_properties(ffi_cucount PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ffi_cucount PRIVATE pybind11::module)
set_target_properties(ffi_cucount PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS ffi_cucount LIBRARY DESTINATION cucountlib PUBLIC_HEADER DESTINATION cucountlib)