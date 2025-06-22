cmake_minimum_required(VERSION 3.18)
project(cucount LANGUAGES CUDA CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)  # or "80-real" for full PTX

enable_language(CUDA)
# Add the CUDA source file
pybind11_add_module(cucount src/cucount.cc src/common.cu src/logger.cc src/mesh.cu src/count2.cu)

target_include_directories(cucount PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

# Enable CUDA separable compilation
set_target_properties(cucount PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link pybind11
target_link_libraries(cucount PRIVATE pybind11::module)

set_target_properties(cucount PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS cucount LIBRARY DESTINATION cucountlib PUBLIC_HEADER DESTINATION cucountlib)