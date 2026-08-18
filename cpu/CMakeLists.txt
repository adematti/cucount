cmake_minimum_required(VERSION 3.20)

# Buildable standalone or as a subdirectory of the root
# project behind -DCUCOUNT_BUILD_CPU=ON.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cucount_cpu LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(CUCOUNT_CPU_ALL_TARGETS
       "Compile every attainable SIMD target, not just Highway's default set" ON)

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

pybind11_add_module(cpucount src/count2_cpu.cpp src/bind.cpp)
target_include_directories(cpucount PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpucount PRIVATE hwy OpenMP::OpenMP_CXX)

target_compile_options(cpucount PRIVATE -O3 -fno-math-errno)

if(CUCOUNT_CPU_ALL_TARGETS)
  target_compile_definitions(cpucount PRIVATE HWY_COMPILE_ALL_ATTAINABLE)
endif()
