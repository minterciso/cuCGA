cmake_minimum_required(VERSION 3.24)

# nvcc only supports a bounded range of host GCC versions (CUDA 13.3: <= 15). Prefer a
# compat compiler if installed (Fedora: dnf install gcc15-c++), otherwise use the system
# one and skip nvcc's version check. Must be decided before the CUDA language is enabled.
find_program(CGA_CUDA_HOST_CXX NAMES g++-15 g++-14 g++-13)
if(CGA_CUDA_HOST_CXX)
    set(CMAKE_CUDA_HOST_COMPILER ${CGA_CUDA_HOST_CXX} CACHE FILEPATH "")
else()
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
endif()
# Target the GPU(s) present on this machine; override with -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuCga VERSION 0.1 LANGUAGES C CUDA)

# The shared C sources are the same as cga's (see src/backend.h): pinned to C17 there too.
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

find_package(Threads REQUIRED)

# Host code shared with cga
add_library(cga_core STATIC
    src/ca.c
    src/cga.c
    src/utils.c
    src/params.c
    src/templates.c
)
target_include_directories(cga_core PUBLIC src)
target_compile_options(cga_core PUBLIC $<$<COMPILE_LANGUAGE:C>:-O1 -ggdb -g3>)
target_link_libraries(cga_core PUBLIC m Threads::Threads)

# The CUDA backend: the only code that differs from cga
add_library(cga_backend_cuda STATIC src/kernel.cu)
target_include_directories(cga_backend_cuda PUBLIC src)
target_compile_options(cga_backend_cuda PRIVATE -O2)

add_executable(cuCga src/main.c)
target_link_libraries(cuCga PRIVATE cga_core cga_backend_cuda)
set_target_properties(cuCga PROPERTIES LINKER_LANGUAGE CUDA)

enable_testing()
add_executable(test_templates tests/test_templates.c)
target_link_libraries(test_templates PRIVATE cga_core)
add_test(NAME templates COMMAND test_templates)

add_executable(test_backend tests/test_backend.c)
target_link_libraries(test_backend PRIVATE cga_core cga_backend_cuda)
set_target_properties(test_backend PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME backend COMMAND test_backend)

install(TARGETS cuCga RUNTIME DESTINATION bin)
