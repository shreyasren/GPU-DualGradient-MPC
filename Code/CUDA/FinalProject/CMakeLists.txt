cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

project(main LANGUAGES CXX CUDA)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

find_package(CUDA REQUIRED)

if (NOT WIN32)
    find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)

set(CMAKE_CUDA_ARCHITECTURES 37)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

find_path(LIBWB_ROOT NAMES wb.h HINTS libwb ../libwb ../../libwb)
include_directories(${LIBWB_ROOT})
find_path(LIBWB_LIB_DIR NAMES libwb.so HINTS ${LIBWB_ROOT}/lib ${LIBWB_ROOT}/build) 
link_directories(${LIBWB_LIB_DIR})

set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} --use_fast_math")

add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main wb rt)