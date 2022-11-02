cmake_minimum_required(VERSION 3.13...3.18)

project(cusz_example CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

## Uncommenting to specify GPU architecture is only for testing purpose.
## uncomment for pre-Turing
# set(CMAKE_CUDA_ARCHITECTURES 60 62 70 72)
## uncomment for V100
# set(CMAKE_CUDA_ARCHITECTURES 70)
## uncomment for Turing
# set(CMAKE_CUDA_ARCHITECTURES 75)
## uncmoment for Ampere
# set(CMAKE_CUDA_ARCHITECTURES 80 86)
## uncmoment for A100
# set(CMAKE_CUDA_ARCHITECTURES 80)

enable_language(CUDA)

include_directories(/opt/zfp/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE SHARED)

#-----------
option(USE_NVTX "Enable support for NVIDIA Tools Extension (NVTX)" ON)
if(USE_NVTX)
  find_package(NVTX)
  if(NVTX_FOUND)
    include_directories(${NVTX_INCLUDE_DIRS})
    message(STATUS "Found NVIDIA Tools Extension (NVTX): ${NVTX_INCLUDE_DIRS}")
    add_definitions("-DUSE_NVTX")
    list(APPEND CMAKE_LIBRARY_PATH "${NVTX_LIBRARIES}")
  else()
    message(FATAL_ERROR "Could not find NVIDIA Tools Extension (NVTX)")
  endif()
  set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
  set(NVTX_LIBRARIES ${NVTX_LIBRARY})

endif()
#-----------

add_executable(ex-api src/main.cu)
target_link_libraries(ex-api
	PUBLIC
  /opt/zfp/lib/libzfp.so
  /opt/zfp/lib/libzfp.so.1
  /opt/zfp/lib/libzfp.so.1.0.0
	-lcuda
	-lcusparse)
set_target_properties(ex-api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
