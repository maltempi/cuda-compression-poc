# Try to find NVTX
#
# The following variables are optionally searched for defaults
#  NVTX_ROOT_DIR: Base directory where all NVTX components are found
#  NVTX_INCLUDE_DIR: Directory where NVTX header is found
#  NVTX_LIB_DIR: Directory where NVTX library is found
#
# The following are set after configuration is done:
#  NVTX_FOUND
#  NVTX_INCLUDE_DIRS
#  NVTX_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NVTX in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(NVTX_ROOT_DIR $ENV{NVTX_ROOT_DIR} CACHE PATH "Folder contains NVIDIA NVTX")

find_path(NVTX_INCLUDE_DIR
  NAMES nvToolsExt.h nvtx.h
  PATHS
    ${NVTX_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_CUDART_LIBRARY_DIR}
    /usr/local/cuda
    /opt/cuda
    ENV CPATH
  PATH_SUFFIXES
    include
    x86_64-linux/include)

find_library(NVTX_LIBRARY
  NAMES nvToolsExt
  PATHS
    ${NVTX_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_CUDART_LIBRARY_DIR}
    /usr/local/cuda
    /opt/cuda
    ENV LD_LIBRARY_PATH
  PATH_SUFFIXES
    lib
    lib64
    x86_64-linux/lib
    x86_64-linux/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIR NVTX_LIBRARY)

mark_as_advanced(NVTX_ROOT_DIR NVTX_INCLUDE_DIRS NVTX_LIBRARIES)

set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
set(NVTX_LIBRARIES ${NVTX_LIBRARY})
