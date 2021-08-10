cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
include(cmake/modules/set_ifndef.cmake)

project(trtx
        LANGUAGES CXX CUDA
        VERSION "1.0.0"
        DESCRIPTION "TensorRTX via cpp"
        HOMEPAGE_URL "https://github.com/pohsi/trtx")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set_ifndef(TRTX_OUT_DIR ${CMAKE_BINARY_DIR}/bin)

message(STATUS "TRTX_OUT_DIR is ${TRTX_OUT_DIR}")

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
endif()

set(CMAKE_SKIP_BUILD_RPATH True)

#### Debug settings
set(TRTX_DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode ${DEBUG_POSTFIX}")
endif()

#### Dependencies

set(DEFAULT_CUDA_VERSION 11.3.1)
set(DEFAULT_CUDNN_VERSION 8.2)
set(DEFAULT_PROTOBUF_VERSION 3.0.0)
set(DEFAULT_CUB_VERSION 1.8.0)

# Dependency Version Resolution
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")
set_ifndef(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")

find_package(Threads REQUIRED)

## find_package(CUDA) is broken for cross-compilation. Enable CUDA language instead.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_package(CUDA ${CUDA_VERSION} REQUIRED)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)

find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)

find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

set(CUDA_LIBRARIES ${CUDART_LIB})

#### CUDA targets
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      35
      53
      61
      70
      75
    )

  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  else()
    message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")


add_subdirectory(src)