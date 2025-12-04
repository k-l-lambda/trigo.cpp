cmake_minimum_required(VERSION 3.18)
project(trigo_cpp CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.17.0")

if(NOT EXISTS ${ONNXRUNTIME_ROOT_DIR})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_DIR}")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT_DIR}/lib
    REQUIRED)

message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_ROOT_DIR}/include")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")  # RTX 2060+, adjust as needed
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Source files (will be added as we implement)
set(TRIGO_INFERENCE_SOURCES
    src/tgn_tokenizer.cpp
    # src/shared_model_inferencer.cpp
    # src/prefix_tree_builder.cpp
)

# CUDA source files
set(TRIGO_CUDA_SOURCES
    # kernels/mcts_select.cu
    # kernels/mcts_expand.cu
    # kernels/mcts_backup.cu
)

# Core inference library
add_library(trigo_inference SHARED
    ${TRIGO_INFERENCE_SOURCES}
    ${TRIGO_CUDA_SOURCES}
)

target_link_libraries(trigo_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    CUDA::cudart
)

target_include_directories(trigo_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Test executable for ONNX Runtime verification
add_executable(test_onnxruntime_cuda tests/test_onnxruntime_cuda.cpp)
target_link_libraries(test_onnxruntime_cuda PRIVATE
    ${ONNXRUNTIME_LIB}
    CUDA::cudart
)

# Set RPATH so executable can find ONNX Runtime library
set_target_properties(test_onnxruntime_cuda PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test executable for TGN tokenizer
add_executable(test_tgn_tokenizer tests/test_tgn_tokenizer.cpp)
target_link_libraries(test_tgn_tokenizer PRIVATE trigo_inference)

# Python bindings (will be added later with pybind11)
# find_package(pybind11 REQUIRED)
# pybind11_add_module(cuda_mcts_inference src/bindings.cpp)
# target_link_libraries(cuda_mcts_inference PRIVATE trigo_inference)

# Testing
enable_testing()

# Installation rules (for later)
# install(TARGETS trigo_inference
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )
#
# install(DIRECTORY include/
#     DESTINATION include
#     FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
# )

message(STATUS "")
message(STATUS "Trigo.cpp Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "")
