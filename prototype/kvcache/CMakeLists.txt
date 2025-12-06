cmake_minimum_required(VERSION 3.18)
project(kvcache_prototype CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA support
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	find_package(CUDA REQUIRED)
	find_package(CUDAToolkit REQUIRED)

	message(STATUS "CUDA support enabled")
else()
	message(STATUS "CUDA support disabled (CPU-only build)")
endif()

# Find ONNX Runtime
set(ONNXRUNTIME_SEARCH_PATHS
	"${CMAKE_CURRENT_SOURCE_DIR}/../../onnxruntime-linux-x64-gpu-1.17.0"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../onnxruntime-linux-x64-1.17.0"
	"/usr/local"
	"/usr"
)

foreach(SEARCH_PATH ${ONNXRUNTIME_SEARCH_PATHS})
	if(EXISTS "${SEARCH_PATH}/include/onnxruntime_cxx_api.h")
		set(ONNXRUNTIME_ROOT_DIR "${SEARCH_PATH}")
		message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_ROOT_DIR}")
		break()
	endif()
endforeach()

if(NOT ONNXRUNTIME_ROOT_DIR)
	message(FATAL_ERROR "ONNX Runtime not found. Please install or set ONNXRUNTIME_ROOT_DIR")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
	HINTS ${ONNXRUNTIME_ROOT_DIR}/lib
	REQUIRED)

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Include directories
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${ONNXRUNTIME_ROOT_DIR}/include
)

if(USE_CUDA)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Find nlohmann/json (header-only library)
# Try to find it in common locations
find_path(JSON_INCLUDE_DIR
	NAMES nlohmann/json.hpp
	PATHS
		/usr/include
		/usr/local/include
		${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/json/include
)

if(JSON_INCLUDE_DIR)
	message(STATUS "Found nlohmann/json: ${JSON_INCLUDE_DIR}")
	include_directories(${JSON_INCLUDE_DIR})
else()
	message(WARNING "nlohmann/json not found. Will download if needed.")
	# Download single header file
	file(DOWNLOAD
		"https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp"
		TIMEOUT 30
		STATUS DOWNLOAD_STATUS
	)
	list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
	if(STATUS_CODE EQUAL 0)
		message(STATUS "Downloaded nlohmann/json.hpp successfully")
	else()
		message(FATAL_ERROR "Failed to download nlohmann/json.hpp. Please install manually.")
	endif()
endif()

# Build KV cache inferencer library
add_library(kvcache_inferencer STATIC
	src/kvcache_inferencer.cpp
)

target_link_libraries(kvcache_inferencer PUBLIC
	${ONNXRUNTIME_LIB}
)

if(USE_CUDA)
	target_link_libraries(kvcache_inferencer PUBLIC CUDA::cudart)
	target_compile_definitions(kvcache_inferencer PUBLIC USE_CUDA)
endif()

# Build test executable
add_executable(test_kvcache_prototype
	test_kvcache.cpp
)

target_link_libraries(test_kvcache_prototype PRIVATE
	kvcache_inferencer
	${ONNXRUNTIME_LIB}
)

if(USE_CUDA)
	target_link_libraries(test_kvcache_prototype PRIVATE CUDA::cudart)
endif()

# Set RPATH for ONNX Runtime
set_target_properties(test_kvcache_prototype PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Print configuration
message(STATUS "")
message(STATUS "KV Cache Prototype Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
if(USE_CUDA)
	message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "")
