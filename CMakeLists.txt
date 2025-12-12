cmake_minimum_required(VERSION 3.18)
project(trigo_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenSSL for SHA256 hashing
find_package(OpenSSL REQUIRED)

# Optional CUDA support
option(USE_CUDA "Enable CUDA support" OFF)

# Optional profiling
option(ENABLE_MCTS_PROFILING "Enable MCTS profiling output" OFF)

if(USE_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	# Find CUDA
	find_package(CUDA REQUIRED)
	find_package(CUDAToolkit REQUIRED)

	message(STATUS "CUDA support enabled")
else()
	message(STATUS "CUDA support disabled (CPU-only build)")
endif()

# ONNX Runtime configuration
if(USE_CUDA)
	set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.17.0")
else()
	# Try multiple locations for CPU-only ONNX Runtime
	set(ONNXRUNTIME_SEARCH_PATHS
		"${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.0"
		"${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.17.0"
		"/usr/local"
		"/usr"
	)
	foreach(SEARCH_PATH ${ONNXRUNTIME_SEARCH_PATHS})
		if(EXISTS "${SEARCH_PATH}/include/onnxruntime_c_api.h")
			set(ONNXRUNTIME_ROOT_DIR "${SEARCH_PATH}")
			break()
		endif()
	endforeach()
endif()

if(NOT ONNXRUNTIME_ROOT_DIR)
	message(FATAL_ERROR "ONNX Runtime not found. Please install or set ONNXRUNTIME_ROOT_DIR")
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
)

if(USE_CUDA)
	include_directories()
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# CUDA flags (only if CUDA is enabled)
if(USE_CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")  # RTX 2060+, adjust as needed
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
	set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

# MCTS profiling (optional)
if(ENABLE_MCTS_PROFILING)
	add_compile_definitions(MCTS_ENABLE_PROFILING)
	message(STATUS "MCTS profiling enabled")
endif()

# Source files (will be added as we implement)
set(TRIGO_INFERENCE_SOURCES
    src/tgn_tokenizer.cpp
    src/shared_model_inferencer.cpp
    src/prefix_cache_inferencer.cpp
    src/prefix_tree_builder.cpp
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
)
if(USE_CUDA)
	target_link_libraries(trigo_inference PRIVATE CUDA::cudart)
endif()

target_include_directories(trigo_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Helper function to link targets with optional CUDA
function(link_with_onnxruntime target_name)
	target_link_libraries(${target_name} PRIVATE ${ONNXRUNTIME_LIB})
	if(USE_CUDA)
		target_link_libraries(${target_name} PRIVATE CUDA::cudart)
	endif()
endfunction()

# Test executable for ONNX Runtime verification
if(USE_CUDA)
	add_executable(test_onnxruntime_cuda tests/test_onnxruntime_cuda.cpp)
	link_with_onnxruntime(test_onnxruntime_cuda)

	# Set RPATH so executable can find ONNX Runtime library
	set_target_properties(test_onnxruntime_cuda PROPERTIES
		BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
		INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	)
endif()

# Test executable for TGN tokenizer
add_executable(test_tgn_tokenizer tests/test_tgn_tokenizer.cpp)
target_link_libraries(test_tgn_tokenizer PRIVATE trigo_inference)

# Test executable for shared model inferencer
add_executable(test_shared_model_inferencer tests/test_shared_model_inferencer.cpp)
target_link_libraries(test_shared_model_inferencer PRIVATE
    trigo_inference
    ${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_shared_model_inferencer PRIVATE CUDA::cudart)
endif()
set_target_properties(test_shared_model_inferencer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test executable for prefix cache inferencer
add_executable(test_prefix_cache_inferencer tests/test_prefix_cache_inferencer.cpp)
target_link_libraries(test_prefix_cache_inferencer PRIVATE
    trigo_inference
    ${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_prefix_cache_inferencer PRIVATE CUDA::cudart)
endif()
set_target_properties(test_prefix_cache_inferencer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test executable for cross-language equivalence validation
add_executable(test_inference_equivalence_cpp tests/test_inference_equivalence_cpp.cpp)
target_link_libraries(test_inference_equivalence_cpp PRIVATE
    trigo_inference
    ${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_inference_equivalence_cpp PRIVATE CUDA::cudart)
endif()
set_target_properties(test_inference_equivalence_cpp PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test executable for TGN coordinate system
add_executable(test_trigo_coords tests/test_trigo_coords.cpp)
# Header-only test, no library linking needed

# Test executable for game utils (Go rules)
add_executable(test_trigo_game_utils tests/test_trigo_game_utils.cpp)
# Header-only test, no library linking needed

# Game library with core game engine
add_library(trigo_game STATIC
    src/trigo_game.cpp
)

# Test executable for game replay (cross-language validation)
add_executable(test_game_replay tests/test_game_replay.cpp)
# Header-only test, no library linking needed

# Test executable for TrigoGame class
add_executable(test_trigo_game tests/test_trigo_game.cpp)
target_link_libraries(test_trigo_game PRIVATE trigo_game)

# Self-play data generator
add_executable(self_play_generator src/self_play_generator.cpp src/prefix_tree_builder.cpp)
target_link_libraries(self_play_generator PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
	OpenSSL::Crypto
)
if(USE_CUDA)
	target_link_libraries(self_play_generator PRIVATE CUDA::cudart)
endif()
set_target_properties(self_play_generator PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test trained model loading
add_executable(test_trained_model_loading tests/test_trained_model_loading.cpp)
target_include_directories(test_trained_model_loading PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(test_trained_model_loading PRIVATE ${ONNXRUNTIME_LIB})

# Test prefix tree builder
add_executable(test_prefix_tree_builder tests/test_prefix_tree_builder.cpp src/prefix_tree_builder.cpp)

# Test MCTS implementation
add_executable(test_mcts tests/test_mcts.cpp)
target_link_libraries(test_mcts PRIVATE
	trigo_game
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Simple MCTS debug test
add_executable(test_mcts_simple tests/test_mcts_simple.cpp)
target_link_libraries(test_mcts_simple PRIVATE
	trigo_game
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts_simple PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts_simple PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# AlphaZero MCTS test
add_executable(test_alphazero_mcts tests/test_alphazero_mcts.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_alphazero_mcts PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_alphazero_mcts PRIVATE CUDA::cudart)
endif()
set_target_properties(test_alphazero_mcts PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test NeuralPolicy full inference
add_executable(test_neural_policy_inference tests/test_neural_policy_inference.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_neural_policy_inference PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_neural_policy_inference PRIVATE CUDA::cudart)
endif()
set_target_properties(test_neural_policy_inference PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test TGN generation consistency
add_executable(test_tgn_consistency tests/test_tgn_consistency.cpp)
target_link_libraries(test_tgn_consistency PRIVATE trigo_game)

# Test prefix cache correctness (vs standard inference)
add_executable(test_prefix_cache_correctness tests/test_prefix_cache_correctness.cpp)
target_link_libraries(test_prefix_cache_correctness PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_prefix_cache_correctness PRIVATE CUDA::cudart)
endif()
set_target_properties(test_prefix_cache_correctness PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test cached inference with real game
add_executable(test_cached_inference_game tests/test_cached_inference_game.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_cached_inference_game PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_cached_inference_game PRIVATE CUDA::cudart)
endif()
set_target_properties(test_cached_inference_game PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Benchmark dynamic shape performance
add_executable(benchmark_dynamic_shapes tests/benchmark_dynamic_shapes.cpp src/prefix_tree_builder.cpp)
target_link_libraries(benchmark_dynamic_shapes PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(benchmark_dynamic_shapes PRIVATE CUDA::cudart)
endif()
set_target_properties(benchmark_dynamic_shapes PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test CachedNeuralPolicy integration
add_executable(test_cached_neural_policy tests/test_cached_neural_policy.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_cached_neural_policy PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_cached_neural_policy PRIVATE CUDA::cudart)
endif()
set_target_properties(test_cached_neural_policy PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test CachedAlphaZeroPolicy integration (Phase 5.7)
add_executable(test_cached_alphazero_policy tests/test_cached_alphazero_policy.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_cached_alphazero_policy PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_cached_alphazero_policy PRIVATE CUDA::cudart)
endif()
set_target_properties(test_cached_alphazero_policy PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Benchmark value inference with cache (Phase 5.7)
add_executable(benchmark_value_cache tests/benchmark_value_cache.cpp)
target_link_libraries(benchmark_value_cache PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(benchmark_value_cache PRIVATE CUDA::cudart)
endif()
set_target_properties(benchmark_value_cache PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Simplified value cache benchmark
add_executable(benchmark_value_cache_simple tests/benchmark_value_cache_simple.cpp)
target_link_libraries(benchmark_value_cache_simple PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(benchmark_value_cache_simple PRIVATE CUDA::cudart)
endif()
set_target_properties(benchmark_value_cache_simple PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test CachedMCTS full integration (Phase 5.8)
add_executable(test_cached_mcts tests/test_cached_mcts.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_cached_mcts PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_cached_mcts PRIVATE CUDA::cudart)
endif()
set_target_properties(test_cached_mcts PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test policy inference comparison (C++ vs TypeScript)
add_executable(test_policy_comparison tests/test_policy_comparison.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_policy_comparison PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_policy_comparison PRIVATE CUDA::cudart)
endif()
set_target_properties(test_policy_comparison PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test policy and value validation
add_executable(test_policy_value_validation tests/test_policy_value_validation.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_policy_value_validation PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_policy_value_validation PRIVATE CUDA::cudart)
endif()
set_target_properties(test_policy_value_validation PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test C++ vs TypeScript comparison (uses hardcoded tree structure)
add_executable(test_compare_with_ts tests/test_compare_with_ts.cpp)
target_link_libraries(test_compare_with_ts PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_compare_with_ts PRIVATE CUDA::cudart)
endif()
set_target_properties(test_compare_with_ts PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test MCTS from specific prefix
add_executable(test_mcts_prefix tests/test_mcts_prefix.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_mcts_prefix PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts_prefix PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts_prefix PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test equivalence between prefix cache and tree models
add_executable(test_cache_vs_tree_equivalence tests/test_cache_vs_tree_equivalence.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_cache_vs_tree_equivalence PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_cache_vs_tree_equivalence PRIVATE CUDA::cudart)
endif()
set_target_properties(test_cache_vs_tree_equivalence PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test prefix tree size fix (TypeScript compatibility)
add_executable(test_tree_size_fix tests/test_tree_size_fix.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_tree_size_fix PRIVATE trigo_game trigo_inference)

# Test evaluated_ids match TypeScript
add_executable(test_evaluated_ids_match tests/test_evaluated_ids_match.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_evaluated_ids_match PRIVATE trigo_game trigo_inference)

# Test single MCTS simulation (node visit counts)
add_executable(test_mcts_single_sim tests/test_mcts_single_sim.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_mcts_single_sim PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts_single_sim PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts_single_sim PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test full MCTS search (C++ vs TypeScript comparison)
add_executable(test_mcts_full_search tests/test_mcts_full_search.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_mcts_full_search PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts_full_search PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts_full_search PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test 3D board tree building
add_executable(test_tree_build_3d tests/test_tree_build_3d.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_tree_build_3d PRIVATE trigo_game trigo_inference)

# Test MCTS single step consistency (C++ vs TypeScript)
add_executable(test_mcts_single_step tests/test_mcts_single_step.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_mcts_single_step PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_mcts_single_step PRIVATE CUDA::cudart)
endif()
set_target_properties(test_mcts_single_step PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

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

add_executable(test_tree_build_5x5 tests/test_tree_build_5x5.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_tree_build_5x5 PRIVATE trigo_game trigo_inference)

# KV cache comparison test
add_executable(test_kv_cache_comparison tests/test_kv_cache_comparison.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_kv_cache_comparison PRIVATE trigo_game trigo_inference ${ONNXRUNTIME_LIB})
set_target_properties(test_kv_cache_comparison PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

add_executable(test_value_comparison tests/test_value_comparison.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_value_comparison PRIVATE trigo_game trigo_inference ${ONNXRUNTIME_LIB})
set_target_properties(test_value_comparison PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test policy consistency (no noise, no temperature)
add_executable(test_policy_consistency tests/test_policy_consistency.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_policy_consistency PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_policy_consistency PRIVATE CUDA::cudart)
endif()
set_target_properties(test_policy_consistency PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test tree model policy inference (C++ vs TypeScript comparison)
add_executable(test_tree_model_policy tests/test_tree_model_policy.cpp src/prefix_tree_builder.cpp)
target_link_libraries(test_tree_model_policy PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_tree_model_policy PRIVATE CUDA::cudart)
endif()
set_target_properties(test_tree_model_policy PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test terminal state detection (TypeScript consistency)
add_executable(test_terminal_detection tests/test_terminal_detection.cpp)
target_link_libraries(test_terminal_detection PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_terminal_detection PRIVATE CUDA::cudart)
endif()
set_target_properties(test_terminal_detection PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test extend_cache() functionality (Phase 5.10)
add_executable(test_extend_cache tests/test_extend_cache.cpp)
target_link_libraries(test_extend_cache PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_extend_cache PRIVATE CUDA::cudart)
endif()
set_target_properties(test_extend_cache PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test incremental cache consistency (Phase 5.10)
add_executable(test_incremental_consistency tests/test_incremental_consistency.cpp)
target_link_libraries(test_incremental_consistency PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_incremental_consistency PRIVATE CUDA::cudart)
endif()
set_target_properties(test_incremental_consistency PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Test IncrementalCachedMCTSPolicy GPU vs CPU
add_executable(test_incremental_gpu tests/test_incremental_gpu.cpp)
target_link_libraries(test_incremental_gpu PRIVATE
	trigo_game
	trigo_inference
	${ONNXRUNTIME_LIB}
)
if(USE_CUDA)
	target_link_libraries(test_incremental_gpu PRIVATE CUDA::cudart)
endif()
set_target_properties(test_incremental_gpu PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)
