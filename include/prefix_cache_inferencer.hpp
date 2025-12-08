/**
 * Prefix Cache Inferencer - MCTS-optimized inference with fixed prefix cache
 *
 * This implementation supports the MCTS prefix-reuse pattern:
 * 1. Compute game state (prefix) once → generate cache
 * 2. Evaluate multiple moves (evaluated) with the same fixed cache
 * 3. Cache stays fixed (doesn't accumulate) across evaluations
 *
 * Uses two ONNX models from Phase 5.4:
 * - base_model_prefix.onnx: prefix_ids → cache
 * - base_model_eval_cached.onnx: cache + evaluated_ids → hidden_states
 *
 * Performance benefits:
 * - 1.46-1.52× speedup for MCTS move evaluation
 * - 30-34% time savings per MCTS iteration
 * - 10-16 seconds saved per 1000 nodes
 *
 * Example usage:
 *     PrefixCacheInferencer inferencer(
 *         "models/base_model_prefix.onnx",
 *         "models/base_model_eval_cached.onnx",
 *         "models/policy_head.onnx",
 *         true, // use_gpu
 *         0     // device_id
 *     );
 *
 *     // Step 1: Compute prefix once
 *     inferencer.compute_prefix_cache(prefix_ids, batch_size, prefix_len);
 *
 *     // Step 2: Evaluate multiple moves with same cache
 *     for (auto& move : candidate_moves) {
 *         auto logits = inferencer.evaluate_with_cache(
 *             move.evaluated_ids,
 *             move.evaluated_mask,
 *             batch_size,
 *             eval_len
 *         );
 *         // Process logits...
 *     }
 *
 *     // Cache stays fixed throughout all evaluations!
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <chrono>


namespace trigo
{


/**
 * Performance metrics for prefix cache benchmarking
 */
struct PrefixCacheMetrics
{
	double prefix_computation_ms = 0.0;
	double avg_eval_latency_ms = 0.0;
	double speedup_vs_standard = 0.0;

	int num_prefix_computations = 0;
	int num_evaluations = 0;

	size_t cache_memory_bytes = 0;
	int cache_length = 0;
};


/**
 * Prefix Cache Inferencer for MCTS
 *
 * Manages two-stage inference:
 * 1. Prefix computation (once per MCTS node)
 * 2. Cached evaluation (multiple times per node)
 */
class PrefixCacheInferencer
{
public:
	/**
	 * Constructor - Load ONNX models for prefix cache inference
	 *
	 * @param prefix_model_path Path to base_model_prefix.onnx
	 * @param eval_cached_model_path Path to base_model_eval_cached.onnx
	 * @param policy_head_path Path to policy_head.onnx
	 * @param value_head_path Path to value_head.onnx (optional, can be empty)
	 * @param use_gpu Enable CUDA execution provider
	 * @param device_id GPU device ID
	 */
	PrefixCacheInferencer(
		const std::string& prefix_model_path,
		const std::string& eval_cached_model_path,
		const std::string& policy_head_path,
		const std::string& value_head_path = "",
		bool use_gpu = true,
		int device_id = 0
	);

	~PrefixCacheInferencer() = default;


	/**
	 * Step 1: Compute prefix cache (once per MCTS node)
	 *
	 * Runs prefix-only model: prefix_ids → cache
	 * Cache is stored internally and reused for subsequent evaluations.
	 *
	 * @param prefix_ids Prefix token IDs [batch, prefix_len]
	 * @param batch_size Batch size
	 * @param prefix_len Prefix sequence length
	 */
	void compute_prefix_cache(
		const std::vector<int64_t>& prefix_ids,
		int batch_size,
		int prefix_len
	);


	/**
	 * Step 2: Evaluate with fixed cache (multiple times per MCTS node)
	 *
	 * Runs eval-cached model: cache + evaluated_ids → hidden_states
	 * Cache remains unchanged. Can be called multiple times with different
	 * evaluated sequences to simulate different moves.
	 *
	 * @param evaluated_ids Evaluated token IDs [batch, eval_len]
	 * @param evaluated_mask Tree attention mask [batch, eval_len, eval_len]
	 * @param batch_size Batch size
	 * @param eval_len Evaluated sequence length
	 * @return Policy logits [batch, eval_len+1, vocab_size]
	 */
	std::vector<float> evaluate_with_cache(
		const std::vector<int64_t>& evaluated_ids,
		const std::vector<float>& evaluated_mask,
		int batch_size,
		int eval_len
	);


	/**
	 * Combined inference (for comparison/baseline)
	 *
	 * Runs standard model without cache optimization.
	 * Used for benchmarking to measure speedup.
	 *
	 * @param prefix_ids Prefix token IDs [batch, prefix_len]
	 * @param evaluated_ids Evaluated token IDs [batch, eval_len]
	 * @param evaluated_mask Tree attention mask [batch, eval_len, eval_len]
	 * @param batch_size Batch size
	 * @param prefix_len Prefix length
	 * @param eval_len Evaluated length
	 * @return Policy logits [batch, eval_len+1, vocab_size]
	 */
	std::vector<float> evaluate_standard(
		const std::vector<int64_t>& prefix_ids,
		const std::vector<int64_t>& evaluated_ids,
		const std::vector<float>& evaluated_mask,
		int batch_size,
		int prefix_len,
		int eval_len
	);


	/**
	 * Check if prefix cache is ready
	 */
	bool has_cache() const { return cache_ready_; }


	/**
	 * Clear prefix cache
	 */
	void clear_cache();


	/**
	 * Get cache dimensions
	 */
	struct CacheDimensions {
		int num_layers;
		int num_heads;
		int prefix_len;
		int head_dim;
		int batch_size;
	};

	CacheDimensions get_cache_dimensions() const {
		return cache_dims_;
	}


	/**
	 * Get performance metrics
	 */
	const PrefixCacheMetrics& get_metrics() const { return metrics_; }


	/**
	 * Print performance report
	 */
	void print_metrics() const;


	/**
	 * Print model information
	 */
	void print_model_info() const;


private:
	// ONNX Runtime environment and sessions
	Ort::Env env_;
	Ort::SessionOptions session_options_;

	std::unique_ptr<Ort::Session> prefix_session_;
	std::unique_ptr<Ort::Session> eval_cached_session_;
	std::unique_ptr<Ort::Session> policy_session_;
	std::unique_ptr<Ort::Session> value_session_;  // Optional

	// Memory allocator
	Ort::AllocatorWithDefaultOptions allocator_;

	// GPU settings
	bool use_gpu_;
	int device_id_;

	// Cache state
	bool cache_ready_;
	CacheDimensions cache_dims_;
	std::vector<std::vector<float>> cached_keys_;   // [num_layers][batch * num_heads * prefix_len * head_dim]
	std::vector<std::vector<float>> cached_values_; // [num_layers][batch * num_heads * prefix_len * head_dim]

	// Performance tracking
	PrefixCacheMetrics metrics_;


	/**
	 * Helper: Create causal evaluated mask (lower triangular)
	 */
	static std::vector<float> create_causal_mask(int eval_len);


	/**
	 * Helper: Expand mask to batch dimension
	 */
	static std::vector<float> expand_mask_to_batch(
		const std::vector<float>& mask,
		int batch_size,
		int eval_len
	);
};


}  // namespace trigo
