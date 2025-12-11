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
	 * @param evaluation_model_path Path to evaluation.onnx for direct value inference (optional)
	 * @param eval_extend_model_path Path to base_model_eval_extend.onnx for incremental cache (optional)
	 */
	PrefixCacheInferencer(
		const std::string& prefix_model_path,
		const std::string& eval_cached_model_path,
		const std::string& policy_head_path,
		const std::string& value_head_path = "",
		bool use_gpu = true,
		int device_id = 0,
		const std::string& evaluation_model_path = "",
		const std::string& eval_extend_model_path = ""
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
	 * @return Hidden states [batch, eval_len, hidden_dim]
	 */
	std::vector<float> evaluate_with_cache(
		const std::vector<int64_t>& evaluated_ids,
		const std::vector<float>& evaluated_mask,
		int batch_size,
		int eval_len
	);


	/**
	 * Value inference with cache (for MCTS leaf evaluation)
	 *
	 * Uses the same fixed cache as policy inference.
	 * Evaluates a single VALUE token to get position value.
	 *
	 * @param value_token_id Token ID for VALUE (default: 3)
	 * @return Value estimation [-1, 1] from current player perspective
	 */
	float value_inference_with_cache(int value_token_id = 3);


	/**
	 * Extend cache with new tokens (incremental cache update)
	 *
	 * Runs eval_extend model: cache + new_tokens → hidden_states + new_cache
	 * Cache is updated internally with the new KV tensors.
	 *
	 * Used for incremental self-play where the game state grows:
	 *   1. Start game: compute_prefix_cache([START])
	 *   2. After each move: extend_cache([move_tokens])
	 *   3. Cache grows incrementally without recomputation
	 *
	 * @param new_token_ids New tokens to add [batch, new_len]
	 * @param new_mask Attention mask for new tokens [batch, new_len, new_len]
	 * @param batch_size Batch size
	 * @param new_len Number of new tokens
	 * @return Hidden states for new tokens [batch, new_len+1, hidden_dim]
	 */
	std::vector<float> extend_cache(
		const std::vector<int64_t>& new_token_ids,
		const std::vector<float>& new_mask,
		int batch_size,
		int new_len
	);


	/**
	 * Direct value inference using evaluation model
	 *
	 * Uses a separate evaluation model that directly processes the full sequence.
	 * More accurate than value_inference_with_cache for MCTS value estimation.
	 *
	 * @param input_ids Full input token IDs [batch, seq_len] with TGN + VALUE token
	 * @param batch_size Batch size
	 * @param seq_len Sequence length
	 * @return Value estimation [-1, 1]
	 */
	float value_inference_direct(
		const std::vector<int64_t>& input_ids,
		int batch_size,
		int seq_len
	);


	/**
	 * Check if evaluation model is loaded
	 */
	bool has_evaluation_model() const { return evaluation_session_ != nullptr; }


	/**
	 * Check if eval_extend model is loaded (for incremental cache)
	 */
	bool has_extend_model() const { return eval_extend_session_ != nullptr; }


	/**
	 * Run policy head on hidden states
	 *
	 * Takes hidden states from base model and runs policy head to get logits.
	 * This is needed for proper move evaluation in MCTS.
	 *
	 * @param hidden_states Hidden states [batch, seq_len, hidden_dim]
	 * @param batch_size Batch size
	 * @param seq_len Sequence length
	 * @param hidden_dim Hidden dimension
	 * @return Policy logits [batch, seq_len, vocab_size]
	 */
	std::vector<float> policy_inference_from_hidden(
		const std::vector<float>& hidden_states,
		int batch_size,
		int seq_len,
		int hidden_dim
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
		int seq_len;      // Current cached sequence length (may grow after extend_cache)
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


	/**
	 * Get cached key tensors (for debugging/comparison)
	 */
	const std::vector<std::vector<float>>& get_cached_keys() const {
		return cached_keys_;
	}


	/**
	 * Get cached value tensors (for debugging/comparison)
	 */
	const std::vector<std::vector<float>>& get_cached_values() const {
		return cached_values_;
	}


private:
	// ONNX Runtime environment and sessions
	Ort::Env env_;
	Ort::SessionOptions session_options_;

	std::unique_ptr<Ort::Session> prefix_session_;
	std::unique_ptr<Ort::Session> eval_cached_session_;
	std::unique_ptr<Ort::Session> eval_extend_session_;  // For incremental cache extension
	std::unique_ptr<Ort::Session> policy_session_;
	std::unique_ptr<Ort::Session> value_session_;  // Optional
	std::unique_ptr<Ort::Session> evaluation_session_;  // Direct value inference (optional)

	// Memory allocator
	Ort::AllocatorWithDefaultOptions allocator_;

	// GPU settings
	bool use_gpu_;
	int device_id_;

	// Cache state
	bool cache_ready_;
	CacheDimensions cache_dims_;
	std::vector<std::vector<float>> cached_keys_;   // [num_layers][batch * num_heads * seq_len * head_dim]
	std::vector<std::vector<float>> cached_values_; // [num_layers][batch * num_heads * seq_len * head_dim]

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
