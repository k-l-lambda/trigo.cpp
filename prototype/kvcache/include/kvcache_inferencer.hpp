/**
 * KV Cache Inferencer - ONNX Runtime with persistent GPU memory for KV cache
 *
 * This is a prototype implementation to validate KV cache design and measure performance.
 *
 * Key features:
 * - IOBinding API for zero-copy GPU tensor management
 * - Persistent GPU tensors for KV cache across inference calls
 * - Supports both CPU and GPU execution
 *
 * Performance expectations:
 * - First token: Same latency as without cache
 * - Subsequent tokens: 10-100× speedup (depends on sequence length)
 * - Memory overhead: 2 * num_layers * batch * num_heads * max_seq_len * head_dim * sizeof(float)
 *
 * Example:
 *     KVCacheInferencer inferencer("model.onnx", true, 0);
 *
 *     // First token (cold start)
 *     auto logits1 = inferencer.forward({42});  // ~50ms
 *
 *     // Subsequent tokens (with cache)
 *     auto logits2 = inferencer.forward({128}); // ~5ms (10× faster)
 *     auto logits3 = inferencer.forward({256}); // ~5ms
 *
 *     // Reset for new sequence
 *     inferencer.reset_cache();
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


namespace trigo
{


/**
 * Performance metrics for KV cache benchmarking
 */
struct KVCacheMetrics
{
	double first_token_latency_ms = 0.0;
	double avg_subsequent_token_latency_ms = 0.0;
	double speedup_factor = 0.0;

	size_t memory_usage_bytes = 0;
	size_t cache_memory_bytes = 0;

	int num_tokens_generated = 0;
	int current_seq_len = 0;
	int max_seq_len = 0;
};


/**
 * KV Cache Inferencer with IOBinding
 *
 * Manages persistent GPU tensors for key-value cache to accelerate
 * sequential token generation.
 */
class KVCacheInferencer
{
public:
	/**
	 * Constructor - Load ONNX model and initialize KV cache
	 *
	 * @param model_path Path to ONNX model with KV cache I/O
	 * @param use_gpu Enable CUDA execution provider
	 * @param device_id GPU device ID
	 * @param max_seq_len Maximum sequence length (for cache allocation)
	 * @param num_layers Number of transformer layers
	 * @param num_heads Number of attention heads
	 * @param head_dim Dimension per attention head
	 */
	KVCacheInferencer(
		const std::string& model_path,
		bool use_gpu = true,
		int device_id = 0,
		int max_seq_len = 2048,
		int num_layers = 12,
		int num_heads = 12,
		int head_dim = 64
	);

	~KVCacheInferencer();


	/**
	 * Forward pass with KV cache
	 *
	 * Generates logits for next token using cached key-value states.
	 * First call will initialize cache, subsequent calls reuse it.
	 *
	 * @param input_ids Input token IDs (typically 1 token for generation)
	 * @return Logits for next token prediction
	 */
	std::vector<float> forward(const std::vector<int64_t>& input_ids);


	/**
	 * Forward pass without KV cache (baseline comparison)
	 *
	 * Recomputes full sequence every time. Used for benchmarking
	 * to measure KV cache speedup.
	 *
	 * @param input_ids Full sequence of token IDs
	 * @return Logits for next token prediction
	 */
	std::vector<float> forward_no_cache(const std::vector<int64_t>& input_ids);


	/**
	 * Reset KV cache for new sequence
	 */
	void reset_cache();


	/**
	 * Get current performance metrics
	 */
	const KVCacheMetrics& get_metrics() const { return metrics_; }


	/**
	 * Print detailed performance report
	 */
	void print_metrics() const;


private:
	// ONNX Runtime
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	std::unique_ptr<Ort::Session> session_;
	Ort::AllocatorWithDefaultOptions allocator_;

	// GPU settings
	bool use_gpu_;
	int device_id_;
	std::unique_ptr<Ort::MemoryInfo> memory_info_cpu_;
	std::unique_ptr<Ort::MemoryInfo> memory_info_gpu_;

	// Model configuration
	int max_seq_len_;
	int num_layers_;
	int num_heads_;
	int head_dim_;
	int hidden_dim_;  // num_heads * head_dim
	int vocab_size_;

	// KV cache tensors (persistent GPU memory)
	std::vector<Ort::Value> past_key_cache_;   // [num_layers][batch, num_heads, seq_len, head_dim]
	std::vector<Ort::Value> past_value_cache_; // [num_layers][batch, num_heads, seq_len, head_dim]

	// Sequence state
	int current_seq_len_;
	std::vector<int64_t> full_sequence_;  // For no-cache baseline comparison

	// Performance tracking
	KVCacheMetrics metrics_;
	std::chrono::high_resolution_clock::time_point start_time_;


	/**
	 * Initialize KV cache tensors in GPU memory
	 */
	void init_kv_cache();


	/**
	 * Update performance metrics
	 */
	void update_metrics(double latency_ms, bool is_first_token);
};


}  // namespace trigo
