/**
 * Shared Model Inferencer - C++ ONNX Runtime wrapper for shared architecture models
 *
 * This class provides high-level inference interface for the shared architecture:
 * - base_model.onnx: GPT-2 transformer with tree attention → hidden_states
 * - policy_head.onnx: Output projection → policy logits
 * - value_head.onnx: Value MLP → value prediction
 *
 * Design benefits:
 * - 48% memory savings (411MB vs 800MB)
 * - 50% inference speedup (single base model forward pass)
 * - Mathematically equivalent to monolithic models (validated)
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>


namespace trigo
{


class SharedModelInferencer
{
public:
	/**
	 * Constructor - Load ONNX models from disk
	 *
	 * @param base_model_path Path to base_model.onnx
	 * @param policy_head_path Path to policy_head.onnx
	 * @param value_head_path Path to value_head.onnx
	 * @param use_gpu Enable CUDA execution provider (default: true)
	 * @param device_id GPU device ID (default: 0)
	 */
	SharedModelInferencer(
		const std::string& base_model_path,
		const std::string& policy_head_path,
		const std::string& value_head_path,
		bool use_gpu = true,
		int device_id = 0
	);


	/**
	 * Policy inference (TreeLM mode)
	 *
	 * Evaluates multiple candidate moves given a game prefix.
	 *
	 * @param prefix_ids Prefix token IDs [batch, n]
	 * @param evaluated_ids Evaluated token IDs [batch, m]
	 * @param evaluated_mask Tree attention mask [batch, m, m]
	 * @return Policy logits [batch, m+1, vocab_size]
	 *
	 * Example:
	 *     auto logits = inferencer.policy_inference(
	 *         prefix_ids,      // [2, 128] - game prefix
	 *         evaluated_ids,   // [2, 64]  - moves to evaluate
	 *         evaluated_mask   // [2, 64, 64] - attention pattern
	 *     );
	 *     // logits shape: [2, 65, 128] (65 = 64 evaluated + 1 last prefix)
	 */
	std::vector<float> policy_inference(
		const std::vector<int64_t>& prefix_ids,
		const std::vector<int64_t>& evaluated_ids,
		const std::vector<float>& evaluated_mask,
		int batch_size,
		int prefix_len,
		int eval_len
	);


	/**
	 * Value inference (EvaluationLM mode)
	 *
	 * Predicts game outcome value for a given position.
	 *
	 * @param input_ids Input token IDs [batch, seq_len]
	 * @param batch_size Batch size
	 * @param seq_len Sequence length
	 * @param value_token_id VALUE token ID to append (default: 3)
	 * @return Values [batch] in range [-1, 1]
	 *
	 * Example:
	 *     auto values = inferencer.value_inference(
	 *         input_ids,   // [2, 256] - game state
	 *         2,           // batch_size
	 *         256,         // seq_len
	 *         3            // VALUE token
	 *     );
	 *     // values: [0.75, -0.32] (batch of 2 predictions)
	 */
	std::vector<float> value_inference(
		const std::vector<int64_t>& input_ids,
		int batch_size,
		int seq_len,
		int value_token_id = 3
	);


	/**
	 * Get model input/output information
	 */
	void print_model_info() const;


private:
	// ONNX Runtime environment and sessions
	Ort::Env env_;
	Ort::SessionOptions session_options_;

	std::unique_ptr<Ort::Session> base_session_;
	std::unique_ptr<Ort::Session> policy_session_;
	std::unique_ptr<Ort::Session> value_session_;

	// Memory allocator
	Ort::AllocatorWithDefaultOptions allocator_;

	// GPU settings
	bool use_gpu_;
	int device_id_;


	/**
	 * Helper: Create causal evaluated mask (lower triangular)
	 *
	 * @param eval_len Length of evaluated sequence (m)
	 * @return Mask [m, m] with lower triangular pattern
	 */
	static std::vector<float> create_causal_mask(int eval_len);


	/**
	 * Helper: Expand mask to batch dimension
	 *
	 * @param mask Single mask [m, m]
	 * @param batch_size Batch size
	 * @return Batched mask [batch, m, m]
	 */
	static std::vector<float> expand_mask_to_batch(
		const std::vector<float>& mask,
		int batch_size,
		int eval_len
	);
};


}  // namespace trigo
