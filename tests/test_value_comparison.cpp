/**
 * Test VALUE inference comparison with Python
 *
 * Compare C++ cached value inference with Python results from:
 * /home/camus/work/trigoRL/tests/test_value_approaches.py
 *
 * Expected results from Python:
 * - Baseline (direct):       -0.085780
 * - Cache + Direct VALUE:    -0.085780 (diff: 0.000000)
 * - ONNX cached:             -0.085773 (diff: 0.000007)
 */

#include "../include/prefix_cache_inferencer.hpp"
#include "../include/tgn_tokenizer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>


using namespace trigo;


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "============================================================================" << std::endl;
	std::cout << "VALUE Inference Comparison Test (C++ vs Python)" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;

	// Load ONNX models
	std::cout << "Loading models from: " << model_dir << std::endl;
	PrefixCacheInferencer inferencer(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // use_cuda
		0       // device_id
	);
	std::cout << "✓ Models loaded" << std::endl;
	std::cout << std::endl;

	// Same TGN as Python test
	std::string tgn_text = "[Board 5x5]\n\n1. Pass";
	std::cout << "Test TGN: " << tgn_text << std::endl;

	// Tokenize - same as Python
	TGNTokenizer tokenizer;
	auto tokens_raw = tokenizer.encode(tgn_text, 256, false, false, false, false);

	// Add START token (1) at beginning
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START token
	for (auto t : tokens_raw)
	{
		prefix_tokens.push_back(t);
	}

	int n = static_cast<int>(prefix_tokens.size());
	std::cout << "Prefix tokens (" << n << "): [";
	for (size_t i = 0; i < prefix_tokens.size() && i < 25; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << prefix_tokens[i];
	}
	if (prefix_tokens.size() > 25) std::cout << ", ...";
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Expected from Python:
	// [1, 91, 66, 111, 97, 114, 100, 32, 53, 120, 53, 93, 10, 10, 49, 46, 32, 80, 97, 115, 115]
	// Length: 21

	std::cout << "============================================================================" << std::endl;
	std::cout << "Step 1: Compute Prefix Cache" << std::endl;
	std::cout << "============================================================================" << std::endl;

	inferencer.compute_prefix_cache(prefix_tokens, 1, n);
	auto dims = inferencer.get_cache_dimensions();

	std::cout << "Cache dimensions:" << std::endl;
	std::cout << "  Layers: " << dims.num_layers << std::endl;
	std::cout << "  Heads: " << dims.num_heads << std::endl;
	std::cout << "  Prefix len: " << dims.prefix_len << std::endl;
	std::cout << "  Head dim: " << dims.head_dim << std::endl;

	// Print KV cache values for comparison with Python
	const auto& cached_keys = inferencer.get_cached_keys();
	std::cout << std::endl;
	std::cout << "KV cache layer 0 key (first 5 values at position 0, head 0):" << std::endl;
	std::cout << "  [";
	for (int i = 0; i < 5 && i < dims.head_dim; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << std::fixed << std::setprecision(6) << cached_keys[0][i];
	}
	std::cout << "]" << std::endl;

	// Expected from Python ONNX:
	// [-9.104119, 8.767575, 10.63566, -9.13075, 8.939606]

	std::cout << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << "Step 2: VALUE Token Inference with Cache" << std::endl;
	std::cout << "============================================================================" << std::endl;

	// VALUE token ID = 3
	int value_token_id = 3;

	// Get hidden states using cached inference
	std::vector<int64_t> value_ids = {value_token_id};
	std::vector<float> value_mask = {1.0f};  // Single token

	auto hidden_states = inferencer.evaluate_with_cache(value_ids, value_mask, 1, 1);
	int hidden_dim = static_cast<int>(hidden_states.size());

	std::cout << "Hidden states size: " << hidden_dim << std::endl;
	std::cout << "Hidden states first 5: [";
	for (int i = 0; i < 5 && i < hidden_dim; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << std::fixed << std::setprecision(6) << hidden_states[i];
	}
	std::cout << "]" << std::endl;

	// Expected from Python ONNX:
	// [-0.03880358, 0.16187283, 1.7240716, -1.0846195, 1.2041575]

	// Calculate hidden norm
	float norm = 0.0f;
	for (auto v : hidden_states)
	{
		norm += v * v;
	}
	norm = std::sqrt(norm);
	std::cout << "Hidden states norm: " << std::fixed << std::setprecision(6) << norm << std::endl;

	// Expected norm from Python ONNX: ~8.302364

	std::cout << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << "Step 3: Run Value Head" << std::endl;
	std::cout << "============================================================================" << std::endl;

	float value = inferencer.value_inference_with_cache(value_token_id);
	std::cout << "VALUE: " << std::fixed << std::setprecision(6) << value << std::endl;

	// Expected from Python ONNX: -0.085773

	std::cout << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << "COMPARISON" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << "Expected (Python ONNX): -0.085773" << std::endl;
	std::cout << "C++ result:             " << std::fixed << std::setprecision(6) << value << std::endl;
	std::cout << "Difference:             " << std::fixed << std::setprecision(6) << std::abs(value - (-0.085773)) << std::endl;

	if (std::abs(value - (-0.085773)) < 0.001)
	{
		std::cout << std::endl;
		std::cout << "✓ VALUES MATCH!" << std::endl;
	}
	else
	{
		std::cout << std::endl;
		std::cout << "✗ VALUES DO NOT MATCH" << std::endl;
		std::cout << std::endl;
		std::cout << "Debug: Compare intermediate values with Python ONNX test" << std::endl;
	}

	return 0;
}
