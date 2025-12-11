/**
 * Test KV cache values comparison with PyTorch
 *
 * Output KV cache tensor values for comparison with:
 * /home/camus/work/trigoRL/tests/test_pytorch_kv_direct.py
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
	std::cout << "KV Cache Comparison Test (C++ ONNX)" << std::endl;
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

	// Same prefix as PyTorch test
	std::string tgn_prefix = "[Board 5x5]\n\n1. ";
	std::cout << "Test prefix: '" << tgn_prefix << "'" << std::endl;

	// Tokenize - same as PyTorch
	TGNTokenizer tokenizer;
	auto prefix_tensor = tokenizer.encode(tgn_prefix, 256, false, false, false, false);

	// Add START token (1) at beginning
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START token
	for (auto t : prefix_tensor)
	{
		prefix_tokens.push_back(t);
	}

	std::cout << "Prefix tokens (" << prefix_tokens.size() << "): [";
	for (size_t i = 0; i < prefix_tokens.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << prefix_tokens[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Compute prefix cache
	std::cout << "Computing prefix cache..." << std::endl;
	inferencer.compute_prefix_cache(prefix_tokens, 1, static_cast<int>(prefix_tokens.size()));
	std::cout << std::endl;

	// Print KV cache values for comparison
	std::cout << "KV Cache values (first 4 elements of each tensor):" << std::endl;
	auto dims = inferencer.get_cache_dimensions();
	std::cout << "  Layers: " << dims.num_layers << std::endl;
	std::cout << "  Heads: " << dims.num_heads << std::endl;
	std::cout << "  Seq len: " << dims.seq_len << std::endl;
	std::cout << "  Head dim: " << dims.head_dim << std::endl;
	std::cout << std::endl;

	// Access cached tensors
	const auto& cached_keys = inferencer.get_cached_keys();
	const auto& cached_values = inferencer.get_cached_values();

	for (int layer = 0; layer < dims.num_layers; layer++)
	{
		const auto& key = cached_keys[layer];
		const auto& value = cached_values[layer];

		// Print [0,0,0,:4] - same format as PyTorch
		std::cout << "  Layer " << layer << " key[0,0,0,:4]: [";
		for (int i = 0; i < 4 && i < dims.head_dim; i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << std::fixed << std::setprecision(6) << key[i];
		}
		std::cout << "]" << std::endl;

		std::cout << "  Layer " << layer << " value[0,0,0,:4]: [";
		for (int i = 0; i < 4 && i < dims.head_dim; i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << std::fixed << std::setprecision(6) << value[i];
		}
		std::cout << "]" << std::endl;
	}
	std::cout << std::endl;

	// Test policy inference - get logits for "aa" (first token = 97)
	std::cout << "============================================================================" << std::endl;
	std::cout << "Policy Logit Test" << std::endl;
	std::cout << "============================================================================" << std::endl;

	// Create tree for single move "aa" (excluding last token)
	auto aa_tokens = tokenizer.encode("aa", 256, false, false, false, false);
	std::cout << "\"aa\" tokens: [";
	for (size_t i = 0; i < aa_tokens.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << aa_tokens[i];
	}
	std::cout << "]" << std::endl;

	// For tree building, we exclude last token
	std::vector<int64_t> tree_tokens;
	if (aa_tokens.size() > 1)
	{
		tree_tokens.assign(aa_tokens.begin(), aa_tokens.end() - 1);
	}
	else if (!aa_tokens.empty())
	{
		tree_tokens.push_back(aa_tokens[0]);
	}

	std::cout << "Tree tokens (excluding last): [";
	for (size_t i = 0; i < tree_tokens.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree_tokens[i];
	}
	std::cout << "]" << std::endl;

	// Create trivial mask for single token
	std::vector<float> mask = {1.0f};

	// Run eval with cache
	auto hidden_states = inferencer.evaluate_with_cache(tree_tokens, mask, 1, 1);
	std::cout << "Hidden states size: " << hidden_states.size() << std::endl;

	// Run policy head
	int hidden_dim = static_cast<int>(hidden_states.size());  // For single token
	auto logits = inferencer.policy_inference_from_hidden(hidden_states, 1, 1, hidden_dim);

	std::cout << "Logits size: " << logits.size() << std::endl;

	// Print logit for token 97 ('a')
	if (logits.size() > 97)
	{
		std::cout << "Logit for token 97 ('a'): " << std::fixed << std::setprecision(6) << logits[97] << std::endl;
	}

	std::cout << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << "Expected from PyTorch:" << std::endl;
	std::cout << "  Logit for 'a' (token 97): ~4.103072" << std::endl;
	std::cout << "============================================================================" << std::endl;

	return 0;
}
