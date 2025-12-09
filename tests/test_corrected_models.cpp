/**
 * Test C++ Prefix Cache with Corrected ONNX Models
 *
 * This test uses the newly exported ONNX models (with corrected attention mask)
 * and verifies that C++ inference now matches PyTorch.
 */

#include "../include/prefix_cache_inferencer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_tokenizer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>


using namespace trigo;


int main()
{
	std::cout << "=== C++ Prefix Cache Test with Corrected Models ===" << std::endl;
	std::cout << std::endl;

	// Model paths for lr500 checkpoint (ep0019) - newly exported with corrected mask
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached";

	// Test prefix - same as PyTorch test
	std::string tgn_prefix = "[Board 5x5]\n\n1. a0 ";

	std::cout << "Test prefix: " << tgn_prefix << std::endl;
	std::cout << std::endl;

	// Tokenize
	TGNTokenizer tokenizer;
	auto tgn_tokens = tokenizer.encode(tgn_prefix, 8192, false, false, false, false);

	// Add START token
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START
	prefix_tokens.insert(prefix_tokens.end(), tgn_tokens.begin(), tgn_tokens.end());

	std::cout << "Prefix tokens (" << prefix_tokens.size() << "): ";
	for (size_t i = 0; i < prefix_tokens.size(); i++)
	{
		std::cout << prefix_tokens[i];
		if (i < prefix_tokens.size() - 1) std::cout << ", ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	// Load models
	std::cout << "Loading models..." << std::endl;

	try
	{
		PrefixCacheInferencer inferencer(
			model_dir + "/base_model_prefix.onnx",
			model_dir + "/base_model_eval_cached.onnx",
			model_dir + "/policy_head.onnx",
			model_dir + "/value_head.onnx",
			false,  // CPU
			0
		);

		std::cout << "✓ Models loaded successfully" << std::endl;
		std::cout << std::endl;

		// Build candidate sequences (same as PyTorch test)
		std::vector<std::vector<int64_t>> candidate_sequences;
		std::vector<std::string> move_names = {
			"aa", "ab", "a0", "az", "ay",
			"ba", "bb", "b0", "bz", "by",
			"0b", "00", "0z", "0y",
			"za", "zb", "z0", "zz", "zy",
			"ya", "yb", "y0", "yz", "yy",
			"PASS"
		};

		for (const auto& move : move_names)
		{
			auto move_tokens = tokenizer.encode(move, 2048, false, false, false, false);
			std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end());
			candidate_sequences.push_back(seq);
		}

		// Build prefix tree
		std::cout << "Building prefix tree..." << std::endl;
		PrefixTreeBuilder tree_builder;
		auto tree = tree_builder.build_tree(candidate_sequences);
		std::cout << "  Num nodes: " << tree.num_nodes << std::endl;
		std::cout << std::endl;

		// Run inference
		std::cout << "Running C++ prefix cache inference..." << std::endl;
		auto results = inferencer.infer(
			prefix_tokens,
			tree.evaluated_ids,
			tree.evaluated_mask
		);

		std::cout << "✓ Inference complete" << std::endl;
		std::cout << std::endl;

		// Print results for first 10 moves
		std::cout << "=== C++ Policy Inference Results ===" << std::endl;
		std::cout << std::endl;

		std::cout << std::fixed << std::setprecision(6);
		std::cout << "    Move |   Leaf Pos |   Last Token |        Logit" << std::endl;
		std::cout << "--------------------------------------------------" << std::endl;

		for (size_t i = 0; i < std::min(size_t(10), move_names.size()); i++)
		{
			std::cout << std::setw(8) << move_names[i] << " | ";
			std::cout << std::setw(10) << tree.move_to_leaf[i] << " | ";
			std::cout << std::setw(12) << tree.last_token_ids[i] << " | ";
			std::cout << std::setw(12) << results[i].policy_logit << std::endl;
		}

		std::cout << std::endl;

		// Highlight the "aa" move
		std::cout << "KEY COMPARISON:" << std::endl;
		std::cout << "  C++ logit for \"aa\": " << results[0].policy_logit << std::endl;
		std::cout << "  Expected (PyTorch): 4.147368" << std::endl;
		std::cout << std::endl;

		float diff = std::abs(results[0].policy_logit - 4.147368f);
		if (diff < 0.01f)
		{
			std::cout << "✓ SUCCESS! C++ matches PyTorch (diff = " << diff << ")" << std::endl;
		}
		else
		{
			std::cout << "✗ MISMATCH! C++ differs from PyTorch (diff = " << diff << ")" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
