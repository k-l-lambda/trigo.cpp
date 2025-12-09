/**
 * Compare C++ Policy Inference with TypeScript ONNX Inference
 *
 * Test prefix from actual self-play game (before first PASS)
 * Uses hardcoded tree structure from PyTorch test to ensure exact comparison
 */

#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_cache_inferencer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>


using namespace trigo;


int main()
{
	std::cout << "=== C++ vs TypeScript Policy Inference Comparison ===" << std::endl;
	std::cout << std::endl;

	// Model configuration - USE CORRECTED LR500 MODEL
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached";
	BoardShape shape{5, 5, 1};

	// Test prefix - MATCH PyTorch test ("a0" not "y0")
	// [Board 5x5]
	//
	// 1. a0
	TrigoGame game(shape);
	game.start_game();

	// Play move: a0 (instead of y0)
	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};
	auto pos_vec = decode_ab0yz("a0", shape_vec);
	Position move_a0{pos_vec[0], pos_vec[1], pos_vec[2]};
	game.drop(move_a0);

	// Get TGN representation
	std::string tgn_text = game_to_tgn(game, false);
	std::cout << "TGN prefix:" << std::endl;
	std::cout << tgn_text << std::endl;
	std::cout << std::endl;

	// Tokenize
	TGNTokenizer tokenizer;
	auto tgn_tokens = tokenizer.encode(tgn_text, 8192, false, false, false, false);

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

	// Use EXACT same tree structure as PyTorch test (hardcoded from TypeScript)
	// This ensures we're testing the same inference scenario
	std::vector<int64_t> evaluated_ids = {97, 98, 48, 122, 121, 80, 97, 115};
	std::vector<float> mask_flat = {
		1,0,0,0,0,0,0,0,
		0,1,0,0,0,0,0,0,
		0,0,1,0,0,0,0,0,
		0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,0,0,
		0,0,0,0,0,1,0,0,
		0,0,0,0,0,0,1,1,
		0,0,0,0,0,0,1,1,1
	};
	int num_nodes = 8;

	std::cout << "Tree structure (hardcoded from PyTorch test):" << std::endl;
	std::cout << "  Num nodes: " << num_nodes << std::endl;
	std::cout << "  Evaluated IDs: [";
	for (int i = 0; i < num_nodes; i++)
	{
		std::cout << evaluated_ids[i];
		if (i < num_nodes - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Load model
	std::cout << "Loading models..." << std::endl;
	PrefixCacheInferencer inferencer(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // CPU
		0
	);
	std::cout << "Models loaded" << std::endl;
	std::cout << std::endl;

	// Compute prefix cache
	int prefix_len = static_cast<int>(prefix_tokens.size());
	inferencer.compute_prefix_cache(prefix_tokens, 1, prefix_len);

	// Evaluate with cache using hardcoded tree structure
	auto hidden_states = inferencer.evaluate_with_cache(
		evaluated_ids,
		mask_flat,
		1,  // batch_size
		num_nodes
	);

	// Run policy head
	int hidden_dim = static_cast<int>(hidden_states.size()) / num_nodes;
	auto logits = inferencer.policy_inference_from_hidden(
		hidden_states,
		1,  // batch_size
		num_nodes,
		hidden_dim
	);

	int vocab_size = 128;

	// Extract and print results - match PyTorch test format
	std::cout << "=== C++ Policy Inference Results ===" << std::endl;
	std::cout << std::endl;

	// Move names and mappings from PyTorch test
	std::vector<std::string> moves = {"aa", "ab", "a0", "az", "ay", "ba", "bb", "b0", "bz", "by",
	                                   "0b", "00", "0z", "0y", "za", "zb", "z0", "zz", "zy",
	                                   "ya", "yb", "y0", "yz", "yy", "PASS"};
	std::vector<int> move_to_leaf = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 7};
	std::vector<int64_t> last_tokens = {
		97, 98, 48, 122, 121,  // aa, ab, a0, az, ay
		97, 98, 48, 122, 121,  // ba, bb, b0, bz, by
		98, 48, 122, 121,      // 0b, 00, 0z, 0y
		97, 98, 48, 122, 121,  // za, zb, z0, zz, zy
		97, 98, 48, 122, 121,  // ya, yb, y0, yz, yy
		83                      // PASS -> 'S'
	};

	std::cout << std::setw(10) << "Move" << " | "
	          << std::setw(10) << "Leaf Pos" << " | "
	          << std::setw(15) << "Last Token" << " | "
	          << std::setw(12) << "Logit" << std::endl;
	std::cout << std::string(65, '-') << std::endl;

	for (size_t i = 0; i < moves.size(); i++)
	{
		int leaf_pos = move_to_leaf[i];
		int64_t last_token = last_tokens[i];

		int logit_idx = leaf_pos * vocab_size + static_cast<int>(last_token);
		float logit = logits[logit_idx];

		std::cout << std::setw(10) << moves[i] << " | "
		          << std::setw(10) << leaf_pos << " | "
		          << std::setw(3) << last_token << " ('" << static_cast<char>(last_token) << "')      | "
		          << std::setw(12) << std::fixed << std::setprecision(6) << logit
		          << std::endl;
	}
	std::cout << std::endl;

	// Highlight key comparison
	std::cout << "KEY COMPARISON:" << std::endl;
	float aa_logit = logits[0 * vocab_size + 97];  // leaf_pos=0, token=97
	std::cout << "  C++ logit for \"aa\": " << std::fixed << std::setprecision(6) << aa_logit << std::endl;
	std::cout << "  Expected (PyTorch): 4.092103" << std::endl;
	std::cout << std::endl;

	float diff = std::abs(aa_logit - 4.092103f);
	if (diff < 0.01f)
	{
		std::cout << "✓ SUCCESS! C++ matches PyTorch (diff = " << diff << ")" << std::endl;
	}
	else
	{
		std::cout << "✗ MISMATCH! C++ differs from PyTorch (diff = " << diff << ")" << std::endl;
	}
	std::cout << std::endl;

	// Also compute value
	std::cout << "=== C++ Value Inference ===" << std::endl;
	float value = inferencer.value_inference_with_cache(3);  // VALUE token
	std::cout << "Value: " << value << std::endl;
	std::cout << std::endl;

	std::cout << "=== Compare with TypeScript output ===" << std::endl;

	return 0;
}
