/**
 * Compare C++ Policy Inference with TypeScript ONNX Inference
 *
 * Test prefix from actual self-play game (before first PASS)
 */

#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_cache_inferencer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>


using namespace trigo;


int main()
{
	std::cout << "=== C++ vs TypeScript Policy Inference Comparison ===" << std::endl;
	std::cout << std::endl;

	// Model configuration
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached";
	BoardShape shape{5, 5, 1};

	// Test prefix from game_0.tgn (before first PASS)
	// [Board 5x5]
	//
	// 1. y0
	TrigoGame game(shape);
	game.start_game();

	// Play move: y0
	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};
	auto pos_vec = decode_ab0yz("y0", shape_vec);
	Position move_y0{pos_vec[0], pos_vec[1], pos_vec[2]};
	game.drop(move_y0);

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

	// Get all valid moves for White (current player)
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << std::endl;
	std::cout << std::endl;

	// Build candidate sequences for ALL valid moves + PASS
	std::vector<std::vector<int64_t>> candidate_sequences;
	std::vector<std::string> move_names;

	for (const auto& move : valid_moves)
	{
		std::string coord = encode_ab0yz(move, shape);
		move_names.push_back(coord);

		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);
		std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);
	}

	// Add PASS
	auto pass_tokens = tokenizer.encode("PASS", 2048, false, false, false, false);
	std::vector<int64_t> pass_seq(pass_tokens.begin(), pass_tokens.end());
	candidate_sequences.push_back(pass_seq);
	move_names.push_back("PASS");

	// Build prefix tree
	PrefixTreeBuilder tree_builder;
	auto tree = tree_builder.build_tree(candidate_sequences);

	std::cout << "Tree structure:" << std::endl;
	std::cout << "  Num nodes: " << tree.num_nodes << std::endl;
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

	// Evaluate with cache
	auto hidden_states = inferencer.evaluate_with_cache(
		tree.evaluated_ids,
		tree.evaluated_mask,
		1,  // batch_size
		tree.num_nodes
	);

	// Run policy head
	int hidden_dim = static_cast<int>(hidden_states.size()) / tree.num_nodes;
	auto logits = inferencer.policy_inference_from_hidden(
		hidden_states,
		1,  // batch_size
		tree.num_nodes,
		hidden_dim
	);

	int vocab_size = 128;

	// Extract and print results as table
	std::cout << "=== C++ Policy Inference Results ===" << std::endl;
	std::cout << std::endl;
	std::cout << std::setw(8) << "Move" << " | "
	          << std::setw(10) << "Leaf Pos" << " | "
	          << std::setw(12) << "Last Token" << " | "
	          << std::setw(12) << "Logit" << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	for (size_t i = 0; i < move_names.size(); i++)
	{
		int leaf_pos = tree.move_to_leaf[i];
		const auto& seq = candidate_sequences[i];
		int64_t last_token = seq.back();

		int logit_idx = leaf_pos * vocab_size + static_cast<int>(last_token);
		float logit = logits[logit_idx];

		std::cout << std::setw(8) << move_names[i] << " | "
		          << std::setw(10) << leaf_pos << " | "
		          << std::setw(12) << last_token << " | "
		          << std::setw(12) << std::fixed << std::setprecision(6) << logit
		          << std::endl;
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
