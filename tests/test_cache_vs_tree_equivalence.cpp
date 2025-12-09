/**
 * Test Equivalence: Prefix Cache vs Tree Model
 *
 * Verify that prefix cache model (4 separate models) produces
 * the same policy logits as the tree model (single model).
 *
 * Both models are exported from the same trained checkpoint.
 */

#include "trigo_game.hpp"
#include "prefix_cache_inferencer.hpp"
#include "prefix_tree_builder.hpp"
#include "tgn_tokenizer.hpp"
#include "tgn_utils.hpp"
#include "trigo_coords.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace trigo;


int main()
{
	std::cout << "=== Prefix Cache vs Tree Model Equivalence Test ===" << std::endl;
	std::cout << std::endl;

	// Use lr500 model (ep0019) - freshly exported with dynamic shapes
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached";

	// Test prefix: after move 1
	std::string tgn_prefix = "[Board 5x5]\n\n1. a0 ";

	std::cout << "Test prefix:" << std::endl;
	std::cout << tgn_prefix << std::endl;
	std::cout << std::endl;

	// Setup game state
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();
	game.drop(Position{2, 0, 0});  // a0

	// Get all valid moves (25 moves for 5x5)
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << std::endl;

	// Initialize prefix cache inferencer
	std::cout << "Loading prefix cache models from:" << std::endl;
	std::cout << "  " << model_dir << std::endl;
	auto inferencer = std::make_shared<PrefixCacheInferencer>(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // CPU
		0
	);
	std::cout << std::endl;

	// Tokenize prefix
	TGNTokenizer tokenizer;
	auto encoded = tokenizer.encode(tgn_prefix, 8192, false, false, false, false);
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START token
	prefix_tokens.insert(prefix_tokens.end(), encoded.begin(), encoded.end());

	std::cout << "Prefix tokens: " << prefix_tokens.size() << std::endl;
	std::cout << std::endl;

	// Compute prefix cache
	inferencer->compute_prefix_cache(prefix_tokens, 1, static_cast<int>(prefix_tokens.size()));
	std::cout << "Prefix cache computed" << std::endl;
	std::cout << std::endl;

	// Build candidate sequences (move tokens only, not prefix)
	std::vector<std::vector<int64_t>> candidate_sequences;
	for (const auto& move : valid_moves)
	{
		std::string coord = encode_ab0yz(move, shape);
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);
		std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);
	}

	// Add PASS
	auto pass_tokens = tokenizer.encode("PASS", 2048, false, false, false, false);
	std::vector<int64_t> pass_seq(pass_tokens.begin(), pass_tokens.end());
	candidate_sequences.push_back(pass_seq);

	std::cout << "Candidate sequences: " << candidate_sequences.size() << std::endl;
	std::cout << std::endl;

	// Build prefix tree
	PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);

	std::cout << "Tree structure:" << std::endl;
	std::cout << "  Nodes: " << tree_structure.num_nodes << std::endl;
	std::cout << std::endl;

	// Evaluate with cache
	auto hidden_states = inferencer->evaluate_with_cache(
		tree_structure.evaluated_ids,
		tree_structure.evaluated_mask,
		1,
		tree_structure.num_nodes
	);

	// Run policy head
	int hidden_dim = static_cast<int>(hidden_states.size()) / tree_structure.num_nodes;
	auto logits = inferencer->policy_inference_from_hidden(
		hidden_states,
		1,
		tree_structure.num_nodes,
		hidden_dim
	);

	// Extract logits for each move
	int vocab_size = static_cast<int>(logits.size()) / tree_structure.num_nodes;

	std::cout << "=== Prefix Cache Model Results ===" << std::endl;
	std::cout << std::endl;
	std::cout << std::setw(8) << "Move" << " | "
	          << std::setw(10) << "Leaf Pos" << " | "
	          << std::setw(12) << "Last Token" << " | "
	          << std::setw(12) << "Logit" << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	for (size_t i = 0; i < candidate_sequences.size(); i++)
	{
		int leaf_pos = tree_structure.move_to_leaf[i];
		const auto& move_seq = candidate_sequences[i];
		int64_t last_token = move_seq.back();

		int logit_idx = leaf_pos * vocab_size + static_cast<int>(last_token);
		float logit = logits[logit_idx];

		std::string move_str;
		if (i < valid_moves.size())
		{
			move_str = encode_ab0yz(valid_moves[i], shape);
		}
		else
		{
			move_str = "PASS";
		}

		std::cout << std::setw(8) << move_str << " | "
		          << std::setw(10) << leaf_pos << " | "
		          << std::setw(12) << last_token << " | "
		          << std::fixed << std::setprecision(6)
		          << std::setw(12) << logit << std::endl;
	}
	std::cout << std::endl;

	// Value inference
	float value = inferencer->value_inference_with_cache(3);

	// Convert to current player perspective
	Stone current_player = game.get_current_player();
	if (current_player == Stone::Black)
	{
		value = -value;
	}

	std::cout << "=== Value Inference ===" << std::endl;
	std::cout << "Raw value (White perspective): " << value << std::endl;
	std::cout << "Current player: " << (current_player == Stone::White ? "White" : "Black") << std::endl;
	std::cout << "Value (current player perspective): " << value << std::endl;
	std::cout << std::endl;

	std::cout << "=== Compare with TypeScript ===" << std::endl;
	std::cout << "Run: cd /home/camus/work/trigo/trigo-web && npm exec tsx tests/testTreeModelInference.ts" << std::endl;
	std::cout << std::endl;
	std::cout << "Expected: Logits should match within floating-point precision" << std::endl;

	return 0;
}
