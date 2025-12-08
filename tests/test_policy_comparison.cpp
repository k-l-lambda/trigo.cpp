/**
 * Test: Compare C++ policy inference with TypeScript tree agent
 *
 * This test verifies that C++ properly runs policy head and produces
 * the same move scores as the TypeScript implementation.
 */

#include "../include/trigo_game.hpp"
#include "../include/prefix_cache_inferencer.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/trigo_coords.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>


/**
 * Test policy inference for a simple position
 */
void test_simple_position(const std::string& model_dir)
{
	std::cout << "==============================================================================" << std::endl;
	std::cout << "Test: Policy Inference Comparison (C++ vs TypeScript)" << std::endl;
	std::cout << "==============================================================================" << std::endl;
	std::cout << std::endl;

	// Initialize inferencer
	std::cout << "Loading model from: " << model_dir << std::endl;
	trigo::PrefixCacheInferencer inferencer(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // CPU mode
		0
	);
	std::cout << "Model loaded successfully" << std::endl << std::endl;

	// Create a simple test position: empty 5×5 board
	trigo::TrigoGame game({5, 5, 1});
	std::cout << "Test Position: Empty 5×5 board" << std::endl;
	std::cout << "Valid moves: " << game.valid_move_positions().size() << std::endl;
	std::cout << std::endl;

	// Get all valid moves
	auto valid_moves = game.valid_move_positions();
	auto board_shape = game.get_shape();

	// Convert game to tokens
	std::string tgn_text = trigo::game_to_tgn(game, false);
	std::cout << "TGN: " << tgn_text << std::endl;

	trigo::TGNTokenizer tokenizer;
	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START token
	prefix_tokens.insert(prefix_tokens.end(), encoded.begin(), encoded.end());

	std::cout << "Prefix length: " << prefix_tokens.size() << " tokens" << std::endl;
	std::cout << std::endl;

	// Build candidate sequences for first 5 moves (for comparison)
	// IMPORTANT: Only include move tokens, NOT prefix (prefix is already in cache)
	std::vector<std::vector<int64_t>> candidate_sequences;
	std::vector<std::string> move_names;

	int num_test_moves = std::min(5, static_cast<int>(valid_moves.size()));
	for (int i = 0; i < num_test_moves; i++)
	{
		std::string coord = trigo::encode_ab0yz(valid_moves[i], board_shape);
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		// Create sequence with ONLY move tokens (divergent part)
		std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);
		move_names.push_back(coord);

		// Debug: Print move sequence
		std::cout << "Move " << coord << " tokens (" << seq.size() << "): ";
		for (size_t j = 0; j < seq.size(); j++)
		{
			std::cout << seq[j] << " ";
		}
		std::cout << std::endl;
	}

	// Build prefix tree
	trigo::PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);

	std::cout << "Tree structure:" << std::endl;
	std::cout << "  Nodes: " << tree_structure.num_nodes << std::endl;
	std::cout << "  Candidates: " << candidate_sequences.size() << std::endl;
	std::cout << "  Evaluated IDs: ";
	for (size_t i = 0; i < std::min(size_t(30), tree_structure.evaluated_ids.size()); i++)
	{
		std::cout << tree_structure.evaluated_ids[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "  Move to leaf: ";
	for (size_t i = 0; i < tree_structure.move_to_leaf.size(); i++)
	{
		std::cout << move_names[i] << "→" << tree_structure.move_to_leaf[i] << " ";
	}
	std::cout << std::endl;

	// Print first few rows of mask for debugging
	std::cout << "  Evaluated mask (first 5 rows):" << std::endl;
	for (int i = 0; i < std::min(5, tree_structure.num_nodes); i++)
	{
		std::cout << "    Row " << i << ": ";
		for (int j = 0; j < std::min(20, tree_structure.num_nodes); j++)
		{
			std::cout << tree_structure.evaluated_mask[i * tree_structure.num_nodes + j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	// STEP 1: Compute prefix cache
	std::cout << "[STEP 1] Computing prefix cache..." << std::endl;
	inferencer.compute_prefix_cache(prefix_tokens, 1, static_cast<int>(prefix_tokens.size()));
	std::cout << "  ✓ Cache computed" << std::endl;

	// Debug: Get cache dimensions and print sample
	auto cache_dims = inferencer.get_cache_dimensions();
	std::cout << "  Cache dimensions: " << cache_dims.num_layers << " layers, "
	          << cache_dims.num_heads << " heads, " << cache_dims.head_dim << " head_dim" << std::endl;
	// Note: Can't easily print cache contents from here without exposing internals
	std::cout << std::endl;

	// STEP 2: Evaluate with cache to get hidden states
	std::cout << "[STEP 2] Evaluating with cache..." << std::endl;
	auto hidden_states = inferencer.evaluate_with_cache(
		tree_structure.evaluated_ids,
		tree_structure.evaluated_mask,
		1,
		tree_structure.num_nodes
	);
	int hidden_dim = static_cast<int>(hidden_states.size()) / tree_structure.num_nodes;
	std::cout << "  ✓ Hidden states computed" << std::endl;
	std::cout << "  Hidden dim: " << hidden_dim << std::endl;
	std::cout << "  Total size: " << hidden_states.size() << std::endl;

	// Debug: Print first 10 dims of hidden state at position 18
	std::cout << "  Hidden states at position 18 (first 10 dims): ";
	for (int d = 0; d < 10; d++)
	{
		std::cout << hidden_states[18 * hidden_dim + d] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	// STEP 3: Run policy head (PROPERLY!)
	std::cout << "[STEP 3] Running policy head properly..." << std::endl;
	auto logits = inferencer.policy_inference_from_hidden(
		hidden_states,
		1,  // batch_size
		tree_structure.num_nodes,
		hidden_dim
	);
	int vocab_size = static_cast<int>(logits.size()) / tree_structure.num_nodes;
	std::cout << "  ✓ Policy logits computed" << std::endl;
	std::cout << "  Vocab size: " << vocab_size << std::endl;
	std::cout << "  Logits size: " << logits.size() << std::endl;
	std::cout << std::endl;

	// Extract move logits (CORRECT METHOD!)
	std::cout << "[CORRECT METHOD] Using proper policy head logits:" << std::endl;
	for (size_t i = 0; i < candidate_sequences.size(); i++)
	{
		int leaf_pos = tree_structure.move_to_leaf[i];
		const auto& move_seq = candidate_sequences[i];
		int64_t last_token = move_seq.back();

		// Get logit for the last token at this leaf position
		int logit_idx = leaf_pos * vocab_size + static_cast<int>(last_token);
		float logit = logits[logit_idx];

		std::cout << "  " << move_names[i] << ": leaf_pos=" << leaf_pos
		          << ", last_token=" << last_token
		          << ", evaluated_ids[" << leaf_pos << "]=" << tree_structure.evaluated_ids[leaf_pos]
		          << ", logit=" << std::fixed << std::setprecision(4) << logit << std::endl;
	}
	std::cout << std::endl;

	// CURRENT BROKEN METHOD: Use magnitude as proxy
	std::cout << "[CURRENT METHOD] Using magnitude heuristic (INCORRECT!):" << std::endl;
	for (size_t i = 0; i < candidate_sequences.size(); i++)
	{
		int leaf_pos = tree_structure.move_to_leaf[i];
		float score = 0.0f;
		for (int d = 0; d < hidden_dim; d++)
		{
			int idx = leaf_pos * hidden_dim + d;
			score += std::abs(hidden_states[idx]);
		}
		score /= hidden_dim;

		std::cout << "  " << move_names[i] << ": " << std::fixed << std::setprecision(4) << score << std::endl;
	}
	std::cout << std::endl;

	// Save data for TypeScript comparison
	std::ofstream out_file("/tmp/policy_test_input.txt");
	out_file << "TGN: " << tgn_text << std::endl;
	out_file << "Prefix length: " << prefix_tokens.size() << std::endl;
	out_file << "Moves: ";
	for (const auto& move : move_names)
	{
		out_file << move << " ";
	}
	out_file << std::endl;
	out_file << "Model: " << model_dir << std::endl;
	out_file.close();

	std::cout << "Test data saved to: /tmp/policy_test_input.txt" << std::endl;
	std::cout << std::endl;
	std::cout << "Next steps:" << std::endl;
	std::cout << "1. Run TypeScript tree agent with this position" << std::endl;
	std::cout << "2. Compare move scores" << std::endl;
	std::cout << "3. Implement proper policy head inference in C++" << std::endl;
	std::cout << std::endl;
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	try
	{
		test_simple_position(model_dir);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
