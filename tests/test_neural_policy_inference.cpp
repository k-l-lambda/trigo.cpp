/**
 * Complete NeuralPolicy Implementation with Real Inference
 *
 * This is a standalone implementation to test the full inference pipeline.
 */

#include "../include/trigo_game.hpp"
#include "../include/shared_model_inferencer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

using namespace trigo;


/**
 * Convert game history to TGN format token sequence
 * Uses shared TGN generation logic
 */
std::vector<int64_t> game_to_tokens(const TrigoGame& game)
{
	// Generate TGN text using shared utility
	std::string tgn_text = game_to_tgn(game, false);

	// Tokenize the complete TGN text
	TGNTokenizer tokenizer;
	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

	// Add START token at beginning
	std::vector<int64_t> tokens;
	tokens.push_back(1);  // START token
	tokens.insert(tokens.end(), encoded.begin(), encoded.end());

	return tokens;
}


/**
 * Test neural policy with real inference
 */
int main()
{
	std::cout << "\n=== NeuralPolicy Full Inference Test ===\n\n";

	// Create game
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Make a few moves
	game.drop(Position{2, 2, 2});  // 000
	game.drop(Position{1, 2, 2});  // a00

	std::cout << "Game state: 2 moves played\n";
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << "\n\n";

	// Get valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << "\n";

	// Limit to first 10 for testing
	if (valid_moves.size() > 10)
	{
		valid_moves.resize(10);
	}

	std::cout << "Testing with " << valid_moves.size() << " candidate moves\n\n";

	// Convert game state to tokens
	auto prefix_tokens = game_to_tokens(game);
	std::cout << "Prefix tokens: " << prefix_tokens.size() << " tokens\n";

	// Build token sequences for each candidate move
	TGNTokenizer tokenizer;
	auto board_shape = game.get_shape();

	std::vector<std::vector<int64_t>> candidate_sequences;
	for (const auto& move : valid_moves)
	{
		std::vector<int64_t> seq = prefix_tokens;

		// Encode move (no padding, no special tokens)
		std::string coord = encode_ab0yz(move, board_shape);
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		seq.insert(seq.end(), move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);
	}

	std::cout << "Built " << candidate_sequences.size() << " candidate sequences\n";

	// Build prefix tree
	PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);

	std::cout << "Prefix tree: " << tree_structure.num_nodes << " nodes, "
	          << tree_structure.num_moves << " moves\n";
	std::cout << "move_to_leaf: [";
	for (size_t i = 0; i < std::min(size_t(5), tree_structure.move_to_leaf.size()); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree_structure.move_to_leaf[i];
	}
	std::cout << "...]\n\n";

	// Load model
	std::string model_path = "/home/camus/work/trigo.cpp/models/trained_shared";
	std::cout << "Loading ONNX models from: " << model_path << "\n";

	SharedModelInferencer inferencer(
		model_path + "/base_model.onnx",
		model_path + "/policy_head.onnx",
		model_path + "/value_head.onnx",
		false,  // use_gpu = false (CPU for testing)
		0
	);

	std::cout << "Models loaded successfully\n\n";

	// Run policy inference
	std::cout << "Running policy inference...\n";

	int prefix_len = static_cast<int>(prefix_tokens.size());
	int eval_len = tree_structure.num_nodes;

	try
	{
		auto logits = inferencer.policy_inference(
			prefix_tokens,
			tree_structure.evaluated_ids,
			tree_structure.evaluated_mask,
			1,  // batch_size
			prefix_len,
			eval_len
		);

		std::cout << "Inference completed!\n";
		std::cout << "Logits size: " << logits.size() << "\n";
		std::cout << "Expected size: " << (eval_len + 1) * 128 << " (" << (eval_len + 1) << " positions × 128 vocab)\n\n";

		// Extract move probabilities
		std::cout << "Move probabilities:\n";

		std::vector<float> move_logits;
		for (size_t i = 0; i < valid_moves.size(); i++)
		{
			int leaf_pos = tree_structure.move_to_leaf[i];

			// Get the last token of this move
			const auto& move_seq = candidate_sequences[i];
			int64_t last_token = move_seq.back();

			// Get logit for this token at this position
			// logits shape: [eval_len+1, vocab_size]
			int logit_idx = leaf_pos * 128 + static_cast<int>(last_token);
			float logit = logits[logit_idx];

			move_logits.push_back(logit);
		}

		// Apply softmax
		float max_logit = *std::max_element(move_logits.begin(), move_logits.end());
		std::vector<float> exp_vals(move_logits.size());
		float sum = 0.0f;

		for (size_t i = 0; i < move_logits.size(); i++)
		{
			exp_vals[i] = std::exp(move_logits[i] - max_logit);
			sum += exp_vals[i];
		}

		std::vector<float> probs(move_logits.size());
		for (size_t i = 0; i < move_logits.size(); i++)
		{
			probs[i] = exp_vals[i] / sum;
		}

		// Print top 5 moves
		std::vector<size_t> indices(probs.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::sort(indices.begin(), indices.end(),
		          [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

		for (size_t i = 0; i < std::min(size_t(5), probs.size()); i++)
		{
			size_t idx = indices[i];
			const auto& move = valid_moves[idx];
			std::string coord = encode_ab0yz(move, board_shape);
			std::cout << "  " << (i+1) << ". " << coord
			          << " - prob: " << (probs[idx] * 100) << "%"
			          << " (logit: " << move_logits[idx] << ")\n";
		}

		std::cout << "\n✓ NeuralPolicy inference test PASSED\n\n";
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
}
