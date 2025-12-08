#include "prefix_cache_inferencer.hpp"
#include "trigo_game.hpp"
#include "trigo_coords.hpp"
#include "tgn_utils.hpp"
#include "tgn_tokenizer.hpp"
#include "prefix_tree_builder.hpp"
#include <iostream>
#include <vector>
#include <chrono>


/**
 * Test Cached Prefix Inference with Real Game
 *
 * Tests the MCTS prefix-reuse pattern:
 * 1. Encode game state (prefix) → compute cache once
 * 2. Evaluate multiple candidate moves with same cache
 */
int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		std::cerr << "\nModel directory should contain:" << std::endl;
		std::cerr << "  - base_model_prefix.onnx" << std::endl;
		std::cerr << "  - base_model_eval_cached.onnx" << std::endl;
		std::cerr << "  - policy_head.onnx" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "==============================================================================" << std::endl;
	std::cout << "Test: Cached Prefix Inference with Real Game" << std::endl;
	std::cout << "==============================================================================" << std::endl;

	// Initialize game
	trigo::TrigoGame game({5, 5, 1});  // 5×5×1 board
	std::cout << "\nInitializing 5×5×1 game..." << std::endl;

	// Play a few moves to create game history
	std::vector<trigo::Position> moves = {
		{1, 1, 0},  // Black
		{3, 3, 0},  // White
		{2, 2, 0},  // Black
		{1, 3, 0},  // White
	};

	std::cout << "Playing moves:" << std::endl;
	for (const auto& move : moves) {
		game.drop(move);
		std::cout << "  " << trigo::encode_ab0yz(move, game.get_shape())
		          << " (" << (game.get_current_player() == trigo::Stone::Black ? "Black" : "White") << ")" << std::endl;
	}

	// Get valid moves for current position
	auto valid_moves = game.valid_move_positions();
	std::cout << "\nValid moves for current position: " << valid_moves.size() << std::endl;
	if (valid_moves.size() > 5) {
		// Limit to 5 moves for testing
		valid_moves.resize(5);
		std::cout << "  (limiting to first 5 for testing)" << std::endl;
	}

	// Initialize inferencer
	std::cout << "\nLoading ONNX models..." << std::endl;
	trigo::PrefixCacheInferencer inferencer(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		"",  // No value head
		false,  // CPU only
		0
	);

	// Initialize tokenizer and tree builder
	trigo::TGNTokenizer tokenizer;
	trigo::PrefixTreeBuilder tree_builder;

	// Convert game history to tokens
	std::cout << "\nConverting game history to tokens..." << std::endl;
	std::string tgn_text = trigo::game_to_tgn(game, false);
	std::cout << "  TGN: " << tgn_text << std::endl;

	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START token
	prefix_tokens.insert(prefix_tokens.end(), encoded.begin(), encoded.end());

	std::cout << "  Prefix length: " << prefix_tokens.size() << " tokens" << std::endl;

	// Build candidate sequences
	std::cout << "\nBuilding candidate sequences for " << valid_moves.size() << " moves..." << std::endl;
	auto board_shape = game.get_shape();
	std::vector<std::vector<int64_t>> candidate_sequences;

	for (const auto& move : valid_moves) {
		std::vector<int64_t> seq = prefix_tokens;

		// Encode move
		std::string coord = trigo::encode_ab0yz(move, board_shape);
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		seq.insert(seq.end(), move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);
	}

	// Build prefix tree
	auto tree_structure = tree_builder.build_tree(candidate_sequences);
	std::cout << "  Tree nodes: " << tree_structure.num_nodes << std::endl;

	int batch_size = 1;
	int prefix_len = static_cast<int>(prefix_tokens.size());
	int eval_len = tree_structure.num_nodes;

	std::cout << "  Batch size: " << batch_size << std::endl;
	std::cout << "  Prefix length: " << prefix_len << std::endl;
	std::cout << "  Evaluated length: " << eval_len << std::endl;

	// STEP 1: Compute prefix cache
	std::cout << "\n[1/2] Computing prefix cache..." << std::endl;
	auto start_prefix = std::chrono::high_resolution_clock::now();

	inferencer.compute_prefix_cache(
		prefix_tokens,
		batch_size,
		prefix_len
	);

	auto end_prefix = std::chrono::high_resolution_clock::now();
	double prefix_time = std::chrono::duration<double, std::milli>(end_prefix - start_prefix).count();

	std::cout << "  ✓ Prefix cache computed in " << prefix_time << " ms" << std::endl;

	// STEP 2: Evaluate with cache
	std::cout << "\n[2/2] Evaluating " << valid_moves.size() << " moves with fixed cache..." << std::endl;
	auto start_eval = std::chrono::high_resolution_clock::now();

	auto hidden_states = inferencer.evaluate_with_cache(
		tree_structure.evaluated_ids,
		tree_structure.evaluated_mask,
		batch_size,
		eval_len
	);

	auto end_eval = std::chrono::high_resolution_clock::now();
	double eval_time = std::chrono::duration<double, std::milli>(end_eval - start_eval).count();

	std::cout << "  ✓ Evaluation completed in " << eval_time << " ms" << std::endl;
	std::cout << "  Hidden states size: " << hidden_states.size() << std::endl;

	// Extract scores for each move
	int hidden_dim = static_cast<int>(hidden_states.size() / (batch_size * eval_len));
	std::cout << "  Hidden dim: " << hidden_dim << std::endl;

	std::cout << "\nMove scores (based on hidden state magnitudes):" << std::endl;
	for (size_t i = 0; i < valid_moves.size(); i++) {
		int leaf_pos = tree_structure.move_to_leaf[i];
		std::string coord = trigo::encode_ab0yz(valid_moves[i], board_shape);

		// Calculate mean magnitude of hidden states at this position
		float score = 0.0f;
		for (int d = 0; d < hidden_dim; d++) {
			int idx = leaf_pos * hidden_dim + d;
			score += std::abs(hidden_states[idx]);
		}
		score /= hidden_dim;

		std::cout << "  " << coord << ": " << score << std::endl;
	}

	// Print performance metrics
	std::cout << "\n" << std::string(80, '=') << std::endl;
	std::cout << "Performance Summary" << std::endl;
	std::cout << std::string(80, '=') << std::endl;
	std::cout << "Prefix computation: " << prefix_time << " ms (once)" << std::endl;
	std::cout << "Move evaluation: " << eval_time << " ms (" << valid_moves.size() << " moves)" << std::endl;
	std::cout << "Per-move time: " << eval_time / valid_moves.size() << " ms" << std::endl;
	std::cout << "Total time: " << (prefix_time + eval_time) << " ms" << std::endl;

	inferencer.print_metrics();

	std::cout << "\n✓ Test completed successfully!" << std::endl;

	return 0;
}
