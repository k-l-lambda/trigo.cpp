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
 * Benchmark Dynamic Shape Performance
 *
 * Tests inference with different prefix lengths to validate dynamic shape overhead.
 * Expected: < 5% variation across different lengths (after warmup).
 */
int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "==============================================================================\n";
	std::cout << "Benchmark: Dynamic Shape Performance\n";
	std::cout << "==============================================================================\n\n";

	// Initialize inferencer
	std::cout << "Loading ONNX models...\n";
	trigo::PrefixCacheInferencer inferencer(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		"",  // No value head
		false,  // CPU only
		0
	);

	trigo::TGNTokenizer tokenizer;
	trigo::PrefixTreeBuilder tree_builder;

	// Test configurations: different numbers of moves (different prefix lengths)
	struct TestCase {
		int num_moves;
		int expected_prefix_len;
	};

	std::vector<TestCase> test_cases = {
		{2, 16},   // Short game
		{4, 32},   // Medium game
		{8, 64},   // Long game
		{12, 96},  // Very long game
	};

	std::cout << "\nWarmup run (to cache ONNX execution plans)...\n";

	// Warmup: Run each configuration once
	for (const auto& tc : test_cases) {
		trigo::TrigoGame game({5, 5, 1});

		// Play moves
		auto valid_positions = game.valid_move_positions();
		for (int i = 0; i < tc.num_moves && i < static_cast<int>(valid_positions.size()); i++) {
			game.drop(valid_positions[i]);
			valid_positions = game.valid_move_positions();
		}

		// Convert to tokens
		std::string tgn_text = trigo::game_to_tgn(game, false);
		auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

		std::vector<int64_t> prefix_tokens;
		prefix_tokens.push_back(1);  // START token
		prefix_tokens.insert(prefix_tokens.end(), encoded.begin(), encoded.end());

		// Build candidate sequences (just 1 move for simplicity)
		valid_positions = game.valid_move_positions();
		if (valid_positions.empty()) continue;

		std::vector<std::vector<int64_t>> candidate_sequences;
		std::vector<int64_t> seq = prefix_tokens;
		std::string coord = trigo::encode_ab0yz(valid_positions[0], game.get_shape());
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);
		seq.insert(seq.end(), move_tokens.begin(), move_tokens.end());
		candidate_sequences.push_back(seq);

		auto tree_structure = tree_builder.build_tree(candidate_sequences);

		// Run inference
		inferencer.compute_prefix_cache(prefix_tokens, 1, static_cast<int>(prefix_tokens.size()));
		inferencer.evaluate_with_cache(
			tree_structure.evaluated_ids,
			tree_structure.evaluated_mask,
			1,
			tree_structure.num_nodes
		);

		std::cout << "  Warmed up prefix_len=" << prefix_tokens.size() << "\n";
	}

	std::cout << "\nRunning benchmarks...\n\n";
	std::cout << "Moves | Prefix Len | Prefix Time | Eval Time | Total Time\n";
	std::cout << "------|------------|-------------|-----------|------------\n";

	// Benchmark: Run each configuration 10 times
	const int num_iterations = 10;

	for (const auto& tc : test_cases) {
		double total_prefix_time = 0.0;
		double total_eval_time = 0.0;
		int actual_prefix_len = 0;

		for (int iter = 0; iter < num_iterations; iter++) {
			trigo::TrigoGame game({5, 5, 1});

			// Play moves
			auto valid_positions = game.valid_move_positions();
			for (int i = 0; i < tc.num_moves && i < static_cast<int>(valid_positions.size()); i++) {
				game.drop(valid_positions[i]);
				valid_positions = game.valid_move_positions();
			}

			// Convert to tokens
			std::string tgn_text = trigo::game_to_tgn(game, false);
			auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

			std::vector<int64_t> prefix_tokens;
			prefix_tokens.push_back(1);  // START token
			prefix_tokens.insert(prefix_tokens.end(), encoded.begin(), encoded.end());

			actual_prefix_len = static_cast<int>(prefix_tokens.size());

			// Build candidate sequences
			valid_positions = game.valid_move_positions();
			if (valid_positions.empty()) continue;

			std::vector<std::vector<int64_t>> candidate_sequences;
			std::vector<int64_t> seq = prefix_tokens;
			std::string coord = trigo::encode_ab0yz(valid_positions[0], game.get_shape());
			auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);
			seq.insert(seq.end(), move_tokens.begin(), move_tokens.end());
			candidate_sequences.push_back(seq);

			auto tree_structure = tree_builder.build_tree(candidate_sequences);

			// Benchmark prefix computation
			auto start = std::chrono::high_resolution_clock::now();
			inferencer.compute_prefix_cache(prefix_tokens, 1, static_cast<int>(prefix_tokens.size()));
			auto mid = std::chrono::high_resolution_clock::now();

			// Benchmark evaluation
			inferencer.evaluate_with_cache(
				tree_structure.evaluated_ids,
				tree_structure.evaluated_mask,
				1,
				tree_structure.num_nodes
			);
			auto end = std::chrono::high_resolution_clock::now();

			double prefix_time = std::chrono::duration<double, std::milli>(mid - start).count();
			double eval_time = std::chrono::duration<double, std::milli>(end - mid).count();

			total_prefix_time += prefix_time;
			total_eval_time += eval_time;
		}

		double avg_prefix = total_prefix_time / num_iterations;
		double avg_eval = total_eval_time / num_iterations;
		double avg_total = avg_prefix + avg_eval;

		printf("%5d | %10d | %10.2f | %9.2f | %10.2f\n",
			tc.num_moves, actual_prefix_len, avg_prefix, avg_eval, avg_total);
	}

	std::cout << "\n==============================================================================\n";
	std::cout << "Analysis:\n";
	std::cout << "==============================================================================\n";
	std::cout << "Expected: Prefix time varies linearly with prefix length (O(n)).\n";
	std::cout << "Expected: < 5% overhead from dynamic shapes after warmup (first run per shape).\n";
	std::cout << "Expected: Eval time relatively constant (same eval_len).\n\n";

	inferencer.print_metrics();

	std::cout << "\n✓ Benchmark completed!\n";

	return 0;
}
