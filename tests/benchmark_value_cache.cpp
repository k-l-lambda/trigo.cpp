#include "prefix_cache_inferencer.hpp"
#include "shared_model_inferencer.hpp"
#include "trigo_game.hpp"
#include "tgn_tokenizer.hpp"
#include "tgn_utils.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>


/**
 * Benchmark: Value Inference with Prefix Cache
 *
 * Compares performance of value inference:
 * 1. Standard inference (SharedModelInferencer, no cache)
 * 2. Cached inference (PrefixCacheInferencer, shared cache)
 *
 * Tests MCTS simulation pattern:
 * - Multiple value evaluations per game position
 * - Cache should be computed once and reused
 */


std::vector<int64_t> game_to_tokens(const trigo::TrigoGame& game)
{
	trigo::TGNTokenizer tokenizer;
	std::string tgn_text = trigo::game_to_tgn(game, false);
	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

	std::vector<int64_t> tokens;
	tokens.push_back(1);  // START token
	tokens.insert(tokens.end(), encoded.begin(), encoded.end());
	return tokens;
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "========================================================================" << std::endl;
	std::cout << "Benchmark: Value Inference with Prefix Cache" << std::endl;
	std::cout << "========================================================================\n" << std::endl;

	// Test configuration
	const int num_iterations = 20;
	const int num_evaluations_per_position = 10;  // Simulate MCTS pattern

	try
	{
		// Load both inferencers
		std::cout << "Loading models..." << std::endl;

		auto shared_inferencer = std::make_shared<trigo::SharedModelInferencer>(
			model_dir + "/base_model.onnx",
			model_dir + "/policy_head.onnx",
			model_dir + "/value_head.onnx",
			false  // CPU for fair comparison
		);

		trigo::PrefixCacheInferencer cache_inferencer(
			model_dir + "/base_model_prefix.onnx",
			model_dir + "/base_model_eval_cached.onnx",
			model_dir + "/policy_head.onnx",
			model_dir + "/value_head.onnx",
			false,  // CPU
			0
		);

		std::cout << "✓ Models loaded\n" << std::endl;

		// Create test game
		trigo::TrigoGame game({5, 5, 1});

		// Play a few moves
		std::vector<trigo::Position> moves = {{1, 1, 0}, {3, 3, 0}, {2, 2, 0}};
		for (const auto& move : moves)
		{
			game.drop(move);
		}

		auto prefix_tokens = game_to_tokens(game);
		int prefix_len = static_cast<int>(prefix_tokens.size());

		std::cout << "Test configuration:" << std::endl;
		std::cout << "  Board: 5×5×1" << std::endl;
		std::cout << "  Moves played: 3" << std::endl;
		std::cout << "  Prefix length: " << prefix_len << " tokens" << std::endl;
		std::cout << "  Evaluations per position: " << num_evaluations_per_position << std::endl;
		std::cout << "  Iterations: " << num_iterations << "\n" << std::endl;

		// ===== Benchmark 1: Standard Inference (no cache) =====
		std::cout << "[Benchmark 1] Standard value inference (no cache)" << std::endl;

		std::vector<double> standard_times;

		for (int iter = 0; iter < num_iterations; iter++)
		{
			auto start = std::chrono::high_resolution_clock::now();

			// Simulate MCTS: multiple value evaluations
			for (int eval = 0; eval < num_evaluations_per_position; eval++)
			{
				// Each evaluation recomputes full sequence
				auto values = shared_inferencer->value_inference(
					prefix_tokens,
					1,
					prefix_len,
					3  // VALUE token
				);
			}

			auto end = std::chrono::high_resolution_clock::now();
			double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
			standard_times.push_back(time_ms);
		}

		double standard_avg = std::accumulate(standard_times.begin(), standard_times.end(), 0.0) / standard_times.size();
		double standard_per_eval = standard_avg / num_evaluations_per_position;

		std::cout << "  Total time: " << standard_avg << " ms" << std::endl;
		std::cout << "  Per evaluation: " << standard_per_eval << " ms" << std::endl;

		// ===== Benchmark 2: Cached Inference =====
		std::cout << "\n[Benchmark 2] Cached value inference (prefix cache)" << std::endl;

		std::vector<double> cached_times;
		std::vector<double> prefix_compute_times;
		std::vector<double> eval_times;

		for (int iter = 0; iter < num_iterations; iter++)
		{
			// Compute prefix cache once
			auto prefix_start = std::chrono::high_resolution_clock::now();
			cache_inferencer.compute_prefix_cache(prefix_tokens, 1, prefix_len);
			auto prefix_end = std::chrono::high_resolution_clock::now();
			double prefix_time = std::chrono::duration<double, std::milli>(prefix_end - prefix_start).count();
			prefix_compute_times.push_back(prefix_time);

			// Multiple value evaluations using cache
			auto eval_start = std::chrono::high_resolution_clock::now();
			for (int eval = 0; eval < num_evaluations_per_position; eval++)
			{
				float value = cache_inferencer.value_inference_with_cache(3);
			}
			auto eval_end = std::chrono::high_resolution_clock::now();
			double eval_time = std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
			eval_times.push_back(eval_time);

			cached_times.push_back(prefix_time + eval_time);
		}

		double cached_avg = std::accumulate(cached_times.begin(), cached_times.end(), 0.0) / cached_times.size();
		double prefix_avg = std::accumulate(prefix_compute_times.begin(), prefix_compute_times.end(), 0.0) / prefix_compute_times.size();
		double eval_avg = std::accumulate(eval_times.begin(), eval_times.end(), 0.0) / eval_times.size();
		double cached_per_eval = eval_avg / num_evaluations_per_position;

		std::cout << "  Prefix computation: " << prefix_avg << " ms (once)" << std::endl;
		std::cout << "  Evaluations: " << eval_avg << " ms (" << num_evaluations_per_position << " evals)" << std::endl;
		std::cout << "  Total time: " << cached_avg << " ms" << std::endl;
		std::cout << "  Per evaluation: " << cached_per_eval << " ms" << std::endl;

		// ===== Results =====
		std::cout << "\n========================================================================" << std::endl;
		std::cout << "Results" << std::endl;
		std::cout << "========================================================================" << std::endl;

		double speedup = standard_avg / cached_avg;

		std::cout << "\nStandard inference:" << std::endl;
		std::cout << "  Total: " << standard_avg << " ms" << std::endl;
		std::cout << "  Per evaluation: " << standard_per_eval << " ms" << std::endl;

		std::cout << "\nCached inference:" << std::endl;
		std::cout << "  Prefix: " << prefix_avg << " ms (1×)" << std::endl;
		std::cout << "  Evaluations: " << eval_avg << " ms (" << num_evaluations_per_position << "×)" << std::endl;
		std::cout << "  Total: " << cached_avg << " ms" << std::endl;
		std::cout << "  Per evaluation: " << cached_per_eval << " ms" << std::endl;

		std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2)
		          << speedup << "× faster" << std::endl;

		std::cout << "\nTime saved: " << std::fixed << std::setprecision(2)
		          << (standard_avg - cached_avg) << " ms per position ("
		          << ((standard_avg - cached_avg) / standard_avg * 100.0) << "%)" << std::endl;

		// Extrapolate to MCTS
		std::cout << "\n========================================================================" << std::endl;
		std::cout << "MCTS Extrapolation (50 simulations)" << std::endl;
		std::cout << "========================================================================" << std::endl;

		double mcts_standard = standard_per_eval * 50;
		double mcts_cached = prefix_avg + (cached_per_eval * 50);

		std::cout << "Standard inference: " << mcts_standard << " ms per move" << std::endl;
		std::cout << "Cached inference: " << mcts_cached << " ms per move" << std::endl;
		std::cout << "Speedup: " << (mcts_standard / mcts_cached) << "×" << std::endl;
		std::cout << "Time saved: " << (mcts_standard - mcts_cached) << " ms per move" << std::endl;

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n✗ Benchmark failed: " << e.what() << std::endl;
		return 1;
	}
}
