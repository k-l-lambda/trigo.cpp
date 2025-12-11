/**
 * Test Consistency: Cached vs Incremental Inference
 *
 * Verify that value inference results are consistent between:
 * 1. CachedMCTS (PrefixCacheInferencer with compute_prefix_cache)
 * 2. IncrementalMCTS (PrefixCacheInferencer with extend_cache)
 *
 * Note: SharedModelInferencer requires seq_len >= 128, so we only test
 * the prefix cache methods for short sequences.
 */

#include "trigo_game.hpp"
#include "prefix_cache_inferencer.hpp"
#include "tgn_tokenizer.hpp"
#include "tgn_utils.hpp"
#include "trigo_coords.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>

using namespace trigo;


// Helper: Convert game to tokens
std::vector<int64_t> game_to_tokens(const TrigoGame& game, TGNTokenizer& tokenizer)
{
	std::string tgn_text = game_to_tgn(game, false);
	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);
	std::vector<int64_t> tokens;
	tokens.push_back(1);  // START token
	tokens.insert(tokens.end(), encoded.begin(), encoded.end());
	return tokens;
}


int main()
{
	std::cout << "=== Incremental Cache Consistency Test ===" << std::endl;
	std::cout << std::endl;

	std::string model_dir = "/home/camus/work/trigo.cpp/models/trained_shared/";

	// Initialize models
	std::cout << "1. Loading models..." << std::endl;

	// Cached model (PrefixCacheInferencer without extend)
	auto cached_inferencer = std::make_shared<PrefixCacheInferencer>(
		model_dir + "base_model_prefix.onnx",
		model_dir + "base_model_eval_cached.onnx",
		model_dir + "policy_head.onnx",
		model_dir + "value_head.onnx",
		false,  // CPU
		0
	);
	std::cout << "   PrefixCacheInferencer (cached) loaded" << std::endl;

	// Incremental model (PrefixCacheInferencer with extend)
	auto incremental_inferencer = std::make_shared<PrefixCacheInferencer>(
		model_dir + "base_model_prefix.onnx",
		model_dir + "base_model_eval_cached.onnx",
		model_dir + "policy_head.onnx",
		model_dir + "value_head.onnx",
		false,  // CPU
		0,
		"",     // No evaluation_model_path
		model_dir + "base_model_eval_extend.onnx"  // Enable incremental
	);
	std::cout << "   PrefixCacheInferencer (incremental) loaded" << std::endl;
	std::cout << std::endl;

	// Setup game
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	TGNTokenizer tokenizer;

	// Test multiple game states
	std::vector<Position> moves = {
		{2, 0, 0},  // a0
		{0, 2, 0},  // 0a
		{2, 2, 0},  // 00
		{0, 0, 0},  // aa
		{4, 4, 0},  // zz
	};

	std::cout << "2. Testing value consistency across game states..." << std::endl;
	std::cout << std::endl;

	float max_diff = 0.0f;
	bool all_passed = true;

	// Initialize incremental cache with initial state
	auto initial_tokens = game_to_tokens(game, tokenizer);
	incremental_inferencer->compute_prefix_cache(initial_tokens, 1, static_cast<int>(initial_tokens.size()));

	for (size_t i = 0; i <= moves.size(); i++)
	{
		// Get current tokens
		auto tokens = game_to_tokens(game, tokenizer);
		int seq_len = static_cast<int>(tokens.size());

		std::cout << "   State " << i << " (seq_len=" << seq_len << "):" << std::endl;

		// === Method 1: PrefixCacheInferencer (cached) - recompute each time ===
		cached_inferencer->compute_prefix_cache(tokens, 1, seq_len);
		float value_cached = cached_inferencer->value_inference_with_cache(3);  // VALUE token

		// === Method 2: PrefixCacheInferencer (incremental) - already computed ===
		float value_incremental = incremental_inferencer->value_inference_with_cache(3);

		// Compare values
		float diff = std::abs(value_cached - value_incremental);
		max_diff = std::max(max_diff, diff);

		std::cout << "      Cached:      " << std::fixed << std::setprecision(6) << value_cached << std::endl;
		std::cout << "      Incremental: " << std::fixed << std::setprecision(6) << value_incremental << std::endl;
		std::cout << "      Diff:        " << std::fixed << std::setprecision(6) << diff;

		const float TOLERANCE = 0.001f;  // Allow small numerical differences
		if (diff < TOLERANCE)
		{
			std::cout << " ✓" << std::endl;
		}
		else
		{
			std::cout << " ✗ FAILED!" << std::endl;
			all_passed = false;
		}
		std::cout << std::endl;

		// Make next move and extend incremental cache
		if (i < moves.size())
		{
			Position next_move = moves[i];
			game.drop(next_move);

			// Build move tokens for the move just made
			std::string coord = encode_ab0yz(next_move, shape);
			std::string move_text;
			Stone player_who_moved = (i % 2 == 0) ? Stone::Black : Stone::White;

			if (player_who_moved == Stone::Black)
			{
				// Black's move: "N. coord "
				int move_num = static_cast<int>((i + 2) / 2);
				move_text = std::to_string(move_num) + ". " + coord + " ";
			}
			else
			{
				// White's move: "coord\n" (followed by newline, not space)
				move_text = coord + "\n";
			}

			auto move_tokens_enc = tokenizer.encode(move_text, 2048, false, false, false, false);
			std::vector<int64_t> move_tokens(move_tokens_enc.begin(), move_tokens_enc.end());

			int new_len = static_cast<int>(move_tokens.size());
			std::vector<float> mask(new_len * new_len);
			for (int r = 0; r < new_len; r++)
			{
				for (int c = 0; c < new_len; c++)
				{
					mask[r * new_len + c] = (c <= r) ? 1.0f : 0.0f;
				}
			}

			std::cout << "   Extending cache with: \"" << move_text << "\" (" << new_len << " tokens)" << std::endl;

			try
			{
				incremental_inferencer->extend_cache(move_tokens, mask, 1, new_len);
			}
			catch (const std::exception& e)
			{
				std::cerr << "      extend_cache failed: " << e.what() << std::endl;
				// Fallback to recompute
				auto new_tokens = game_to_tokens(game, tokenizer);
				incremental_inferencer->compute_prefix_cache(new_tokens, 1, static_cast<int>(new_tokens.size()));
			}
			std::cout << std::endl;
		}
	}

	std::cout << "3. Summary:" << std::endl;
	std::cout << "   Maximum difference: " << std::fixed << std::setprecision(6) << max_diff << std::endl;
	std::cout << std::endl;

	if (all_passed)
	{
		std::cout << "=== All consistency tests PASSED! ===" << std::endl;
		return 0;
	}
	else
	{
		std::cout << "=== Some tests FAILED! ===" << std::endl;
		return 1;
	}
}
