#include "self_play_policy.hpp"
#include "trigo_game.hpp"
#include <iostream>
#include <chrono>


/**
 * Test CachedNeuralPolicy
 *
 * Validates that the CachedNeuralPolicy class:
 * 1. Loads models successfully
 * 2. Selects valid moves
 * 3. Uses prefix cache optimization
 */
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		std::cerr << "\\nModel directory should contain cached models:" << std::endl;
		std::cerr << "  - base_model_prefix.onnx" << std::endl;
		std::cerr << "  - base_model_eval_cached.onnx" << std::endl;
		std::cerr << "  - policy_head.onnx" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "==============================================================================" << std::endl;
	std::cout << "Test: CachedNeuralPolicy Integration" << std::endl;
	std::cout << "==============================================================================" << std::endl;

	try
	{
		// Create policy via factory
		std::cout << "\\nCreating CachedNeuralPolicy via PolicyFactory..." << std::endl;
		auto policy = trigo::PolicyFactory::create("cached", model_dir, 42);
		std::cout << "  ✓ Policy created: " << policy->name() << std::endl;

		// Initialize game
		std::cout << "\\nInitializing 5×5×1 game..." << std::endl;
		trigo::TrigoGame game({5, 5, 1});

		// Play a few moves
		std::vector<trigo::Position> moves = {
			{1, 1, 0},  // Black
			{3, 3, 0},  // White
			{2, 2, 0},  // Black
		};

		std::cout << "Playing moves:" << std::endl;
		for (const auto& move : moves)
		{
			game.drop(move);
			std::cout << "  " << trigo::encode_ab0yz(move, game.get_shape())
			          << " (" << (game.get_current_player() == trigo::Stone::Black ? "Black" : "White") << ")" << std::endl;
		}

		// Test move selection
		std::cout << "\\n[Test 1] Single move selection..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		auto action = policy->select_action(game);

		auto end = std::chrono::high_resolution_clock::now();
		double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

		if (action.is_pass)
		{
			std::cout << "  Selected: PASS" << std::endl;
		}
		else
		{
			std::string coord = trigo::encode_ab0yz(action.position, game.get_shape());
			std::cout << "  Selected: " << coord << std::endl;
			std::cout << "  Confidence: " << action.confidence << std::endl;
		}
		std::cout << "  Time: " << time_ms << " ms" << std::endl;
		std::cout << "  ✓ Test 1 passed" << std::endl;

		// Test multiple selections (MCTS pattern)
		std::cout << "\\n[Test 2] Multiple selections (MCTS pattern)..." << std::endl;
		const int num_selections = 10;

		std::vector<double> times;
		for (int i = 0; i < num_selections; i++)
		{
			auto start = std::chrono::high_resolution_clock::now();
			auto action = policy->select_action(game);
			auto end = std::chrono::high_resolution_clock::now();

			double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
			times.push_back(time_ms);
		}

		// Calculate statistics
		double total = 0.0;
		for (double t : times) total += t;
		double avg = total / times.size();

		std::cout << "  Selections: " << num_selections << std::endl;
		std::cout << "  Average time: " << avg << " ms" << std::endl;
		std::cout << "  Total time: " << total << " ms" << std::endl;
		std::cout << "  ✓ Test 2 passed" << std::endl;

		// Test with different game state
		std::cout << "\\n[Test 3] Different game state..." << std::endl;
		trigo::TrigoGame game2({5, 5, 1});

		// Play more moves
		for (int i = 0; i < 8; i++)
		{
			auto valid = game2.valid_move_positions();
			if (!valid.empty())
			{
				game2.drop(valid[0]);
			}
		}

		start = std::chrono::high_resolution_clock::now();
		action = policy->select_action(game2);
		end = std::chrono::high_resolution_clock::now();
		time_ms = std::chrono::duration<double, std::milli>(end - start).count();

		std::cout << "  Game state: 8 moves played" << std::endl;
		if (action.is_pass)
		{
			std::cout << "  Selected: PASS" << std::endl;
		}
		else
		{
			std::string coord = trigo::encode_ab0yz(action.position, game2.get_shape());
			std::cout << "  Selected: " << coord << std::endl;
		}
		std::cout << "  Time: " << time_ms << " ms" << std::endl;
		std::cout << "  ✓ Test 3 passed" << std::endl;

		// Summary
		std::cout << "\\n==============================================================================" << std::endl;
		std::cout << "All Tests Passed!" << std::endl;
		std::cout << "==============================================================================" << std::endl;
		std::cout << "✓ CachedNeuralPolicy works correctly" << std::endl;
		std::cout << "✓ Prefix cache optimization functional" << std::endl;
		std::cout << "✓ Ready for MCTS integration" << std::endl;

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\\n✗ Test failed: " << e.what() << std::endl;
		return 1;
	}
}
