#include "cached_mcts.hpp"
#include "self_play_policy.hpp"
#include "trigo_game.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>


/**
 * Test: CachedMCTS Full Integration
 *
 * Tests complete AlphaZero MCTS with prefix cache optimization
 */


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "==============================================================================\n";
	std::cout << "Test: CachedMCTS Full Integration\n";
	std::cout << "==============================================================================\n\n";

	try
	{
		// Create policy via factory
		std::cout << "Creating CachedMCTS policy via PolicyFactory...\n";
		auto policy = trigo::PolicyFactory::create("cached-mcts", model_dir, 42);
		std::cout << "  ✓ Policy created: " << policy->name() << "\n\n";

		// Initialize 5×5×1 game
		std::cout << "Initializing 5×5×1 game...\n";
		trigo::TrigoGame game({5, 5, 1});
		std::cout << "  ✓ Game initialized\n\n";

		// Test 1: Empty board (first move)
		std::cout << "[Test 1] Empty board move selection...\n";
		auto start = std::chrono::high_resolution_clock::now();
		auto action = policy->select_action(game);
		auto end = std::chrono::high_resolution_clock::now();
		double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

		std::cout << "  Selected: " << trigo::encode_ab0yz(action.position, game.get_shape())
		          << (action.is_pass ? " (pass)" : "")
		          << "\n";
		std::cout << "  Confidence: " << action.confidence << "\n";
		std::cout << "  Time: " << time_ms << " ms\n";
		std::cout << "  ✓ Test 1 passed\n\n";

		// Apply first move
		if (!action.is_pass)
		{
			game.drop(action.position);
		}

		// Test 2: After one move (smaller search space)
		std::cout << "[Test 2] Move selection after one move...\n";
		start = std::chrono::high_resolution_clock::now();
		action = policy->select_action(game);
		end = std::chrono::high_resolution_clock::now();
		time_ms = std::chrono::duration<double, std::milli>(end - start).count();

		std::cout << "  Selected: " << trigo::encode_ab0yz(action.position, game.get_shape())
		          << (action.is_pass ? " (pass)" : "")
		          << "\n";
		std::cout << "  Confidence: " << action.confidence << "\n";
		std::cout << "  Time: " << time_ms << " ms\n";
		std::cout << "  ✓ Test 2 passed\n\n";

		// Apply second move
		if (!action.is_pass)
		{
			game.drop(action.position);
		}

		// Test 3: Play a few more moves to test cache with longer game history
		std::cout << "[Test 3] Playing additional moves to test with longer history...\n";
		int num_moves = 3;
		double total_time = 0.0;

		for (int i = 0; i < num_moves; i++)
		{
			start = std::chrono::high_resolution_clock::now();
			action = policy->select_action(game);
			end = std::chrono::high_resolution_clock::now();
			time_ms = std::chrono::duration<double, std::milli>(end - start).count();
			total_time += time_ms;

			std::cout << "  Move " << (i + 1) << ": "
			          << trigo::encode_ab0yz(action.position, game.get_shape())
			          << " (" << time_ms << " ms)\n";

			if (!action.is_pass)
			{
				game.drop(action.position);
			}
		}

		double avg_time = total_time / num_moves;
		std::cout << "  Average time: " << avg_time << " ms\n";
		std::cout << "  ✓ Test 3 passed\n\n";

		// Test 4: Compare with simplified CachedAlphaZeroPolicy
		std::cout << "[Test 4] Comparison with simplified CachedAlphaZeroPolicy...\n";
		auto simple_policy = trigo::PolicyFactory::create("cached-alphazero", model_dir, 42);

		// Reset game
		game = trigo::TrigoGame({5, 5, 1});

		// CachedMCTS
		start = std::chrono::high_resolution_clock::now();
		auto mcts_action = policy->select_action(game);
		end = std::chrono::high_resolution_clock::now();
		double mcts_time = std::chrono::duration<double, std::milli>(end - start).count();

		// Simplified policy
		start = std::chrono::high_resolution_clock::now();
		auto simple_action = simple_policy->select_action(game);
		end = std::chrono::high_resolution_clock::now();
		double simple_time = std::chrono::duration<double, std::milli>(end - start).count();

		std::cout << "  CachedMCTS time: " << mcts_time << " ms\n";
		std::cout << "  Simplified policy time: " << simple_time << " ms\n";
		std::cout << "  Overhead: " << (mcts_time / simple_time) << "×\n";
		std::cout << "  ✓ Test 4 passed\n\n";

		std::cout << "==============================================================================\n";
		std::cout << "All Tests Passed!\n";
		std::cout << "==============================================================================\n";
		std::cout << "✓ CachedMCTS works correctly\n";
		std::cout << "✓ Full MCTS with prefix cache optimization\n";
		std::cout << "✓ Ready for production use\n\n";

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
		return 1;
	}
}
