/**
 * Simple MCTS Debug Test
 *
 * Minimal test with very few simulations to diagnose performance issue
 */

// Enable MCTS profiling for this test
#define MCTS_ENABLE_PROFILING

#include "../include/mcts.hpp"
#include "../include/trigo_game.hpp"
#include <iostream>
#include <chrono>

using namespace trigo;


int main()
{
	std::cout << "\n=== MCTS Simple Debug Test ===\n\n";

	// Create a simple game state
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	std::cout << "Game initialized. Board: 5x5x5\n";
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << "\n";

	// Check valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << "\n\n";

	// Create MCTS with VERY few simulations
	std::cout << "Creating MCTS with 5 simulations...\n";
	MCTS mcts(5, 1.414f, 42);

	std::cout << "Starting MCTS search...\n";
	auto start = std::chrono::steady_clock::now();

	// Run search
	PolicyAction action = mcts.search(game);

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - start
	).count();

	std::cout << "\n=== Search Complete ===\n";
	std::cout << "Time: " << elapsed << "ms\n";

	if (action.is_pass)
	{
		std::cout << "Best move: Pass\n";
	}
	else
	{
		std::cout << "Best move: (" << action.position.x << ", "
		          << action.position.y << ", " << action.position.z << ")\n";
	}
	std::cout << "Confidence: " << action.confidence << "\n";

	std::cout << "\n✓ Test complete\n\n";
	return 0;
}
