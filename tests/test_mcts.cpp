/**
 * Test MCTS Implementation
 */

#include "../include/trigo_game.hpp"
#include "../include/mcts_moc.hpp"
#include <iostream>

using namespace trigo;


int main()
{
	std::cout << "\n=== PureMCTS Test ===\n\n";

	// Create simple 5x5x5 game
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	std::cout << "Initial board setup complete\n";
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << "\n\n";

	// Test with small number of simulations
	PureMCTS mcts_engine(50, 1.414f, 42);  // Only 50 simulations for test

	std::cout << "Running PureMCTS search (50 simulations)...\n";
	auto action = mcts_engine.search(game);

	std::cout << "MCTS selected action:\n";
	if (action.is_pass)
	{
		std::cout << "  Move: PASS\n";
	}
	else
	{
		std::cout << "  Move: (" << action.position.x << ","
		          << action.position.y << "," << action.position.z << ")\n";
	}
	std::cout << "  Confidence: " << action.confidence << "\n\n";

	// Apply the move
	if (action.is_pass)
	{
		game.pass();
	}
	else
	{
		bool success = game.drop(action.position);
		if (success)
		{
			std::cout << "Move applied successfully\n";
		}
		else
		{
			std::cout << "ERROR: Move failed\n";
			return 1;
		}
	}

	std::cout << "\n✓ MCTS test passed\n\n";

	return 0;
}
