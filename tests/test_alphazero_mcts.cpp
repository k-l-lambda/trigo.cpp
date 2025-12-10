/**
 * AlphaZero MCTS Performance Test
 *
 * Tests MCTS with value network evaluation
 * Should be ~250× faster than random rollouts
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
	std::cout << "\n=== AlphaZero MCTS Performance Test ===\n\n";

	// Load trained model
	std::string base_path = "../models/trained_shared/";
	std::string base_model = base_path + "base_model.onnx";
	std::string policy_head = base_path + "policy_head.onnx";
	std::string value_head = base_path + "value_head.onnx";

	std::cout << "Loading ONNX models...\n";
	std::cout << "  Base: " << base_model << "\n";
	std::cout << "  Policy: " << policy_head << "\n";
	std::cout << "  Value: " << value_head << "\n\n";

	auto inferencer = std::make_shared<SharedModelInferencer>(
		base_model,
		policy_head,
		value_head
	);

	std::cout << "Models loaded successfully.\n\n";

	// Create a simple game state (5x5x1 matches training data)
	TrigoGame game(BoardShape{5, 5, 1});
	game.start_game();

	std::cout << "Game initialized. Board: 5x5x1\n";
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << "\n";

	// Check valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << "\n\n";

	// Create MCTS with 100 simulations (AlphaZero-style with value network)
	std::cout << "Creating MCTS (AlphaZero) with 100 simulations...\n";
	MCTS mcts(inferencer, 100, 1.0f, 42);

	std::cout << "Starting MCTS search...\n";
	auto start = std::chrono::steady_clock::now();

	// Run search
	PolicyAction action = mcts.search(game);

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - start
	).count();

	std::cout << "\n=== Search Complete ===\n";
	std::cout << "Time: " << elapsed << "ms\n";
	std::cout << "Simulations per second: " << (100.0 / elapsed * 1000.0) << "\n\n";

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

	std::cout << "\n=== Performance Comparison ===\n";
	std::cout << "PureMCTS (random rollouts): ~923ms per simulation\n";
	std::cout << "MCTS (value network): ~" << (elapsed / 100.0) << "ms per simulation\n";
	float speedup = 923.0 / (elapsed / 100.0);
	std::cout << "Speedup: " << speedup << "×\n";

	std::cout << "\n✓ Test complete\n\n";
	return 0;
}
