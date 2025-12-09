/**
 * Test full MCTS search (50 simulations) for comparison with TypeScript
 *
 * Runs on empty board, outputs detailed visit counts and move selection
 */

#include "../include/cached_mcts.hpp"
#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>


using namespace trigo;


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "============================================================================" << std::endl;
	std::cout << "MCTS Full Search - C++ vs TypeScript Comparison" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;

	// Load model
	std::cout << "Loading models from: " << model_dir << std::endl;
	auto inferencer = std::make_shared<PrefixCacheInferencer>(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // use_cuda
		0       // device_id
	);
	std::cout << "✓ Models loaded" << std::endl;
	std::cout << std::endl;

	// Setup game: 5x5 board, empty (Move 1)
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	std::cout << "Test Configuration:" << std::endl;
	std::cout << "  Board: 5×5×1" << std::endl;
	std::cout << "  Simulations: 50" << std::endl;
	std::cout << "  c_puct: 1.0" << std::endl;
	std::cout << "  Position: Empty board (Move 1)" << std::endl;
	std::cout << "  Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;
	std::cout << "  Valid moves: " << game.valid_move_positions().size() << std::endl;
	std::cout << std::endl;

	// Create MCTS with 50 simulations
	std::cout << "Creating CachedMCTS with 50 simulations..." << std::endl;
	CachedMCTS mcts(
		inferencer,
		50,     // num_simulations
		1.0f,   // c_puct
		42      // seed
	);
	std::cout << "✓ MCTS created" << std::endl;
	std::cout << std::endl;

	// Run search
	std::cout << "Running MCTS search..." << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;
	auto start_time = std::chrono::steady_clock::now();
	auto result = mcts.search(game);
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - start_time
	).count();
	std::cout << "------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	// Print result
	std::cout << "Search Results:" << std::endl;
	std::cout << "  Total time: " << elapsed << "ms" << std::endl;
	std::cout << "  Avg per simulation: " << (elapsed / 50.0f) << "ms" << std::endl;
	std::cout << std::endl;

	if (result.is_pass)
	{
		std::cout << "  Selected move: Pass" << std::endl;
	}
	else
	{
		std::string move_name = encode_ab0yz(result.position, shape);
		std::cout << "  Selected move: " << move_name << std::endl;
	}
	std::cout << "  Confidence: " << std::fixed << std::setprecision(6) << result.confidence << std::endl;
	std::cout << std::endl;

	std::cout << "============================================================================" << std::endl;
	std::cout << "Compare with TypeScript MCTS output" << std::endl;
	std::cout << "============================================================================" << std::endl;

	return 0;
}
