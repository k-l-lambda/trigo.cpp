/**
 * Test MCTS single step consistency
 *
 * Run a single MCTS simulation and output detailed intermediate results
 * for comparison with TypeScript implementation
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
	std::cout << "MCTS Single Step Consistency Test (C++)" << std::endl;
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

	// Setup game: 5x5 board, empty
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	std::cout << "Game Configuration:" << std::endl;
	std::cout << "  Board: 5×5×1" << std::endl;
	std::cout << "  Position: Empty board (Move 1)" << std::endl;
	std::cout << "  Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;
	std::cout << "  Valid moves: " << game.valid_move_positions().size() << std::endl;
	std::cout << std::endl;

	// Create MCTS with specific seed
	std::cout << "Creating MCTS with seed=42, c_puct=1.0, simulations=1" << std::endl;
	CachedMCTS mcts(
		inferencer,
		1,      // num_simulations (just 1 for single step test)
		1.0f,   // c_puct
		42      // seed
	);
	std::cout << std::endl;

	// Run single simulation
	std::cout << "Running single MCTS simulation..." << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;
	auto result = mcts.search(game);
	std::cout << "------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	// Print result
	std::cout << "Single Step Result:" << std::endl;
	if (result.is_pass)
	{
		std::cout << "  Selected move: Pass" << std::endl;
	}
	else
	{
		std::string move_name = encode_ab0yz(result.position, shape);
		std::cout << "  Selected move: " << move_name << std::endl;
		std::cout << "  Position: (" << result.position.x << ", " << result.position.y << ", " << result.position.z << ")" << std::endl;
	}
	std::cout << "  Confidence: " << std::fixed << std::setprecision(6) << result.confidence << std::endl;
	std::cout << std::endl;

	std::cout << "============================================================================" << std::endl;
	std::cout << "Compare with TypeScript MCTS single step output" << std::endl;
	std::cout << "Expected: Both should expand the same first move and select the same action" << std::endl;
	std::cout << "============================================================================" << std::endl;

	return 0;
}
