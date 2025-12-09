/**
 * Test single MCTS simulation - Compare C++ vs TypeScript node visit counts
 *
 * Runs 1 MCTS simulation and outputs detailed node visit statistics
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
	std::cout << "Single MCTS Simulation - Node Visit Count Test" << std::endl;
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

	// Setup game: 5x5 board, first move at a0
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};
	auto pos_vec = decode_ab0yz("a0", shape_vec);
	Position move_a0{pos_vec[0], pos_vec[1], pos_vec[2]};
	game.drop(move_a0);

	std::cout << "Game state:" << std::endl;
	std::cout << "  Board: 5×5×1" << std::endl;
	std::cout << "  Moves played: a0" << std::endl;
	std::cout << "  Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;
	std::cout << "  Valid moves: " << game.valid_move_positions().size() << std::endl;
	std::cout << std::endl;

	// Create MCTS with 1 simulation
	std::cout << "Creating CachedMCTS with 1 simulation..." << std::endl;
	CachedMCTS mcts(
		inferencer,
		1,      // num_simulations = 1
		1.0f,   // c_puct
		42      // seed
	);
	std::cout << "✓ MCTS created" << std::endl;
	std::cout << std::endl;

	// Run search
	std::cout << "Running MCTS search with 1 simulation..." << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;
	auto result = mcts.search(game);
	std::cout << "------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	// Print result
	std::cout << "Search completed:" << std::endl;
	if (result.is_pass)
	{
		std::cout << "  Selected move: PASS" << std::endl;
	}
	else
	{
		std::string move_name = encode_ab0yz(result.position, shape);
		std::cout << "  Selected move: " << move_name << std::endl;
	}
	std::cout << "  Confidence: " << std::fixed << std::setprecision(4) << result.confidence << std::endl;
	std::cout << std::endl;

	std::cout << "============================================================================" << std::endl;
	std::cout << "Test complete - Compare node visit counts with TypeScript output" << std::endl;
	std::cout << "============================================================================" << std::endl;

	return 0;
}
