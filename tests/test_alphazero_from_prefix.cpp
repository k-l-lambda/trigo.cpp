/**
 * Test AlphaZero MCTS from a prefix sequence
 *
 * Usage:
 *   ./test_alphazero_from_prefix <board_shape> <moves> <num_simulations> [model_path] [seed]
 *
 * Example:
 *   ./test_alphazero_from_prefix "5x3x4" "a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz" 200
 *
 * This will:
 * 1. Parse the move sequence
 * 2. Replay moves
 * 3. Run AlphaZero MCTS from that position
 * 4. Print detailed statistics
 *
 * Uses the same policy as self_play_generator with --black-policy alphazero
 */

#include "trigo_game.hpp"
#include "self_play_policy.hpp"
#include "mcts.hpp"
#include "shared_model_inferencer.hpp"
#include "trigo_coords.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <regex>

using namespace trigo;


// Parse board shape string like "5x3x4"
BoardShape parse_board_shape(const std::string& shape_str)
{
	std::regex shape_regex("(\\d+)x(\\d+)(?:x(\\d+))?");
	std::smatch matches;

	if (std::regex_match(shape_str, matches, shape_regex))
	{
		int x = std::stoi(matches[1]);
		int y = std::stoi(matches[2]);
		int z = matches[3].matched ? std::stoi(matches[3]) : 1;
		return BoardShape{x, y, z};
	}

	throw std::invalid_argument("Invalid board shape format: " + shape_str);
}


void print_board_state(const TrigoGame& game)
{
	auto shape = game.get_shape();

	std::cout << "\n=== Board State ===" << std::endl;
	std::cout << "Shape: " << shape.x << "x" << shape.y << "x" << shape.z << std::endl;
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;

	// Print territory
	auto territory = const_cast<TrigoGame&>(game).get_territory();
	std::cout << "Territory: Black=" << territory.black
	          << ", White=" << territory.white
	          << ", Neutral=" << territory.neutral << std::endl;

	// Print valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << std::endl;

	// Print pass count
	std::cout << "Pass count: " << game.get_pass_count() << std::endl;

	std::cout << std::endl;
}


int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cerr << "Usage: " << argv[0] << " <board_shape> <moves> <num_simulations> [model_dir] [seed]" << std::endl;
		std::cerr << "Example: " << argv[0] << " \"5x3x4\" \"a0z b0z aza azz\" 200" << std::endl;
		std::cerr << std::endl;
		std::cerr << "board_shape: Board dimensions (e.g., \"5x3x4\" or \"19x19\")" << std::endl;
		std::cerr << "moves: Space-separated moves (e.g., \"a0z b0z aza\")" << std::endl;
		std::cerr << "num_simulations: MCTS simulations (e.g., 200)" << std::endl;
		std::cerr << "model_dir: Directory containing ONNX models (default: models/trained_shared)" << std::endl;
		std::cerr << "           Expected files: base_model.onnx, policy_head.onnx, value_head.onnx" << std::endl;
		std::cerr << "seed: Random seed for MCTS (default: 42)" << std::endl;
		return 1;
	}

	std::string shape_str = argv[1];
	std::string moves_str = argv[2];
	int num_simulations = std::atoi(argv[3]);
	std::string model_path = (argc >= 5) ? argv[4] : "models/trained_shared";
	int seed = (argc >= 6) ? std::atoi(argv[5]) : 42;

	std::cout << "=== AlphaZero MCTS from Prefix Test ===" << std::endl;
	std::cout << "Board shape: " << shape_str << std::endl;
	std::cout << "Moves: " << moves_str << std::endl;
	std::cout << "Simulations: " << num_simulations << std::endl;
	std::cout << "Model: " << model_path << std::endl;
	std::cout << "Random seed: " << seed << std::endl;
	std::cout << std::endl;

	// Parse board shape
	BoardShape shape;
	try
	{
		shape = parse_board_shape(shape_str);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error parsing board shape: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "Parsed board shape: " << shape.x << "x" << shape.y << "x" << shape.z << std::endl;

	// Parse move strings
	std::vector<std::string> move_strings;
	std::istringstream iss(moves_str);
	std::string move_str;
	while (iss >> move_str)
	{
		move_strings.push_back(move_str);
	}

	std::cout << "Parsed " << move_strings.size() << " moves" << std::endl;
	std::cout << std::endl;

	// Create game
	TrigoGame game(shape);
	game.start_game();

	// Convert BoardShape to vector for decode_ab0yz
	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};

	// Replay moves
	std::cout << "=== Replaying moves ===" << std::endl;
	for (size_t i = 0; i < move_strings.size(); i++)
	{
		const auto& move_str = move_strings[i];
		bool success = false;

		if (move_str == "Pass")
		{
			success = game.pass();
		}
		else
		{
			auto pos_vec = decode_ab0yz(move_str, shape_vec);
			Position pos(pos_vec[0], pos_vec[1], pos_vec[2]);
			success = game.drop(pos);
		}

		if (!success)
		{
			std::cerr << "Failed to replay move " << (i + 1) << ": " << move_str << std::endl;
			return 1;
		}

		std::string player = (i % 2 == 0) ? "Black" : "White";
		std::cout << "Move " << (i + 1) << ": " << move_str << " (" << player << ")" << std::endl;
	}

	// Print board state
	print_board_state(game);

	// Check if game is already finished
	if (!game.is_game_active())
	{
		std::cout << "Game is already finished. No MCTS needed." << std::endl;
		return 0;
	}

	// Initialize AlphaZero MCTS (same as self_play_generator)
	std::cout << "=== Initializing AlphaZero MCTS ===" << std::endl;

	// Construct model paths
	std::string base_model = model_path + "/base_model.onnx";
	std::string policy_head = model_path + "/policy_head.onnx";
	std::string value_head = model_path + "/value_head.onnx";

	// Check if model files exist
	std::vector<std::string> required_files = {base_model, policy_head, value_head};
	for (const auto& file : required_files)
	{
		std::ifstream check(file);
		if (!check.good())
		{
			std::cerr << "Error: Model file not found: " << file << std::endl;
			return 1;
		}
	}

	try
	{
		// Create SharedModelInferencer (same as AlphaZeroPolicy)
		// Try GPU first, fallback to CPU
		std::shared_ptr<SharedModelInferencer> inferencer;
		try
		{
			inferencer = std::make_shared<SharedModelInferencer>(
				base_model,
				policy_head,
				value_head,
				true,  // use_gpu
				0      // device_id
			);
		}
		catch (const std::exception& e)
		{
			std::cout << "GPU failed, falling back to CPU..." << std::endl;
			inferencer = std::make_shared<SharedModelInferencer>(
				base_model,
				policy_head,
				value_head,
				false,  // use_gpu = false (CPU)
				0
			);
		}

		// Create MCTS engine (same parameters as AlphaZeroPolicy)
		float pass_prior = 1e-10f;  // Default minimal prior for Pass
		float pass_value_bias = 0.0f;  // No bias for testing
		float dirichlet_epsilon = 0.0f;  // No noise for deterministic testing
		auto mcts = std::make_unique<MCTS>(
			inferencer,
			num_simulations,
			1.0f,      // c_puct
			seed,
			0.03f,     // dirichlet_alpha
			dirichlet_epsilon,
			pass_prior,
			pass_value_bias
		);

		std::cout << "MCTS initialized with " << num_simulations << " simulations" << std::endl;
		std::cout << "Pass prior: " << pass_prior << std::endl;
		std::cout << std::endl;

		// Run MCTS search
		std::cout << "=== Running AlphaZero MCTS Search ===" << std::endl;
		auto action = mcts->search(game);

		// Print detailed statistics
		mcts->print_root_statistics(game);

		// Print selected action
		std::cout << "\n=== Selected Action ===" << std::endl;
		if (action.is_pass)
		{
			std::cout << "Action: Pass" << std::endl;
		}
		else
		{
			std::cout << "Action: " << encode_ab0yz({action.position.x, action.position.y, action.position.z}, shape_vec) << std::endl;
		}
		std::cout << "Confidence: " << std::fixed << std::setprecision(6) << action.confidence << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
