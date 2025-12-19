/**
 * Test value network output at a specific position
 *
 * Usage:
 *   ./test_value_at_position <board_shape> <moves> [model_path]
 *
 * Example:
 *   ./test_value_at_position "5x3x4" "a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz"
 *   ./test_value_at_position "5x3x4" "a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz Pass"
 */

#include "trigo_game.hpp"
#include "shared_model_inferencer.hpp"
#include "tgn_tokenizer.hpp"
#include "tgn_utils.hpp"
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


int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " <board_shape> <moves> [model_dir]" << std::endl;
		std::cerr << "Example: " << argv[0] << " \"5x3x4\" \"a0z b0z aza azz\" ../models/trained_shared" << std::endl;
		return 1;
	}

	std::string shape_str = argv[1];
	std::string moves_str = argv[2];
	std::string model_path = (argc >= 4) ? argv[3] : "models/trained_shared";

	std::cout << "=== Value Network Test ===" << std::endl;
	std::cout << "Board shape: " << shape_str << std::endl;
	std::cout << "Moves: " << moves_str << std::endl;
	std::cout << "Model: " << model_path << std::endl;
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

	// Parse move strings
	std::vector<std::string> move_strings;
	std::istringstream iss(moves_str);
	std::string move_str;
	while (iss >> move_str)
	{
		move_strings.push_back(move_str);
	}

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
	std::cout << "\n=== Board State ===" << std::endl;
	std::cout << "Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;
	auto territory = game.get_territory();
	std::cout << "Territory: Black=" << territory.black
	          << ", White=" << territory.white
	          << ", Neutral=" << territory.neutral << std::endl;
	std::cout << "Pass count: " << game.get_pass_count() << std::endl;
	std::cout << "Game active: " << (game.is_game_active() ? "Yes" : "No") << std::endl;

	// Load model
	std::cout << "\n=== Loading Model ===" << std::endl;
	try
	{
		auto inferencer = std::make_shared<SharedModelInferencer>(
			model_path + "/base_model.onnx",
			model_path + "/policy_head.onnx",
			model_path + "/value_head.onnx",
			true,  // use_gpu
			0      // device_id
		);

		// Convert game to TGN
		std::string tgn_text = game_to_tgn(game, false);
		std::cout << "\nTGN: " << tgn_text << std::endl;

		// Tokenize
		TGNTokenizer tokenizer;
		auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

		// Add START token
		std::vector<int64_t> tokens;
		tokens.push_back(1);  // START token
		tokens.insert(tokens.end(), encoded.begin(), encoded.end());

		int seq_len = tokens.size();
		std::cout << "Sequence length: " << seq_len << " tokens" << std::endl;

		// Run value inference
		std::cout << "\n=== Value Network Output ===" << std::endl;
		auto values = inferencer->value_inference(tokens, 1, seq_len, 3);
		float raw_value = values[0];

		std::cout << "Raw value (White advantage): " << std::fixed << std::setprecision(6) << raw_value << std::endl;

		// Adjust for current player
		Stone current_player = game.get_current_player();
		float adjusted_value = raw_value;
		if (current_player == Stone::Black)
		{
			adjusted_value = -raw_value;
		}
		std::cout << "Adjusted value (current player perspective): " << std::fixed << std::setprecision(6) << adjusted_value << std::endl;
		std::cout << "Current player: " << (current_player == Stone::Black ? "Black" : "White") << std::endl;

		// Interpretation
		std::cout << "\n=== Interpretation ===" << std::endl;
		if (adjusted_value > 0.1)
		{
			std::cout << "Value network thinks current player is WINNING" << std::endl;
		}
		else if (adjusted_value < -0.1)
		{
			std::cout << "Value network thinks current player is LOSING" << std::endl;
		}
		else
		{
			std::cout << "Value network thinks game is BALANCED" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
