/**
 * Self-Play Data Generator
 *
 * Generates training data through self-play games
 * Supports both offline and online training modes
 *
 * Usage:
 *   ./self_play_generator --config config.yaml
 *   ./self_play_generator --num-games 1000 --board 5x5x5 --policy random
 */

#include "../include/trigo_game.hpp"
#include "../include/self_play_policy.hpp"
#include "../include/game_recorder.hpp"
#include "../include/board_shape_candidates.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>


using namespace trigo;
namespace fs = std::filesystem;


/**
 * Self-Play Configuration
 */
struct SelfPlayConfig
{
	// Game settings
	BoardShape board_shape{5, 5, 5};
	bool random_board{false};  // Enable random board selection
	std::string board_ranges{"2-13x1-13x1-1,2-5x2-5x2-5"};  // Default ranges (220 candidates)
	int max_moves{500};

	// Policy settings
	std::string black_policy{"random"};
	std::string white_policy{"random"};
	std::string model_path{""};

	// Generation settings
	int num_games{100};
	int num_threads{1};
	int random_seed{-1};

	// Output settings
	std::string output_dir{"./selfplay_data"};
	bool save_tgn{true};
	bool save_training_data{false};

	// Logging
	int log_interval{10};
};


/**
 * Self-Play Generator
 */
class SelfPlayGenerator
{
private:
	SelfPlayConfig config;
	std::vector<BoardShape> board_candidates;  // Pre-generated board shape candidates
	int games_completed;
	int total_moves;
	std::chrono::steady_clock::time_point start_time;

public:
	SelfPlayGenerator(const SelfPlayConfig& cfg)
		: config(cfg)
		, games_completed(0)
		, total_moves(0)
	{
		// Create output directory
		fs::create_directories(config.output_dir);

		// Generate board shape candidates if random mode is enabled
		if (config.random_board)
		{
			board_candidates = generate_shapes_from_ranges(config.board_ranges);
			if (board_candidates.empty())
			{
				std::cerr << "Warning: No valid board shapes generated from ranges: "
				          << config.board_ranges << std::endl;
				std::cerr << "Falling back to default ranges" << std::endl;
				board_candidates = generate_default_board_shapes();
			}
		}
	}

	/**
	 * Generate specified number of games
	 */
	void generate()
	{
		std::cout << "=== Trigo Self-Play Data Generator ===" << std::endl;
		if (config.random_board)
		{
			std::cout << "Board: Random (" << board_candidates.size() << " candidates)" << std::endl;
			std::cout << "  Ranges: " << config.board_ranges << std::endl;
		}
		else
		{
			std::cout << "Board: " << config.board_shape.x << "x"
			          << config.board_shape.y << "x"
			          << config.board_shape.z << std::endl;
		}
		std::cout << "Games: " << config.num_games << std::endl;
		std::cout << "Black Policy: " << config.black_policy << std::endl;
		std::cout << "White Policy: " << config.white_policy << std::endl;
		std::cout << "Output: " << config.output_dir << std::endl;
		std::cout << std::endl;

		start_time = std::chrono::steady_clock::now();

		// Generate games
		for (int i = 0; i < config.num_games; i++)
		{
			generate_one_game(i);

			if ((i + 1) % config.log_interval == 0)
			{
				log_progress();
			}
		}

		log_final_stats();
	}

private:
	/**
	 * Generate one self-play game
	 */
	void generate_one_game(int game_id)
	{
		// Select board shape (fixed or random)
		BoardShape actual_shape = config.board_shape;

		if (config.random_board)
		{
			// Create separate RNG for board shape selection
			int seed = config.random_seed >= 0
			           ? config.random_seed + game_id * 1000000  // Large offset to avoid policy RNG collision
			           : std::random_device{}() + game_id;

			std::mt19937 shape_rng(seed);
			actual_shape = select_random_board_shape(board_candidates, shape_rng);
		}

		// Create game with selected board shape
		TrigoGame game(actual_shape);
		game.start_game();

		// Create policies
		int seed = config.random_seed >= 0
		           ? config.random_seed + game_id
		           : -1;

		auto black = PolicyFactory::create(config.black_policy, config.model_path, seed);
		auto white = PolicyFactory::create(config.white_policy, config.model_path, seed + 1);

		// Play game
		int move_count = 0;
		while (game.is_game_active() && move_count < config.max_moves)
		{
			// Get current player's policy
			IPolicy* current_policy = game.get_current_player() == Stone::Black
			                          ? black.get()
			                          : white.get();

			// Select action
			PolicyAction action = current_policy->select_action(game);

			// Execute action
			bool success;
			if (action.is_pass)
			{
				success = game.pass();
			}
			else
			{
				success = game.drop(action.position);
			}

			if (!success)
			{
				std::cerr << "Warning: Invalid action in game " << game_id
				          << " at move " << move_count << std::endl;
				break;
			}

			move_count++;
		}

		// Record game
		SelfPlayRecord record = GameRecorder::record_game(
			game,
			black->name(),
			white->name(),
			"Self-Play Training"
		);

		// Save to file
		if (config.save_tgn)
		{
			std::string filename = config.output_dir + "/game_"
			                       + std::to_string(game_id) + ".tgn";
			GameRecorder::save_tgn(record, filename);
		}

		// Update stats
		games_completed++;
		total_moves += move_count;
	}

	/**
	 * Log progress
	 */
	void log_progress()
	{
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
			now - start_time
		).count();

		float games_per_sec = games_completed / (float)std::max(elapsed, 1L);
		float avg_moves = total_moves / (float)std::max(games_completed, 1);

		std::cout << "Progress: " << games_completed << "/" << config.num_games
		          << " games (" << (games_completed * 100 / config.num_games) << "%)"
		          << " | " << games_per_sec << " games/sec"
		          << " | avg " << avg_moves << " moves/game"
		          << std::endl;
	}

	/**
	 * Log final statistics
	 */
	void log_final_stats()
	{
		auto end_time = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
			end_time - start_time
		).count();

		std::cout << "\n=== Generation Complete ===" << std::endl;
		std::cout << "Total games: " << games_completed << std::endl;
		std::cout << "Total moves: " << total_moves << std::endl;
		std::cout << "Average moves per game: " << (total_moves / (float)games_completed) << std::endl;
		std::cout << "Time elapsed: " << elapsed << " seconds" << std::endl;
		std::cout << "Games per second: " << (games_completed / (float)std::max(elapsed, 1L)) << std::endl;
		std::cout << "Output directory: " << config.output_dir << std::endl;
	}
};


/**
 * Parse command line arguments
 */
SelfPlayConfig parse_args(int argc, char* argv[])
{
	SelfPlayConfig config;

	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];

		if (arg == "--num-games" && i + 1 < argc)
		{
			config.num_games = std::stoi(argv[++i]);
		}
		else if (arg == "--board" && i + 1 < argc)
		{
			// Parse "5x5x5" format
			std::string board_str = argv[++i];
			sscanf(board_str.c_str(), "%dx%dx%d",
			       &config.board_shape.x,
			       &config.board_shape.y,
			       &config.board_shape.z);
		}
		else if (arg == "--random-board")
		{
			config.random_board = true;
		}
		else if (arg == "--black-policy" && i + 1 < argc)
		{
			config.black_policy = argv[++i];
		}
		else if (arg == "--white-policy" && i + 1 < argc)
		{
			config.white_policy = argv[++i];
		}
		else if (arg == "--model" && i + 1 < argc)
		{
			config.model_path = argv[++i];
		}
		else if (arg == "--output" && i + 1 < argc)
		{
			config.output_dir = argv[++i];
		}
		else if (arg == "--max-moves" && i + 1 < argc)
		{
			config.max_moves = std::stoi(argv[++i]);
		}
		else if (arg == "--seed" && i + 1 < argc)
		{
			config.random_seed = std::stoi(argv[++i]);
		}
		else if (arg == "--help" || arg == "-h")
		{
			std::cout << "Usage: self_play_generator [options]\n";
			std::cout << "Options:\n";
			std::cout << "  --num-games N       Number of games to generate (default: 100)\n";
			std::cout << "  --board WxHxD       Board dimensions (default: 5x5x5)\n";
			std::cout << "  --random-board      Enable random board selection (220 candidates)\n";
			std::cout << "  --black-policy P    Black player policy (random/mcts/neural)\n";
			std::cout << "  --white-policy P    White player policy (random/mcts/neural)\n";
			std::cout << "  --model PATH        Path to ONNX model for neural policy\n";
			std::cout << "  --output DIR        Output directory (default: ./selfplay_data)\n";
			std::cout << "  --max-moves N       Max moves per game (default: 500)\n";
			std::cout << "  --seed N            Random seed (default: random)\n";
			std::cout << "  --help              Show this help message\n";
			std::exit(0);
		}
	}

	// Validate mutual exclusivity of --board and --random-board
	if (config.random_board)
	{
		// Check if --board was explicitly set (non-default value)
		if (config.board_shape.x != 5 || config.board_shape.y != 5 || config.board_shape.z != 5)
		{
			std::cerr << "Error: Cannot specify both --board and --random-board" << std::endl;
			std::exit(1);
		}
	}

	return config;
}


int main(int argc, char* argv[])
{
	try
	{
		// Parse configuration
		SelfPlayConfig config = parse_args(argc, argv);

		// Create generator
		SelfPlayGenerator generator(config);

		// Generate data
		generator.generate();

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
