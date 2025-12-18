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
#include "../include/trigo_coords.hpp"
#include "../include/self_play_policy.hpp"
#include "../include/game_recorder.hpp"
#include "../include/board_shape_candidates.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <openssl/sha.h>
#include <thread>
#include <mutex>
#include <atomic>


using namespace trigo;
namespace fs = std::filesystem;


/**
 * Compute SHA256 hash and return first 16 hex characters
 */
std::string sha256_short(const std::string& data)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	SHA256(reinterpret_cast<const unsigned char*>(data.c_str()), data.size(), hash);

	std::ostringstream oss;
	for (int i = 0; i < 8; i++)  // First 8 bytes = 16 hex chars
	{
		oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
	}
	return oss.str();
}


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

	// MCTS settings
	int mcts_simulations{50};
	float mcts_c_puct{1.0f};

	// Generation settings
	int num_games{100};
	int num_threads{1};
	int random_seed{-1};

	// GPU settings
	int num_gpus{0};  // 0 = single-threaded (auto GPU), >0 = multi-GPU parallel

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
	std::atomic<int> games_completed{0};
	std::atomic<int> total_moves{0};
	std::atomic<int> next_game_id{0};  // For multi-threaded game assignment
	std::mutex output_mutex;  // Protect file I/O and console output
	std::chrono::steady_clock::time_point start_time;

public:
	SelfPlayGenerator(const SelfPlayConfig& cfg)
		: config(cfg)
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
		if (config.num_gpus > 0)
		{
			std::cout << "GPUs: " << config.num_gpus << " (parallel threads)" << std::endl;
		}
		std::cout << "Output: " << config.output_dir << std::endl;
		std::cout << std::endl;

		start_time = std::chrono::steady_clock::now();

		if (config.num_gpus > 0)
		{
			// Multi-GPU parallel generation
			generate_parallel();
		}
		else
		{
			// Single-threaded generation (original behavior)
			generate_single();
		}

		log_final_stats();
	}

private:
	/**
	 * Single-threaded generation (original behavior)
	 */
	void generate_single()
	{
		// Create policies ONCE before game loop (avoid reloading ONNX models per game)
		int base_seed = config.random_seed >= 0 ? config.random_seed : -1;

		auto black = PolicyFactory::create(
			config.black_policy,
			config.model_path,
			base_seed,
			config.mcts_simulations,
			config.mcts_c_puct
		);
		auto white = PolicyFactory::create(
			config.white_policy,
			config.model_path,
			base_seed >= 0 ? base_seed + 1 : -1,
			config.mcts_simulations,
			config.mcts_c_puct
		);

		// Generate games
		for (int i = 0; i < config.num_games; i++)
		{
			generate_one_game(i, black.get(), white.get());

			if ((i + 1) % config.log_interval == 0)
			{
				log_progress();
			}
		}
	}

	/**
	 * Multi-GPU parallel generation
	 */
	void generate_parallel()
	{
		std::vector<std::thread> threads;

		for (int gpu_id = 0; gpu_id < config.num_gpus; gpu_id++)
		{
			threads.emplace_back([this, gpu_id]() {
				worker_thread(gpu_id);
			});
		}

		// Wait for all threads to complete
		for (auto& t : threads)
		{
			t.join();
		}
	}

	/**
	 * Worker thread for multi-GPU generation
	 */
	void worker_thread(int gpu_id)
	{
		// Create policies for this GPU
		int base_seed = config.random_seed >= 0
		                ? config.random_seed + gpu_id * 10000
		                : std::random_device{}();

		std::unique_ptr<IPolicy> black, white;

		try
		{
			black = PolicyFactory::create(
				config.black_policy,
				config.model_path,
				base_seed,
				config.mcts_simulations,
				config.mcts_c_puct,
				true,   // use_gpu
				gpu_id  // device_id
			);
			white = PolicyFactory::create(
				config.white_policy,
				config.model_path,
				base_seed + 1,
				config.mcts_simulations,
				config.mcts_c_puct,
				true,   // use_gpu
				gpu_id  // device_id
			);
		}
		catch (const std::exception& e)
		{
			std::lock_guard<std::mutex> lock(output_mutex);
			std::cerr << "Error: GPU " << gpu_id << " initialization failed: " << e.what() << std::endl;
			return;
		}

		{
			std::lock_guard<std::mutex> lock(output_mutex);
			std::cout << "[GPU " << gpu_id << "] Worker started" << std::endl;
		}

		// Process games until all are done
		while (true)
		{
			int game_id = next_game_id.fetch_add(1);
			if (game_id >= config.num_games)
			{
				break;
			}

			generate_one_game(game_id, black.get(), white.get(), gpu_id);

			int completed = games_completed.load();
			if (completed % config.log_interval == 0 && completed > 0)
			{
				std::lock_guard<std::mutex> lock(output_mutex);
				log_progress();
			}
		}

		{
			std::lock_guard<std::mutex> lock(output_mutex);
			std::cout << "[GPU " << gpu_id << "] Worker finished" << std::endl;
		}
	}

	/**
	 * Print game header to output stream
	 * Helper to avoid duplication between immediate and batch print modes
	 */
	void print_game_header(std::ostream& os, int gpu_id, int game_id, const BoardShape& shape)
	{
		if (gpu_id >= 0)
		{
			os << "\n[GPU " << gpu_id << "][Game " << game_id << "] ";
		}
		else
		{
			os << "\n[Game " << game_id << "] ";
		}
		os << shape.x << "x" << shape.y << "x" << shape.z << ": ";
	}

	/**
	 * Check if natural terminal condition is met
	 * Natural terminal: stone count >= (totalPositions - 1) / 2 AND all territory claimed (neutral == 0)
	 *
	 * This is extracted as a separate function for clarity and potential reuse
	 */
	bool check_natural_terminal(TrigoGame& game, int stone_count)
	{
		const auto& shape = game.get_shape();
		int totalPositions = shape.x * shape.y * shape.z;

		// Safety check (should not happen with valid board shapes)
		if (totalPositions <= 0) return false;

		// Early exit if not enough stones yet
		// For 5x1x1: need >= 2 stones instead of > 2.5 (50%)
		int minStones = (totalPositions - 1) / 2;
		if (stone_count < minStones) return false;

		// Check if all territory has been claimed
		auto territory = game.get_territory();
		return (territory.neutral == 0);
	}

	/**
	 * Generate one self-play game
	 */
	void generate_one_game(int game_id, IPolicy* black, IPolicy* white, int gpu_id = -1)
	{
		// Select board shape (fixed or random)
		BoardShape actual_shape = config.board_shape;

		if (config.random_board)
		{
			// Create separate RNG for board shape selection
			// Use 64-bit arithmetic to avoid integer overflow
			uint64_t base_seed = (config.random_seed >= 0)
				? static_cast<uint64_t>(config.random_seed)
				: static_cast<uint64_t>(std::random_device{}());

			// Mix seed with game_id using a large prime
			uint64_t seed = base_seed ^ (static_cast<uint64_t>(game_id) * 0x9E3779B97F4A7C15ULL);
			std::mt19937 shape_rng(static_cast<std::mt19937::result_type>(seed));
			actual_shape = select_random_board_shape(board_candidates, shape_rng);
		}

		// Create game with selected board shape
		TrigoGame game(actual_shape);
		game.start_game();

		// Determine if we should do immediate printing (only in single-threaded mode)
		bool immediate_print = (config.num_threads == 1 && config.num_gpus == 0);

		// Print game header (only if immediate printing)
		if (immediate_print)
		{
			std::lock_guard<std::mutex> lock(output_mutex);
			print_game_header(std::cout, gpu_id, game_id, actual_shape);
			// Don't flush here - let buffer accumulate and flush at end
		}

		// Play game
		int move_count = 0;
		int stone_count = 0;  // Track stone count incrementally (optimization: avoid repeated board traversal)
		bool had_error = false;  // Track if game ended due to invalid action
		std::ostringstream moves_stream;  // Collect moves for final output and TGN saving
		bool first_move = true;  // Track first move to avoid trailing space

		while (game.is_game_active() && move_count < config.max_moves)
		{
			// Get current player's policy
			Stone current_player = game.get_current_player();
			IPolicy* current_policy = current_player == Stone::Black
			                          ? black
			                          : white;

			// Select action
			PolicyAction action = current_policy->select_action(game);

			// Format move notation
			std::string move_str;
			if (action.is_pass)
			{
				move_str = "Pass";
			}
			else
			{
				move_str = encode_ab0yz(action.position, game.get_shape());
			}

			// Execute action
			bool success;
			if (action.is_pass)
			{
				success = game.pass();
				// stone_count unchanged for pass
			}
			else
			{
				success = game.drop(action.position);
				if (success)
				{
					stone_count++;  // Increment stone count on successful drop
				}
			}

			if (!success)
			{
				std::lock_guard<std::mutex> lock(output_mutex);
				std::cerr << "\nWarning: Invalid action in game " << game_id
				          << " at move " << move_count << " (action: " << move_str << ")" << std::endl;
				had_error = true;
				break;
			}

			// Collect move for final output (avoid trailing space)
			if (!first_move)
			{
				moves_stream << ' ';
			}
			moves_stream << move_str;
			first_move = false;

			// Print move immediately (only in single-threaded mode)
			// Don't flush each move - let buffer accumulate for better performance
			if (immediate_print)
			{
				std::lock_guard<std::mutex> lock(output_mutex);
				std::cout << move_str << " ";
			}

			move_count++;

			// Check for natural terminal condition after each move
			// Uses optimized stone_count instead of traversing board
			if (check_natural_terminal(game, stone_count))
			{
				// Natural terminal: stone count >= threshold and all territory claimed
				break;
			}
		}

		// Skip saving games that ended with errors (optional: can be configured)
		// For now, we log the error but still save the partial game for analysis
		if (had_error)
		{
			std::lock_guard<std::mutex> lock(output_mutex);
			std::cerr << "[Game " << game_id << "] Ended with error after "
			          << move_count << " moves (partial game will be saved)" << std::endl;
		}

		// Calculate territory result
		auto territory = game.get_territory();
		int score = territory.white - territory.black;  // Positive = White wins

		// Print game summary
		{
			std::lock_guard<std::mutex> lock(output_mutex);
			if (immediate_print)
			{
				// Just print the summary (moves already printed)
				// Flush at end to ensure all output is visible
				std::cout << "; " << move_count << " moves, score: "
				          << (score >= 0 ? "+" : "") << score << std::endl;
			}
			else
			{
				// Print complete game on one line (header + moves + summary)
				print_game_header(std::cout, gpu_id, game_id, actual_shape);
				std::cout << moves_stream.str() << "; " << move_count << " moves, score: "
				          << (score >= 0 ? "+" : "") << score << std::endl;
			}
		}

		// Record game
		SelfPlayRecord record = GameRecorder::record_game(
			game,
			black->name(),
			white->name(),
			"Self-Play Training"
		);

		// Save to file with hash-based filename (thread-safe due to unique filenames)
		if (config.save_tgn)
		{
			std::string tgn_content = GameRecorder::to_tgn(record);
			std::string hash = sha256_short(tgn_content);
			std::string filename = config.output_dir + "/game_" + hash + ".tgn";
			GameRecorder::save_tgn(record, filename);
		}

		// Update stats (atomic)
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

		int completed = games_completed.load();
		int moves = total_moves.load();
		float games_per_sec = completed / (float)std::max(elapsed, 1L);
		float avg_moves = moves / (float)std::max(completed, 1);

		std::cout << "Progress: " << completed << "/" << config.num_games
		          << " games (" << (completed * 100 / config.num_games) << "%)"
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

		int completed = games_completed.load();
		int moves = total_moves.load();

		std::cout << "\n=== Generation Complete ===" << std::endl;
		std::cout << "Total games: " << completed << std::endl;
		std::cout << "Total moves: " << moves << std::endl;
		std::cout << "Average moves per game: " << (moves / (float)completed) << std::endl;
		std::cout << "Time elapsed: " << elapsed << " seconds" << std::endl;
		std::cout << "Games per second: " << (completed / (float)std::max(elapsed, 1L)) << std::endl;
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
		else if (arg == "--board-ranges" && i + 1 < argc)
		{
			config.board_ranges = argv[++i];
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
		else if (arg == "--mcts-simulations" && i + 1 < argc)
		{
			config.mcts_simulations = std::stoi(argv[++i]);
		}
		else if (arg == "--mcts-c-puct" && i + 1 < argc)
		{
			config.mcts_c_puct = std::stof(argv[++i]);
		}
		else if (arg == "--num-gpus" && i + 1 < argc)
		{
			config.num_gpus = std::stoi(argv[++i]);
		}
		else if (arg == "--help" || arg == "-h")
		{
			std::cout << "Usage: self_play_generator [options]\n";
			std::cout << "Options:\n";
			std::cout << "  --num-games N          Number of games to generate (default: 100)\n";
			std::cout << "  --board WxHxD          Board dimensions (default: 5x5x5)\n";
			std::cout << "  --random-board         Enable random board selection\n";
			std::cout << "  --board-ranges R       Custom board ranges for random selection\n";
			std::cout << "                         Format: \"minX-maxXxminY-maxYxminZ-maxZ,...\"\n";
			std::cout << "                         Example: \"2-13x1-13x1-1,2-5x2-5x2-5\"\n";
			std::cout << "                         Default: \"2-13x1-13x1-1,2-5x2-5x2-5\" (220 candidates)\n";
			std::cout << "  --black-policy P       Black player policy (random/mcts/alphazero/cached-mcts)\n";
			std::cout << "  --white-policy P       White player policy (random/mcts/alphazero/cached-mcts)\n";
			std::cout << "  --model PATH           Path to ONNX model for neural policy\n";
			std::cout << "  --output DIR           Output directory (default: ./selfplay_data)\n";
			std::cout << "  --max-moves N          Max moves per game (default: 500)\n";
			std::cout << "  --seed N               Random seed (default: random)\n";
			std::cout << "  --mcts-simulations N   MCTS simulations per move (default: 50)\n";
			std::cout << "  --mcts-c-puct F        MCTS exploration constant (default: 1.0)\n";
			std::cout << "  --num-gpus N           Number of GPUs for parallel generation (default: 0 = single)\n";
			std::cout << "  --help                 Show this help message\n";
			std::exit(0);
		}
	}

	// Validate parameter combinations
	if (config.random_board)
	{
		// Check if --board was explicitly set (non-default value)
		if (config.board_shape.x != 5 || config.board_shape.y != 5 || config.board_shape.z != 5)
		{
			std::cerr << "Error: Cannot specify both --board and --random-board" << std::endl;
			std::exit(1);
		}
	}
	else
	{
		// Check if --board-ranges was specified without --random-board
		if (config.board_ranges != "2-13x1-13x1-1,2-5x2-5x2-5")
		{
			std::cerr << "Error: --board-ranges requires --random-board" << std::endl;
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
