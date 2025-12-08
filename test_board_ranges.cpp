/**
 * Simple test for --board-ranges parameter parsing
 * Tests the board shape range generation without needing full dependencies
 */

#include "include/board_shape_candidates.hpp"
#include <iostream>
#include <string>

using namespace trigo;


/**
 * Test configuration
 */
struct TestConfig
{
	std::string board_ranges{"2-13x1-13x1-1,2-5x2-5x2-5"};
	bool random_board{false};
};


/**
 * Parse command line arguments
 */
TestConfig parse_args(int argc, char* argv[])
{
	TestConfig config;

	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];

		if (arg == "--random-board")
		{
			config.random_board = true;
		}
		else if (arg == "--board-ranges" && i + 1 < argc)
		{
			config.board_ranges = argv[++i];
		}
		else if (arg == "--help" || arg == "-h")
		{
			std::cout << "Test program for --board-ranges parameter\n";
			std::cout << "Usage: test_board_ranges [options]\n";
			std::cout << "Options:\n";
			std::cout << "  --random-board      Enable random board selection\n";
			std::cout << "  --board-ranges R    Custom board ranges\n";
			std::cout << "  --help              Show this help message\n";
			std::exit(0);
		}
	}

	// Validate
	if (!config.random_board && config.board_ranges != "2-13x1-13x1-1,2-5x2-5x2-5")
	{
		std::cerr << "Error: --board-ranges requires --random-board" << std::endl;
		std::exit(1);
	}

	return config;
}


int main(int argc, char* argv[])
{
	std::cout << "=== Board Ranges Parameter Test ===" << std::endl;

	// Parse arguments
	TestConfig config = parse_args(argc, argv);

	// Show configuration
	std::cout << "Random board: " << (config.random_board ? "enabled" : "disabled") << std::endl;
	std::cout << "Board ranges: " << config.board_ranges << std::endl;
	std::cout << std::endl;

	if (config.random_board)
	{
		// Generate board shape candidates
		std::cout << "Generating board shape candidates..." << std::endl;
		auto candidates = generate_shapes_from_ranges(config.board_ranges);

		if (candidates.empty())
		{
			std::cerr << "Error: No valid board shapes generated from ranges" << std::endl;
			return 1;
		}

		std::cout << "Generated " << candidates.size() << " candidate board shapes:" << std::endl;
		std::cout << std::endl;

		// Show first 10 and last 10
		size_t show_count = std::min<size_t>(10, candidates.size());

		std::cout << "First " << show_count << " shapes:" << std::endl;
		for (size_t i = 0; i < show_count; i++)
		{
			std::cout << "  [" << i << "] "
			          << candidates[i].x << "×"
			          << candidates[i].y << "×"
			          << candidates[i].z << std::endl;
		}

		if (candidates.size() > 20)
		{
			std::cout << "  ..." << std::endl;
			std::cout << "Last 10 shapes:" << std::endl;
			for (size_t i = candidates.size() - 10; i < candidates.size(); i++)
			{
				std::cout << "  [" << i << "] "
				          << candidates[i].x << "×"
				          << candidates[i].y << "×"
				          << candidates[i].z << std::endl;
			}
		}
		else if (candidates.size() > 10)
		{
			std::cout << "Remaining shapes:" << std::endl;
			for (size_t i = 10; i < candidates.size(); i++)
			{
				std::cout << "  [" << i << "] "
				          << candidates[i].x << "×"
				          << candidates[i].y << "×"
				          << candidates[i].z << std::endl;
			}
		}

		std::cout << std::endl;
		std::cout << "✓ Test passed! Generated " << candidates.size() << " shapes." << std::endl;
	}
	else
	{
		std::cout << "Random board disabled. Use --random-board to test." << std::endl;
	}

	return 0;
}
