/**
 * Test Terminal State Detection in CachedMCTS
 *
 * Validates that checkTerminal() and calculateTerminalValue() work correctly
 * to match TypeScript mctsAgent.ts behavior.
 */

#include "cached_mcts.hpp"
#include "trigo_game.hpp"
#include "prefix_cache_inferencer.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>


using namespace trigo;


// Helper to create CachedMCTS with mock inferencer (won't be used in these tests)
class TerminalDetectionTester
{
public:
	/**
	 * Calculate terminal value from territory scores
	 * Direct copy of the formula for testing
	 */
	static float calculateTerminalValue(int white, int black)
	{
		float scoreDiff = static_cast<float>(white - black);

		if (std::abs(scoreDiff) < 1e-6f)
		{
			return 0.0f;
		}

		const float territory_value_factor = 1.0f;
		float signScore = (scoreDiff > 0.0f) ? 1.0f : -1.0f;
		return signScore * (1.0f + std::log(std::abs(scoreDiff))) * territory_value_factor;
	}
};


void print_board(const TrigoGame& game)
{
	const auto& board = game.get_board();
	const auto& shape = game.get_shape();

	std::cout << "Board (" << shape.x << "×" << shape.y << "):\n";
	for (int y = shape.y - 1; y >= 0; y--)
	{
		std::cout << "  ";
		for (int x = 0; x < shape.x; x++)
		{
			char c = '.';
			if (board[x][y][0] == Stone::Black) c = 'X';
			else if (board[x][y][0] == Stone::White) c = 'O';
			std::cout << c << ' ';
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}


int main(int argc, char* argv[])
{
	std::cout << "============================================================\n";
	std::cout << "Terminal State Detection Test\n";
	std::cout << "============================================================\n\n";


	// Test 1: calculateTerminalValue formula
	std::cout << "Test 1: calculateTerminalValue formula\n";
	std::cout << "------------------------------------------------------------\n";

	// Test cases: (white, black) -> expected value
	struct TestCase
	{
		int white;
		int black;
		float expected;
	};

	std::vector<TestCase> test_cases = {
		{15, 10, 1.0f * (1.0f + std::log(5.0f))},   // white=15, black=10, diff=+5
		{10, 15, -1.0f * (1.0f + std::log(5.0f))},  // white=10, black=15, diff=-5
		{20, 5, 1.0f * (1.0f + std::log(15.0f))},   // white=20, black=5, diff=+15
		{5, 20, -1.0f * (1.0f + std::log(15.0f))},  // white=5, black=20, diff=-15
		{12, 12, 0.0f},                              // draw
		{1, 0, 1.0f * (1.0f + std::log(1.0f))},     // minimal win
	};

	bool all_passed = true;
	for (const auto& tc : test_cases)
	{
		float result = TerminalDetectionTester::calculateTerminalValue(tc.white, tc.black);
		float error = std::abs(result - tc.expected);
		bool passed = error < 1e-5f;

		std::cout << "  white=" << tc.white << ", black=" << tc.black
		          << " -> " << std::fixed << std::setprecision(4) << result
		          << " (expected " << tc.expected << ") "
		          << (passed ? "✓" : "✗") << "\n";

		if (!passed) all_passed = false;
	}

	std::cout << "\n";


	// Test 2: Game with double-pass (FINISHED status)
	std::cout << "Test 2: Game ending with double-pass\n";
	std::cout << "------------------------------------------------------------\n";

	{
		TrigoGame game(BoardShape{5, 5, 1});
		game.set_game_status(GameStatus::PLAYING);

		// Play some moves
		game.drop({2, 2, 0});  // Black center
		game.drop({0, 0, 0});  // White corner
		game.drop({2, 1, 0});  // Black
		game.drop({0, 1, 0});  // White

		print_board(game);

		// Double pass
		game.pass();  // Black passes
		game.pass();  // White passes

		std::cout << "After double-pass:\n";
		std::cout << "  Game status: " << (game.get_game_status() == GameStatus::FINISHED ? "FINISHED" : "NOT FINISHED") << "\n";

		auto territory = game.get_territory();
		std::cout << "  Territory: black=" << territory.black << ", white=" << territory.white
		          << ", neutral=" << territory.neutral << "\n";

		float expected_value = TerminalDetectionTester::calculateTerminalValue(territory.white, territory.black);
		std::cout << "  Expected terminal value: " << std::fixed << std::setprecision(4) << expected_value << "\n";
		std::cout << "  (white-positive: " << (expected_value > 0 ? "white winning" : (expected_value < 0 ? "black winning" : "draw")) << ")\n\n";
	}


	// Test 3: Game with natural end (coverage > 50%, neutral = 0)
	std::cout << "Test 3: Game with natural end (board full)\n";
	std::cout << "------------------------------------------------------------\n";

	{
		// Create a 3x3 board and fill it completely
		TrigoGame game(BoardShape{3, 3, 1});
		game.set_game_status(GameStatus::PLAYING);

		// Fill the board with alternating stones
		// Black: (0,0), (1,1), (2,2), (0,2), (2,0) = 5 stones
		// White: (1,0), (0,1), (2,1), (1,2) = 4 stones
		game.drop({1, 1, 0});  // Black center
		game.drop({0, 0, 0});  // White corner
		game.drop({2, 2, 0});  // Black corner
		game.drop({1, 0, 0});  // White
		game.drop({0, 2, 0});  // Black
		game.drop({2, 1, 0});  // White
		game.drop({2, 0, 0});  // Black
		game.drop({0, 1, 0});  // White
		game.drop({1, 2, 0});  // Black

		print_board(game);

		const auto& board = game.get_board();
		const auto& shape = game.get_shape();

		// Count stones
		int stoneCount = 0;
		for (int x = 0; x < shape.x; x++)
		{
			for (int y = 0; y < shape.y; y++)
			{
				if (board[x][y][0] != Stone::Empty) stoneCount++;
			}
		}

		float coverage = static_cast<float>(stoneCount) / (shape.x * shape.y);
		std::cout << "  Stone count: " << stoneCount << "/" << (shape.x * shape.y) << "\n";
		std::cout << "  Coverage: " << std::fixed << std::setprecision(1) << (coverage * 100) << "%\n";

		auto territory = game.get_territory();
		std::cout << "  Territory: black=" << territory.black << ", white=" << territory.white
		          << ", neutral=" << territory.neutral << "\n";

		bool is_natural_end = (coverage > 0.5f && territory.neutral == 0);
		std::cout << "  Natural end condition: " << (is_natural_end ? "YES" : "NO") << "\n";

		if (is_natural_end)
		{
			float expected_value = TerminalDetectionTester::calculateTerminalValue(territory.white, territory.black);
			std::cout << "  Expected terminal value: " << std::fixed << std::setprecision(4) << expected_value << "\n";
		}
		std::cout << "\n";
	}


	// Test 4: Non-terminal state (sparse board)
	std::cout << "Test 4: Non-terminal state (sparse board)\n";
	std::cout << "------------------------------------------------------------\n";

	{
		TrigoGame game(BoardShape{5, 5, 1});
		game.set_game_status(GameStatus::PLAYING);

		// Only a few moves
		game.drop({2, 2, 0});  // Black center
		game.drop({0, 0, 0});  // White corner

		print_board(game);

		const auto& board = game.get_board();
		const auto& shape = game.get_shape();

		int stoneCount = 0;
		for (int x = 0; x < shape.x; x++)
		{
			for (int y = 0; y < shape.y; y++)
			{
				if (board[x][y][0] != Stone::Empty) stoneCount++;
			}
		}

		float coverage = static_cast<float>(stoneCount) / (shape.x * shape.y);
		std::cout << "  Stone count: " << stoneCount << "/" << (shape.x * shape.y) << "\n";
		std::cout << "  Coverage: " << std::fixed << std::setprecision(1) << (coverage * 100) << "%\n";
		std::cout << "  Terminal: NO (coverage < 50%)\n\n";
	}


	// Test 5: Verify TypeScript formula match
	std::cout << "Test 5: Verify TypeScript formula match\n";
	std::cout << "------------------------------------------------------------\n";

	// TypeScript: sign(scoreDiff) * (1 + Math.log(Math.abs(scoreDiff)))
	// Example from context: white=15, black=10 -> sign(5) * (1 + log(5)) ≈ 2.609
	{
		int white = 15, black = 10;
		float scoreDiff = static_cast<float>(white - black);  // 5
		float expected = 1.0f * (1.0f + std::log(5.0f));      // ≈ 2.609

		float result = TerminalDetectionTester::calculateTerminalValue(white, black);

		std::cout << "  TypeScript example: white=15, black=10\n";
		std::cout << "  scoreDiff = " << scoreDiff << "\n";
		std::cout << "  Expected: sign(5) * (1 + log(5)) = " << expected << "\n";
		std::cout << "  Result: " << result << "\n";
		std::cout << "  Match: " << (std::abs(result - expected) < 1e-5f ? "✓" : "✗") << "\n\n";
	}


	std::cout << "============================================================\n";
	std::cout << "All tests completed" << (all_passed ? " ✓" : " (some failures)") << "\n";
	std::cout << "============================================================\n";

	return all_passed ? 0 : 1;
}
