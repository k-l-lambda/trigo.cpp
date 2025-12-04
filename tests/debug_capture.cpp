/**
 * Simple debug test for capture
 */

#include "../include/trigo_game.hpp"
#include <iostream>

using namespace trigo;

void print_board_slice(const TrigoGame& game, int z)
{
	auto shape = game.get_shape();
	std::cout << "Board at z=" << z << ":" << std::endl;
	for (int y = shape.y - 1; y >= 0; y--)
	{
		for (int x = 0; x < shape.x; x++)
		{
			Stone stone = game.get_stone(Position(x, y, z));
			char c = stone == Stone::Empty ? '.' :
			         stone == Stone::Black ? 'B' : 'W';
			std::cout << c << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	std::cout << "=== Testing Simple Capture ===" << std::endl;

	// Simpler test: surround a white stone in 2D plane
	std::cout << "\n1. Black at (2,2,2)" << std::endl;
	game.drop(Position(2, 2, 2));
	print_board_slice(game, 2);

	std::cout << "2. White at (2,3,2)" << std::endl;
	game.drop(Position(2, 3, 2));
	print_board_slice(game, 2);

	std::cout << "3. Black at (1,3,2)" << std::endl;
	game.drop(Position(1, 3, 2));
	print_board_slice(game, 2);

	std::cout << "4. White at (0,0,0)" << std::endl;
	game.drop(Position(0, 0, 0));
	print_board_slice(game, 2);

	std::cout << "5. Black at (2,4,2)" << std::endl;
	game.drop(Position(2, 4, 2));
	print_board_slice(game, 2);

	std::cout << "6. White at (0,0,1)" << std::endl;
	game.drop(Position(0, 0, 1));
	print_board_slice(game, 2);

	std::cout << "7. Black at (3,3,2) - should capture white at (2,3,2)" << std::endl;

	// Manual check before drop
	{
		Board temp_board = create_board(game.get_shape());
		// Manually reconstruct current state
		set_stone(temp_board, Position(2,2,2), Stone::Black);
		set_stone(temp_board, Position(2,3,2), Stone::White);
		set_stone(temp_board, Position(1,3,2), Stone::Black);
		set_stone(temp_board, Position(0,0,0), Stone::White);
		set_stone(temp_board, Position(2,4,2), Stone::Black);
		set_stone(temp_board, Position(0,0,1), Stone::White);

		// Simulate placing black at (3,3,2)
		auto captured = find_captured_groups(Position(3,3,2), Stone::Black, temp_board, game.get_shape());
		std::cout << "Manual check: Found " << captured.size() << " captured groups" << std::endl;
		for (const auto& group : captured)
		{
			std::cout << "  Group size: " << group.positions.size() << std::endl;
		}
	}

	game.drop(Position(3, 3, 2));
	print_board_slice(game, 2);

	// Check if white stone was captured
	Stone stone_at_target = game.get_stone(Position(2, 3, 2));
	std::cout << "Stone at (2,3,2): " << (int)stone_at_target << " (0=Empty, 1=Black, 2=White)" << std::endl;

	if (stone_at_target == Stone::Empty)
	{
		std::cout << "✓ Capture successful!" << std::endl;
	}
	else
	{
		std::cout << "❌ Capture failed - stone still present" << std::endl;

		// Print neighbors
		auto neighbors = get_neighbors(Position(2, 3, 2), game.get_shape());
		std::cout << "\nNeighbors of (2,3,2):" << std::endl;
		for (const auto& n : neighbors)
		{
			Stone nStone = game.get_stone(n);
			std::cout << "  (" << n.x << "," << n.y << "," << n.z << "): "
			          << (int)nStone << std::endl;
		}
	}

	return 0;
}
