#include "../include/trigo_game_utils.hpp"
#include <iostream>

using namespace trigo;

void print_board_slice(const Board& board, const BoardShape& shape, int z)
{
	std::cout << "Board at z=" << z << ":" << std::endl;
	for (int y = shape.y - 1; y >= 0; y--)
	{
		for (int x = 0; x < shape.x; x++)
		{
			Stone stone = board[x][y][z];
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
	BoardShape shape{5, 5, 5};
	Board board = create_board(shape);

	// Set up the board state
	set_stone(board, Position(2,2,2), Stone::Black);
	set_stone(board, Position(2,3,2), Stone::White);
	set_stone(board, Position(1,3,2), Stone::Black);
	set_stone(board, Position(2,4,2), Stone::Black);

	std::cout << "Initial board (z=2):" << std::endl;
	print_board_slice(board, shape, 2);

	// Check white stone at (2,3,2)
	Position white_pos(2, 3, 2);
	auto group = find_group(white_pos, board, shape);

	std::cout << "White group at (2,3,2):" << std::endl;
	std::cout << "  Color: " << (int)group.color << std::endl;
	std::cout << "  Size: " << group.positions.size() << std::endl;

	// Calculate liberties
	auto liberties = group.get_liberties(board, shape);
	std::cout << "  Liberties: " << liberties.size() << std::endl;

	// Now place black at (3,3,2)
	set_stone(board, Position(3,3,2), Stone::Black);
	std::cout << "\nAfter placing black at (3,3,2):" << std::endl;
	print_board_slice(board, shape, 2);

	// Recount liberties
	group = find_group(white_pos, board, shape);
	liberties = group.get_liberties(board, shape);
	std::cout << "White group liberties now: " << liberties.size() << std::endl;

	// Test capture detection
	bool captured = is_group_captured(group, board, shape);
	std::cout << "Is group captured? " << (captured ? "YES" : "NO") << std::endl;

	return 0;
}
