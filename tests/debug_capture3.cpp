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

	// Set up the board state exactly as the test does
	set_stone(board, Position(2,2,2), Stone::Black);
	set_stone(board, Position(2,3,2), Stone::White);
	set_stone(board, Position(1,3,2), Stone::Black);
	set_stone(board, Position(2,4,2), Stone::Black);
	set_stone(board, Position(3,3,2), Stone::Black);

	std::cout << "Board with all stones:" << std::endl;
	print_board_slice(board, shape, 2);

	// Print neighbors of white stone
	auto neighbors = get_neighbors(Position(2,3,2), shape);
	std::cout << "Neighbors of white stone at (2,3,2):" << std::endl;
	for (const auto& n : neighbors)
	{
		Stone s = board[n.x][n.y][n.z];
		std::cout << "  (" << n.x << "," << n.y << "," << n.z << "): "
		          << (s == Stone::Empty ? "Empty" : s == Stone::Black ? "Black" : "White") << std::endl;
	}

	// Get the white group and its liberties
	auto group = find_group(Position(2,3,2), board, shape);
	auto liberties = group.get_liberties(board, shape);
	
	std::cout << "\nLiberties of white group:" << std::endl;
	liberties.for_each([](const Position& p) {
		std::cout << "  (" << p.x << "," << p.y << "," << p.z << ")" << std::endl;
	});

	return 0;
}
