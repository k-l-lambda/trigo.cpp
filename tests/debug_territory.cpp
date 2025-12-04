#include "../include/trigo_game.hpp"
#include <iostream>

using namespace trigo;

int main()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Place some stones
	game.drop(Position(1, 1, 1));
	game.drop(Position(3, 3, 3));

	auto territory = game.get_territory();

	std::cout << "Territory:" << std::endl;
	std::cout << "  Black: " << territory.black << std::endl;
	std::cout << "  White: " << territory.white << std::endl;
	std::cout << "  Neutral: " << territory.neutral << std::endl;

	int total = territory.black + territory.white + territory.neutral;
	int boardSize = 5 * 5 * 5;

	std::cout << "\nTotal territory: " << total << std::endl;
	std::cout << "Board size: " << boardSize << std::endl;
	std::cout << "Stones placed: 2" << std::endl;
	std::cout << "Expected (without stones): " << (boardSize - 2) << std::endl;

	return 0;
}
