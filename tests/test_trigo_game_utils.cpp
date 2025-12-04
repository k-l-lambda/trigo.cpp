/**
 * Test Trigo Game Utils - Go Rules
 */

#include "../include/trigo_types.hpp"
#include "../include/trigo_game_utils.hpp"
#include <iostream>
#include <cassert>


using namespace trigo;


void test_board_creation()
{
	std::cout << "\n[TEST] Board Creation\n";
	std::cout << "=====================\n";

	BoardShape shape(5, 5, 5);
	Board board = create_board(shape);

	// Verify all positions are empty
	for (int x = 0; x < 5; x++)
	{
		for (int y = 0; y < 5; y++)
		{
			for (int z = 0; z < 5; z++)
			{
				Position pos(x, y, z);
				assert(get_stone(board, pos) == Stone::Empty);
			}
		}
	}

	std::cout << "✓ Created 5×5×5 board with all empty positions\n";
}


void test_neighbors()
{
	std::cout << "\n[TEST] Neighbor Detection\n";
	std::cout << "==========================\n";

	BoardShape shape(5, 5, 5);

	// Center position has 6 neighbors
	Position center(2, 2, 2);
	auto neighbors = get_neighbors(center, shape);
	std::cout << "Center (2,2,2) has " << neighbors.size() << " neighbors\n";
	assert(neighbors.size() == 6);

	// Corner has 3 neighbors
	Position corner(0, 0, 0);
	neighbors = get_neighbors(corner, shape);
	std::cout << "Corner (0,0,0) has " << neighbors.size() << " neighbors\n";
	assert(neighbors.size() == 3);

	// Edge has 4 or 5 neighbors
	Position edge(0, 2, 2);
	neighbors = get_neighbors(edge, shape);
	std::cout << "Edge (0,2,2) has " << neighbors.size() << " neighbors\n";
	assert(neighbors.size() == 5);

	std::cout << "✓ Neighbor detection correct\n";
}


void test_capture_single_stone()
{
	std::cout << "\n[TEST] Capture Single Stone\n";
	std::cout << "============================\n";

	BoardShape shape(5, 5, 5);
	Board board = create_board(shape);

	// Place white stone at center
	Position center(2, 2, 2);
	set_stone(board, center, Stone::White);

	// Surround with black stones (5 of 6 neighbors)
	set_stone(board, Position(1, 2, 2), Stone::Black);
	set_stone(board, Position(3, 2, 2), Stone::Black);
	set_stone(board, Position(2, 1, 2), Stone::Black);
	set_stone(board, Position(2, 3, 2), Stone::Black);
	set_stone(board, Position(2, 2, 1), Stone::Black);

	// White stone has 1 liberty remaining
	Patch white_group = find_group(center, board, shape);
	auto liberties = white_group.get_liberties(board, shape);
	std::cout << "White stone has " << liberties.size() << " liberty\n";
	assert(liberties.size() == 1);

	// Place final black stone - should capture white
	Position last_move(2, 2, 3);
	auto captured = find_captured_groups(last_move, Stone::Black, board, shape);
	std::cout << "Captured " << captured.size() << " group(s)\n";
	assert(captured.size() == 1);
	assert(captured[0].size() == 1);

	std::cout << "✓ Single stone capture detected correctly\n";
}


void test_capture_group()
{
	std::cout << "\n[TEST] Capture Group of Stones\n";
	std::cout << "===============================\n";

	BoardShape shape(5, 5, 5);
	Board board = create_board(shape);

	// Create a white group (3 stones in a line)
	set_stone(board, Position(2, 2, 2), Stone::White);
	set_stone(board, Position(3, 2, 2), Stone::White);
	set_stone(board, Position(4, 2, 2), Stone::White);

	// Find the group
	Patch white_group = find_group(Position(2, 2, 2), board, shape);
	std::cout << "White group size: " << white_group.size() << "\n";
	assert(white_group.size() == 3);

	// Surround partially with black
	set_stone(board, Position(1, 2, 2), Stone::Black);
	set_stone(board, Position(2, 1, 2), Stone::Black);
	set_stone(board, Position(2, 3, 2), Stone::Black);
	set_stone(board, Position(3, 1, 2), Stone::Black);
	set_stone(board, Position(3, 3, 2), Stone::Black);
	set_stone(board, Position(4, 1, 2), Stone::Black);
	set_stone(board, Position(4, 3, 2), Stone::Black);

	// Still has liberties in z direction
	auto liberties = white_group.get_liberties(board, shape);
	std::cout << "Group has " << liberties.size() << " liberties\n";
	assert(liberties.size() > 0);

	std::cout << "✓ Group detection and liberty calculation correct\n";
}


void test_ko_rule()
{
	std::cout << "\n[TEST] Ko Rule\n";
	std::cout << "==============\n";

	// Use 2D board like TypeScript test (5×5×1)
	BoardShape shape(5, 5, 1);
	Board board = create_board(shape);

	// Ko example from TypeScript test:
	// Board setup:    After BLACK captures:
	//   a b c           a b c
	// a . W B         a B . B   (WHITE at ba captured)
	// b W B .         b W B .

	// Place stones (all at z=0)
	set_stone(board, Position(1, 1, 0), Stone::Black);  // bb - Black
	set_stone(board, Position(0, 1, 0), Stone::White);  // ab - White
	set_stone(board, Position(2, 0, 0), Stone::Black);  // ca - Black
	set_stone(board, Position(1, 0, 0), Stone::White);  // ba - White (will be captured)

	// BLACK at aa captures WHITE at ba
	Position capture_move(0, 0, 0);  // aa
	auto captured = find_captured_groups(capture_move, Stone::Black, board, shape);

	std::cout << "Black captures " << captured.size() << " group(s)\n";
	if (!captured.empty())
	{
		std::cout << "  Group size: " << captured[0].size() << " stone(s)\n";
	}
	assert(captured.size() == 1);
	assert(captured[0].size() == 1);

	// Execute capture
	auto captured_positions = execute_captures(captured, board);
	set_stone(board, capture_move, Stone::Black);

	// White cannot immediately recapture at ba (Ko violation)
	Position ko_move(1, 0, 0);  // ba
	bool is_ko = is_ko_violation(ko_move, Stone::White, board, shape, &captured_positions);
	std::cout << "Ko violation detected: " << (is_ko ? "yes" : "no") << "\n";
	assert(is_ko == true);

	std::cout << "✓ Ko rule correctly prevents immediate recapture\n";
}


void test_suicide_prevention()
{
	std::cout << "\n[TEST] Suicide Prevention\n";
	std::cout << "=========================\n";

	// Use 2D board (5×5×1) to properly test suicide
	BoardShape shape(5, 5, 1);
	Board board = create_board(shape);

	// Create situation where placing stone would be suicide
	// . B .
	// B . B
	// . B .
	set_stone(board, Position(2, 1, 0), Stone::Black);
	set_stone(board, Position(1, 2, 0), Stone::Black);
	set_stone(board, Position(3, 2, 0), Stone::Black);
	set_stone(board, Position(2, 3, 0), Stone::Black);

	// White trying to play in center would be suicide
	Position suicide_move(2, 2, 0);
	bool is_suicide = is_suicide_move(suicide_move, Stone::White, board, shape);
	std::cout << "Move is suicide: " << (is_suicide ? "yes" : "no") << "\n";
	assert(is_suicide == true);

	std::cout << "✓ Suicide move correctly detected\n";
}


void test_territory_calculation()
{
	std::cout << "\n[TEST] Territory Calculation\n";
	std::cout << "============================\n";

	BoardShape shape(5, 5, 5);
	Board board = create_board(shape);

	// Create simple territory
	// Place black stones in one corner
	set_stone(board, Position(0, 0, 0), Stone::Black);
	set_stone(board, Position(1, 0, 0), Stone::Black);
	set_stone(board, Position(0, 1, 0), Stone::Black);

	// Place white stones in opposite corner
	set_stone(board, Position(4, 4, 4), Stone::White);
	set_stone(board, Position(3, 4, 4), Stone::White);
	set_stone(board, Position(4, 3, 4), Stone::White);

	auto territory = calculate_territory(board, shape);

	std::cout << "Black territory: " << territory.black << "\n";
	std::cout << "White territory: " << territory.white << "\n";
	std::cout << "Neutral: " << territory.neutral << "\n";
	std::cout << "Total: " << (territory.black + territory.white + territory.neutral) << " / " << (5*5*5) << "\n";

	assert(territory.black >= 3);  // At least the 3 stones
	assert(territory.white >= 3);  // At least the 3 stones
	assert(territory.black + territory.white + territory.neutral == 125);  // Total 5×5×5

	std::cout << "✓ Territory calculation complete\n";
}


void test_move_validation()
{
	std::cout << "\n[TEST] Move Validation\n";
	std::cout << "======================\n";

	BoardShape shape(5, 5, 5);
	Board board = create_board(shape);

	// Valid move
	Position valid_pos(2, 2, 2);
	auto validation = validate_move(valid_pos, Stone::Black, board, shape);
	std::cout << "Valid move: " << (validation.valid ? "yes" : "no") << "\n";
	assert(validation.valid == true);

	// Out of bounds
	Position out_of_bounds(10, 10, 10);
	validation = validate_move(out_of_bounds, Stone::Black, board, shape);
	std::cout << "Out of bounds: " << validation.reason << "\n";
	assert(validation.valid == false);

	// Occupied position
	set_stone(board, valid_pos, Stone::White);
	validation = validate_move(valid_pos, Stone::Black, board, shape);
	std::cout << "Occupied: " << validation.reason << "\n";
	assert(validation.valid == false);

	std::cout << "✓ Move validation working correctly\n";
}


int main()
{
	std::cout << "Trigo Game Utils Test Suite\n";
	std::cout << "============================\n";

	try
	{
		test_board_creation();
		test_neighbors();
		test_capture_single_stone();
		test_capture_group();
		test_ko_rule();
		test_suicide_prevention();
		test_territory_calculation();
		test_move_validation();

		std::cout << "\n" << std::string(70, '=') << "\n";
		std::cout << "✅ ALL TESTS PASSED!\n";
		std::cout << "Go rules implementation is working correctly.\n";
		std::cout << std::string(70, '=') << "\n";

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
		return 1;
	}
}
