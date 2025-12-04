/**
 * Test suite for TrigoGame class
 *
 * Comprehensive tests for game state management, move validation,
 * undo/redo, territory calculation, and game flow.
 */

#include "../include/trigo_game.hpp"
#include <iostream>
#include <cassert>
#include <string>


using namespace trigo;


// === Test Helpers ===

void test_assert(bool condition, const std::string& testName)
{
	if (!condition)
	{
		std::cerr << "❌ FAILED: " << testName << std::endl;
		std::exit(1);
	}
	std::cout << "✓ " << testName << std::endl;
}


// === Basic Construction and Initialization ===

void test_constructor_default()
{
	TrigoGame game;

	test_assert(game.get_shape().x == 5, "Default shape x = 5");
	test_assert(game.get_shape().y == 5, "Default shape y = 5");
	test_assert(game.get_shape().z == 5, "Default shape z = 5");
	test_assert(game.get_current_player() == Stone::Black, "Current player is Black");
	test_assert(game.get_current_step() == 0, "Initial step is 0");
	test_assert(game.get_game_status() == GameStatus::IDLE, "Game status is IDLE");
}


void test_constructor_custom_shape()
{
	TrigoGame game(BoardShape{19, 19, 1});

	test_assert(game.get_shape().x == 19, "Custom shape x = 19");
	test_assert(game.get_shape().y == 19, "Custom shape y = 19");
	test_assert(game.get_shape().z == 1, "Custom shape z = 1");
}


void test_empty_board()
{
	TrigoGame game(BoardShape{3, 3, 3});
	auto board = game.get_board();

	// Check all positions are empty
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int z = 0; z < 3; z++)
			{
				test_assert(board[x][y][z] == Stone::Empty, "Board position is empty");
			}
		}
	}
}


// === Basic Move Operations ===

void test_drop_single_stone()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	Position pos(2, 2, 2);
	bool result = game.drop(pos);

	test_assert(result, "Drop returns true");
	test_assert(game.get_stone(pos) == Stone::Black, "Stone placed is black");
	test_assert(game.get_current_player() == Stone::White, "Current player switched to white");
	test_assert(game.get_current_step() == 1, "Step count is 1");
}


void test_drop_alternating_players()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Black's move
	bool result1 = game.drop(Position(2, 2, 2));
	test_assert(result1 && game.get_current_player() == Stone::White, "Black played, white's turn");

	// White's move
	bool result2 = game.drop(Position(2, 2, 3));
	test_assert(result2 && game.get_current_player() == Stone::Black, "White played, black's turn");
}


void test_invalid_drop_occupied()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	Position pos(2, 2, 2);
	game.drop(pos);

	// Try to place stone on occupied position
	bool result = game.drop(pos);
	test_assert(!result, "Cannot drop on occupied position");
	test_assert(game.get_current_step() == 1, "Step count unchanged after invalid move");
}


void test_invalid_drop_out_of_bounds()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	bool result = game.drop(Position(10, 10, 10));
	test_assert(!result, "Cannot drop out of bounds");
}


// === Pass and Surrender ===

void test_pass_move()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	bool result = game.pass();

	test_assert(result, "Pass returns true");
	test_assert(game.get_current_player() == Stone::White, "Player switched after pass");
	test_assert(game.get_pass_count() == 1, "Pass count is 1");
	test_assert(game.get_game_status() == GameStatus::PLAYING, "Game still playing after one pass");
}


void test_double_pass_ends_game()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.pass(); // Black passes
	game.pass(); // White passes

	test_assert(game.get_pass_count() == 2, "Pass count is 2");
	test_assert(game.get_game_status() == GameStatus::FINISHED, "Game finished after double pass");
	test_assert(game.get_game_result().has_value(), "Game result is set");
}


void test_pass_count_reset_after_drop()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.pass(); // Pass count = 1
	test_assert(game.get_pass_count() == 1, "Pass count is 1");

	game.drop(Position(2, 2, 2)); // Drop resets pass count
	test_assert(game.get_pass_count() == 0, "Pass count reset after drop");
}


void test_surrender()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Black surrenders
	bool result = game.surrender();

	test_assert(result, "Surrender returns true");
	test_assert(game.get_game_status() == GameStatus::FINISHED, "Game finished after surrender");

	auto gameResult = game.get_game_result();
	test_assert(gameResult.has_value(), "Game result is set");
	test_assert(gameResult->winner == GameResult::Winner::White, "White wins after black surrenders");
	test_assert(gameResult->reason == GameResult::Reason::Resignation, "Win reason is resignation");
}


// === Capture Detection ===

void test_simple_capture()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Create a proper 3D capture scenario
	// White stone at (1,2,2) will be surrounded from all 6 directions:
	// (0,2,2), (2,2,2), (1,1,2), (1,3,2), (1,2,1), (1,2,3)

	game.drop(Position(2, 2, 2)); // Black right of target
	game.drop(Position(1, 2, 2)); // White (will be captured)

	game.drop(Position(0, 2, 2)); // Black left of white
	game.drop(Position(4, 4, 4)); // White elsewhere

	game.drop(Position(1, 3, 2)); // Black front of white
	game.drop(Position(4, 4, 3)); // White elsewhere

	game.drop(Position(1, 1, 2)); // Black back of white
	game.drop(Position(4, 4, 2)); // White elsewhere

	game.drop(Position(1, 2, 3)); // Black top of white
	game.drop(Position(4, 3, 4)); // White elsewhere

	game.drop(Position(1, 2, 1)); // Black bottom - captures white at (1,2,2)

	// Check white stone was captured
	test_assert(game.get_stone(Position(1, 2, 2)) == Stone::Empty, "Captured stone removed");

	// Check capture count
	auto counts = game.get_captured_counts();
	test_assert(counts.white == 1, "White captured count is 1");
	test_assert(counts.black == 0, "Black captured count is 0");
}


// === Undo/Redo ===

void test_undo_single_move()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	Position pos(2, 2, 2);
	game.drop(pos);

	bool result = game.undo();

	test_assert(result, "Undo returns true");
	test_assert(game.get_stone(pos) == Stone::Empty, "Stone removed after undo");
	test_assert(game.get_current_player() == Stone::Black, "Player reverted to black");
	test_assert(game.get_current_step() == 0, "Step count back to 0");
}


void test_undo_multiple_moves()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	Position pos1(2, 2, 2);
	Position pos2(2, 2, 3);
	game.drop(pos1);
	game.drop(pos2);

	game.undo();
	test_assert(game.get_stone(pos2) == Stone::Empty, "Second stone removed");
	test_assert(game.get_current_step() == 1, "Step count is 1");

	game.undo();
	test_assert(game.get_stone(pos1) == Stone::Empty, "First stone removed");
	test_assert(game.get_current_step() == 0, "Step count is 0");
}


void test_undo_restores_captures()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Use same scenario as test_simple_capture
	// White stone at (1,2,2) will be surrounded and captured

	game.drop(Position(2, 2, 2)); // Black
	game.drop(Position(1, 2, 2)); // White (will be captured)

	game.drop(Position(0, 2, 2)); // Black
	game.drop(Position(4, 4, 4)); // White elsewhere

	game.drop(Position(1, 3, 2)); // Black
	game.drop(Position(4, 4, 3)); // White elsewhere

	game.drop(Position(1, 1, 2)); // Black
	game.drop(Position(4, 4, 2)); // White elsewhere

	game.drop(Position(1, 2, 3)); // Black
	game.drop(Position(4, 3, 4)); // White elsewhere

	game.drop(Position(1, 2, 1)); // Black - captures white

	// Verify capture
	test_assert(game.get_stone(Position(1, 2, 2)) == Stone::Empty, "Stone captured");

	// Undo capture move
	game.undo();

	// Verify stone restored
	test_assert(game.get_stone(Position(1, 2, 2)) == Stone::White, "Captured stone restored after undo");
}


void test_redo_after_undo()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	Position pos(2, 2, 2);
	game.drop(pos);

	game.undo();
	test_assert(game.get_stone(pos) == Stone::Empty, "Stone removed after undo");

	bool result = game.redo();
	test_assert(result, "Redo returns true");
	test_assert(game.get_stone(pos) == Stone::Black, "Stone restored after redo");
	test_assert(game.get_current_step() == 1, "Step count back to 1");
}


void test_can_redo()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	test_assert(!game.can_redo(), "Cannot redo initially");

	game.drop(Position(2, 2, 2));
	test_assert(!game.can_redo(), "Cannot redo after move");

	game.undo();
	test_assert(game.can_redo(), "Can redo after undo");

	game.redo();
	test_assert(!game.can_redo(), "Cannot redo after redo");
}


// === Jump to Step ===

void test_jump_to_step()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Make 4 moves
	game.drop(Position(2, 2, 2)); // Step 1
	game.drop(Position(2, 2, 3)); // Step 2
	game.drop(Position(2, 3, 2)); // Step 3
	game.drop(Position(3, 2, 2)); // Step 4

	// Jump to step 2 (after 2 moves)
	bool result = game.jump_to_step(2);

	test_assert(result, "Jump to step returns true");
	test_assert(game.get_current_step() == 2, "Current step is 2");
	test_assert(game.get_stone(Position(2, 2, 2)) == Stone::Black, "First stone present");
	test_assert(game.get_stone(Position(2, 2, 3)) == Stone::White, "Second stone present");
	test_assert(game.get_stone(Position(2, 3, 2)) == Stone::Empty, "Third stone absent");
	test_assert(game.get_stone(Position(3, 2, 2)) == Stone::Empty, "Fourth stone absent");
}


void test_jump_to_initial_state()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.drop(Position(2, 2, 2));
	game.drop(Position(2, 2, 3));

	bool result = game.jump_to_step(0);

	test_assert(result, "Jump to step 0 returns true");
	test_assert(game.get_current_step() == 0, "Current step is 0");
	test_assert(game.get_stone(Position(2, 2, 2)) == Stone::Empty, "Board is empty");
}


// === History ===

void test_history_tracking()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.drop(Position(2, 2, 2));
	game.drop(Position(2, 2, 3));

	auto history = game.get_history();

	test_assert(history.size() == 2, "History has 2 moves");
	test_assert(history[0].type == StepType::DROP, "First move is DROP");
	test_assert(history[0].player == Stone::Black, "First move by black");
	test_assert(history[1].player == Stone::White, "Second move by white");
}


void test_last_step()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	auto lastStep1 = game.get_last_step();
	test_assert(!lastStep1.has_value(), "No last step initially");

	game.drop(Position(2, 2, 2));

	auto lastStep2 = game.get_last_step();
	test_assert(lastStep2.has_value(), "Last step exists after move");
	test_assert(lastStep2->type == StepType::DROP, "Last step is DROP");
	test_assert(lastStep2->player == Stone::Black, "Last step by black");
}


// === Territory Calculation ===

void test_territory_calculation()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	// Place some stones
	game.drop(Position(1, 1, 1));
	game.drop(Position(3, 3, 3));

	auto territory = game.get_territory();

	// Basic sanity check
	test_assert(territory.black >= 0, "Black territory non-negative");
	test_assert(territory.white >= 0, "White territory non-negative");
	test_assert(territory.neutral >= 0, "Neutral territory non-negative");

	int total = territory.black + territory.white + territory.neutral;
	int boardSize = 5 * 5 * 5; // Territory includes the stones themselves

	test_assert(total == boardSize, "Territory sums to board size");
}


// === Clone and Reset ===

void test_clone()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.drop(Position(2, 2, 2));

	TrigoGame cloned = game.clone();

	test_assert(cloned.get_current_step() == 1, "Cloned game has same step");
	test_assert(cloned.get_stone(Position(2, 2, 2)) == Stone::Black, "Cloned game has same board");

	// Verify independence
	game.drop(Position(2, 2, 3));
	test_assert(cloned.get_current_step() == 1, "Cloned game unaffected by original");
}


void test_reset()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.drop(Position(2, 2, 2));
	game.drop(Position(2, 2, 3));

	game.reset();

	test_assert(game.get_current_step() == 0, "Step reset to 0");
	test_assert(game.get_game_status() == GameStatus::IDLE, "Status reset to IDLE");
	test_assert(game.get_stone(Position(2, 2, 2)) == Stone::Empty, "Board cleared");
	test_assert(game.get_history().empty(), "History cleared");
}


// === Statistics ===

void test_game_stats()
{
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	game.drop(Position(2, 2, 2)); // Black
	game.drop(Position(2, 2, 3)); // White
	game.drop(Position(2, 3, 2)); // Black

	auto stats = game.get_stats();

	test_assert(stats.totalMoves == 3, "Total moves is 3");
	test_assert(stats.blackMoves == 2, "Black made 2 moves");
	test_assert(stats.whiteMoves == 1, "White made 1 move");
}


// === Valid Move Positions ===

void test_valid_move_positions()
{
	TrigoGame game(BoardShape{3, 3, 3});
	game.start_game();

	auto validPos = game.valid_move_positions();

	// Initially all positions should be valid (3^3 = 27)
	test_assert(validPos.size() == 27, "All 27 positions initially valid");

	// Place a stone
	game.drop(Position(1, 1, 1));

	validPos = game.valid_move_positions();
	test_assert(validPos.size() == 26, "26 positions valid after one move");
}


// === Main Test Runner ===

int main()
{
	std::cout << "\n=== TrigoGame Test Suite ===\n" << std::endl;

	// Basic construction
	std::cout << "--- Basic Construction ---" << std::endl;
	test_constructor_default();
	test_constructor_custom_shape();
	test_empty_board();

	// Basic moves
	std::cout << "\n--- Basic Move Operations ---" << std::endl;
	test_drop_single_stone();
	test_drop_alternating_players();
	test_invalid_drop_occupied();
	test_invalid_drop_out_of_bounds();

	// Pass and surrender
	std::cout << "\n--- Pass and Surrender ---" << std::endl;
	test_pass_move();
	test_double_pass_ends_game();
	test_pass_count_reset_after_drop();
	test_surrender();

	// Captures
	std::cout << "\n--- Capture Detection ---" << std::endl;
	test_simple_capture();

	// Undo/Redo
	std::cout << "\n--- Undo/Redo ---" << std::endl;
	test_undo_single_move();
	test_undo_multiple_moves();
	test_undo_restores_captures();
	test_redo_after_undo();
	test_can_redo();

	// Jump to step
	std::cout << "\n--- Jump to Step ---" << std::endl;
	test_jump_to_step();
	test_jump_to_initial_state();

	// History
	std::cout << "\n--- History ---" << std::endl;
	test_history_tracking();
	test_last_step();

	// Territory
	std::cout << "\n--- Territory ---" << std::endl;
	test_territory_calculation();

	// Clone and reset
	std::cout << "\n--- Clone and Reset ---" << std::endl;
	test_clone();
	test_reset();

	// Statistics
	std::cout << "\n--- Statistics ---" << std::endl;
	test_game_stats();

	// Valid moves
	std::cout << "\n--- Valid Move Positions ---" << std::endl;
	test_valid_move_positions();

	std::cout << "\n=== All Tests Passed! ===\n" << std::endl;

	return 0;
}
