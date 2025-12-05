/**
 * Test TGN Generation Consistency
 *
 * Verifies that game_to_tgn() and GameRecorder::to_tgn() produce
 * consistent TGN format for the same game state
 */

#include "../include/trigo_game.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/game_recorder.hpp"
#include <iostream>
#include <cassert>
#include <string>

using namespace trigo;


/**
 * Extract moves section from TGN (excluding metadata and score)
 */
std::string extract_moves_section(const std::string& tgn)
{
	// Find the start of moves (after "[Board ...]\n\n")
	size_t moves_start = tgn.find("]\n\n");
	if (moves_start == std::string::npos)
		return "";

	moves_start += 3;  // Skip "]\n\n"

	// Find the end of moves (before score comment or end)
	size_t moves_end = tgn.find(";", moves_start);
	if (moves_end == std::string::npos)
		moves_end = tgn.length();

	return tgn.substr(moves_start, moves_end - moves_start);
}


int main()
{
	std::cout << "\n=== TGN Consistency Test ===\n\n";

	// Create and play a game
	TrigoGame game(BoardShape{5, 5, 5});
	game.start_game();

	std::cout << "Playing test game...\n";

	// Play some moves
	game.drop(Position{2, 2, 2});  // Black: 000
	game.drop(Position{1, 2, 2});  // White: a00
	game.drop(Position{3, 2, 2});  // Black: z00
	game.drop(Position{2, 1, 2});  // White: 0a0
	game.drop(Position{2, 3, 2});  // Black: 0z0
	game.pass();                    // White: Pass
	game.drop(Position{2, 2, 1});  // Black: 00a
	game.drop(Position{2, 2, 3});  // White: 00z

	std::cout << "Played " << game.get_history().size() << " moves\n\n";

	// Method 1: Use game_to_tgn() directly
	std::string tgn_from_game = game_to_tgn(game, false);
	std::cout << "TGN from game_to_tgn():\n";
	std::cout << tgn_from_game << "\n";

	// Method 2: Create SelfPlayRecord and use GameRecorder::to_tgn()
	SelfPlayRecord record = GameRecorder::record_game(game, "NeuralPolicy", "RandomPolicy");
	std::string tgn_from_recorder = GameRecorder::to_tgn(record);
	std::cout << "TGN from GameRecorder::to_tgn():\n";
	std::cout << tgn_from_recorder << "\n";

	// Extract moves sections (excluding final score comment)
	std::string moves_from_game = extract_moves_section(tgn_from_game);
	std::string moves_from_recorder = extract_moves_section(tgn_from_recorder);

	// Verify consistency
	std::cout << "\n--- Consistency Check ---\n";
	std::cout << "Moves from game_to_tgn():\n[" << moves_from_game << "]\n\n";
	std::cout << "Moves from GameRecorder::to_tgn():\n[" << moves_from_recorder << "]\n\n";

	if (moves_from_game == moves_from_recorder)
	{
		std::cout << "✓ TGN formats are CONSISTENT!\n";
		std::cout << "Both methods produce identical move sequences.\n\n";
		return 0;
	}
	else
	{
		std::cout << "✗ TGN formats are INCONSISTENT!\n";
		std::cout << "The two methods produce different outputs.\n\n";

		// Show differences
		std::cout << "Length difference: "
		          << (int)moves_from_game.length() - (int)moves_from_recorder.length()
		          << " bytes\n";

		// Character-by-character comparison
		size_t min_len = std::min(moves_from_game.length(), moves_from_recorder.length());
		for (size_t i = 0; i < min_len; i++)
		{
			if (moves_from_game[i] != moves_from_recorder[i])
			{
				std::cout << "First difference at position " << i << ":\n";
				std::cout << "  game_to_tgn():         '" << moves_from_game[i] << "' (ASCII " << (int)moves_from_game[i] << ")\n";
				std::cout << "  GameRecorder::to_tgn(): '" << moves_from_recorder[i] << "' (ASCII " << (int)moves_from_recorder[i] << ")\n";
				std::cout << "  Context: \"" << moves_from_game.substr(i > 10 ? i-10 : 0, 20) << "\"\n";
				break;
			}
		}

		return 1;
	}
}
