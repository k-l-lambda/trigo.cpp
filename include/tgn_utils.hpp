/**
 * TGN Utilities
 *
 * Reusable functions for generating TGN (Trigo Game Notation) format
 * Used by both GameRecorder and NeuralPolicy for consistent formatting
 */

#pragma once

#include "trigo_game.hpp"
#include "trigo_coords.hpp"
#include <string>
#include <sstream>


namespace trigo
{


/**
 * Convert game state to TGN format string
 *
 * Generates TGN notation for the current game state (in-progress or completed)
 * Format matches training data exactly:
 *   [Board 5x5x5]
 *
 *   1. 000 a00
 *   2. b00 0b0
 *   ...
 *
 * @param game The game to convert
 * @param include_result Include final score comment (only for finished games)
 * @return TGN format string
 */
inline std::string game_to_tgn(const TrigoGame& game, bool include_result = false)
{
	const auto& steps = game.get_history();
	auto board_shape = game.get_shape();

	std::ostringstream tgn;

	// Board header
	std::string board_str;
	if (board_shape.z == 1)
	{
		board_str = std::to_string(board_shape.x) + "x" +
		            std::to_string(board_shape.y);
	}
	else
	{
		board_str = std::to_string(board_shape.x) + "x" +
		            std::to_string(board_shape.y) + "x" +
		            std::to_string(board_shape.z);
	}
	tgn << "[Board " << board_str << "]\n";

	tgn << "\n";  // Empty line after metadata

	// Move sequence
	int move_number = 1;
	for (size_t i = 0; i < steps.size(); i++)
	{
		const auto& step = steps[i];

		// Add move number for black's move
		if (step.player == Stone::Black)
		{
			tgn << move_number << ". ";
		}

		// Format the move
		if (step.type == StepType::DROP && step.position)
		{
			auto coord = encode_ab0yz(*step.position, board_shape);
			tgn << coord;
		}
		else if (step.type == StepType::PASS)
		{
			tgn << "Pass";
		}
		else if (step.type == StepType::SURRENDER)
		{
			tgn << "Resign";
		}

		// Add space after black's move, newline after white's move
		if (step.player == Stone::White)
		{
			tgn << "\n";
			move_number++;
		}
		else
		{
			tgn << " ";
		}
	}

	// Add final score comment if requested
	if (include_result)
	{
		// Need non-const game to call get_territory
		auto game_copy = const_cast<TrigoGame&>(game);
		auto territory = game_copy.get_territory();
		int score_diff = territory.black - territory.white;
		std::string sign = score_diff > 0 ? "-" : score_diff < 0 ? "+" : "";
		tgn << "; " << sign << std::abs(score_diff) << "\n";
	}

	return tgn.str();
}


}  // namespace trigo
