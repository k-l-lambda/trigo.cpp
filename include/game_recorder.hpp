/**
 * Game Recorder and TGN Export
 *
 * Records self-play games and exports to TGN format
 * TGN (Trigo Game Notation) is similar to PGN for chess
 *
 * Supports:
 * - Game metadata (players, date, result)
 * - Move history
 * - Final position and score
 * - Training data annotations (policy, value)
 */

#pragma once

#include "trigo_game.hpp"
#include "trigo_coords.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>


namespace trigo
{


/**
 * Training data for one position (optional)
 * Used for offline training
 */
struct TrainingData
{
	std::vector<float> policy;  // Policy distribution over all actions
	float value;                 // Value estimate [-1, 1]

	TrainingData() : value(0.0f) {}
};


/**
 * Self-play game record (extends GameRecord with training data)
 */
struct SelfPlayRecord
{
	// Game configuration
	BoardShape board_shape;
	std::string black_player;
	std::string white_player;

	// Game result
	GameResult result;
	TerritoryResult final_territory;

	// Move history (using Step from TrigoGame)
	std::vector<Step> steps;

	// Optional: Training data per position
	std::vector<TrainingData> training_data;

	// Metadata
	std::string event;
	std::string date;
	int move_count;

	SelfPlayRecord()
		: board_shape(5, 5, 5)
		, black_player("Unknown")
		, white_player("Unknown")
		, result(GameResult::Winner::Draw, GameResult::Reason::Completion)
		, final_territory()
		, move_count(0)
	{
		// Set current date
		auto now = std::chrono::system_clock::now();
		auto time_t = std::chrono::system_clock::to_time_t(now);
		std::tm tm = *std::localtime(&time_t);
		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y.%m.%d");
		date = oss.str();
	}
};


/**
 * Game Recorder
 *
 * Records games during self-play
 */
class GameRecorder
{
public:
	/**
	 * Create record from finished game
	 */
	static SelfPlayRecord record_game(
		const TrigoGame& game,
		const std::string& black_player,
		const std::string& white_player,
		const std::string& event = "Self-Play"
	)
	{
		SelfPlayRecord record;

		// Basic info
		record.board_shape = game.get_shape();
		record.black_player = black_player;
		record.white_player = white_player;
		record.event = event;

		// Game result
		auto result = game.get_game_result();
		if (result.has_value())
		{
			record.result = *result;
		}

		// Final territory
		auto game_copy = const_cast<TrigoGame&>(game);  // get_territory is non-const
		record.final_territory = game_copy.get_territory();

		// Move history
		record.steps = game.get_history();
		record.move_count = record.steps.size();

		return record;
	}

	/**
	 * Export game to TGN format
	 */
	static std::string to_tgn(const SelfPlayRecord& record)
	{
		std::ostringstream tgn;

		// Only Board metadata (other headers removed to avoid training data leakage)
		std::string board_str;
		if (record.board_shape.z == 1)
		{
			board_str = std::to_string(record.board_shape.x) + "x" +
			            std::to_string(record.board_shape.y);
		}
		else
		{
			board_str = std::to_string(record.board_shape.x) + "x" +
			            std::to_string(record.board_shape.y) + "x" +
			            std::to_string(record.board_shape.z);
		}
		tgn << "[Board " << board_str << "]\n";

		tgn << "\n";  // Empty line after metadata

		// Move sequence
		int move_number = 1;
		for (size_t i = 0; i < record.steps.size(); i++)
		{
			const auto& step = record.steps[i];

			// Add move number for black's move
			if (step.player == Stone::Black)
			{
				tgn << move_number << ". ";
			}

			// Format the move
			if (step.type == StepType::DROP && step.position)
			{
				// Convert position to TGN coordinate
				auto coord = encode_ab0yz(*step.position, record.board_shape);
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

			// Add space after white's move or newline after pair
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

		// Add final score comment
		int score_diff = record.final_territory.black - record.final_territory.white;
		std::string sign = score_diff > 0 ? "-" : score_diff < 0 ? "+" : "";
		tgn << "; " << sign << std::abs(score_diff) << "\n";

		return tgn.str();
	}

	/**
	 * Save game to TGN file
	 */
	static bool save_tgn(const SelfPlayRecord& record, const std::string& filename)
	{
		std::ofstream file(filename);
		if (!file.is_open())
		{
			return false;
		}

		file << to_tgn(record);
		file.close();
		return true;
	}

	/**
	 * Export training data to binary format
	 *
	 * Format: positions with policy/value annotations
	 * Used for offline neural network training
	 */
	static bool save_training_data(
		const SelfPlayRecord& record,
		const std::string& filename
	)
	{
		// TODO: Implement binary format for efficient loading
		// Format could be:
		// - NumPy .npz
		// - HDF5
		// - Custom binary
		// - Protocol Buffers

		return false;  // Not implemented yet
	}
};


} // namespace trigo
