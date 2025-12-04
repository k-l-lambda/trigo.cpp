/**
 * C++ Game Replay and Validation
 *
 * Reads JSON move sequences from TypeScript converter,
 * replays moves using C++ game utils, and outputs results for comparison.
 */

#include "../include/trigo_types.hpp"
#include "../include/trigo_game_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>


using namespace trigo;


/**
 * Simple JSON parser for our specific format
 * We only need to parse: boardShape, moves array
 */
struct MoveData
{
	int x, y, z;
	int player;  // 1=BLACK, 2=WHITE
};


struct GameData
{
	std::string tgnFile;
	BoardShape boardShape;
	std::vector<MoveData> moves;
};


/**
 * Parse a simple JSON value (number)
 */
int parseJsonNumber(const std::string& json, const std::string& key, size_t& pos)
{
	std::string searchKey = "\"" + key + "\":";
	pos = json.find(searchKey, pos);
	if (pos == std::string::npos)
	{
		throw std::runtime_error("Key not found: " + key);
	}
	pos += searchKey.length();

	// Skip whitespace
	while (pos < json.length() && std::isspace(json[pos])) pos++;

	// Parse number
	size_t endPos = pos;
	while (endPos < json.length() && (std::isdigit(json[endPos]) || json[endPos] == '-')) endPos++;

	int value = std::stoi(json.substr(pos, endPos - pos));
	pos = endPos;
	return value;
}


/**
 * Parse a string value
 */
std::string parseJsonString(const std::string& json, const std::string& key, size_t& pos)
{
	std::string searchKey = "\"" + key + "\":";
	pos = json.find(searchKey, pos);
	if (pos == std::string::npos)
	{
		throw std::runtime_error("Key not found: " + key);
	}
	pos += searchKey.length();

	// Skip whitespace and opening quote
	while (pos < json.length() && (std::isspace(json[pos]) || json[pos] == '"')) pos++;

	// Find closing quote
	size_t endPos = json.find('"', pos);
	std::string value = json.substr(pos, endPos - pos);
	pos = endPos + 1;
	return value;
}


/**
 * Parse move sequence from JSON
 */
GameData parseGameJson(const std::string& jsonStr)
{
	GameData data;
	size_t pos = 0;

	// Parse tgnFile
	data.tgnFile = parseJsonString(jsonStr, "tgnFile", pos);

	// Parse boardShape
	pos = jsonStr.find("\"boardShape\"", pos);
	data.boardShape.x = parseJsonNumber(jsonStr, "x", pos);
	data.boardShape.y = parseJsonNumber(jsonStr, "y", pos);
	data.boardShape.z = parseJsonNumber(jsonStr, "z", pos);

	// Parse moves array
	pos = jsonStr.find("\"moves\":", pos);
	pos = jsonStr.find('[', pos);

	while (pos < jsonStr.length())
	{
		// Find next move object
		size_t moveStart = jsonStr.find('{', pos);
		if (moveStart == std::string::npos || moveStart > jsonStr.find(']', pos))
		{
			break;
		}

		size_t moveEnd = jsonStr.find('}', moveStart);
		std::string moveJson = jsonStr.substr(moveStart, moveEnd - moveStart + 1);

		// Check if it's null (pass move)
		size_t nullPos = jsonStr.find("null", pos);
		if (nullPos != std::string::npos && nullPos < moveStart)
		{
			// Pass move - skip for now
			pos = nullPos + 4;
			continue;
		}

		// Parse move
		MoveData move;
		size_t movePos = 0;
		move.x = parseJsonNumber(moveJson, "x", movePos);
		move.y = parseJsonNumber(moveJson, "y", movePos);
		move.z = parseJsonNumber(moveJson, "z", movePos);
		move.player = parseJsonNumber(moveJson, "player", movePos);

		data.moves.push_back(move);
		pos = moveEnd + 1;
	}

	return data;
}


/**
 * Replay game from move sequence
 */
void replayGame(const GameData& gameData)
{
	std::cout << "\n=== Replaying: " << gameData.tgnFile << " ===" << std::endl;
	std::cout << "Board: " << gameData.boardShape.x << "×" << gameData.boardShape.y
	          << "×" << gameData.boardShape.z << std::endl;
	std::cout << "Moves: " << gameData.moves.size() << std::endl;

	// Create board
	Board board = create_board(gameData.boardShape);
	std::vector<Position> lastCapturedPositions;

	// Replay each move
	int moveNum = 0;
	int invalidMoves = 0;
	for (const auto& moveData : gameData.moves)
	{
		moveNum++;
		Position pos(moveData.x, moveData.y, moveData.z);
		Stone player = (moveData.player == 1) ? Stone::Black : Stone::White;

		// Validate move
		auto validation = validate_move(pos, player, board, gameData.boardShape, &lastCapturedPositions);

		if (!validation.valid)
		{
			std::cerr << "  Move " << moveNum << ": INVALID - " << validation.reason << std::endl;
			std::cerr << "    Position: (" << pos.x << "," << pos.y << "," << pos.z << ")" << std::endl;
			std::cerr << "    Player: " << (player == Stone::Black ? "BLACK" : "WHITE") << std::endl;
			invalidMoves++;
			continue;
		}

		// Find captured groups
		auto captured = find_captured_groups(pos, player, board, gameData.boardShape);

		// Execute captures
		lastCapturedPositions = execute_captures(captured, board);

		// Place stone
		set_stone(board, pos, player);

		if (!captured.empty())
		{
			int totalCaptured = 0;
			for (const auto& group : captured)
			{
				totalCaptured += group.size();
			}
			std::cout << "  Move " << moveNum << ": (" << pos.x << "," << pos.y << "," << pos.z
			          << ") captures " << totalCaptured << " stone(s)" << std::endl;
		}
	}

	// Show invalid moves count if any
	if (invalidMoves > 0)
	{
		std::cerr << "\n⚠️  WARNING: " << invalidMoves << " invalid move(s) rejected!" << std::endl;
	}

	// Calculate final territory
	auto territory = calculate_territory(board, gameData.boardShape);

	std::cout << "\nFinal Territory:" << std::endl;
	std::cout << "  Black: " << territory.black << std::endl;
	std::cout << "  White: " << territory.white << std::endl;
	std::cout << "  Neutral: " << territory.neutral << std::endl;
	std::cout << "  Total: " << (territory.black + territory.white + territory.neutral)
	          << " / " << (gameData.boardShape.x * gameData.boardShape.y * gameData.boardShape.z) << std::endl;

	// Output in parseable format for comparison
	std::cout << "\n[RESULT]" << std::endl;
	std::cout << "FILE=" << gameData.tgnFile << std::endl;
	std::cout << "MOVES=" << gameData.moves.size() << std::endl;
	std::cout << "BLACK=" << territory.black << std::endl;
	std::cout << "WHITE=" << territory.white << std::endl;
	std::cout << "NEUTRAL=" << territory.neutral << std::endl;

	// Output board state (compact format)
	std::cout << "BOARD_HASH=";
	size_t hash = 0;
	for (int x = 0; x < gameData.boardShape.x; x++)
	{
		for (int y = 0; y < gameData.boardShape.y; y++)
		{
			for (int z = 0; z < gameData.boardShape.z; z++)
			{
				int stone = static_cast<int>(board[x][y][z]);
				hash = hash * 3 + stone;
			}
		}
	}
	std::cout << hash << std::endl;
}


int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <json_file>" << std::endl;
		std::cerr << "" << std::endl;
		std::cerr << "Replays games from JSON move sequences (from tgnToMoveSequence.ts)" << std::endl;
		return 1;
	}

	std::string jsonFile = argv[1];

	// Read JSON file
	std::ifstream file(jsonFile);
	if (!file.is_open())
	{
		std::cerr << "Error: Cannot open file: " << jsonFile << std::endl;
		return 1;
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string jsonContent = buffer.str();
	file.close();

	// Parse JSON array
	// Find each game object in array
	size_t pos = 0;
	int gameCount = 0;

	while (pos < jsonContent.length())
	{
		size_t objStart = jsonContent.find('{', pos);
		if (objStart == std::string::npos)
		{
			break;
		}

		// Find matching closing brace
		int braceCount = 1;
		size_t objEnd = objStart + 1;
		while (objEnd < jsonContent.length() && braceCount > 0)
		{
			if (jsonContent[objEnd] == '{') braceCount++;
			else if (jsonContent[objEnd] == '}') braceCount--;
			objEnd++;
		}

		std::string gameJson = jsonContent.substr(objStart, objEnd - objStart);

		try
		{
			GameData gameData = parseGameJson(gameJson);
			replayGame(gameData);
			gameCount++;
		}
		catch (const std::exception& e)
		{
			std::cerr << "\nError parsing game: " << e.what() << std::endl;
		}

		pos = objEnd;
	}

	std::cout << "\n=== Replayed " << gameCount << " game(s) ===" << std::endl;

	return 0;
}
