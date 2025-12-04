/**
 * Trigo Game Engine - Core Type Definitions
 *
 * This file defines the fundamental data structures for the Trigo 3D Go game.
 * Port from TypeScript: inc/trigo/types.ts
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <chrono>
#include <stdexcept>


namespace trigo
{


/**
 * 3D position on the board
 */
struct Position
{
	int x;
	int y;
	int z;

	Position() : x(0), y(0), z(0) {}
	Position(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

	bool operator==(const Position& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}

	bool operator!=(const Position& other) const
	{
		return !(*this == other);
	}
};


/**
 * Board dimensions
 */
struct BoardShape
{
	int x;
	int y;
	int z;

	BoardShape() : x(0), y(0), z(0) {}
	BoardShape(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

	bool operator==(const BoardShape& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}
};


/**
 * Stone types on the board
 */
enum class Stone : uint8_t
{
	Empty = 0,
	Black = 1,
	White = 2
};


/**
 * Player identifiers
 */
enum class Player
{
	Black,
	White
};


/**
 * Convert Player to Stone
 */
inline Stone player_to_stone(Player player)
{
	return player == Player::Black ? Stone::Black : Stone::White;
}


/**
 * Convert Stone to Player (throws if Empty)
 */
inline Player stone_to_player(Stone stone)
{
	if (stone == Stone::Black) return Player::Black;
	if (stone == Stone::White) return Player::White;
	throw std::runtime_error("Cannot convert Empty stone to player");
}


/**
 * Get opponent player
 */
inline Player opponent(Player player)
{
	return player == Player::Black ? Player::White : Player::Black;
}


/**
 * Get opponent stone
 */
inline Stone opponent_stone(Stone stone)
{
	if (stone == Stone::Black) return Stone::White;
	if (stone == Stone::White) return Stone::Black;
	return Stone::Empty;
}


/**
 * A move in the game
 */
struct Move
{
	std::optional<Position> position;  // None means pass
	Player player;
	std::optional<std::chrono::system_clock::time_point> timestamp;

	Move() : player(Player::Black) {}

	Move(Position pos, Player p)
		: position(pos), player(p) {}

	Move(Player p)  // Pass move
		: position(std::nullopt), player(p) {}

	bool is_pass() const
	{
		return !position.has_value();
	}
};


/**
 * Game configuration
 */
struct GameConfig
{
	BoardShape board_shape;
	std::optional<int> time_limit;  // seconds
	bool allow_undo;

	GameConfig()
		: board_shape(5, 5, 5),
		  time_limit(std::nullopt),
		  allow_undo(true) {}

	GameConfig(BoardShape shape, bool undo = true)
		: board_shape(shape),
		  time_limit(std::nullopt),
		  allow_undo(undo) {}
};


/**
 * Game result
 */
struct GameResult
{
	enum class Winner { Black, White, Draw };
	enum class Reason { Resignation, Timeout, Completion, DoublePass };

	Winner winner;
	Reason reason;

	GameResult(Winner w, Reason r) : winner(w), reason(r) {}
};


/**
 * Game record metadata
 */
struct GameRecord
{
	std::string id;
	std::vector<Move> moves;
	std::optional<GameResult> result;
	std::chrono::system_clock::time_point created_at;
	std::string black_player;
	std::string white_player;

	GameRecord() : created_at(std::chrono::system_clock::now()) {}
};


}  // namespace trigo
