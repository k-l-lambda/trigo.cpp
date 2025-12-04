/**
 * Trigo Game Utils - Core Go Rules Implementation
 *
 * Implements:
 * - Capture detection (flood fill for connected groups)
 * - Liberty calculation
 * - Ko rule (打劫 prohibition)
 * - Suicide prevention
 * - Territory counting
 *
 * Port from TypeScript: inc/trigo/gameUtils.ts (566 lines)
 */

#pragma once

#include "trigo_types.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <functional>


namespace trigo
{


/**
 * Position hash function for unordered_set
 */
struct PositionHash
{
	std::size_t operator()(const Position& p) const
	{
		return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1) ^ (std::hash<int>()(p.z) << 2);
	}
};


/**
 * CoordSet - Manages a set of positions
 * Used for tracking stones in a group, liberties, visited positions, etc.
 */
class CoordSet
{
private:
	std::unordered_set<Position, PositionHash> positions_;

public:
	CoordSet() = default;

	bool has(const Position& pos) const
	{
		return positions_.find(pos) != positions_.end();
	}

	bool insert(const Position& pos)
	{
		return positions_.insert(pos).second;
	}

	void remove(const Position& pos)
	{
		positions_.erase(pos);
	}

	size_t size() const
	{
		return positions_.size();
	}

	bool empty() const
	{
		return positions_.empty();
	}

	void clear()
	{
		positions_.clear();
	}

	std::vector<Position> to_array() const
	{
		return std::vector<Position>(positions_.begin(), positions_.end());
	}

	void for_each(std::function<void(const Position&)> callback) const
	{
		for (const auto& pos : positions_)
		{
			callback(pos);
		}
	}

	// Iterator support
	auto begin() const { return positions_.begin(); }
	auto end() const { return positions_.end(); }
};


/**
 * Check if position is within board bounds
 */
inline bool is_in_bounds(const Position& pos, const BoardShape& shape)
{
	return pos.x >= 0 && pos.x < shape.x &&
	       pos.y >= 0 && pos.y < shape.y &&
	       pos.z >= 0 && pos.z < shape.z;
}


/**
 * Get all neighboring positions (up to 6 in 3D: ±x, ±y, ±z)
 */
inline std::vector<Position> get_neighbors(const Position& pos, const BoardShape& shape)
{
	std::vector<Position> neighbors;
	neighbors.reserve(6);

	// Check all 6 directions in 3D space
	const Position directions[6] = {
		{1, 0, 0}, {-1, 0, 0},
		{0, 1, 0}, {0, -1, 0},
		{0, 0, 1}, {0, 0, -1}
	};

	for (const auto& dir : directions)
	{
		Position neighbor(pos.x + dir.x, pos.y + dir.y, pos.z + dir.z);

		if (is_in_bounds(neighbor, shape))
		{
			neighbors.push_back(neighbor);
		}
	}

	return neighbors;
}


/**
 * Board representation: 3D array of stones
 * board[x][y][z] = Stone
 */
using Board = std::vector<std::vector<std::vector<Stone>>>;


/**
 * Create empty board
 */
inline Board create_board(const BoardShape& shape)
{
	return Board(
		shape.x,
		std::vector<std::vector<Stone>>(
			shape.y,
			std::vector<Stone>(shape.z, Stone::Empty)
		)
	);
}


/**
 * Deep copy a board
 */
inline Board copy_board(const Board& board)
{
	return board;  // std::vector has proper deep copy
}


/**
 * Get stone at position (with bounds checking)
 */
inline Stone get_stone(const Board& board, const Position& pos)
{
	return board[pos.x][pos.y][pos.z];
}


/**
 * Set stone at position
 */
inline void set_stone(Board& board, const Position& pos, Stone stone)
{
	board[pos.x][pos.y][pos.z] = stone;
}


/**
 * Patch - Connected group of same-colored stones
 */
class Patch
{
public:
	CoordSet positions;
	Stone color;

	Patch(Stone c = Stone::Empty) : color(c) {}

	void add_stone(const Position& pos)
	{
		positions.insert(pos);
	}

	size_t size() const
	{
		return positions.size();
	}

	/**
	 * Get all liberties (empty adjacent positions) for this group
	 */
	CoordSet get_liberties(const Board& board, const BoardShape& shape) const
	{
		CoordSet liberties;

		positions.for_each([&](const Position& stone_pos)
		{
			auto neighbors = get_neighbors(stone_pos, shape);
			for (const auto& neighbor : neighbors)
			{
				if (get_stone(board, neighbor) == Stone::Empty)
				{
					liberties.insert(neighbor);
				}
			}
		});

		return liberties;
	}
};


/**
 * Find connected group of stones at position (flood fill)
 */
inline Patch find_group(const Position& pos, const Board& board, const BoardShape& shape)
{
	Stone color = get_stone(board, pos);
	Patch group(color);

	if (color == Stone::Empty)
	{
		return group;
	}

	// Flood fill to find all connected stones of same color
	CoordSet visited;
	std::vector<Position> stack = {pos};

	while (!stack.empty())
	{
		Position current = stack.back();
		stack.pop_back();

		if (visited.has(current))
		{
			continue;
		}

		visited.insert(current);

		if (get_stone(board, current) == color)
		{
			group.add_stone(current);

			auto neighbors = get_neighbors(current, shape);
			for (const auto& neighbor : neighbors)
			{
				if (!visited.has(neighbor))
				{
					stack.push_back(neighbor);
				}
			}
		}
	}

	return group;
}


/**
 * Get all neighboring groups (different color from position)
 */
inline std::vector<Patch> get_neighbor_groups(
	const Position& pos,
	const Board& board,
	const BoardShape& shape,
	bool exclude_empty = false
)
{
	auto neighbors = get_neighbors(pos, shape);
	std::vector<Patch> groups;
	CoordSet processed_positions;

	for (const auto& neighbor : neighbors)
	{
		if (processed_positions.has(neighbor))
		{
			continue;
		}

		Stone stone = get_stone(board, neighbor);

		if (exclude_empty && stone == Stone::Empty)
		{
			continue;
		}

		if (stone != Stone::Empty)
		{
			Patch group = find_group(neighbor, board, shape);
			group.positions.for_each([&](const Position& p)
			{
				processed_positions.insert(p);
			});
			groups.push_back(group);
		}
	}

	return groups;
}


/**
 * Check if group is captured (has no liberties)
 */
inline bool is_group_captured(const Patch& group, const Board& board, const BoardShape& shape)
{
	auto liberties = group.get_liberties(board, shape);
	return liberties.size() == 0;
}


/**
 * Find all groups that would be captured by placing stone at position
 */
inline std::vector<Patch> find_captured_groups(
	const Position& pos,
	Stone player_color,
	const Board& board,
	const BoardShape& shape
)
{
	Stone enemy_color = opponent_stone(player_color);
	std::vector<Patch> captured;

	// Create temporary board with new stone placed
	Board temp_board = copy_board(board);
	set_stone(temp_board, pos, player_color);

	// Check all neighboring enemy groups
	auto neighbor_groups = get_neighbor_groups(pos, temp_board, shape, true);

	for (const auto& group : neighbor_groups)
	{
		if (group.color == enemy_color)
		{
			if (is_group_captured(group, temp_board, shape))
			{
				captured.push_back(group);
			}
		}
	}

	return captured;
}


/**
 * Check if placing stone would result in self-capture (suicide)
 * Exception: Move allowed if it captures enemy stones first
 */
inline bool is_suicide_move(
	const Position& pos,
	Stone player_color,
	const Board& board,
	const BoardShape& shape
)
{
	// Create temporary board with new stone
	Board temp_board = copy_board(board);
	set_stone(temp_board, pos, player_color);

	// If this move captures enemy stones, it's not suicide
	auto captured_groups = find_captured_groups(pos, player_color, board, shape);
	if (!captured_groups.empty())
	{
		return false;
	}

	// Check if placed stone's group has any liberties
	Patch placed_group = find_group(pos, temp_board, shape);
	auto liberties = placed_group.get_liberties(temp_board, shape);

	return liberties.size() == 0;
}


/**
 * Ko Detection - Check if move would recreate previous board state
 *
 * Ko rule: Cannot immediately recapture a single stone if it would
 * return board to previous position
 */
inline bool is_ko_violation(
	const Position& pos,
	Stone player_color,
	const Board& board,
	const BoardShape& shape,
	const std::vector<Position>* last_captured_positions
)
{
	// Ko only applies when:
	// 1. Previous move captured exactly one stone
	// 2. This move would capture exactly one stone
	// 3. We're placing at position of previously captured stone

	if (!last_captured_positions || last_captured_positions->size() != 1)
	{
		return false;
	}

	auto captured_groups = find_captured_groups(pos, player_color, board, shape);

	// Check if this move would capture exactly one stone
	if (captured_groups.size() != 1 || captured_groups[0].size() != 1)
	{
		return false;
	}

	// Check if placing at position that was just captured
	const Position& previously_captured = (*last_captured_positions)[0];
	if (pos == previously_captured)
	{
		return true;
	}

	return false;
}


/**
 * Execute captures on board
 * Returns positions of captured stones
 */
inline std::vector<Position> execute_captures(
	const std::vector<Patch>& captured_groups,
	Board& board
)
{
	std::vector<Position> captured_positions;

	for (const auto& group : captured_groups)
	{
		group.positions.for_each([&](const Position& pos)
		{
			set_stone(board, pos, Stone::Empty);
			captured_positions.push_back(pos);
		});
	}

	return captured_positions;
}


/**
 * Territory calculation result
 */
struct TerritoryResult
{
	int black;
	int white;
	int neutral;
	std::vector<Position> black_territory;
	std::vector<Position> white_territory;
	std::vector<Position> neutral_territory;

	TerritoryResult()
		: black(0), white(0), neutral(0) {}
};


/**
 * Find all connected empty positions starting from position
 */
inline CoordSet find_empty_region(
	const Position& start_pos,
	const Board& board,
	const BoardShape& shape,
	CoordSet& visited
)
{
	CoordSet region;
	std::vector<Position> stack = {start_pos};

	while (!stack.empty())
	{
		Position pos = stack.back();
		stack.pop_back();

		if (visited.has(pos))
		{
			continue;
		}

		visited.insert(pos);

		if (get_stone(board, pos) == Stone::Empty)
		{
			region.insert(pos);

			auto neighbors = get_neighbors(pos, shape);
			for (const auto& neighbor : neighbors)
			{
				if (!visited.has(neighbor))
				{
					stack.push_back(neighbor);
				}
			}
		}
	}

	return region;
}


/**
 * Determine which player owns an empty region
 * Returns BLACK, WHITE, or EMPTY (neutral)
 *
 * An empty region belongs to a player if ALL bordering stones are that color
 */
inline Stone determine_region_owner(
	const CoordSet& region,
	const Board& board,
	const BoardShape& shape
)
{
	Stone owner = Stone::Empty;

	for (const auto& pos : region)
	{
		auto neighbors = get_neighbors(pos, shape);

		for (const auto& neighbor : neighbors)
		{
			Stone stone = get_stone(board, neighbor);

			if (stone != Stone::Empty)
			{
				if (owner == Stone::Empty)
				{
					// First colored stone found
					owner = stone;
				}
				else if (owner != stone)
				{
					// Found different colored stone - region is neutral
					return Stone::Empty;
				}
			}
		}
	}

	return owner;
}


/**
 * Calculate territory for both players
 *
 * Algorithm:
 * 1. Count all stones as territory
 * 2. Find all empty regions and determine ownership
 */
inline TerritoryResult calculate_territory(const Board& board, const BoardShape& shape)
{
	TerritoryResult result;
	CoordSet visited;
	std::vector<CoordSet> empty_regions;

	// FIRST PASS: Count stones and find empty regions
	for (int x = 0; x < shape.x; x++)
	{
		for (int y = 0; y < shape.y; y++)
		{
			for (int z = 0; z < shape.z; z++)
			{
				Position pos(x, y, z);
				Stone stone = get_stone(board, pos);

				if (stone == Stone::Black)
				{
					result.black++;
					result.black_territory.push_back(pos);
				}
				else if (stone == Stone::White)
				{
					result.white++;
					result.white_territory.push_back(pos);
				}
				else if (!visited.has(pos))
				{
					// Found empty position - explore region
					CoordSet region = find_empty_region(pos, board, shape, visited);
					empty_regions.push_back(region);
				}
			}
		}
	}

	// SECOND PASS: Determine ownership of each empty region
	for (const auto& region : empty_regions)
	{
		Stone owner = determine_region_owner(region, board, shape);
		auto region_array = region.to_array();

		if (owner == Stone::Black)
		{
			result.black += region.size();
			result.black_territory.insert(
				result.black_territory.end(),
				region_array.begin(),
				region_array.end()
			);
		}
		else if (owner == Stone::White)
		{
			result.white += region.size();
			result.white_territory.insert(
				result.white_territory.end(),
				region_array.begin(),
				region_array.end()
			);
		}
		else
		{
			result.neutral += region.size();
			result.neutral_territory.insert(
				result.neutral_territory.end(),
				region_array.begin(),
				region_array.end()
			);
		}
	}

	return result;
}


/**
 * Move validation result
 */
struct MoveValidation
{
	bool valid;
	std::string reason;

	MoveValidation(bool v = true, const std::string& r = "")
		: valid(v), reason(r) {}
};


/**
 * Validate if move is legal
 * Checks: bounds, occupation, Ko rule, suicide rule
 */
inline MoveValidation validate_move(
	const Position& pos,
	Stone player_color,
	const Board& board,
	const BoardShape& shape,
	const std::vector<Position>* last_captured_positions = nullptr
)
{
	// Check bounds
	if (!is_in_bounds(pos, shape))
	{
		return MoveValidation(false, "Position out of bounds");
	}

	// Check if position is empty
	if (get_stone(board, pos) != Stone::Empty)
	{
		return MoveValidation(false, "Position already occupied");
	}

	// Check for Ko violation
	if (is_ko_violation(pos, player_color, board, shape, last_captured_positions))
	{
		return MoveValidation(false, "Ko rule violation");
	}

	// Check for suicide
	if (is_suicide_move(pos, player_color, board, shape))
	{
		return MoveValidation(false, "Suicide move not allowed");
	}

	return MoveValidation(true);
}


}  // namespace trigo
