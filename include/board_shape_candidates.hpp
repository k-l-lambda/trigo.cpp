/**
 * Board Shape Candidates
 *
 * Generates candidate board shapes from range specifications
 * Supports flexible range definitions like "2-13x1-13x1-1,2-5x2-5x2-5"
 *
 * Range format: "minX-maxXxminY-maxYxminZ-maxZ"
 * Multiple ranges can be concatenated with commas
 */

#pragma once

#include "trigo_types.hpp"
#include <vector>
#include <random>
#include <string>
#include <sstream>


namespace trigo
{


/**
 * Board shape range specification
 */
struct BoardShapeRange
{
	int min_x, max_x;
	int min_y, max_y;
	int min_z, max_z;

	BoardShapeRange(int min_x_, int max_x_, int min_y_, int max_y_, int min_z_, int max_z_)
		: min_x(min_x_), max_x(max_x_)
		, min_y(min_y_), max_y(max_y_)
		, min_z(min_z_), max_z(max_z_)
	{}
};


/**
 * Parse a single range specification
 *
 * Format: "minX-maxXxminY-maxYxminZ-maxZ"
 * Example: "2-13x1-13x1-1" or "2-5x2-5x2-5"
 *
 * @param range_str Range specification string
 * @return Parsed range, or empty optional if parse fails
 */
inline std::optional<BoardShapeRange> parse_range(const std::string& range_str)
{
	int min_x, max_x, min_y, max_y, min_z, max_z;

	// Try to parse "minX-maxXxminY-maxYxminZ-maxZ"
	int matched = sscanf(range_str.c_str(), "%d-%dx%d-%dx%d-%d",
	                     &min_x, &max_x, &min_y, &max_y, &min_z, &max_z);

	if (matched == 6)
	{
		return BoardShapeRange(min_x, max_x, min_y, max_y, min_z, max_z);
	}

	return std::nullopt;
}


/**
 * Generate board shapes from a single range
 *
 * Creates all combinations within the specified range
 *
 * @param range Range specification
 * @return Vector of board shapes
 */
inline std::vector<BoardShape> generate_shapes_from_range(const BoardShapeRange& range)
{
	std::vector<BoardShape> shapes;

	for (int x = range.min_x; x <= range.max_x; x++)
	{
		for (int y = range.min_y; y <= range.max_y; y++)
		{
			for (int z = range.min_z; z <= range.max_z; z++)
			{
				shapes.push_back(BoardShape{x, y, z});
			}
		}
	}

	return shapes;
}


/**
 * Generate board shapes from multiple range specifications
 *
 * Parses comma-separated range strings and generates all candidates
 *
 * Format: "range1,range2,range3,..."
 * Example: "2-13x1-13x1-1,2-5x2-5x2-5"
 *
 * @param ranges_str Comma-separated range specifications
 * @return Vector of all candidate board shapes
 */
inline std::vector<BoardShape> generate_shapes_from_ranges(const std::string& ranges_str)
{
	std::vector<BoardShape> all_shapes;

	if (ranges_str.empty())
	{
		return all_shapes;
	}

	// Split by comma
	std::stringstream ss(ranges_str);
	std::string range_str;

	while (std::getline(ss, range_str, ','))
	{
		// Trim whitespace
		range_str.erase(0, range_str.find_first_not_of(" \t"));
		range_str.erase(range_str.find_last_not_of(" \t") + 1);

		if (range_str.empty())
		{
			continue;
		}

		// Parse range
		auto range = parse_range(range_str);
		if (range)
		{
			// Generate shapes from this range
			auto shapes = generate_shapes_from_range(*range);
			all_shapes.insert(all_shapes.end(), shapes.begin(), shapes.end());
		}
		else
		{
			std::cerr << "Warning: Failed to parse board range: " << range_str << std::endl;
		}
	}

	return all_shapes;
}


/**
 * Generate default candidate board shapes
 *
 * Default: 2D boards (2-13x1-13x1-1) + 3D boards (2-5x2-5x2-5)
 * Total: 156 + 64 = 220 shapes
 *
 * @return Vector of default board shapes
 */
inline std::vector<BoardShape> generate_default_board_shapes()
{
	return generate_shapes_from_ranges("2-13x1-13x1-1,2-5x2-5x2-5");
}


/**
 * Select random board shape from candidates
 *
 * Selects uniformly from provided candidate list
 *
 * @param candidates Vector of candidate board shapes
 * @param rng Random number generator (e.g., std::mt19937)
 * @return Randomly selected board shape
 */
inline BoardShape select_random_board_shape(
	const std::vector<BoardShape>& candidates,
	std::mt19937& rng)
{
	if (candidates.empty())
	{
		// Fallback to default 5x5x5 if no candidates
		return BoardShape{5, 5, 5};
	}

	std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
	return candidates[dist(rng)];
}


}  // namespace trigo
