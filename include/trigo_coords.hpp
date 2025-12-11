/**
 * TGN Coordinate System - Encoding and Decoding
 *
 * TGN (Trigo Game Notation) uses a center-symmetric coordinate system:
 * - '0' represents the center position on each axis
 * - 'a', 'b', 'c', ... from one edge toward center
 * - 'z', 'y', 'x', ... from opposite edge toward center
 *
 * For 2D boards (trailing dimension = 1), coordinates are compacted.
 * Example: 19x19x1 board uses 2-character codes like "aa" instead of "aa0"
 *
 * Port from TypeScript: inc/trigo/ab0yz.ts
 */

#pragma once

#include "trigo_types.hpp"
#include <string>
#include <vector>
#include <stdexcept>


namespace trigo
{


/**
 * Compact board shape by removing trailing 1s
 *
 * @param shape Board dimensions
 * @return Compacted shape (e.g., [19, 19, 1] -> [19, 19])
 */
inline std::vector<int> compact_shape(const std::vector<int>& shape)
{
	if (shape.empty() || shape.back() != 1)
	{
		return shape;
	}

	std::vector<int> compacted(shape.begin(), shape.end() - 1);
	return compact_shape(compacted);  // Recursive removal
}


/**
 * Encode position to TGN coordinate string
 *
 * @param pos Position [x, y, z, ...] with 0-based indices
 * @param board_shape Board dimensions [sizeX, sizeY, sizeZ, ...]
 * @return TGN coordinate string (e.g., "000", "aa0", "bzz", "aa")
 *
 * Examples:
 *   encodeAb0yz([2, 2, 2], [5, 5, 5]) → "000" (center of 5x5x5)
 *   encodeAb0yz([0, 0, 2], [5, 5, 5]) → "aa0" (corner)
 *   encodeAb0yz([4, 2, 2], [5, 5, 5]) → "z00"
 *   encodeAb0yz([0, 0, 0], [19, 19, 1]) → "aa" (2D board)
 */
inline std::string encode_ab0yz(const std::vector<int>& pos, const std::vector<int>& board_shape)
{
	auto compacted_shape = compact_shape(board_shape);
	std::string result;
	result.reserve(compacted_shape.size());

	for (size_t i = 0; i < compacted_shape.size(); i++)
	{
		int size = compacted_shape[i];
		double center = (size - 1) / 2.0;
		int index = pos[i];

		// Only output '0' for center if size is odd (center is integer)
		// For even size, center is x.5, so no position equals center
		if (size % 2 == 1 && index == static_cast<int>(center))
		{
			// Center position (only valid for odd-sized dimensions)
			result += '0';
		}
		else if (index < center)
		{
			// Left side: a, b, c, ...
			result += static_cast<char>('a' + index);
		}
		else
		{
			// Right side: z, y, x, ...
			int offset = size - 1 - index;
			result += static_cast<char>('z' - offset);
		}
	}

	return result;
}


/**
 * Encode Position struct to TGN coordinate string
 */
inline std::string encode_ab0yz(const Position& pos, const BoardShape& board_shape)
{
	return encode_ab0yz(
		std::vector<int>{pos.x, pos.y, pos.z},
		std::vector<int>{board_shape.x, board_shape.y, board_shape.z}
	);
}


/**
 * Decode TGN coordinate string to position array
 *
 * @param code TGN coordinate string (e.g., "000", "aa0", "bzz", "aa")
 * @param board_shape Board dimensions [sizeX, sizeY, sizeZ, ...]
 * @return Position [x, y, z, ...] with 0-based indices
 *
 * Examples:
 *   decodeAb0yz("000", [5, 5, 5]) → [2, 2, 2] (center)
 *   decodeAb0yz("aa0", [5, 5, 5]) → [0, 0, 2]
 *   decodeAb0yz("z00", [5, 5, 5]) → [4, 2, 2]
 *   decodeAb0yz("aa", [19, 19, 1]) → [0, 0, 0] (2D board)
 */
inline std::vector<int> decode_ab0yz(const std::string& code, const std::vector<int>& board_shape)
{
	auto compacted_shape = compact_shape(board_shape);

	if (code.length() != compacted_shape.size())
	{
		throw std::invalid_argument(
			"Invalid TGN coordinate: \"" + code + "\" (must be " +
			std::to_string(compacted_shape.size()) + " characters for board shape " +
			std::to_string(board_shape[0]) + "x" + std::to_string(board_shape[1]) + "x" + std::to_string(board_shape[2]) + ")"
		);
	}

	std::vector<int> result;
	result.reserve(compacted_shape.size());

	for (size_t i = 0; i < compacted_shape.size(); i++)
	{
		char ch = code[i];
		int size = compacted_shape[i];
		double center = (size - 1) / 2.0;

		if (ch == '0')
		{
			// Center position - only valid for odd-sized dimensions
			if (size % 2 == 0)
			{
				throw std::invalid_argument(
					"Invalid TGN coordinate: \"" + code + "\" ('0' is not valid for even-sized dimension " +
					std::to_string(size) + " on axis " + std::to_string(i) + ")"
				);
			}
			result.push_back(static_cast<int>(center));
		}
		else if (ch >= 'a' && ch <= 'z')
		{
			// Calculate distance from 'a' and 'z'
			int dist_from_a = ch - 'a';
			int dist_from_z = 'z' - ch;

			// Determine if it's left side (closer to 'a') or right side (closer to 'z')
			if (dist_from_a < dist_from_z)
			{
				// Left side: a=0, b=1, c=2, ...
				int index = dist_from_a;
				if (index >= center)
				{
					throw std::invalid_argument(
						"Invalid TGN coordinate: \"" + code + "\" (position " +
						std::to_string(index) + " >= center " + std::to_string(static_cast<int>(center)) + " on axis " + std::to_string(i) + ")"
					);
				}
				result.push_back(index);
			}
			else
			{
				// Right side: z=size-1, y=size-2, x=size-3, ...
				int index = size - 1 - dist_from_z;
				if (index <= center)
				{
					throw std::invalid_argument(
						"Invalid TGN coordinate: \"" + code + "\" (position " +
						std::to_string(index) + " <= center " + std::to_string(static_cast<int>(center)) + " on axis " + std::to_string(i) + ")"
					);
				}
				result.push_back(index);
			}
		}
		else
		{
			throw std::invalid_argument(
				"Invalid TGN coordinate: \"" + code + "\" (character '" + std::string(1, ch) +
				"' at position " + std::to_string(i) + " must be '0' or a-z)"
			);
		}
	}

	// Fill remaining dimensions with 0 if board_shape has trailing 1s
	while (result.size() < board_shape.size())
	{
		result.push_back(0);
	}

	return result;
}


/**
 * Decode TGN coordinate string to Position struct
 */
inline Position decode_ab0yz_position(const std::string& code, const BoardShape& board_shape)
{
	auto pos_vec = decode_ab0yz(
		code,
		std::vector<int>{board_shape.x, board_shape.y, board_shape.z}
	);

	return Position(pos_vec[0], pos_vec[1], pos_vec[2]);
}


}  // namespace trigo
