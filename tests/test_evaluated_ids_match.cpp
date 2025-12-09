/**
 * Test that C++ evaluated_ids match TypeScript exactly
 *
 * Tests multiple game scenarios to ensure tree building is consistent
 */

#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>


using namespace trigo;


void print_evaluated_ids(const std::vector<int64_t>& ids)
{
	std::cout << "[";
	for (size_t i = 0; i < ids.size(); i++)
	{
		std::cout << ids[i];
		if (i < ids.size() - 1) std::cout << ", ";
	}
	std::cout << "]";
}


void print_evaluated_ids_as_chars(const std::vector<int64_t>& ids)
{
	std::cout << "\"";
	for (auto id : ids)
	{
		std::cout << static_cast<char>(id);
	}
	std::cout << "\"";
}


std::vector<std::vector<int64_t>> build_candidate_sequences(
	const TrigoGame& game,
	const TGNTokenizer& tokenizer
)
{
	auto valid_moves = game.valid_move_positions();
	std::vector<std::vector<int64_t>> candidate_sequences;

	// Encode each move (excluding last token)
	for (const auto& move : valid_moves)
	{
		std::string coord = encode_ab0yz(move, game.get_shape());
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		if (!move_tokens.empty())
		{
			std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end() - 1);
			candidate_sequences.push_back(seq);
		}
	}

	// Add PASS (excluding last token)
	auto pass_tokens = tokenizer.encode("PASS", 2048, false, false, false, false);
	if (!pass_tokens.empty())
	{
		std::vector<int64_t> seq(pass_tokens.begin(), pass_tokens.end() - 1);
		candidate_sequences.push_back(seq);
	}

	return candidate_sequences;
}


bool test_scenario(
	const std::string& name,
	const BoardShape& shape,
	const std::vector<std::string>& moves,
	const std::vector<int64_t>& expected_ids
)
{
	std::cout << "\n=== Test: " << name << " ===" << std::endl;

	// Setup game
	TrigoGame game(shape);
	game.start_game();

	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};

	// Play moves
	for (const auto& move_str : moves)
	{
		auto pos_vec = decode_ab0yz(move_str, shape_vec);
		Position move{pos_vec[0], pos_vec[1], pos_vec[2]};
		game.drop(move);
		std::cout << "  Played: " << move_str << std::endl;
	}

	// Build tree
	TGNTokenizer tokenizer;
	auto candidate_sequences = build_candidate_sequences(game, tokenizer);

	PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);

	std::cout << "Valid moves: " << game.valid_move_positions().size() << std::endl;
	std::cout << "Candidates: " << candidate_sequences.size() << std::endl;
	std::cout << "Tree nodes: " << tree_structure.num_nodes << std::endl;

	std::cout << "Evaluated IDs (numeric): ";
	print_evaluated_ids(tree_structure.evaluated_ids);
	std::cout << std::endl;

	std::cout << "Evaluated IDs (chars):   ";
	print_evaluated_ids_as_chars(tree_structure.evaluated_ids);
	std::cout << std::endl;

	// Compare with expected
	if (tree_structure.evaluated_ids == expected_ids)
	{
		std::cout << "✓ MATCH! C++ matches expected TypeScript values" << std::endl;
		return true;
	}
	else
	{
		std::cout << "✗ MISMATCH!" << std::endl;
		std::cout << "Expected IDs (numeric): ";
		print_evaluated_ids(expected_ids);
		std::cout << std::endl;
		std::cout << "Expected IDs (chars):   ";
		print_evaluated_ids_as_chars(expected_ids);
		std::cout << std::endl;
		return false;
	}
}


int main()
{
	std::cout << "===================================================================" << std::endl;
	std::cout << "Test: C++ evaluated_ids Match TypeScript" << std::endl;
	std::cout << "===================================================================" << std::endl;

	int passed = 0;
	int total = 0;

	// Test 1: Empty board after first move (from test_compare_with_ts.cpp)
	// Expected: [97, 98, 48, 122, 121, 80, 97, 115]
	// Chars: "ab0zyPas"
	total++;
	if (test_scenario(
		"5x5 board, first move a0",
		BoardShape{5, 5, 1},
		{"a0"},
		{97, 98, 48, 122, 121, 80, 97, 115}
	))
	{
		passed++;
	}

	// Test 2: Empty board (no moves yet)
	// All 25 positions available (5x5 board)
	// Tree should have 5 first-level nodes (a, b, 0, z, y) + second level + PASS
	total++;
	if (test_scenario(
		"5x5 board, no moves",
		BoardShape{5, 5, 1},
		{},
		{97, 98, 48, 122, 121, 80, 97, 115}  // Same structure as after first move
	))
	{
		passed++;
	}

	// Test 3: 3x3 board, first move at center (0,0)
	// Smaller board = different tree structure
	total++;
	if (test_scenario(
		"3x3 board, first move 00",
		BoardShape{3, 3, 1},
		{"00"},
		{97, 98, 122, 121, 80, 97, 115}  // Only 3x3: a, b, z, y (no '0' column)
	))
	{
		passed++;
	}

	// Test 4: 5x5 board, multiple moves
	total++;
	if (test_scenario(
		"5x5 board, moves: a0, zz",
		BoardShape{5, 5, 1},
		{"a0", "zz"},
		{97, 98, 48, 122, 121, 80, 97, 115}  // Same structure (23 moves left)
	))
	{
		passed++;
	}

	std::cout << "\n===================================================================" << std::endl;
	std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
	std::cout << "===================================================================" << std::endl;

	return (passed == total) ? 0 : 1;
}
