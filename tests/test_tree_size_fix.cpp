/**
 * Test that prefix tree building now matches TypeScript implementation
 *
 * Verifies that excluding the last token results in 8 nodes (not 33)
 * for the same test case from test_compare_with_ts.cpp
 */

#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include <iostream>
#include <vector>


using namespace trigo;


int main()
{
	std::cout << "=== Test: Prefix Tree Size Fix ===" << std::endl;
	std::cout << std::endl;

	// Setup: 5x5 board, first move at a0
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	std::vector<int> shape_vec = {shape.x, shape.y, shape.z};
	auto pos_vec = decode_ab0yz("a0", shape_vec);
	Position move_a0{pos_vec[0], pos_vec[1], pos_vec[2]};
	game.drop(move_a0);

	// Get all valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << std::endl;

	// Tokenize each move (EXCLUDING last token as per TypeScript)
	TGNTokenizer tokenizer;
	std::vector<std::vector<int64_t>> candidate_sequences;

	for (const auto& move : valid_moves)
	{
		std::string coord = encode_ab0yz(move, shape);
		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		// Exclude last token (following TypeScript trigoTreeAgent.ts:226-227)
		if (!move_tokens.empty())
		{
			std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end() - 1);
			candidate_sequences.push_back(seq);
		}
	}

	// Add PASS move (excluding last token)
	auto pass_tokens = tokenizer.encode("PASS", 2048, false, false, false, false);
	if (!pass_tokens.empty())
	{
		std::vector<int64_t> seq(pass_tokens.begin(), pass_tokens.end() - 1);
		candidate_sequences.push_back(seq);
	}

	std::cout << "Candidate sequences: " << candidate_sequences.size() << std::endl;

	// Build tree
	PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);

	std::cout << "Tree nodes: " << tree_structure.num_nodes << std::endl;
	std::cout << std::endl;

	// Verify: TypeScript builds 8 nodes for this scenario
	// (5 first-level tokens: 'a', 'b', '0', 'z', 'y' and 3 second-level tokens + PASS)
	if (tree_structure.num_nodes == 8)
	{
		std::cout << "✓ SUCCESS! Tree now has 8 nodes (matches TypeScript)" << std::endl;
		return 0;
	}
	else
	{
		std::cout << "✗ FAILURE! Expected 8 nodes, got " << tree_structure.num_nodes << std::endl;
		std::cout << "  (Before fix: 33 nodes)" << std::endl;
		return 1;
	}
}
