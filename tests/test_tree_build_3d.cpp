/**
 * Test prefix tree building for 3D board moves
 *
 * Compare C++ tree builder with TypeScript implementation
 */

#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/trigo_coords.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>


using namespace trigo;


void print_tree_structure(const TreeStructure& tree, const std::vector<std::string>& notations)
{
	std::cout << "Tree structure:" << std::endl;
	std::cout << "  num_nodes: " << tree.num_nodes << std::endl;
	std::cout << "  num_moves: " << tree.num_moves << std::endl;
	std::cout << std::endl;

	std::cout << "evaluated_ids: [";
	for (int i = 0; i < tree.num_nodes; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.evaluated_ids[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	std::cout << "parent: [";
	for (int i = 0; i < tree.num_nodes; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.parent[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	std::cout << "move_to_leaf:" << std::endl;
	for (size_t i = 0; i < notations.size(); i++)
	{
		std::cout << "  " << notations[i] << " -> " << tree.move_to_leaf[i] << std::endl;
	}
	std::cout << std::endl;

	// Print mask matrix
	std::cout << "mask (ancestor attention):" << std::endl;
	std::cout << "    ";
	for (int j = 0; j < tree.num_nodes; j++)
	{
		std::cout << std::setw(3) << j;
	}
	std::cout << std::endl;

	for (int i = 0; i < tree.num_nodes; i++)
	{
		std::cout << std::setw(3) << i << " ";
		for (int j = 0; j < tree.num_nodes; j++)
		{
			int val = static_cast<int>(tree.evaluated_mask[i * tree.num_nodes + j]);
			std::cout << std::setw(3) << val;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


int main()
{
	std::cout << "============================================================================" << std::endl;
	std::cout << "3D Board Prefix Tree Build Test" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;

	TGNTokenizer tokenizer;

	// Test 1: Simple 3D coordinates
	std::cout << "Test 1: Simple 3D coordinates (3x3x3 board)" << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;

	BoardShape shape{3, 3, 3};
	std::vector<Position> moves = {
		{0, 0, 0},  // aaa
		{0, 0, 1},  // aab
		{0, 1, 0},  // aba
		{1, 0, 0},  // baa
	};

	std::vector<std::string> notations;
	std::vector<std::vector<int64_t>> token_sequences;

	for (const auto& move : moves)
	{
		std::string coord = encode_ab0yz(move, shape);
		notations.push_back(coord);

		auto tokens = tokenizer.encode(coord, 2048, false, false, false, false);
		std::cout << "Move " << coord << ": tokens = [";
		for (size_t i = 0; i < tokens.size(); i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << tokens[i];
		}
		std::cout << "]" << std::endl;

		// Exclude last token (as per scoreMoves logic)
		if (tokens.size() > 1)
		{
			std::vector<int64_t> seq(tokens.begin(), tokens.end() - 1);
			token_sequences.push_back(seq);
		}
		else
		{
			token_sequences.push_back(std::vector<int64_t>());
		}
	}
	std::cout << std::endl;

	// Build tree
	PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_sequences);

	print_tree_structure(tree, notations);


	// Test 2: Larger 3D board with more moves
	std::cout << std::endl;
	std::cout << "Test 2: Larger 3D board (5x5x5)" << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;

	BoardShape shape2{5, 5, 5};
	std::vector<Position> moves2 = {
		{0, 0, 0},  // aaa
		{0, 0, 1},  // aab
		{0, 0, 2},  // aa0
		{4, 4, 4},  // zzz
		{4, 4, 3},  // zzy
		{2, 2, 2},  // 000
	};

	std::vector<std::string> notations2;
	std::vector<std::vector<int64_t>> token_sequences2;

	for (const auto& move : moves2)
	{
		std::string coord = encode_ab0yz(move, shape2);
		notations2.push_back(coord);

		auto tokens = tokenizer.encode(coord, 2048, false, false, false, false);
		std::cout << "Move " << coord << ": tokens = [";
		for (size_t i = 0; i < tokens.size(); i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << tokens[i];
		}
		std::cout << "]" << std::endl;

		// Exclude last token
		if (tokens.size() > 1)
		{
			std::vector<int64_t> seq(tokens.begin(), tokens.end() - 1);
			token_sequences2.push_back(seq);
		}
		else
		{
			token_sequences2.push_back(std::vector<int64_t>());
		}
	}
	std::cout << std::endl;

	auto tree2 = builder.build_tree(token_sequences2);
	print_tree_structure(tree2, notations2);

	std::cout << "============================================================================" << std::endl;
	std::cout << "Compare these results with TypeScript output" << std::endl;
	std::cout << "============================================================================" << std::endl;

	return 0;
}
