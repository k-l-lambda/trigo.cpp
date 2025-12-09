/**
 * Test tree building for 5x5 board - compare with TypeScript
 */

#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>


int main()
{
	std::cout << "============================================================================" << std::endl;
	std::cout << "Tree Building Comparison: C++" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;

	// Initialize tokenizer
	trigo::TGNTokenizer tokenizer;

	// Generate all 2-char coords for 5x5 board
	std::vector<std::string> coords = {
		"aa", "ab", "a0", "ay", "az",
		"ba", "bb", "b0", "by", "bz",
		"0a", "0b", "00", "0y", "0z",
		"ya", "yb", "y0", "yy", "yz",
		"za", "zb", "z0", "zy", "zz"
	};

	std::vector<std::vector<int64_t>> token_sequences;

	std::cout << "Move tokens (excluding last):" << std::endl;
	for (const auto& coord : coords)
	{
		auto tokens = tokenizer.encode(coord, 256, false, false, false, false);
		// Exclude last token for tree building
		std::vector<int64_t> tree_tokens;
		if (tokens.size() > 1)
		{
			tree_tokens.assign(tokens.begin(), tokens.end() - 1);
		}
		token_sequences.push_back(tree_tokens);

		std::cout << "  " << coord << ": full=[";
		for (size_t i = 0; i < tokens.size(); i++)
		{
			if (i > 0) std::cout << ",";
			std::cout << tokens[i];
		}
		std::cout << "] tree=[";
		for (size_t i = 0; i < tree_tokens.size(); i++)
		{
			if (i > 0) std::cout << ",";
			std::cout << tree_tokens[i];
		}
		std::cout << "]" << std::endl;
	}

	// Add Pass
	auto pass_tokens = tokenizer.encode("Pass", 256, false, false, false, false);
	std::vector<int64_t> pass_tree;
	if (pass_tokens.size() > 1)
	{
		pass_tree.assign(pass_tokens.begin(), pass_tokens.end() - 1);
	}
	token_sequences.push_back(pass_tree);

	std::cout << "  Pass: full=[";
	for (size_t i = 0; i < pass_tokens.size(); i++)
	{
		if (i > 0) std::cout << ",";
		std::cout << pass_tokens[i];
	}
	std::cout << "] tree=[";
	for (size_t i = 0; i < pass_tree.size(); i++)
	{
		if (i > 0) std::cout << ",";
		std::cout << pass_tree[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Build tree
	trigo::PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_sequences);

	std::cout << "C++ tree structure for full 5x5:" << std::endl;
	std::cout << "  num_nodes: " << tree.num_nodes << std::endl;

	std::cout << "  evaluated_ids: [";
	for (int i = 0; i < tree.num_nodes; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.evaluated_ids[i];
	}
	std::cout << "]" << std::endl;

	std::cout << "  parent: [";
	for (int i = 0; i < tree.num_nodes; i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.parent[i];
	}
	std::cout << "]" << std::endl;

	std::cout << "  move_to_leaf: [";
	for (size_t i = 0; i < tree.move_to_leaf.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.move_to_leaf[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Expected from TypeScript
	std::cout << "============================================================================" << std::endl;
	std::cout << "Expected from TypeScript:" << std::endl;
	std::cout << "  num_nodes: 8" << std::endl;
	std::cout << "  evaluated_ids: [97, 98, 48, 121, 122, 80, 97, 115]" << std::endl;
	std::cout << "  parent: [-1, -1, -1, -1, -1, -1, 5, 6]" << std::endl;
	std::cout << "  move_to_leaf: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 7]" << std::endl;
	std::cout << "============================================================================" << std::endl;

	// Compare
	bool match = true;
	if (tree.num_nodes != 8)
	{
		std::cout << "❌ num_nodes MISMATCH: " << tree.num_nodes << " vs 8" << std::endl;
		match = false;
	}

	std::vector<int64_t> expected_ids = {97, 98, 48, 121, 122, 80, 97, 115};
	std::vector<int> expected_parent = {-1, -1, -1, -1, -1, -1, 5, 6};
	std::vector<int> expected_leaf = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 7};

	for (int i = 0; i < std::min(tree.num_nodes, 8); i++)
	{
		if (tree.evaluated_ids[i] != expected_ids[i])
		{
			std::cout << "❌ evaluated_ids[" << i << "] MISMATCH: " << tree.evaluated_ids[i] << " vs " << expected_ids[i] << std::endl;
			match = false;
		}
		if (tree.parent[i] != expected_parent[i])
		{
			std::cout << "❌ parent[" << i << "] MISMATCH: " << tree.parent[i] << " vs " << expected_parent[i] << std::endl;
			match = false;
		}
	}

	for (size_t i = 0; i < std::min(tree.move_to_leaf.size(), expected_leaf.size()); i++)
	{
		if (tree.move_to_leaf[i] != expected_leaf[i])
		{
			std::cout << "❌ move_to_leaf[" << i << "] MISMATCH: " << tree.move_to_leaf[i] << " vs " << expected_leaf[i] << std::endl;
			match = false;
		}
	}

	if (match)
	{
		std::cout << "✓ Tree structures MATCH!" << std::endl;
	}

	return match ? 0 : 1;
}
