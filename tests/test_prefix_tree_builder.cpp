/**
 * Test PrefixTreeBuilder
 *
 * Verify that the C++ implementation produces the same results as TypeScript
 */

#include "../include/prefix_tree_builder.hpp"
#include <iostream>
#include <iomanip>


using namespace trigo;


void print_tree_structure(const TreeStructure& tree)
{
	std::cout << "Tree Structure:\n";
	std::cout << "  Nodes: " << tree.num_nodes << "\n";
	std::cout << "  Moves: " << tree.num_moves << "\n";
	std::cout << "\n";

	std::cout << "  Evaluated IDs: [";
	for (size_t i = 0; i < tree.evaluated_ids.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.evaluated_ids[i];
	}
	std::cout << "]\n\n";

	std::cout << "  Move to Leaf: [";
	for (size_t i = 0; i < tree.move_to_leaf.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tree.move_to_leaf[i];
	}
	std::cout << "]\n\n";

	std::cout << "  Attention Mask:\n";
	for (int i = 0; i < tree.num_nodes; i++)
	{
		std::cout << "    [" << i << "]: ";
		for (int j = 0; j < tree.num_nodes; j++)
		{
			std::cout << static_cast<int>(tree.evaluated_mask[i * tree.num_nodes + j]);
			if (j < tree.num_nodes - 1) std::cout << " ";
		}
		std::cout << "\n";
	}
}


bool test_simple_tree()
{
	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "Test 1: Simple Tree\n";
	std::cout << "======================================================================\n\n";

	// Example from header comment:
	//   Input:  [[1,2,3], [1,2,4], [1,5]]
	//   Tree:   1 -> 2 -> [3, 4]
	//           └--> 5
	//   Position assignment (during tree build, DFS order):
	//     1: pos 0
	//     2: pos 1 (child of 1)
	//       3: pos 2 (child of 2)
	//       4: pos 3 (child of 2)
	//     5: pos 4 (child of 1)
	//   Output: evaluated_ids = [1, 2, 3, 4, 5]
	//           move_to_leaf[0] = 2  (move [1,2,3] ends at pos 2)
	//           move_to_leaf[1] = 3  (move [1,2,4] ends at pos 3)
	//           move_to_leaf[2] = 4  (move [1,5] ends at pos 4)

	std::vector<std::vector<int64_t>> token_arrays = {
		{1, 2, 3},
		{1, 2, 4},
		{1, 5}
	};

	PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_arrays);

	print_tree_structure(tree);

	// Verify results
	std::vector<int64_t> expected_ids = {1, 2, 3, 4, 5};
	std::vector<int> expected_leaf = {2, 3, 4};

	if (tree.evaluated_ids != expected_ids)
	{
		std::cout << "\n✗ FAILED: evaluated_ids mismatch\n";
		return false;
	}

	if (tree.move_to_leaf != expected_leaf)
	{
		std::cout << "\n✗ FAILED: move_to_leaf mismatch\n";
		return false;
	}

	std::cout << "\n✓ Test 1 PASSED\n";
	return true;
}


bool test_single_move()
{
	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "Test 2: Single Move\n";
	std::cout << "======================================================================\n\n";

	std::vector<std::vector<int64_t>> token_arrays = {
		{10, 20, 30}
	};

	PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_arrays);

	print_tree_structure(tree);

	if (tree.num_nodes != 3)
	{
		std::cout << "\n✗ FAILED: Expected 3 nodes, got " << tree.num_nodes << "\n";
		return false;
	}

	if (tree.move_to_leaf[0] != 2)
	{
		std::cout << "\n✗ FAILED: Expected leaf at position 2\n";
		return false;
	}

	std::cout << "\n✓ Test 2 PASSED\n";
	return true;
}


bool test_no_shared_prefix()
{
	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "Test 3: No Shared Prefix\n";
	std::cout << "======================================================================\n\n";

	std::vector<std::vector<int64_t>> token_arrays = {
		{1, 2},
		{3, 4},
		{5, 6}
	};

	PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_arrays);

	print_tree_structure(tree);

	if (tree.num_nodes != 6)
	{
		std::cout << "\n✗ FAILED: Expected 6 nodes (3 roots, 3 children)\n";
		return false;
	}

	std::cout << "\n✓ Test 3 PASSED\n";
	return true;
}


bool test_all_same_prefix()
{
	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "Test 4: All Same Prefix\n";
	std::cout << "======================================================================\n\n";

	std::vector<std::vector<int64_t>> token_arrays = {
		{1, 2, 3},
		{1, 2, 4},
		{1, 2, 5}
	};

	PrefixTreeBuilder builder;
	auto tree = builder.build_tree(token_arrays);

	print_tree_structure(tree);

	if (tree.num_nodes != 5)
	{
		std::cout << "\n✗ FAILED: Expected 5 nodes\n";
		return false;
	}

	// All moves should end at different leaf positions
	if (tree.move_to_leaf[0] == tree.move_to_leaf[1] ||
	    tree.move_to_leaf[1] == tree.move_to_leaf[2] ||
	    tree.move_to_leaf[0] == tree.move_to_leaf[2])
	{
		std::cout << "\n✗ FAILED: Moves should end at different leaves\n";
		return false;
	}

	std::cout << "\n✓ Test 4 PASSED\n";
	return true;
}


int main()
{
	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "PrefixTreeBuilder Test Suite\n";
	std::cout << "======================================================================\n";

	int passed = 0;
	int total = 4;

	if (test_simple_tree()) passed++;
	if (test_single_move()) passed++;
	if (test_no_shared_prefix()) passed++;
	if (test_all_same_prefix()) passed++;

	std::cout << "\n";
	std::cout << "======================================================================\n";
	std::cout << "Results: " << passed << "/" << total << " tests passed\n";
	std::cout << "======================================================================\n\n";

	return (passed == total) ? 0 : 1;
}
