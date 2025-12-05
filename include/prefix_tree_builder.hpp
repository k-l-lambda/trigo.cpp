/**
 * Prefix Tree Builder for Tree Attention
 *
 * Builds a prefix tree structure from multiple token sequences (moves) to enable
 * efficient batch evaluation with custom attention patterns.
 *
 * Algorithm (from trigoTreeAgent.ts:77-168):
 * 1. Group sequences by first token recursively
 * 2. Flatten tree into linear array (evaluated_ids)
 * 3. Build ancestor attention mask (nodes can attend to ancestors)
 * 4. Track which position corresponds to each move's end
 */

#pragma once

#include <vector>
#include <map>
#include <cstdint>


namespace trigo
{


/**
 * Result of prefix tree building
 */
struct TreeStructure
{
	// Flattened tree token sequence
	std::vector<int64_t> evaluated_ids;  // [m] tokens in tree order

	// Attention mask: evaluated_mask[i][j] = 1 if node i can attend to node j
	std::vector<float> evaluated_mask;   // [m * m] flattened 2D mask

	// Mapping from move index to its leaf position in evaluated_ids
	std::vector<int> move_to_leaf;       // [num_moves] -> position in evaluated_ids

	// Dimensions
	int num_nodes;   // m (total nodes in tree)
	int num_moves;   // number of input moves
};


/**
 * Prefix Tree Builder
 *
 * Converts multiple token sequences into a tree structure with shared prefixes.
 */
class PrefixTreeBuilder
{
public:
	/**
	 * Build prefix tree from token arrays
	 *
	 * @param token_arrays Array of token sequences, one per move
	 *                     token_arrays[i] = tokens for move i
	 * @return TreeStructure with evaluated_ids, mask, and move_to_leaf mapping
	 *
	 * Example:
	 *   Input:  [[1,2,3], [1,2,4], [1,5]]
	 *   Tree:   1 -> 2 -> [3, 4]
	 *           └--> 5
	 *   Output: evaluated_ids = [1, 2, 3, 4, 5]
	 *           move_to_leaf[0] = 2  (move [1,2,3] ends at pos 2)
	 *           move_to_leaf[1] = 3  (move [1,2,4] ends at pos 3)
	 *           move_to_leaf[2] = 4  (move [1,5] ends at pos 4)
	 */
	TreeStructure build_tree(const std::vector<std::vector<int64_t>>& token_arrays);


private:
	// Internal node structure during tree building
	struct TreeNode
	{
		int64_t token;
		int pos;
		int parent;  // -1 for root
		std::vector<TreeNode*> children;
		std::vector<int> move_ends;  // indices of moves that end at this node

		TreeNode(int64_t t, int p, int par)
			: token(t), pos(p), parent(par) {}
	};

	// Sequence with move index
	struct Sequence
	{
		int move_index;
		std::vector<int64_t> tokens;
	};

	int next_pos_;  // Counter for assigning positions

	/**
	 * Recursive tree building (from TypeScript build() function)
	 *
	 * Groups sequences by first token, creates nodes, and recurses on residues.
	 */
	std::vector<TreeNode*> build_recursive(
		const std::vector<Sequence>& seqs,
		int parent_pos
	);

	/**
	 * Flatten tree via DFS traversal
	 *
	 * Fills evaluated_ids and parent arrays, and updates move_to_leaf mapping.
	 */
	void flatten_tree(
		TreeNode* node,
		std::vector<int64_t>& evaluated_ids,
		std::vector<int>& parent,
		std::vector<int>& move_to_leaf
	);

	/**
	 * Build ancestor attention mask
	 *
	 * mask[i * total + j] = 1 if node i can attend to node j (j is ancestor of i)
	 */
	void build_ancestor_mask(
		const std::vector<int>& parent,
		int total,
		std::vector<float>& mask
	);

	/**
	 * Clean up tree nodes (called after building)
	 */
	void cleanup_tree(std::vector<TreeNode*>& roots);
};


} // namespace trigo
