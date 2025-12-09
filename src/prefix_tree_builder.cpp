/**
 * Prefix Tree Builder Implementation
 *
 * Port of TypeScript buildPrefixTree from trigoTreeAgent.ts:77-168
 */

#include "../include/prefix_tree_builder.hpp"
#include <algorithm>


namespace trigo
{


TreeStructure PrefixTreeBuilder::build_tree(const std::vector<std::vector<int64_t>>& token_arrays)
{
	if (token_arrays.empty())
	{
		return TreeStructure{
			std::vector<int64_t>(),
			std::vector<float>(),
			std::vector<int>(),
			std::vector<int>(),
			0,
			0
		};
	}

	// Reset position counter
	next_pos_ = 0;

	// Convert to Sequence format
	std::vector<Sequence> seqs;
	seqs.reserve(token_arrays.size());
	for (size_t i = 0; i < token_arrays.size(); i++)
	{
		seqs.push_back(Sequence{static_cast<int>(i), token_arrays[i]});
	}

	// Build tree recursively
	auto roots = build_recursive(seqs, -1);
	int total = next_pos_;

	// Prepare output arrays
	std::vector<int64_t> evaluated_ids(total);
	std::vector<int> parent(total, -1);
	std::vector<int> move_to_leaf(token_arrays.size(), -1);

	// Flatten tree via DFS
	for (auto* root : roots)
	{
		flatten_tree(root, evaluated_ids, parent, move_to_leaf);
	}

	// Build ancestor attention mask
	std::vector<float> mask(total * total, 0.0f);
	build_ancestor_mask(parent, total, mask);

	// Cleanup tree nodes
	cleanup_tree(roots);

	return TreeStructure{
		std::move(evaluated_ids),
		std::move(mask),
		std::move(move_to_leaf),
		std::move(parent),
		total,
		static_cast<int>(token_arrays.size())
	};
}


std::vector<PrefixTreeBuilder::TreeNode*> PrefixTreeBuilder::build_recursive(
	const std::vector<Sequence>& seqs,
	int parent_pos
)
{
	// Group sequences by first token
	std::map<int64_t, std::vector<Sequence>> groups;

	for (const auto& s : seqs)
	{
		if (s.tokens.empty()) continue;

		int64_t first_token = s.tokens[0];
		groups[first_token].push_back(s);
	}

	std::vector<TreeNode*> level_nodes;

	// Create node for each unique first token
	for (const auto& [token, group] : groups)
	{
		int pos = next_pos_++;
		TreeNode* node = new TreeNode(token, pos, parent_pos);

		// Split into ends (single token left) and residues (more tokens)
		std::vector<int> ends;
		std::vector<Sequence> residues;

		for (const auto& g : group)
		{
			if (g.tokens.size() == 1)
			{
				// This move ends at current node
				ends.push_back(g.move_index);
			}
			else
			{
				// More tokens remaining, create residue
				Sequence residue;
				residue.move_index = g.move_index;
				residue.tokens.assign(g.tokens.begin() + 1, g.tokens.end());
				residues.push_back(residue);
			}
		}

		node->move_ends = ends;

		// Recursively create children
		if (!residues.empty())
		{
			node->children = build_recursive(residues, pos);
		}

		level_nodes.push_back(node);
	}

	return level_nodes;
}


void PrefixTreeBuilder::flatten_tree(
	TreeNode* node,
	std::vector<int64_t>& evaluated_ids,
	std::vector<int>& parent,
	std::vector<int>& move_to_leaf
)
{
	// Fill position in flattened arrays
	evaluated_ids[node->pos] = node->token;
	parent[node->pos] = node->parent;

	// Update move_to_leaf for moves that end at this node
	for (int move_idx : node->move_ends)
	{
		move_to_leaf[move_idx] = node->pos;
	}

	// Recursively flatten children
	for (TreeNode* child : node->children)
	{
		flatten_tree(child, evaluated_ids, parent, move_to_leaf);
	}
}


void PrefixTreeBuilder::build_ancestor_mask(
	const std::vector<int>& parent,
	int total,
	std::vector<float>& mask
)
{
	// For each node i, mark all ancestors (including self) as attendable
	for (int i = 0; i < total; i++)
	{
		int p = i;
		while (p != -1)
		{
			mask[i * total + p] = 1.0f;
			p = parent[p];
		}
	}
}


void PrefixTreeBuilder::cleanup_tree(std::vector<TreeNode*>& roots)
{
	// DFS deletion
	std::vector<TreeNode*> stack = roots;

	while (!stack.empty())
	{
		TreeNode* node = stack.back();
		stack.pop_back();

		// Add children to stack
		for (TreeNode* child : node->children)
		{
			stack.push_back(child);
		}

		// Delete node
		delete node;
	}
}


} // namespace trigo
