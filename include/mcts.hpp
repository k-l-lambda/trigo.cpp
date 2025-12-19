/**
 * AlphaZero-Style MCTS Implementation
 *
 * Combines MCTS with neural network guidance:
 * - Policy network for move selection (prior probabilities)
 * - Value network for position evaluation (replaces random rollouts)
 * - PUCT formula for exploration
 *
 * Performance: ~250× faster than pure MCTS with random rollouts
 *
 * NOTE: For reference implementation with random rollouts, see mcts_moc.hpp
 */

#pragma once

#include "trigo_game.hpp"
#include "shared_model_inferencer.hpp"
#include "tgn_tokenizer.hpp"
#include "prefix_tree_builder.hpp"
#include "tgn_utils.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <iomanip>


namespace trigo
{

/**
 * Action selection result
 */
struct PolicyAction
{
	Position position;
	bool is_pass;
	float confidence;  // Policy probability or value estimate

	PolicyAction() : is_pass(true), confidence(0.0f) {}
	PolicyAction(const Position& pos, float conf = 1.0f)
		: position(pos), is_pass(false), confidence(conf) {}

	static PolicyAction Pass(float conf = 1.0f)
	{
		PolicyAction action;
		action.is_pass = true;
		action.confidence = conf;
		return action;
	}
};


/**
 * MCTS Node
 *
 * Represents a game state in the search tree
 */
class MCTSNode
{
public:
	Position move;           // Move that led to this node
	bool is_pass;            // Whether this is a pass move
	MCTSNode* parent;        // Parent node
	std::vector<std::unique_ptr<MCTSNode>> children;  // Child nodes

	int visit_count;         // Number of visits (N)
	float total_value;       // Sum of values (W)
	float prior_prob;        // Prior probability (for AlphaZero-style MCTS)

	bool is_fully_expanded;  // Whether all children have been created


	MCTSNode(const Position& m, bool pass, MCTSNode* p = nullptr, float prior = 1.0f)
		: move(m)
		, is_pass(pass)
		, parent(p)
		, visit_count(0)
		, total_value(0.0f)
		, prior_prob(prior)
		, is_fully_expanded(false)
	{
	}


	/**
	 * Get average value (Q value)
	 */
	float q_value() const
	{
		if (visit_count == 0)
			return 0.0f;

		return total_value / visit_count;
	}


	/**
	 * UCB1 score for action selection
	 *
	 * UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
	 *
	 * NOTE: This method is NOT used in the main MCTS flow.
	 * It assumes Q is white-positive and does NOT handle player perspective.
	 * For white-positive MCTS, use select_best_puct_child() which applies
	 * player perspective adjustment: (is_white ? q : -q) + u
	 */
	float ucb1_score(float exploration_constant) const
	{
		if (visit_count == 0)
			return std::numeric_limits<float>::infinity();

		float exploitation = q_value();
		float exploration = exploration_constant * std::sqrt(std::log(parent->visit_count) / visit_count);

		return exploitation + exploration;
	}


	/**
	 * PUCT score for AlphaZero-style MCTS
	 *
	 * PUCT = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
	 *
	 * NOTE: This method is NOT used in the main MCTS flow.
	 * It assumes Q is white-positive and does NOT handle player perspective.
	 * For white-positive MCTS, use select_best_puct_child() which applies
	 * player perspective adjustment: (is_white ? q : -q) + u
	 */
	float puct_score(float c_puct) const
	{
		float q = q_value();
		float u = c_puct * prior_prob * std::sqrt(parent->visit_count) / (1.0f + visit_count);

		return q + u;
	}
};


/**
 * MCTS Engine (AlphaZero-Style)
 *
 * Uses neural networks for both move selection and position evaluation
 */
class MCTS
{
private:
	std::unique_ptr<MCTSNode> root;
	int num_simulations;
	float c_puct;  // Exploration constant for PUCT formula

	// Dirichlet noise parameters (for root exploration)
	float dirichlet_alpha;    // Alpha parameter (default: 0.03 for Go-like games)
	float dirichlet_epsilon;  // Noise mixing weight (default: 0.25)

	// Neural network components
	std::shared_ptr<SharedModelInferencer> inferencer;
	TGNTokenizer tokenizer;

	std::mt19937 rng;

	// Pass move prior probability (default: 1e-10, minimal prior)
	float pass_prior;


public:
	MCTS(
		std::shared_ptr<SharedModelInferencer> inf,
		int num_sims = 800,
		float exploration = 1.0f,
		int seed = 42,
		float dir_alpha = 0.03f,
		float dir_epsilon = 0.25f,
		float pass_prob = 1e-10f  // Default minimal prior for Pass
	)
		: num_simulations(num_sims)
		, c_puct(exploration)
		, dirichlet_alpha(dir_alpha)
		, dirichlet_epsilon(dir_epsilon)
		, inferencer(inf)
		, rng(seed)
		, pass_prior(pass_prob)
	{
	}


	/**
	 * Run AlphaZero MCTS search and return best move
	 *
	 * @param game Current game state
	 * @return Best move found by MCTS
	 */
	PolicyAction search(const TrigoGame& game)
	{
		// Create root node
		root = std::make_unique<MCTSNode>(Position{0, 0, 0}, false);
		root->visit_count = 1;  // Root is always visited

		bool dirichlet_applied = false;  // Track if Dirichlet noise has been added

#ifdef MCTS_ENABLE_PROFILING
		std::cout << "[AlphaZero MCTS] Starting search with " << num_simulations << " simulations\n";
		auto start_time = std::chrono::steady_clock::now();
#endif

		// Run simulations
		for (int i = 0; i < num_simulations; i++)
		{
#ifdef MCTS_ENABLE_PROFILING
			if (i % 10 == 0)
			{
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::steady_clock::now() - start_time
				).count();
				std::cout << "[AlphaZero MCTS] Simulation " << i << "/" << num_simulations
				          << " (" << elapsed << "ms)\n";
			}

			auto sim_start = std::chrono::steady_clock::now();
#endif

			// Make a copy of game state for simulation
			TrigoGame game_copy = game;

			// 1. Selection: Traverse tree using PUCT
			MCTSNode* node = select(root.get(), game_copy);

#ifdef MCTS_ENABLE_PROFILING
			auto select_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count();
#endif

			// 2. Expansion: Add new child node with policy prior
			if (game_copy.is_game_active() && node->visit_count > 0)
			{
				node = expand(node, game_copy);

				// Apply Dirichlet noise to root immediately after first expansion
				// AlphaZero style: add noise as soon as root children exist
				if (!dirichlet_applied && !root->children.empty())
				{
					add_dirichlet_noise_to_root();
					dirichlet_applied = true;
				}
			}

#ifdef MCTS_ENABLE_PROFILING
			auto expand_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count() - select_time;
#endif

			// 3. Evaluation: Use value network (replaces rollout)
			float value = evaluate(game_copy);

#ifdef MCTS_ENABLE_PROFILING
			auto evaluate_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count() - select_time - expand_time;
#endif

			// 4. Backpropagation: Update node statistics
			backpropagate(node, value);

#ifdef MCTS_ENABLE_PROFILING
			auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count();

			if (i == 0)
			{
				std::cout << "[AlphaZero MCTS] First simulation breakdown:\n";
				std::cout << "  Selection: " << select_time << "μs\n";
				std::cout << "  Expansion: " << expand_time << "μs\n";
				std::cout << "  Evaluation: " << evaluate_time << "μs (value network)\n";
				std::cout << "  Total: " << total_time << "μs\n";
			}
#endif
		}

#ifdef MCTS_ENABLE_PROFILING
		auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start_time
		).count();
		std::cout << "[AlphaZero MCTS] Search complete in " << total_elapsed << "ms\n";
#endif

		// Select best move by visit count (most robust)
		return select_best_child();
	}


	/**
	 * Print root children statistics for debugging
	 */
	void print_root_statistics(const TrigoGame& game) const
	{
		if (!root || root->children.empty())
		{
			std::cout << "[MCTS] No root children to display\n";
			return;
		}

		auto board_shape = game.get_shape();

		// Sort children by visit count (descending)
		std::vector<MCTSNode*> sorted_children;
		for (const auto& child : root->children)
		{
			sorted_children.push_back(child.get());
		}
		std::sort(sorted_children.begin(), sorted_children.end(),
			[](MCTSNode* a, MCTSNode* b) { return a->visit_count > b->visit_count; });

		std::cout << "\n[MCTS] Root children statistics (sorted by visits):\n";
		std::cout << std::setw(10) << "Move"
		          << std::setw(10) << "Visits"
		          << std::setw(12) << "Prior"
		          << std::setw(12) << "Q-value"
		          << "\n";
		std::cout << std::string(44, '-') << "\n";

		for (size_t i = 0; i < std::min(size_t(20), sorted_children.size()); i++)
		{
			MCTSNode* child = sorted_children[i];
			std::string move_str;
			if (child->is_pass)
			{
				move_str = "Pass";
			}
			else
			{
				move_str = encode_ab0yz({child->move.x, child->move.y, child->move.z}, board_shape);
			}

			std::cout << std::setw(10) << move_str
			          << std::setw(10) << child->visit_count
			          << std::setw(12) << std::fixed << std::setprecision(6) << child->prior_prob
			          << std::setw(12) << std::fixed << std::setprecision(6) << child->q_value()
			          << "\n";
		}

		// Show Pass stats if not in top 20
		bool pass_shown = false;
		for (size_t i = 0; i < std::min(size_t(20), sorted_children.size()); i++)
		{
			if (sorted_children[i]->is_pass) pass_shown = true;
		}
		if (!pass_shown)
		{
			for (const auto& child : root->children)
			{
				if (child->is_pass)
				{
					std::cout << "...\n";
					std::cout << std::setw(10) << "Pass"
					          << std::setw(10) << child->visit_count
					          << std::setw(12) << std::fixed << std::setprecision(6) << child->prior_prob
					          << std::setw(12) << std::fixed << std::setprecision(6) << child->q_value()
					          << "\n";
					break;
				}
			}
		}
	}


private:
	/**
	 * Selection phase: Traverse tree using PUCT until leaf node
	 */
	MCTSNode* select(MCTSNode* node, TrigoGame& game)
	{
		while (game.is_game_active())
		{
			// If not fully expanded, return this node for expansion
			if (!node->is_fully_expanded)
			{
				return node;
			}

			// Otherwise, select best child by PUCT
			node = select_best_puct_child(node, game.get_current_player());

			// Apply move to game
			if (node->is_pass)
			{
				game.pass();
			}
			else
			{
				game.drop(node->move);
			}
		}

		return node;
	}


	/**
	 * Expansion phase: Add ALL child nodes with policy priors (AlphaZero style)
	 *
	 * Unlike traditional MCTS that expands one node at a time,
	 * AlphaZero expands all children at once and sets their priors from policy network.
	 */
	MCTSNode* expand(MCTSNode* node, TrigoGame& game)
	{
		// AlphaZero-style: Expand ALL children at once on first visit
		// This ensures proper prior distribution without renormalization issues

		// If already expanded, select a child to traverse
		if (!node->children.empty())
		{
			// Find unexpanded child or select best by PUCT
			for (const auto& child : node->children)
			{
				if (child->visit_count == 0)
				{
					// Apply move to game
					if (child->is_pass)
					{
						game.pass();
					}
					else
					{
						game.drop(child->move);
					}
					return child.get();
				}
			}

			// All children have been visited at least once
			node->is_fully_expanded = true;
			return node;
		}

		// Get all valid moves
		auto valid_moves = game.valid_move_positions();
		bool can_pass = game.is_game_active();

		// Handle terminal or no-move situations
		if (valid_moves.empty() && !can_pass)
		{
			node->is_fully_expanded = true;
			return node;
		}

		// Get policy priors for ALL valid moves at once
		std::vector<float> priors = get_move_priors(game, valid_moves, can_pass, pass_prior);

#ifdef _DEBUG
		// Debug: verify priors array size
		if (priors.size() != valid_moves.size() + (can_pass ? 1 : 0))
		{
			std::cerr << "[MCTS expand] ERROR: priors size mismatch!\n";
			std::cerr << "  valid_moves.size() = " << valid_moves.size() << "\n";
			std::cerr << "  can_pass = " << can_pass << "\n";
			std::cerr << "  priors.size() = " << priors.size() << "\n";
			std::cerr << "  expected = " << (valid_moves.size() + (can_pass ? 1 : 0)) << "\n";
		}
#endif	// _DEBUG

#ifdef MCTS_ENABLE_PROFILING
		// Debug: Print top 5 moves with priors
		auto board_shape = game.get_shape();
		std::vector<std::pair<size_t, float>> indexed_priors;
		for (size_t i = 0; i < priors.size(); i++)
		{
			indexed_priors.push_back({i, priors[i]});
		}
		std::sort(indexed_priors.begin(), indexed_priors.end(),
			[](const auto& a, const auto& b) { return a.second > b.second; });

		std::cout << "  [expand] Top 5 moves by prior:" << std::endl;
		for (size_t i = 0; i < std::min(size_t(5), indexed_priors.size()); i++)
		{
			size_t idx = indexed_priors[i].first;
			std::string move_str;
			if (idx < valid_moves.size())
			{
				move_str = encode_ab0yz(valid_moves[idx], board_shape);
			}
			else
			{
				move_str = "Pass";
			}
			std::cout << "    " << (i + 1) << ". " << move_str
				<< " prior=" << std::fixed << std::setprecision(6) << priors[idx] << std::endl;
		}
#endif

		// Create ALL child nodes with their priors
		for (size_t i = 0; i < valid_moves.size(); i++)
		{
			auto new_node = std::make_unique<MCTSNode>(valid_moves[i], false, node, priors[i]);
			node->children.push_back(std::move(new_node));
		}

		// Add Pass node if available
		if (can_pass)
		{
			float pass_prior = priors.back();
			auto pass_node = std::make_unique<MCTSNode>(Position{0, 0, 0}, true, node, pass_prior);
			node->children.push_back(std::move(pass_node));
		}

		// Select first child using prior-weighted sampling
		std::discrete_distribution<size_t> dist(priors.begin(), priors.end());
		size_t idx = dist(rng);

		MCTSNode* selected_child = node->children[idx].get();

		// Apply selected move to game
		if (selected_child->is_pass)
		{
			game.pass();
		}
		else
		{
			game.drop(selected_child->move);
		}

		return selected_child;
	}


	/**
	 * Evaluation phase: Use value network to evaluate position
	 *
	 * Replaces expensive random rollouts with fast neural network inference
	 *
	 * @return White-positive value (positive = White winning, negative = Black winning)
	 *         Consistent with CachedMCTS value system
	 */
	float evaluate(TrigoGame& game)
	{
		// Convert game to TGN format
		std::string tgn_text = game_to_tgn(game, false);

		// Tokenize
		auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

		// Add START token
		std::vector<int64_t> tokens;
		tokens.push_back(1);  // START token
		tokens.insert(tokens.end(), encoded.begin(), encoded.end());

		int seq_len = tokens.size();

#ifdef MCTS_ENABLE_PROFILING
		static bool first_call = true;
		if (first_call)
		{
			std::cout << "[AlphaZero MCTS] First evaluate() call:\n";
			std::cout << "  TGN length: " << tgn_text.length() << " chars\n";
			std::cout << "  Encoded length: " << encoded.size() << " tokens\n";
			std::cout << "  Total seq_len: " << seq_len << " tokens (with START)\n";
			first_call = false;
		}
#endif

		// Note: Dynamic sequence length models don't require padding
		// The model will handle any sequence length

		// Sanity check seq_len
		if (seq_len > 8192)
		{
			std::cerr << "[AlphaZero MCTS] Warning: seq_len=" << seq_len << " exceeds max_length=8192\n";
			return 0.0f;  // Return neutral value on error
		}

		// Run value inference (batch_size=1)
		try
		{
			auto values = inferencer->value_inference(tokens, 1, seq_len, 3);
			float value = values[0];

			// IMPORTANT: Value model outputs White advantage (positive = White winning)
			// We return this directly without conversion (white-positive system)
			// PUCT selection will handle player perspective adjustment
			return value;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[AlphaZero MCTS] Value inference error (seq_len=" << seq_len << "): " << e.what() << "\n";
			return 0.0f;  // Return neutral value on error
		}
	}


	/**
	 * Backpropagation phase: Update statistics up the tree
	 *
	 * White-positive system: Q-values are always White advantage throughout the tree.
	 * Player perspective is handled during PUCT selection, not during backpropagation.
	 */
	void backpropagate(MCTSNode* node, float value)
	{
		while (node != nullptr)
		{
			node->visit_count++;
			node->total_value += value;

			// White-positive system: NO sign flipping
			// All Q-values represent White advantage
			// Player perspective handled in select_best_puct_child()

			node = node->parent;
		}
	}


	/**
	 * Apply softmax to logits (numerical stability)
	 */
	std::vector<float> softmax(const std::vector<float>& logits)
	{
		if (logits.empty())
			return {};

		float max_logit = *std::max_element(logits.begin(), logits.end());

		std::vector<float> exp_vals(logits.size());
		float sum = 0.0f;
		for (size_t i = 0; i < logits.size(); i++)
		{
			exp_vals[i] = std::exp(logits[i] - max_logit);
			sum += exp_vals[i];
		}

		std::vector<float> probs(logits.size());
		for (size_t i = 0; i < logits.size(); i++)
		{
			probs[i] = exp_vals[i] / sum;
		}

		return probs;
	}


	/**
	 * Compute softmax for a single position in logits array
	 */
	std::vector<float> softmax_at_position(
		const std::vector<float>& logits,
		int position,
		int vocab_size
	)
	{
		int offset = position * vocab_size;
		std::vector<float> position_logits(vocab_size);
		for (int i = 0; i < vocab_size; i++)
		{
			position_logits[i] = logits[offset + i];
		}
		return softmax(position_logits);
	}


	/**
	 * Convert log scores to normalized probabilities via exp and normalization
	 */
	std::vector<float> exp_normalize(const std::vector<float>& log_scores)
	{
		if (log_scores.empty())
			return {};

		float max_score = -std::numeric_limits<float>::infinity();
		for (float score : log_scores)
		{
			if (std::isfinite(score) && score > max_score)
			{
				max_score = score;
			}
		}

		if (!std::isfinite(max_score))
		{
			// All scores are -inf or nan, return uniform
			return std::vector<float>(log_scores.size(), 1.0f / log_scores.size());
		}

		std::vector<float> exp_vals(log_scores.size());
		float sum = 0.0f;
		for (size_t i = 0; i < log_scores.size(); i++)
		{
			if (std::isfinite(log_scores[i]))
			{
				exp_vals[i] = std::exp(log_scores[i] - max_score);
			}
			else
			{
				exp_vals[i] = 0.0f;
			}
			sum += exp_vals[i];
		}

		if (sum <= 0.0f)
		{
			return std::vector<float>(log_scores.size(), 1.0f / log_scores.size());
		}

		std::vector<float> probs(log_scores.size());
		for (size_t i = 0; i < log_scores.size(); i++)
		{
			probs[i] = exp_vals[i] / sum;
		}

		return probs;
	}


	/**
	 * Get policy priors for all valid moves using neural network
	 *
	 * Uses PrefixTreeBuilder to construct tree structure,
	 * then SharedModelInferencer::policy_inference for evaluation.
	 *
	 * @param game Current game state
	 * @param valid_moves List of valid positions
	 * @param include_pass Whether to include Pass as an option
	 * @return Probability distribution over moves (normalized)
	 */
	std::vector<float> get_move_priors(
		const TrigoGame& game,
		const std::vector<Position>& valid_moves,
		bool include_pass,
		float pass_prior_value = 1e-10f  // Default minimal prior for Pass move
	)
	{
		// Early exit optimization: if only Pass is valid, skip inference
		if (valid_moves.empty() && include_pass)
		{
			return {1.0f};  // 100% probability for Pass
		}

		auto board_shape = game.get_shape();
		std::vector<std::vector<int64_t>> candidate_sequences;
		std::vector<std::string> move_notations;

		// Add regular moves (ONLY move tokens, not prefix)
		// Exclude last token (following TypeScript trigoTreeAgent.ts pattern)
		for (const auto& move : valid_moves)
		{
			std::string coord = encode_ab0yz(move, board_shape);
			auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

			move_notations.push_back(coord);

			if (move_tokens.size() > 1)
			{
				std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end() - 1);
				candidate_sequences.push_back(seq);
			}
			else
			{
				candidate_sequences.push_back(std::vector<int64_t>());
			}
		}

		// Build prefix tree
		PrefixTreeBuilder tree_builder;
		auto tree_structure = tree_builder.build_tree(candidate_sequences);

		// Build game prefix tokens
		std::string tgn_text = game_to_tgn(game, true);  // Include move number prefix
		auto prefix_tokens = tokenizer.encode(tgn_text, 8192, false, false, false, false);

		// Add START token at beginning
		std::vector<int64_t> prefix_ids;
		prefix_ids.push_back(1);  // START token
		prefix_ids.insert(prefix_ids.end(), prefix_tokens.begin(), prefix_tokens.end());

		int prefix_len = static_cast<int>(prefix_ids.size());
		int eval_len = tree_structure.num_nodes;

		// Handle edge case: all moves are single-token (common on small boards)
		// In this case, tree_structure.num_nodes = 0, which would cause invalid tensor shape
		if (eval_len == 0)
		{
			// All moves are single tokens - can't build prefix tree
			// Fallback to uniform priors (all moves equally likely)
			std::vector<float> probs(valid_moves.size(), 1.0f / valid_moves.size());

			if (include_pass)
			{
				probs.push_back(pass_prior_value);

				// Renormalize
				float sum = 0.0f;
				for (float p : probs) sum += p;
				if (sum > 0.0f)
				{
					for (float& p : probs) p /= sum;
				}
			}

			return probs;
		}

		try
		{
			// Get policy logits from neural network
			auto logits = inferencer->policy_inference(
				prefix_ids,
				tree_structure.evaluated_ids,
				tree_structure.evaluated_mask,
				1,  // batch_size
				prefix_len,
				eval_len
			);

			int vocab_size = static_cast<int>(logits.size()) / (eval_len + 1);

			// Score each move by accumulating log probabilities along path
			const float MIN_PROB = 1e-10f;
			std::vector<float> log_scores;

			for (size_t i = 0; i < candidate_sequences.size(); i++)
			{
				int leaf_pos = tree_structure.move_to_leaf[i];
				float log_prob = 0.0f;

				if (leaf_pos == -1)
				{
					// Single-token move: direct prediction from prefix
					const std::string& notation = move_notations[i];
					auto notation_tokens = tokenizer.encode(notation, 2048, false, false, false, false);
					if (notation_tokens.empty())
					{
						log_prob = std::log(MIN_PROB);
					}
					else
					{
						int64_t token = notation_tokens[0];
						auto probs = softmax_at_position(logits, 0, vocab_size);
						float prob = std::max(probs[token], MIN_PROB);
						log_prob = std::log(prob);
					}
				}
				else
				{
					// Multi-token move: accumulate along path
					std::vector<int> path_reverse;
					int pos = leaf_pos;
					while (pos != -1)
					{
						path_reverse.push_back(pos);
						pos = tree_structure.parent[pos];
					}
					std::reverse(path_reverse.begin(), path_reverse.end());

					// 1. Root token
					if (!path_reverse.empty())
					{
						int root_pos = path_reverse[0];
						int64_t root_token = tree_structure.evaluated_ids[root_pos];
						auto probs = softmax_at_position(logits, 0, vocab_size);
						float prob = std::max(probs[root_token], MIN_PROB);
						log_prob += std::log(prob);
					}

					// 2. Intermediate transitions
					for (size_t j = 1; j < path_reverse.size(); j++)
					{
						int parent_pos = path_reverse[j - 1];
						int child_pos = path_reverse[j];
						int64_t child_token = tree_structure.evaluated_ids[child_pos];

						int logits_index = parent_pos + 1;
						auto probs = softmax_at_position(logits, logits_index, vocab_size);
						float prob = std::max(probs[child_token], MIN_PROB);
						log_prob += std::log(prob);
					}

					// 3. Last token
					if (!path_reverse.empty())
					{
						int leaf = path_reverse.back();
						const std::string& notation = move_notations[i];
						auto notation_tokens = tokenizer.encode(notation, 2048, false, false, false, false);
						if (notation_tokens.empty())
						{
							log_prob += std::log(MIN_PROB);
						}
						else
						{
							int64_t last_token = notation_tokens.back();
							int logits_index = leaf + 1;
							auto probs = softmax_at_position(logits, logits_index, vocab_size);
							float prob = std::max(probs[last_token], MIN_PROB);
							log_prob += std::log(prob);
						}
					}
				}

				log_scores.push_back(log_prob);
			}

			auto probs = exp_normalize(log_scores);

			// Add Pass prior manually if requested
			if (include_pass)
			{
				probs.push_back(pass_prior_value);

				// Renormalize probabilities to sum to 1.0
				float sum = 0.0f;
				for (float p : probs) sum += p;
				if (sum > 0.0f)
				{
					for (float& p : probs) p /= sum;
				}
			}

			return probs;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[MCTS] Policy prior inference error: " << e.what() << "\n";

			std::vector<float> probs;

			if (!valid_moves.empty())
			{
				// Fallback: uniform prior over valid moves
				float base = 1.0f / static_cast<float>(valid_moves.size());
				probs.assign(valid_moves.size(), base);
			}
			else
			{
				// No valid moves: handle pass-only or terminal state
				if (include_pass)
				{
					probs.push_back(1.0f);  // All probability on pass
					return probs;
				}
				else
				{
					// No moves at all: return empty (terminal state)
					std::cerr << "[MCTS] No valid moves and pass not allowed in fallback\n";
					return probs;  // empty
				}
			}

			if (include_pass)
			{
				float pass_prior = std::max(pass_prior_value, 0.0f);
				probs.push_back(pass_prior);

				// Renormalize
				double sum = 0.0;
				for (float p : probs) sum += p;

				if (sum <= 0.0)
				{
					std::cerr << "[MCTS] Fallback prior renormalization failed (non-positive sum)\n";
					float uniform = 1.0f / static_cast<float>(probs.size());
					for (float& p : probs) p = uniform;
					return probs;
				}

				for (float& p : probs) p = static_cast<float>(p / sum);
			}

			return probs;
		}
	}


	/**
	 * Select best child by PUCT score
	 *
	 * White-positive system: Q-values are always White advantage.
	 * We adjust by player perspective during selection:
	 * - White maximizes Q (higher Q = better for White)
	 * - Black maximizes -Q (lower Q = better for Black)
	 */
	MCTSNode* select_best_puct_child(MCTSNode* node, Stone current_player)
	{
		MCTSNode* best = nullptr;
		float best_score = -std::numeric_limits<float>::infinity();

		bool is_white = (current_player == Stone::White);

		for (const auto& child : node->children)
		{
			float q = child->q_value();
			float u = c_puct * child->prior_prob * std::sqrt(node->visit_count) / (1.0f + child->visit_count);

			// White maximizes Q, Black maximizes -Q
			float score = (is_white ? q : -q) + u;

			if (score > best_score)
			{
				best_score = score;
				best = child.get();
			}
		}

		return best;
	}


	/**
	 * Sample from Gamma distribution using Marsaglia and Tsang method (2000)
	 * Used for Dirichlet noise generation
	 *
	 * @param alpha Shape parameter (must be > 0)
	 * @return Sample from Gamma(alpha, 1)
	 */
	float sample_gamma(float alpha)
	{
		if (alpha <= 0.0f)
		{
			return 0.0f;
		}

		// For alpha < 1, use transformation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
		if (alpha < 1.0f)
		{
			std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
			float u = uniform(rng);
			return sample_gamma(alpha + 1.0f) * std::pow(u, 1.0f / alpha);
		}

		// Marsaglia and Tsang's method for alpha >= 1
		float d = alpha - 1.0f / 3.0f;
		float c = 1.0f / std::sqrt(9.0f * d);

		std::normal_distribution<float> normal(0.0f, 1.0f);
		std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

		while (true)
		{
			float x, v;
			do
			{
				x = normal(rng);
				v = 1.0f + c * x;
			}
			while (v <= 0.0f);

			v = v * v * v;
			float u = uniform(rng);

			// Fast acceptance check
			if (u < 1.0f - 0.0331f * x * x * x * x)
			{
				return d * v;
			}

			// Fallback acceptance check
			if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v)))
			{
				return d * v;
			}
		}
	}


	/**
	 * Add Dirichlet noise to root node's children priors
	 *
	 * P(s,a) = (1 - ε) * P(s,a) + ε * η_a
	 * where η ~ Dir(α)
	 *
	 * This encourages exploration at the root during self-play
	 *
	 * NOTE: Pass move is EXCLUDED from Dirichlet noise to preserve minimal prior.
	 *       Without this exclusion, Pass's prior can be boosted from 1e-10 to 5-10%
	 *       by noise, causing incorrect selection on low simulations.
	 */
	void add_dirichlet_noise_to_root()
	{
		if (root->children.empty())
		{
			return;
		}

		// Count non-Pass actions for noise generation
		size_t num_regular_actions = 0;
		for (const auto& child : root->children)
		{
			if (!child->is_pass)
			{
				num_regular_actions++;
			}
		}

		// No regular actions to add noise to
		if (num_regular_actions == 0)
		{
			return;
		}

		// Generate Dirichlet noise ONLY for non-Pass actions
		std::vector<float> noise(num_regular_actions);
		float noise_sum = 0.0f;

		for (size_t i = 0; i < num_regular_actions; i++)
		{
			noise[i] = sample_gamma(dirichlet_alpha);
			noise_sum += noise[i];
		}

		// Handle edge case: if all samples are 0
		if (noise_sum <= 0.0f)
		{
			return;  // Keep original priors
		}

		// Apply noise only to non-Pass children
		size_t noise_idx = 0;
		for (const auto& child : root->children)
		{
			if (child->is_pass)
			{
				// Keep Pass prior unchanged (no noise applied)
				continue;
			}

			float normalized_noise = noise[noise_idx++] / noise_sum;
			float original_prior = child->prior_prob;
			child->prior_prob =
				(1.0f - dirichlet_epsilon) * original_prior +
				dirichlet_epsilon * normalized_noise;
		}
	}


	/**
	 * Select best child by visit count (most robust)
	 */
	PolicyAction select_best_child()
	{
		if (root->children.empty())
		{
			// No children - return pass
			return PolicyAction::Pass();
		}

		MCTSNode* best = nullptr;
		int max_visits = -1;

		for (const auto& child : root->children)
		{
			if (child->visit_count > max_visits)
			{
				max_visits = child->visit_count;
				best = child.get();
			}
		}

		if (best == nullptr)
		{
			return PolicyAction::Pass();
		}

		// Calculate confidence from visit proportion
		float confidence = static_cast<float>(best->visit_count) / num_simulations;

		if (best->is_pass)
		{
			return PolicyAction::Pass(confidence);
		}
		else
		{
			return PolicyAction(best->move, confidence);
		}
	}
};


} // namespace trigo
