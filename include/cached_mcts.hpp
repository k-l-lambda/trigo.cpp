/**
 * CachedMCTS - AlphaZero MCTS with Prefix Cache Optimization
 *
 * Full MCTS implementation using PrefixCacheInferencer for both:
 * - Policy evaluation (expansion with priors)
 * - Value evaluation (leaf assessment)
 *
 * Performance: 3-4× faster than standard MCTS due to cache reuse
 *
 * Key differences from standard MCTS:
 * - Uses PrefixCacheInferencer instead of SharedModelInferencer
 * - Computes prefix cache once at root
 * - Reuses cache across all simulations
 * - Policy priors guide tree expansion
 */

#pragma once

#include "trigo_game.hpp"
#include "trigo_coords.hpp"
#include "prefix_cache_inferencer.hpp"
#include "tgn_tokenizer.hpp"
#include "prefix_tree_builder.hpp"
#include "tgn_utils.hpp"
#include "mcts.hpp"  // For MCTSNode and PolicyAction
#include <iomanip>   // For std::setprecision
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>


namespace trigo
{

/**
 * CachedMCTS Engine (AlphaZero with Prefix Cache)
 *
 * Uses cached neural network inference for optimal performance
 */
class CachedMCTS
{
private:
	std::unique_ptr<MCTSNode> root;
	int num_simulations;
	float c_puct;  // Exploration constant for PUCT formula

	// Dirichlet noise parameters (for root exploration)
	float dirichlet_alpha;    // Alpha parameter (default: 0.03 for Go-like games)
	float dirichlet_epsilon;  // Noise mixing weight (default: 0.25)

	// Cached neural network components
	std::shared_ptr<PrefixCacheInferencer> inferencer;
	TGNTokenizer tokenizer;

	std::mt19937 rng;


public:
	CachedMCTS(
		std::shared_ptr<PrefixCacheInferencer> inf,
		int num_sims = 50,
		float exploration = 1.0f,
		int seed = 42,
		float dir_alpha = 0.03f,
		float dir_epsilon = 0.25f
	)
		: num_simulations(num_sims)
		, c_puct(exploration)
		, dirichlet_alpha(dir_alpha)
		, dirichlet_epsilon(dir_epsilon)
		, inferencer(inf)
		, rng(seed)
	{
	}


	/**
	 * Run AlphaZero MCTS search with cached inference
	 *
	 * @param game Current game state
	 * @return Best move found by MCTS
	 */
	PolicyAction search(const TrigoGame& game)
	{
		// Create root node
		root = std::make_unique<MCTSNode>(Position{0, 0, 0}, false);
		// NOTE: root->visit_count defaults to 0 (matches TypeScript totalN=0 initially)
		// TypeScript: totalN = sum of all child N values, starts at 0
		// PUCT uses sqrt(totalN + 1), so initial U = c * P * sqrt(1) / 1 = c * P

		bool dirichlet_applied = false;  // Track if Dirichlet noise has been added

#ifdef MCTS_ENABLE_PROFILING
		std::cout << "[CachedMCTS] Starting search with " << num_simulations << " simulations\n";
		auto start_time = std::chrono::steady_clock::now();
#endif

		// CRITICAL: Compute prefix cache ONCE for root position
		auto root_tokens = game_to_tokens(game);
		int root_seq_len = static_cast<int>(root_tokens.size());

		try
		{
			inferencer->compute_prefix_cache(root_tokens, 1, root_seq_len);

#ifdef MCTS_ENABLE_PROFILING
			auto cache_time = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - start_time
			).count();
			std::cout << "[CachedMCTS] Root cache computed in " << cache_time << "ms\n";
			std::cout << "[CachedMCTS] Root position: " << root_seq_len << " tokens\n";
#endif
		}
		catch (const std::exception& e)
		{
			std::cerr << "[CachedMCTS] Failed to compute root cache: " << e.what() << "\n";
			// Fallback: return random move
			auto valid_moves = game.valid_move_positions();
			if (valid_moves.empty())
				return PolicyAction::Pass();

			std::uniform_int_distribution<size_t> dist(0, valid_moves.size() - 1);
			return PolicyAction(valid_moves[dist(rng)], 0.0f);
		}

		// Run simulations (all reuse the same root cache)
		for (int i = 0; i < num_simulations; i++)
		{
#ifdef MCTS_ENABLE_PROFILING
			if (i % 10 == 0 && i > 0)
			{
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::steady_clock::now() - start_time
				).count();
				std::cout << "[CachedMCTS] Simulation " << i << "/" << num_simulations
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

			// 3. Evaluation: Check for terminal state first, then use value network
			// TypeScript consistency: Use ground-truth territory value at terminal states
			float value;
			auto terminal_value = checkTerminal(game_copy);
			if (terminal_value.has_value())
			{
				value = terminal_value.value();
#ifdef MCTS_ENABLE_PROFILING
				std::cout << "    [eval] Terminal state detected, ground-truth value=" << std::fixed << std::setprecision(4) << value << "\n";
#endif
			}
			else
			{
				// Non-terminal: Use value network with cache
				value = evaluate_with_cache(game_copy);
			}

#ifdef MCTS_ENABLE_PROFILING
			// Debug: Show what node was evaluated and its value
			if (node->is_pass)
			{
				std::cout << "    [eval] Pass node evaluated, value=" << std::fixed << std::setprecision(4) << value << "\n";
			}
#endif

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
				std::cout << "[CachedMCTS] First simulation breakdown:\n";
				std::cout << "  Selection: " << select_time << "μs\n";
				std::cout << "  Expansion: " << expand_time << "μs\n";
				std::cout << "  Evaluation: " << evaluate_time << "μs (cached value network)\n";
				std::cout << "  Total: " << total_time << "μs\n";
			}
#endif
		}

#ifdef MCTS_ENABLE_PROFILING
		auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start_time
		).count();
		std::cout << "[CachedMCTS] Search complete in " << total_elapsed << "ms\n";
		std::cout << "[CachedMCTS] Average per simulation: "
		          << (static_cast<float>(total_elapsed) / num_simulations) << "ms\n";
#endif

		// Debug: Print visit counts for all children
#ifdef MCTS_ENABLE_PROFILING
		std::cout << "\n[CachedMCTS] Child visit counts after search:\n";
		for (const auto& child : root->children)
		{
			std::string move_name = child->is_pass ? "PASS" : encode_ab0yz(child->move, game.get_shape());
			std::cout << "  " << move_name << ": visits=" << child->visit_count
			          << ", prior=" << std::fixed << std::setprecision(6) << child->prior_prob
			          << ", Q=" << child->q_value() << "\n";
		}
		std::cout << std::endl;
#endif

		// Select best move by visit count (most robust)
		return select_best_child();
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

			// Get current player BEFORE selecting child
			bool is_white = (game.get_current_player() == Stone::White);

			// Otherwise, select best child by PUCT
			node = select_best_puct_child(node, is_white);

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
	 * Expansion phase: Add one new child node with policy prior
	 *
	 * Uses policy network to get priors for all valid moves
	 */
	MCTSNode* expand(MCTSNode* node, TrigoGame& game)
	{
		// AlphaZero-style: Expand ALL children at once on first visit
		// This avoids the prior renormalization issue where the last move gets prior=1.0

		// If already expanded, select a child to traverse
		if (!node->children.empty())
		{
			// Node already has children, select one for traversal
			// Find an unexpanded child (visit_count == 0)
			for (auto& child : node->children)
			{
				if (child->visit_count == 0)
				{
					// Apply this move to game
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

			// All children visited at least once, mark as fully expanded
			node->is_fully_expanded = true;
			return node;
		}

		// First visit to this node - expand ALL children at once
		auto valid_moves = game.valid_move_positions();
		bool can_pass = game.is_game_active();

		if (valid_moves.empty() && !can_pass)
		{
			// No moves available
			node->is_fully_expanded = true;
			return node;
		}

		// CRITICAL: Recompute prefix cache for current position before getting priors
		// The cache may have been invalidated by evaluate_with_cache() in a previous simulation
		auto current_tokens = game_to_tokens(game);
		inferencer->compute_prefix_cache(current_tokens, 1, static_cast<int>(current_tokens.size()));

		// Get policy priors for ALL valid moves at once
		std::vector<float> priors = get_move_priors(game, valid_moves, can_pass);

#ifdef MCTS_ENABLE_PROFILING
		// Debug: Print Pass prior if included
		if (can_pass && !priors.empty())
		{
			float pass_prior = priors.back();
			std::cout << "    [expand] Pass prior = " << std::fixed << std::setprecision(6) << pass_prior
			          << " (total moves = " << priors.size() << ")" << std::endl;
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

		// Select the first child to traverse: use highest prior (deterministic)
		// TypeScript consistency: After expansion, select() uses PUCT which picks highest P when all N=0
		// This is equivalent to picking the child with highest prior deterministically
		size_t best_idx = 0;
		float best_prior = node->children[0]->prior_prob;
		for (size_t i = 1; i < node->children.size(); i++)
		{
			if (node->children[i]->prior_prob > best_prior)
			{
				best_prior = node->children[i]->prior_prob;
				best_idx = i;
			}
		}

		MCTSNode* selected_child = node->children[best_idx].get();

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
	 * Evaluation phase: Use value network with proper cache
	 *
	 * IMPORTANT: We must recompute cache for the current position!
	 * Using root cache gives wrong values for diverged positions.
	 *
	 * Uses prefix cache value inference for optimal performance.
	 * Verified: ONNX cached value matches Python baseline (diff: 0.000007).
	 *
	 * @return Value from perspective of current player in game
	 */
	float evaluate_with_cache(TrigoGame& game)
	{
		try
		{
			// Generate TGN tokens for this position
			auto tokens = game_to_tokens(game);
			int seq_len = static_cast<int>(tokens.size());

			// Compute prefix cache for this position
			inferencer->compute_prefix_cache(tokens, 1, seq_len);

			// Get value using cached inference (VALUE token ID = 3)
			float value = inferencer->value_inference_with_cache(3);

			// Value model outputs White advantage (positive = White winning)
			// Return white-positive value (no conversion to current player perspective)
			// The backup phase does NOT flip signs, and PUCT selection handles player perspective
			return value;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[CachedMCTS] Value inference error: " << e.what() << "\n";
			return 0.0f;  // Return neutral value on error
		}
	}


	/**
	 * Backpropagation phase: Update statistics up the tree
	 *
	 * Uses white-positive value system (no sign flipping)
	 * All Q values are white-positive throughout the tree
	 */
	void backpropagate(MCTSNode* node, float value)
	{
		while (node != nullptr)
		{
			node->visit_count++;
			node->total_value += value;

			// NO sign flipping - Q values are always white-positive
			// PUCT selection will handle player perspective

			node = node->parent;
		}
	}


	/**
	 * Select child with best PUCT score
	 *
	 * Uses white-positive Q values:
	 * - White maximizes Q (PUCT = Q + U)
	 * - Black minimizes Q (PUCT = -Q + U)
	 *
	 * @param node Current node
	 * @param is_white Whether current player is White
	 */
	MCTSNode* select_best_puct_child(MCTSNode* node, bool is_white)
	{
		MCTSNode* best = nullptr;
		float best_score = -std::numeric_limits<float>::infinity();

		for (const auto& child : node->children)
		{
			// Calculate PUCT score with player perspective
			float q = child->q_value();
			// PUCT formula: U(s,a) = c_puct * P(s,a) * sqrt(N(s) + 1) / (1 + N(s,a))
			// The +1 in sqrt ensures exploration term is non-zero when node is first expanded
			float u = c_puct * child->prior_prob * std::sqrt(node->visit_count + 1) / (1.0f + child->visit_count);

			// Black minimizes Q (flips sign), White maximizes Q
			float score = (is_white ? q : -q) + u;

			// NOTE: Removed -1000 penalty for zero-prior moves (December 10, 2025)
			// TypeScript mctsAgent.ts does NOT have this penalty.
			// Allowing Q to drive selection even for low-prior moves is consistent with
			// AlphaZero behavior where value network can override policy network.

			if (score > best_score)
			{
				best_score = score;
				best = child.get();
			}
		}

		return best;
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


	/**
	 * Convert game history to token sequence
	 *
	 * Matches TypeScript TrigoTreeAgent.buildMoveTree behavior:
	 * - Generates TGN from game history
	 * - Appends move number prefix when it's a player's turn
	 * - Black's turn: add "N. " (move number + space)
	 * - White's turn: add " " (just space after Black's move)
	 */
	std::vector<int64_t> game_to_tokens(const TrigoGame& game)
	{
		// Generate TGN text
		std::string tgn_text = game_to_tgn(game, false);

		// Add move number prefix for the next move (TypeScript compatibility)
		// This matches TrigoTreeAgent.buildMoveTree lines 193-209
		const auto& history = game.get_history();
		int move_number = static_cast<int>(history.size()) / 2 + 1;

		if (game.get_current_player() == Stone::Black)
		{
			// Black's turn: add move number
			tgn_text += std::to_string(move_number) + ". ";
		}
		else
		{
			// White's turn: add space after Black's move
			tgn_text += " ";
		}

		// Tokenize
		auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

		// Add START token
		std::vector<int64_t> tokens;
		tokens.push_back(1);  // START token
		tokens.insert(tokens.end(), encoded.begin(), encoded.end());

		return tokens;
	}


	/**
	 * Get policy priors for unexpanded moves using cached inference
	 *
	 * Implements the same scoring algorithm as TypeScript TrigoTreeAgent.scoreMoves:
	 * - Builds complete path from root to leaf using parent array
	 * - Accumulates log probabilities along the path
	 * - Converts to probabilities via exp and normalization
	 *
	 * @param game Current game state
	 * @param unexpanded_moves List of unexpanded move positions
	 * @param include_pass Whether to include pass in the evaluation
	 * @return Prior probabilities for each move
	 */
	std::vector<float> get_move_priors(
		const TrigoGame& game,
		const std::vector<Position>& unexpanded_moves,
		bool include_pass
	)
	{
		// Build token sequences for each candidate move
		auto board_shape = game.get_shape();
		std::vector<std::vector<int64_t>> candidate_sequences;
		std::vector<std::string> move_notations;  // Store original notations

		// Add regular moves (ONLY move tokens, not prefix)
		// CRITICAL: Exclude last token (following TypeScript trigoTreeAgent.ts:226-227)
		for (const auto& move : unexpanded_moves)
		{
			std::string coord = encode_ab0yz(move, board_shape);
			auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

			move_notations.push_back(coord);

			// Exclude the last token
			if (move_tokens.size() > 1)
			{
				std::vector<int64_t> seq(move_tokens.begin(), move_tokens.end() - 1);
				candidate_sequences.push_back(seq);
			}
			else
			{
				// Single token or empty - add empty sequence
				// These will be handled specially in scoring (direct prediction from prefix)
				candidate_sequences.push_back(std::vector<int64_t>());
			}
		}

		// Add pass if needed
		// Note: TGN format uses "Pass" (capital P, lowercase ass)
		if (include_pass)
		{
			auto pass_tokens = tokenizer.encode("Pass", 2048, false, false, false, false);
			move_notations.push_back("Pass");

			// Always push a candidate sequence for pass to keep arrays aligned
			if (pass_tokens.size() > 1)
			{
				std::vector<int64_t> seq(pass_tokens.begin(), pass_tokens.end() - 1);
				candidate_sequences.push_back(seq);
			}
			else
			{
				// Single-token or empty: treat as single-token move (leaf_pos == -1)
				candidate_sequences.push_back(std::vector<int64_t>());
			}
		}

		// Build prefix tree
		PrefixTreeBuilder tree_builder;
		auto tree_structure = tree_builder.build_tree(candidate_sequences);

		try
		{
			// Get hidden states using cached evaluation
			auto hidden_states = inferencer->evaluate_with_cache(
				tree_structure.evaluated_ids,
				tree_structure.evaluated_mask,
				1,  // batch_size
				tree_structure.num_nodes
			);

			// Run policy head to get logits
			int hidden_dim = static_cast<int>(hidden_states.size()) / tree_structure.num_nodes;
			auto logits = inferencer->policy_inference_from_hidden(
				hidden_states,
				1,  // batch_size
				tree_structure.num_nodes,
				hidden_dim
			);

			int vocab_size = static_cast<int>(logits.size()) / tree_structure.num_nodes;

			// Score each move by accumulating log probabilities along path
			// (Following TypeScript TrigoTreeAgent.scoreMoves:336-445)
			const float MIN_PROB = 1e-10f;  // Minimum probability to avoid log(0)
			std::vector<float> log_scores;

			for (size_t i = 0; i < candidate_sequences.size(); i++)
			{
				int leaf_pos = tree_structure.move_to_leaf[i];
				float log_prob = 0.0f;

				// Handle moves with empty sequences (single-token moves)
				if (leaf_pos == -1)
				{
					// Direct prediction from prefix (position 0)
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
					// Build path from leaf to root, then reverse
					std::vector<int> path_reverse;
					int pos = leaf_pos;
					while (pos != -1)
					{
						path_reverse.push_back(pos);
						pos = tree_structure.parent[pos];
					}
					std::reverse(path_reverse.begin(), path_reverse.end());

					// Accumulate log probabilities along path
					// 1. Root token (predicted from prefix last position, logits[0])
					if (!path_reverse.empty())
					{
						int root_pos = path_reverse[0];
						int64_t root_token = tree_structure.evaluated_ids[root_pos];
						auto probs = softmax_at_position(logits, 0, vocab_size);
						float prob = std::max(probs[root_token], MIN_PROB);
						log_prob += std::log(prob);
					}

					// 2. Intermediate transitions (parent→child)
					for (size_t j = 1; j < path_reverse.size(); j++)
					{
						int parent_pos = path_reverse[j - 1];
						int child_pos = path_reverse[j];
						int64_t child_token = tree_structure.evaluated_ids[child_pos];

						// Parent output is at logits[parent_pos + 1]
						int logits_index = parent_pos + 1;
						auto probs = softmax_at_position(logits, logits_index, vocab_size);
						float prob = std::max(probs[child_token], MIN_PROB);
						log_prob += std::log(prob);
					}

					// 3. Last token (predicted from leaf, excluded from tree)
					if (!path_reverse.empty())
					{
						int leaf = path_reverse.back();
						const std::string& notation = move_notations[i];

						// Tokenize the full notation to get the actual last token ID
						auto notation_tokens = tokenizer.encode(notation, 2048, false, false, false, false);
						if (notation_tokens.empty())
						{
							log_prob += std::log(MIN_PROB);
						}
						else
						{
							int64_t last_token = notation_tokens.back();

							// Leaf output is at logits[leaf + 1]
							int logits_index = leaf + 1;
							auto probs = softmax_at_position(logits, logits_index, vocab_size);
							float prob = std::max(probs[last_token], MIN_PROB);
							log_prob += std::log(prob);
						}
					}
				}

				log_scores.push_back(log_prob);
			}

			// Convert log scores to probabilities via exp and normalization
			std::vector<float> probs = exp_normalize(log_scores);

#ifdef MCTS_ENABLE_PROFILING
			// Print top 5 moves with log scores and priors
			std::vector<std::pair<size_t, float>> indexed_scores;
			for (size_t i = 0; i < log_scores.size(); i++) {
				indexed_scores.push_back({i, log_scores[i]});
			}
			std::sort(indexed_scores.begin(), indexed_scores.end(),
					  [](const auto& a, const auto& b) { return a.second > b.second; });

			std::cout << "  Top 5 moves by log score:" << std::endl;
			for (size_t i = 0; i < std::min(size_t(5), indexed_scores.size()); i++) {
				size_t idx = indexed_scores[i].first;
				std::string move_str;
				if (idx < unexpanded_moves.size()) {
					move_str = encode_ab0yz(unexpanded_moves[idx], board_shape);
				} else {
					move_str = "Pass";
				}
				std::cout << "    " << (i+1) << ". " << move_str
						  << " log_score=" << std::fixed << std::setprecision(6) << log_scores[idx]
						  << " prior=" << probs[idx] << std::endl;
			}
#endif

			return probs;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[CachedMCTS] Policy prior inference error: " << e.what() << "\n";
			// Fallback to uniform priors
			size_t num_moves = unexpanded_moves.size() + (include_pass ? 1 : 0);
			return std::vector<float>(num_moves, 1.0f / num_moves);
		}
	}


	/**
	 * Apply softmax to logits
	 */
	std::vector<float> softmax(const std::vector<float>& logits)
	{
		if (logits.empty())
			return {};

		// Find max for numerical stability
		float max_logit = *std::max_element(logits.begin(), logits.end());

		// Compute exp and sum
		std::vector<float> exp_vals(logits.size());
		float sum = 0.0f;
		for (size_t i = 0; i < logits.size(); i++)
		{
			exp_vals[i] = std::exp(logits[i] - max_logit);
			sum += exp_vals[i];
		}

		// Normalize
		std::vector<float> probs(logits.size());
		for (size_t i = 0; i < logits.size(); i++)
		{
			probs[i] = exp_vals[i] / sum;
		}

		return probs;
	}


	/**
	 * Compute softmax for a single position in logits array
	 *
	 * @param logits Full logits array [num_positions * vocab_size]
	 * @param position Position index (0 to num_positions-1)
	 * @param vocab_size Vocabulary size
	 * @return Softmax probabilities for all tokens at this position [vocab_size]
	 */
	std::vector<float> softmax_at_position(
		const std::vector<float>& logits,
		int position,
		int vocab_size
	)
	{
		// Extract logits for this position
		int offset = position * vocab_size;
		std::vector<float> position_logits(vocab_size);
		for (int i = 0; i < vocab_size; i++)
		{
			position_logits[i] = logits[offset + i];
		}

		// Apply softmax
		return softmax(position_logits);
	}


	/**
	 * Convert log scores to normalized probabilities
	 *
	 * exp(log_scores) / sum(exp(log_scores))
	 * Uses log-sum-exp trick for numerical stability
	 */
	std::vector<float> exp_normalize(const std::vector<float>& log_scores)
	{
		if (log_scores.empty())
			return {};

		// Find max for numerical stability (log-sum-exp trick)
		// Filter out -inf and nan values
		float max_score = -std::numeric_limits<float>::infinity();
		for (float score : log_scores)
		{
			if (std::isfinite(score) && score > max_score)
			{
				max_score = score;
			}
		}

		// If all scores are -inf or nan, return uniform distribution
		if (!std::isfinite(max_score))
		{
			size_t n = log_scores.size();
			return std::vector<float>(n, 1.0f / n);
		}

		// Compute exp and sum
		std::vector<float> exp_vals(log_scores.size());
		float sum = 0.0f;
		for (size_t i = 0; i < log_scores.size(); i++)
		{
			// Handle -inf and nan: treat as 0 probability
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

		// Safety check: if sum is 0 or nan, return uniform
		if (sum <= 0.0f || !std::isfinite(sum))
		{
			size_t n = log_scores.size();
			return std::vector<float>(n, 1.0f / n);
		}

		// Normalize
		std::vector<float> probs(log_scores.size());
		for (size_t i = 0; i < log_scores.size(); i++)
		{
			probs[i] = exp_vals[i] / sum;
		}

		return probs;
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

			// Check acceptance
			if (u < 1.0f - 0.0331f * x * x * x * x)
			{
				return d * v;
			}

			if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v)))
			{
				return d * v;
			}
		}
	}


	/**
	 * Add Dirichlet noise to root node priors (AlphaZero style exploration)
	 *
	 * Formula:
	 * P(s,a) = (1 - ε) * P(s,a) + ε * η_a
	 * where η ~ Dir(α)
	 *
	 * This encourages exploration at the root during self-play
	 */
	void add_dirichlet_noise_to_root()
	{
		if (root->children.empty())
		{
			return;
		}

		size_t num_actions = root->children.size();

		// Generate Dirichlet noise by sampling from Gamma distribution
		std::vector<float> noise(num_actions);
		float noise_sum = 0.0f;

		for (size_t i = 0; i < num_actions; i++)
		{
			noise[i] = sample_gamma(dirichlet_alpha);
			noise_sum += noise[i];
		}

		// Handle edge case: if all samples are 0
		if (noise_sum <= 0.0f)
		{
			return;  // Keep original priors
		}

		// Normalize and mix with original priors
		for (size_t i = 0; i < num_actions; i++)
		{
			float normalized_noise = noise[i] / noise_sum;
			float original_prior = root->children[i]->prior_prob;
			root->children[i]->prior_prob =
				(1.0f - dirichlet_epsilon) * original_prior +
				dirichlet_epsilon * normalized_noise;
		}
	}


	/**
	 * Calculate terminal value from territory scores
	 *
	 * Uses logarithmic scaling matching the training code.
	 * Formula: sign(scoreDiff) * (1 + log(|scoreDiff|))
	 *
	 * @param territory Territory counts from game
	 * @return Value (white-positive: positive = white winning)
	 */
	float calculateTerminalValue(const TerritoryResult& territory)
	{
		float scoreDiff = static_cast<float>(territory.white - territory.black);

		if (std::abs(scoreDiff) < 1e-6f)
		{
			// Draw/tie case
			return 0.0f;
		}

		// Match training formula from valueCausalLoss.py:_expand_value_targets
		// target = sign(score) * (1 + log(|score|)) * territory_value_factor
		// The log term incentivizes winning by larger margins (logarithmically)
		const float territory_value_factor = 1.0f;  // Default from training config
		float signScore = (scoreDiff > 0.0f) ? 1.0f : -1.0f;
		return signScore * (1.0f + std::log(std::abs(scoreDiff))) * territory_value_factor;
	}


	/**
	 * Check if game state is terminal and return ground-truth value if so
	 *
	 * Matches TypeScript mctsAgent.checkTerminal() behavior:
	 * 1. Check if game is already finished (double-pass, resignation)
	 * 2. Check for "natural" game end (coverage > 50% && neutral == 0)
	 *
	 * @param game Game state to check
	 * @return Terminal value (white-positive) if terminal, nullopt otherwise
	 */
	std::optional<float> checkTerminal(TrigoGame& game)
	{
		// 1. Check if game is already finished (double-pass, resignation, etc.)
		if (game.get_game_status() == GameStatus::FINISHED)
		{
			auto territory = game.get_territory();
			return calculateTerminalValue(territory);
		}

		// 2. Check for "natural" game end (all territory claimed)
		const auto& board = game.get_board();
		const auto& shape = game.get_shape();
		int totalPositions = shape.x * shape.y * shape.z;

		// Count stones (cheap)
		int stoneCount = 0;
		bool hasBlack = false;
		bool hasWhite = false;

		for (int x = 0; x < shape.x; x++)
		{
			for (int y = 0; y < shape.y; y++)
			{
				for (int z = 0; z < shape.z; z++)
				{
					Stone stone = board[x][y][z];
					if (stone == Stone::Black)
					{
						hasBlack = true;
						stoneCount++;
					}
					else if (stone == Stone::White)
					{
						hasWhite = true;
						stoneCount++;
					}
				}
			}
		}

		float coverageRatio = static_cast<float>(stoneCount) / totalPositions;

		// Only check territory if board is reasonably full
		// (optimization: neutral == 0 is unlikely with sparse board)
		if (hasBlack && hasWhite && coverageRatio > 0.5f)
		{
			auto territory = game.get_territory();
			if (territory.neutral == 0)
			{
				return calculateTerminalValue(territory);
			}
		}

		return std::nullopt;  // Not terminal
	}
};


} // namespace trigo
