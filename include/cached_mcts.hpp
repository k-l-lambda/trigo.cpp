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

	// Cached neural network components
	std::shared_ptr<PrefixCacheInferencer> inferencer;
	TGNTokenizer tokenizer;

	std::mt19937 rng;


public:
	CachedMCTS(
		std::shared_ptr<PrefixCacheInferencer> inf,
		int num_sims = 50,
		float exploration = 1.0f,
		int seed = 42
	)
		: num_simulations(num_sims)
		, c_puct(exploration)
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
		root->visit_count = 1;  // Root is always visited

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
			}

#ifdef MCTS_ENABLE_PROFILING
			auto expand_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count() - select_time;
#endif

			// 3. Evaluation: Use value network with cache
			float value = evaluate_with_cache(game_copy);

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

		// Select the first child (highest prior due to sampling) for traversal
		// Use prior-weighted sampling to select which child to visit first
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
	 * Evaluation phase: Use value network with proper cache
	 *
	 * IMPORTANT: We must recompute cache for the current position!
	 * Using root cache gives wrong values for diverged positions.
	 *
	 * Uses direct evaluation model when available (more accurate),
	 * falls back to cached value inference otherwise.
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

			float value;

			// Prefer direct evaluation model (more accurate for value inference)
			if (inferencer->has_evaluation_model())
			{
				// Build full input sequence: TGN + END token
				// The evaluation model expects: START + TGN + END (padded to fixed length)
				const int64_t END_TOKEN = 2;
				const int64_t PAD_TOKEN = 0;
				const int SEQ_LEN = 256;  // Fixed sequence length for evaluation model

				std::vector<int64_t> eval_tokens = tokens;  // Already has START token
				eval_tokens.push_back(END_TOKEN);

				// Pad to fixed length
				while (eval_tokens.size() < SEQ_LEN)
				{
					eval_tokens.push_back(PAD_TOKEN);
				}

				// Truncate if too long
				if (eval_tokens.size() > SEQ_LEN)
				{
					eval_tokens.resize(SEQ_LEN);
				}

				value = inferencer->value_inference_direct(eval_tokens, 1, SEQ_LEN);
			}
			else
			{
				// Fallback to cached value inference
				inferencer->compute_prefix_cache(tokens, 1, seq_len);
				value = inferencer->value_inference_with_cache(3);  // VALUE token
			}

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
			float u = c_puct * child->prior_prob * std::sqrt(node->visit_count) / (1.0f + child->visit_count);

			// Black minimizes Q (flips sign), White maximizes Q
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
	 */
	std::vector<int64_t> game_to_tokens(const TrigoGame& game)
	{
		// Generate TGN text
		std::string tgn_text = game_to_tgn(game, false);

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

			if (!pass_tokens.empty())
			{
				std::vector<int64_t> seq(pass_tokens.begin(), pass_tokens.end() - 1);
				candidate_sequences.push_back(seq);
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
};


} // namespace trigo
