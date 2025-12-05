/**
 * Monte Carlo Tree Search (MCTS) Implementation
 *
 * Pure MCTS without neural network guidance
 * Uses UCB1 formula for tree exploration
 */

#pragma once

#include "trigo_game.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <iostream>


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
	 */
	float puct_score(float c_puct) const
	{
		float q = q_value();
		float u = c_puct * prior_prob * std::sqrt(parent->visit_count) / (1.0f + visit_count);

		return q + u;
	}
};


/**
 * MCTS Engine
 *
 * Performs Monte Carlo Tree Search on a game state
 */
class MCTS
{
private:
	std::unique_ptr<MCTSNode> root;
	int num_simulations;
	float exploration_constant;
	std::mt19937 rng;


public:
	MCTS(int num_sims = 800, float c = 1.414f, int seed = 42)
		: num_simulations(num_sims)
		, exploration_constant(c)
		, rng(seed)
	{
	}


	/**
	 * Run MCTS search and return best move
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
		std::cout << "[MCTS] Starting search with " << num_simulations << " simulations\n";
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
				std::cout << "[MCTS] Simulation " << i << "/" << num_simulations
				          << " (" << elapsed << "ms)\n";
			}

			auto sim_start = std::chrono::steady_clock::now();
#endif

			// Make a copy of game state for simulation
			TrigoGame game_copy = game;

			// 1. Selection: Traverse tree using UCB1
			MCTSNode* node = select(root.get(), game_copy);

#ifdef MCTS_ENABLE_PROFILING
			auto select_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count();
#endif

			// 2. Expansion: Add new child node
			if (game_copy.is_game_active() && node->visit_count > 0)
			{
				node = expand(node, game_copy);
			}

#ifdef MCTS_ENABLE_PROFILING
			auto expand_time = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - sim_start
			).count() - select_time;
#endif

			// 3. Simulation: Play random game to completion
			float value = simulate(game_copy);

#ifdef MCTS_ENABLE_PROFILING
			auto simulate_time = std::chrono::duration_cast<std::chrono::microseconds>(
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
				std::cout << "[MCTS] First simulation breakdown:\n";
				std::cout << "  Selection: " << select_time << "μs\n";
				std::cout << "  Expansion: " << expand_time << "μs\n";
				std::cout << "  Simulation: " << simulate_time << "μs\n";
				std::cout << "  Total: " << total_time << "μs\n";
			}
#endif
		}

#ifdef MCTS_ENABLE_PROFILING
		auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start_time
		).count();
		std::cout << "[MCTS] Search complete in " << total_elapsed << "ms\n";
#endif

		// Select best move by visit count (most robust)
		return select_best_child();
	}


private:
	/**
	 * Selection phase: Traverse tree using UCB1 until leaf node
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

			// Otherwise, select best child by UCB1
			node = select_best_ucb1_child(node);

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
	 * Expansion phase: Add one new child node
	 */
	MCTSNode* expand(MCTSNode* node, TrigoGame& game)
	{
		// Get all valid moves
		auto valid_moves = game.valid_move_positions();

		// Pass is always available in Trigo when game is active
		bool can_pass = game.is_game_active();

		// Find unexpanded moves
		std::vector<Position> unexpanded_moves;
		for (const auto& move : valid_moves)
		{
			bool already_expanded = false;
			for (const auto& child : node->children)
			{
				if (!child->is_pass && child->move == move)
				{
					already_expanded = true;
					break;
				}
			}
			if (!already_expanded)
			{
				unexpanded_moves.push_back(move);
			}
		}

		// Check if pass is unexpanded
		bool pass_unexpanded = can_pass;
		for (const auto& child : node->children)
		{
			if (child->is_pass)
			{
				pass_unexpanded = false;
				break;
			}
		}

		// If all moves expanded, mark as fully expanded
		if (unexpanded_moves.empty() && !pass_unexpanded)
		{
			node->is_fully_expanded = true;
			return node;
		}

		// Select random unexpanded move
		MCTSNode* new_child = nullptr;

		if (pass_unexpanded && unexpanded_moves.empty())
		{
			// Only pass available
			new_child = new MCTSNode(Position{0, 0, 0}, true, node);
			node->children.push_back(std::unique_ptr<MCTSNode>(new_child));
			game.pass();
		}
		else if (!unexpanded_moves.empty())
		{
			// Select random move
			std::uniform_int_distribution<size_t> dist(0, unexpanded_moves.size() - 1);
			size_t idx = dist(rng);
			Position move = unexpanded_moves[idx];

			new_child = new MCTSNode(move, false, node);
			node->children.push_back(std::unique_ptr<MCTSNode>(new_child));
			game.drop(move);
		}

		// Check if all moves now expanded
		if (unexpanded_moves.size() == 1 && !pass_unexpanded)
		{
			node->is_fully_expanded = true;
		}

		return new_child;
	}


	/**
	 * Simulation phase: Play random game to completion
	 *
	 * @return Value from perspective of player at root
	 */
	float simulate(TrigoGame& game)
	{
		Stone root_player = game.get_current_player();

		// Play random moves until game ends
		int max_moves = 500;  // Prevent infinite loops
		int moves = 0;

		while (game.is_game_active() && moves < max_moves)
		{
			auto valid_moves = game.valid_move_positions();

			if (valid_moves.empty())
			{
				game.pass();
			}
			else
			{
				// Random move
				std::uniform_int_distribution<size_t> dist(0, valid_moves.size() - 1);
				size_t idx = dist(rng);
				game.drop(valid_moves[idx]);
			}

			moves++;
		}

		// Get game result
		auto result_opt = game.get_game_result();

		// If no result yet (max moves reached), return draw
		if (!result_opt.has_value())
		{
			return 0.0f;
		}

		auto result = result_opt.value();

		// Convert to value from root player's perspective
		if (result.winner == GameResult::Winner::Draw)
		{
			return 0.0f;
		}
		else if (result.winner == GameResult::Winner::Black)
		{
			return root_player == Stone::Black ? 1.0f : -1.0f;
		}
		else  // White wins
		{
			return root_player == Stone::White ? 1.0f : -1.0f;
		}
	}


	/**
	 * Backpropagation phase: Update statistics up the tree
	 */
	void backpropagate(MCTSNode* node, float value)
	{
		while (node != nullptr)
		{
			node->visit_count++;
			node->total_value += value;

			// Flip value for opponent
			value = -value;

			node = node->parent;
		}
	}


	/**
	 * Select child with best UCB1 score
	 */
	MCTSNode* select_best_ucb1_child(MCTSNode* node)
	{
		MCTSNode* best = nullptr;
		float best_score = -std::numeric_limits<float>::infinity();

		for (const auto& child : node->children)
		{
			float score = child->ucb1_score(exploration_constant);

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
};


} // namespace trigo
