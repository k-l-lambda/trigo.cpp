/**
 * Policy Interface for Trigo Self-Play
 *
 * Abstract interface for different playing strategies:
 * - Random policy (baseline)
 * - MCTS policy (tree search)
 * - Neural policy (ONNX model inference)
 * - Hybrid policy (MCTS + Neural)
 *
 * Design supports both offline and online training:
 * - Offline: Load fixed ONNX model
 * - Online: Call Python NN via callback/IPC
 */

#pragma once

#include "trigo_game.hpp"
#include "shared_model_inferencer.hpp"
#include "prefix_tree_builder.hpp"
#include "tgn_tokenizer.hpp"
#include "mcts.hpp"
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>


namespace trigo
{


/**
 * Abstract Policy Interface
 *
 * All policies must implement this interface
 * Allows pluggable strategies for self-play
 */
class IPolicy
{
public:
	virtual ~IPolicy() = default;

	/**
	 * Select action for given game state
	 *
	 * @param game Current game state
	 * @return Selected action (move or pass)
	 */
	virtual PolicyAction select_action(const TrigoGame& game) = 0;

	/**
	 * Optional: Update policy after game ends
	 * Used for learning algorithms (MCTS, RL)
	 *
	 * @param game_result Final game result
	 */
	virtual void update_from_result(const GameResult& game_result) {}

	/**
	 * Get policy name for logging
	 */
	virtual std::string name() const = 0;
};


/**
 * Random Policy (Baseline)
 *
 * Selects random valid moves uniformly
 * Useful for:
 * - Baseline comparison
 * - Data diversity
 * - Testing
 */
class RandomPolicy : public IPolicy
{
private:
	std::mt19937 rng;
	float pass_probability;  // Probability of passing when moves available

public:
	RandomPolicy(unsigned int seed = std::random_device{}(), float pass_prob = 0.05f)
		: rng(seed), pass_probability(pass_prob)
	{
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		// Get valid positions
		auto valid_positions = game.valid_move_positions();

		if (valid_positions.empty())
		{
			// Must pass
			return PolicyAction::Pass();
		}

		// Randomly decide to pass (add exploration)
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		if (dist(rng) < pass_probability)
		{
			return PolicyAction::Pass();
		}

		// Select random valid position
		std::uniform_int_distribution<size_t> index_dist(0, valid_positions.size() - 1);
		size_t index = index_dist(rng);

		return PolicyAction(valid_positions[index], 1.0f / valid_positions.size());
	}

	std::string name() const override
	{
		return "Random";
	}
};


/**
 * Neural Policy (ONNX Model)
 *
 * Uses trained transformer model for policy + value estimation
 * Model architecture:
 * - Policy (tree mode): Takes prefix tree, outputs move logits
 * - Value (eval mode): Takes position, outputs win probability
 */
class NeuralPolicy : public IPolicy
{
private:
	std::string model_path;
	std::mt19937 rng;
	float temperature;
	std::unique_ptr<SharedModelInferencer> inferencer;
	TGNTokenizer tokenizer;
	PrefixTreeBuilder tree_builder;

public:
	NeuralPolicy(const std::string& model_path, float temp = 1.0f, int seed = 42)
		: model_path(model_path)
		, rng(seed)
		, temperature(temp)
	{
		// Load ONNX models
		inferencer = std::make_unique<SharedModelInferencer>(
			model_path + "/base_model.onnx",
			model_path + "/policy_head.onnx",
			model_path + "/value_head.onnx"
		);
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		// Get valid moves
		auto valid_moves = game.valid_move_positions();

		if (valid_moves.empty())
		{
			// No valid moves, must pass
			return PolicyAction::Pass();
		}

		// TODO: Full neural policy implementation requires:
		// 1. Understanding token-to-move mapping for the vocabulary
		// 2. Handling the logits output shape [batch, seq_len, vocab_size]
		// 3. Extracting probabilities for each candidate move
		//
		// For now, use a weighted random selection as placeholder
		// This allows testing the infrastructure while we figure out the exact mapping

		std::uniform_int_distribution<size_t> dist(0, valid_moves.size() - 1);
		size_t selected_idx = dist(rng);

		return PolicyAction(valid_moves[selected_idx], 1.0f / valid_moves.size());
	}

	std::string name() const override
	{
		return "Neural";
	}


	// TODO: Implement these helper methods when neural inference is fully implemented
	/*
	private:
		std::vector<int64_t> game_to_tokens(const TrigoGame& game);
		std::vector<float> softmax_with_temperature(const std::vector<float>& logits, float temp);
	*/
};


/**
 * MCTS Policy (Monte Carlo Tree Search)
 *
 * Pure MCTS without neural guidance
 * Uses UCB1 formula for tree exploration
 */
class MCTSPolicy : public IPolicy
{
private:
	std::unique_ptr<MCTS> mcts_engine;
	int num_simulations;
	float exploration_constant;

public:
	MCTSPolicy(int num_sims = 800, float c_puct = 1.414f, int seed = 42)
		: num_simulations(num_sims)
		, exploration_constant(c_puct)
	{
		mcts_engine = std::make_unique<MCTS>(num_sims, c_puct, seed);
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		return mcts_engine->search(game);
	}

	std::string name() const override
	{
		return "MCTS";
	}
};


/**
 * Hybrid Policy (MCTS + Neural)
 *
 * AlphaZero-style policy:
 * - Use NN for prior probabilities
 * - Use NN for value estimation
 * - Use MCTS for action selection
 *
 * TODO: Implement hybrid algorithm
 */
class HybridPolicy : public IPolicy
{
private:
	std::unique_ptr<NeuralPolicy> neural;
	int num_simulations;

public:
	HybridPolicy(const std::string& model_path, int num_sims = 800)
		: neural(std::make_unique<NeuralPolicy>(model_path))
		, num_simulations(num_sims)
	{
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		// TODO: Implement AlphaZero MCTS
		// 1. Use NN for prior and value
		// 2. Run MCTS with NN guidance
		// 3. Return move with temperature sampling

		// Placeholder: use neural policy
		return neural->select_action(game);
	}

	std::string name() const override
	{
		return "Hybrid";
	}
};


/**
 * Policy Factory
 *
 * Creates policies from configuration
 */
class PolicyFactory
{
public:
	static std::unique_ptr<IPolicy> create(
		const std::string& type,
		const std::string& model_path = "",
		int seed = -1
	)
	{
		if (seed < 0)
		{
			seed = std::random_device{}();
		}

		if (type == "random")
		{
			return std::make_unique<RandomPolicy>(seed);
		}
		else if (type == "neural")
		{
			if (model_path.empty())
			{
				throw std::runtime_error("Neural policy requires model_path");
			}
			return std::make_unique<NeuralPolicy>(model_path, 1.0f, seed);
		}
		else if (type == "mcts")
		{
			return std::make_unique<MCTSPolicy>(800, 1.414f, seed);
		}
		else if (type == "hybrid")
		{
			return std::make_unique<HybridPolicy>(model_path);
		}
		else
		{
			throw std::runtime_error("Unknown policy type: " + type);
		}
	}
};


} // namespace trigo
