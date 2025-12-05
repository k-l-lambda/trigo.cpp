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
#include <vector>
#include <random>
#include <memory>


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
 * Neural Policy (ONNX Model Inference)
 *
 * Interface for neural network policies
 * Supports both offline (fixed model) and online (updating model) modes
 *
 * TODO: Implement ONNX Runtime integration
 */
class NeuralPolicy : public IPolicy
{
private:
	std::string model_path;
	// TODO: Add ONNX Runtime session

public:
	NeuralPolicy(const std::string& model_path)
		: model_path(model_path)
	{
		// TODO: Load ONNX model
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		// TODO: Implement NN inference
		// 1. Convert game state to NN input
		// 2. Run forward pass
		// 3. Get policy distribution + value
		// 4. Sample action from policy

		// Placeholder: use random for now
		RandomPolicy fallback;
		return fallback.select_action(game);
	}

	std::string name() const override
	{
		return "Neural";
	}
};


/**
 * MCTS Policy (Monte Carlo Tree Search)
 *
 * Pure MCTS without neural guidance
 * Can be enhanced with neural value/policy estimates
 *
 * TODO: Implement MCTS algorithm
 */
class MCTSPolicy : public IPolicy
{
private:
	int num_simulations;
	float exploration_constant;

public:
	MCTSPolicy(int num_sims = 800, float c_puct = 1.0f)
		: num_simulations(num_sims), exploration_constant(c_puct)
	{
	}

	PolicyAction select_action(const TrigoGame& game) override
	{
		// TODO: Implement MCTS search
		// 1. Build search tree
		// 2. Run simulations
		// 3. Select best move by visit count

		// Placeholder: use random for now
		RandomPolicy fallback;
		return fallback.select_action(game);
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
			return std::make_unique<NeuralPolicy>(model_path);
		}
		else if (type == "mcts")
		{
			return std::make_unique<MCTSPolicy>();
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
