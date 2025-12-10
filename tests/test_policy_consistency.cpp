/**
 * Test policy consistency between C++ and TypeScript
 *
 * Pure policy prior test without MCTS simulation
 * - No noise
 * - No temperature (or equivalently temperature=1.0)
 * - Just raw log scores from model
 */

#include "../include/prefix_cache_inferencer.hpp"
#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include "../include/tgn_utils.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>


using namespace trigo;


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

	// Find max for numerical stability
	float max_logit = *std::max_element(position_logits.begin(), position_logits.end());

	// Compute exp and sum
	std::vector<float> exp_vals(vocab_size);
	float sum = 0.0f;
	for (int i = 0; i < vocab_size; i++)
	{
		exp_vals[i] = std::exp(position_logits[i] - max_logit);
		sum += exp_vals[i];
	}

	// Normalize
	std::vector<float> probs(vocab_size);
	for (int i = 0; i < vocab_size; i++)
	{
		probs[i] = exp_vals[i] / sum;
	}

	return probs;
}


/**
 * Convert log scores to normalized probabilities (exp-normalize)
 */
std::vector<float> exp_normalize(const std::vector<float>& log_scores)
{
	if (log_scores.empty())
		return {};

	float max_score = *std::max_element(log_scores.begin(), log_scores.end());

	std::vector<float> exp_vals(log_scores.size());
	float sum = 0.0f;
	for (size_t i = 0; i < log_scores.size(); i++)
	{
		exp_vals[i] = std::exp(log_scores[i] - max_score);
		sum += exp_vals[i];
	}

	std::vector<float> probs(log_scores.size());
	for (size_t i = 0; i < log_scores.size(); i++)
	{
		probs[i] = exp_vals[i] / sum;
	}

	return probs;
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
		return 1;
	}

	std::string model_dir = argv[1];

	std::cout << "============================================================================" << std::endl;
	std::cout << "Policy Consistency Test (C++ Prefix Cache)" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;

	// Load model
	std::cout << "Loading models from: " << model_dir << std::endl;
	auto inferencer = std::make_shared<PrefixCacheInferencer>(
		model_dir + "/base_model_prefix.onnx",
		model_dir + "/base_model_eval_cached.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false,  // use_cuda
		0       // device_id
	);
	std::cout << "✓ Models loaded" << std::endl;
	std::cout << std::endl;

	// Setup tokenizer
	TGNTokenizer tokenizer;

	// Setup game: 5x5 board, empty
	BoardShape shape{5, 5, 1};
	TrigoGame game(shape);
	game.start_game();

	std::cout << "Game Configuration:" << std::endl;
	std::cout << "  Board: 5×5×1" << std::endl;
	std::cout << "  Position: Empty board (Move 1)" << std::endl;
	std::cout << "  Current player: " << (game.get_current_player() == Stone::Black ? "Black" : "White") << std::endl;
	auto valid_moves = game.valid_move_positions();
	std::cout << "  Valid moves: " << valid_moves.size() << std::endl;
	std::cout << std::endl;

	// Generate TGN for current position
	std::string tgn_text = game_to_tgn(game, false);

	// Add move number prefix (matching TypeScript TrigoTreeAgent.buildMoveTree)
	const auto& history = game.get_history();
	int move_number = static_cast<int>(history.size()) / 2 + 1;

	if (game.get_current_player() == Stone::Black)
	{
		tgn_text += std::to_string(move_number) + ". ";
	}
	else
	{
		tgn_text += " ";
	}

	std::cout << "TGN Text: \"" << tgn_text << "\"" << std::endl;

	// Tokenize
	auto encoded = tokenizer.encode(tgn_text, 8192, false, false, false, false);

	// Add START token
	std::vector<int64_t> tokens;
	tokens.push_back(1);  // START token
	tokens.insert(tokens.end(), encoded.begin(), encoded.end());

	std::cout << "Token count: " << tokens.size() << std::endl;
	std::cout << "Tokens: [";
	for (size_t i = 0; i < tokens.size(); i++)
	{
		if (i > 0) std::cout << ", ";
		std::cout << tokens[i];
	}
	std::cout << "]" << std::endl;
	std::cout << std::endl;

	// Compute prefix cache
	std::cout << "Computing prefix cache..." << std::endl;
	int seq_len = static_cast<int>(tokens.size());
	inferencer->compute_prefix_cache(tokens, 1, seq_len);
	std::cout << "✓ Prefix cache computed" << std::endl;
	std::cout << std::endl;

	// Build candidate move sequences
	std::cout << "Building candidate move sequences..." << std::endl;
	std::vector<std::vector<int64_t>> candidate_sequences;
	std::vector<std::string> move_notations;

	// Add all valid moves (exclude last token as per TypeScript)
	for (const auto& move : valid_moves)
	{
		std::string coord = encode_ab0yz(move, shape);
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

	// Add Pass
	auto pass_tokens = tokenizer.encode("Pass", 2048, false, false, false, false);
	move_notations.push_back("Pass");
	if (!pass_tokens.empty())
	{
		std::vector<int64_t> seq(pass_tokens.begin(), pass_tokens.end() - 1);
		candidate_sequences.push_back(seq);
	}

	std::cout << "Total candidates: " << candidate_sequences.size() << std::endl;
	std::cout << std::endl;

	// Build prefix tree
	PrefixTreeBuilder tree_builder;
	auto tree_structure = tree_builder.build_tree(candidate_sequences);
	std::cout << "Prefix tree nodes: " << tree_structure.num_nodes << std::endl;
	std::cout << std::endl;

	// Evaluate with cache
	std::cout << "Running cached evaluation..." << std::endl;
	auto hidden_states = inferencer->evaluate_with_cache(
		tree_structure.evaluated_ids,
		tree_structure.evaluated_mask,
		1,  // batch_size
		tree_structure.num_nodes
	);

	// Run policy head
	int hidden_dim = static_cast<int>(hidden_states.size()) / tree_structure.num_nodes;
	auto logits = inferencer->policy_inference_from_hidden(
		hidden_states,
		1,  // batch_size
		tree_structure.num_nodes,
		hidden_dim
	);

	int vocab_size = static_cast<int>(logits.size()) / tree_structure.num_nodes;
	std::cout << "Hidden dim: " << hidden_dim << ", Vocab size: " << vocab_size << std::endl;
	std::cout << std::endl;

	// Score each move
	const float MIN_PROB = 1e-10f;
	std::vector<float> log_scores;

	for (size_t i = 0; i < candidate_sequences.size(); i++)
	{
		int leaf_pos = tree_structure.move_to_leaf[i];
		float log_prob = 0.0f;

		if (leaf_pos == -1)
		{
			// Direct prediction from prefix
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
			// Build path from leaf to root
			std::vector<int> path_reverse;
			int pos = leaf_pos;
			while (pos != -1)
			{
				path_reverse.push_back(pos);
				pos = tree_structure.parent[pos];
			}
			std::reverse(path_reverse.begin(), path_reverse.end());

			// Root token
			if (!path_reverse.empty())
			{
				int root_pos = path_reverse[0];
				int64_t root_token = tree_structure.evaluated_ids[root_pos];
				auto probs = softmax_at_position(logits, 0, vocab_size);
				float prob = std::max(probs[root_token], MIN_PROB);
				log_prob += std::log(prob);
			}

			// Intermediate transitions
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

			// Last token
			if (!path_reverse.empty())
			{
				int leaf = path_reverse.back();
				const std::string& notation = move_notations[i];
				auto notation_tokens = tokenizer.encode(notation, 2048, false, false, false, false);
				if (!notation_tokens.empty())
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

	// Convert to priors
	auto priors = exp_normalize(log_scores);

	// Sort by log score descending
	std::vector<std::pair<size_t, float>> indexed_scores;
	for (size_t i = 0; i < log_scores.size(); i++)
	{
		indexed_scores.push_back({i, log_scores[i]});
	}
	std::sort(indexed_scores.begin(), indexed_scores.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	// Print results
	std::cout << "============================================================================" << std::endl;
	std::cout << "Policy Priors (sorted by log score, no noise, no temperature):" << std::endl;
	std::cout << "============================================================================" << std::endl;
	std::cout << std::endl;
	std::cout << "| Rank | Move | Log Score | Prior |" << std::endl;
	std::cout << "|------|------|-----------|-------|" << std::endl;

	for (size_t i = 0; i < std::min(size_t(10), indexed_scores.size()); i++)
	{
		size_t idx = indexed_scores[i].first;
		std::string move_str = move_notations[idx];
		float log_score = log_scores[idx];
		float prior = priors[idx];

		std::cout << "| " << std::setw(4) << (i + 1) << " | "
				  << std::setw(4) << move_str << " | "
				  << std::fixed << std::setprecision(6) << std::setw(9) << log_score << " | "
				  << std::setprecision(6) << std::setw(8) << prior << " |" << std::endl;
	}
	std::cout << std::endl;

	// Print for easy comparison
	std::cout << "For TypeScript comparison (top 5):" << std::endl;
	for (size_t i = 0; i < std::min(size_t(5), indexed_scores.size()); i++)
	{
		size_t idx = indexed_scores[i].first;
		std::cout << "  " << (i + 1) << ". " << move_notations[idx]
				  << " log_score=" << std::fixed << std::setprecision(6) << log_scores[idx]
				  << " prior=" << priors[idx] << std::endl;
	}
	std::cout << std::endl;

	// Get value
	std::cout << "Value estimate:" << std::endl;
	float value = inferencer->value_inference_with_cache(3);  // VALUE token ID = 3
	std::cout << "  Value: " << std::fixed << std::setprecision(6) << value << std::endl;
	std::cout << std::endl;

	return 0;
}
