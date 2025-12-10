/**
 * Test Tree Model Policy Inference (C++ vs TypeScript comparison)
 *
 * This test uses the tree model directly (GPT2CausalLM_ep0019_tree.onnx)
 * to compare with TypeScript's output, bypassing the prefix cache.
 *
 * Goal: Verify that C++ tree model inference matches TypeScript exactly.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

#include "../include/tgn_tokenizer.hpp"
#include "../include/prefix_tree_builder.hpp"


/**
 * Compute softmax over a slice of logits
 */
std::vector<float> softmax(const float* logits, int vocab_size)
{
	std::vector<float> probs(vocab_size);

	// Find max for numerical stability
	float max_val = *std::max_element(logits, logits + vocab_size);

	// Compute exp and sum
	float sum = 0.0f;
	for (int i = 0; i < vocab_size; i++)
	{
		probs[i] = std::exp(logits[i] - max_val);
		sum += probs[i];
	}

	// Normalize
	for (int i = 0; i < vocab_size; i++)
	{
		probs[i] /= sum;
	}

	return probs;
}


int main()
{
	std::cout << "\n";
	std::cout << "============================================================================\n";
	std::cout << "C++ Tree Model Policy Test (matches TypeScript trigoTreeAgent)\n";
	std::cout << "============================================================================\n\n";

	// Model path - same as TypeScript uses
	std::string tree_model_path = "/home/camus/work/trigoRL/outputs/trigor/"
		"20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_tree.onnx";

	std::cout << "Loading tree model: " << tree_model_path << "\n\n";

	try
	{
		// Initialize ONNX Runtime
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tree_policy_test");
		Ort::SessionOptions session_options;

		// Use CPU for consistency
		std::cout << "Using CPU execution provider\n\n";

		// Load tree model
		Ort::Session session(env, tree_model_path.c_str(), session_options);
		Ort::AllocatorWithDefaultOptions allocator;

		// Game setup: 5x5 board, empty
		std::cout << "Game Configuration:\n";
		std::cout << "  Board: 5x5x1\n";
		std::cout << "  Position: Empty board (Move 1)\n";
		std::cout << "  Current player: Black\n\n";

		// Generate TGN prefix (matching TypeScript trigoTreeAgent.buildMoveTree)
		// TypeScript uses: game.toTGN().trim() + " " for empty board
		// which gives "[Board 5x5] " (without newlines and move number)
		std::string tgn_text = "[Board 5x5] ";

		std::cout << "TGN Text: \"" << tgn_text << "\"\n";

		// Tokenize prefix (START + ASCII)
		const int START_TOKEN = 1;
		std::vector<int64_t> prefix_ids;
		prefix_ids.push_back(START_TOKEN);
		for (char c : tgn_text)
		{
			prefix_ids.push_back(static_cast<int64_t>(c));
		}

		std::cout << "Prefix tokens (" << prefix_ids.size() << "): [";
		for (size_t i = 0; i < prefix_ids.size(); i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << prefix_ids[i];
		}
		std::cout << "]\n\n";

		// Build move notations for 5x5 board
		trigo::TGNTokenizer tokenizer;
		std::vector<std::string> coords = {
			"aa", "ab", "a0", "ay", "az",
			"ba", "bb", "b0", "by", "bz",
			"0a", "0b", "00", "0y", "0z",
			"ya", "yb", "y0", "yy", "yz",
			"za", "zb", "z0", "zy", "zz"
		};

		// Store full notations and build token sequences for tree
		std::vector<std::string> move_notations;
		std::vector<std::vector<int64_t>> token_sequences;

		for (const auto& coord : coords)
		{
			move_notations.push_back(coord);
			auto tokens = tokenizer.encode(coord, 256, false, false, false, false);
			// Exclude last token for tree building
			std::vector<int64_t> tree_tokens;
			if (tokens.size() > 1)
			{
				tree_tokens.assign(tokens.begin(), tokens.end() - 1);
			}
			token_sequences.push_back(tree_tokens);
		}

		// Add Pass move
		move_notations.push_back("Pass");
		auto pass_tokens = tokenizer.encode("Pass", 256, false, false, false, false);
		std::vector<int64_t> pass_tree;
		if (pass_tokens.size() > 1)
		{
			pass_tree.assign(pass_tokens.begin(), pass_tokens.end() - 1);
		}
		token_sequences.push_back(pass_tree);

		std::cout << "Total candidates: " << move_notations.size() << "\n\n";

		// Build prefix tree using PrefixTreeBuilder
		trigo::PrefixTreeBuilder builder;
		auto tree = builder.build_tree(token_sequences);

		std::cout << "Tree structure:\n";
		std::cout << "  num_nodes: " << tree.num_nodes << "\n";
		std::cout << "  evaluated_ids: [";
		for (int i = 0; i < tree.num_nodes; i++)
		{
			if (i > 0) std::cout << ", ";
			std::cout << tree.evaluated_ids[i];
		}
		std::cout << "]\n";
		std::cout << "  As chars: \"";
		for (int i = 0; i < tree.num_nodes; i++)
		{
			std::cout << static_cast<char>(tree.evaluated_ids[i]);
		}
		std::cout << "\"\n\n";

		// Prepare inputs for tree model
		int batch_size = 1;
		int prefix_len = static_cast<int>(prefix_ids.size());
		int eval_len = tree.num_nodes;

		std::vector<int64_t> prefix_ids_batched(prefix_ids);  // Already correct shape

		std::vector<int64_t> evaluated_ids_batched;
		for (int i = 0; i < tree.num_nodes; i++)
		{
			evaluated_ids_batched.push_back(tree.evaluated_ids[i]);
		}

		// Build mask from tree structure (m x m)
		std::vector<float> mask_batched(eval_len * eval_len, 0.0f);
		for (int i = 0; i < eval_len; i++)
		{
			int pos = i;
			while (pos >= 0)
			{
				mask_batched[i * eval_len + pos] = 1.0f;
				pos = tree.parent[pos];
			}
		}

		// Create input tensors
		std::vector<int64_t> prefix_shape = {batch_size, prefix_len};
		std::vector<int64_t> eval_shape = {batch_size, eval_len};
		std::vector<int64_t> mask_shape = {batch_size, eval_len, eval_len};

		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value prefix_tensor = Ort::Value::CreateTensor<int64_t>(
			memory_info, prefix_ids_batched.data(), prefix_ids_batched.size(),
			prefix_shape.data(), prefix_shape.size()
		);

		Ort::Value eval_tensor = Ort::Value::CreateTensor<int64_t>(
			memory_info, evaluated_ids_batched.data(), evaluated_ids_batched.size(),
			eval_shape.data(), eval_shape.size()
		);

		Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
			memory_info, mask_batched.data(), mask_batched.size(),
			mask_shape.data(), mask_shape.size()
		);

		// Run inference
		const char* input_names[] = {"prefix_ids", "evaluated_ids", "evaluated_mask"};
		const char* output_names[] = {"logits"};

		std::vector<Ort::Value> input_tensors;
		input_tensors.push_back(std::move(prefix_tensor));
		input_tensors.push_back(std::move(eval_tensor));
		input_tensors.push_back(std::move(mask_tensor));

		std::cout << "Running tree model inference...\n";
		auto outputs = session.Run(
			Ort::RunOptions{nullptr},
			input_names, input_tensors.data(), 3,
			output_names, 1
		);

		// Get output logits
		auto& logits_tensor = outputs[0];
		auto logits_info = logits_tensor.GetTensorTypeAndShapeInfo();
		auto logits_shape = logits_info.GetShape();

		std::cout << "Logits shape: [" << logits_shape[0] << ", "
			<< logits_shape[1] << ", " << logits_shape[2] << "]\n";

		const float* logits = logits_tensor.GetTensorData<float>();
		int vocab_size = static_cast<int>(logits_shape[2]);
		int num_outputs = static_cast<int>(logits_shape[1]);  // m+1

		std::cout << "num_outputs: " << num_outputs << " (should be eval_len+1=" << (eval_len+1) << ")\n\n";

		// Score each move using path accumulation (same algorithm as TypeScript)
		const float MIN_PROB = 1e-10f;

		struct ScoredMove
		{
			std::string notation;
			float log_prob;
		};

		std::vector<ScoredMove> scored_moves;

		// Debug first few moves
		bool debug_mode = true;
		int debug_count = 0;

		for (size_t move_idx = 0; move_idx < move_notations.size(); move_idx++)
		{
			float log_prob = 0.0f;
			int leaf_pos = tree.move_to_leaf[move_idx];
			std::string notation = move_notations[move_idx];

			// Build path from leaf to root
			std::vector<int> path;
			int pos = leaf_pos;
			while (pos >= 0)
			{
				path.push_back(pos);
				pos = tree.parent[pos];
			}
			std::reverse(path.begin(), path.end());

			bool show_debug = debug_mode && (notation == "zz" || notation == "aa" || notation == "za");
			if (show_debug)
			{
				std::cout << "\n--- Debug: " << notation << " ---\n";
				std::cout << "leaf_pos: " << leaf_pos << "\n";
				std::cout << "path: [";
				for (size_t i = 0; i < path.size(); i++)
				{
					if (i > 0) std::cout << ", ";
					std::cout << path[i];
				}
				std::cout << "]\n";
			}

			// Root token (predicted from prefix last position = logits[0])
			if (path.size() > 0)
			{
				int root_pos = path[0];
				int root_token = static_cast<int>(tree.evaluated_ids[root_pos]);

				auto probs = softmax(&logits[0 * vocab_size], vocab_size);
				float prob = std::max(probs[root_token], MIN_PROB);
				if (show_debug)
				{
					std::cout << "Root: token=" << root_token << " ('" << static_cast<char>(root_token)
						<< "') prob=" << prob << " log=" << std::log(prob) << "\n";
				}
				log_prob += std::log(prob);
			}

			// Path transitions (parent output predicts child token)
			for (size_t i = 1; i < path.size(); i++)
			{
				int parent_pos = path[i - 1];
				int child_token = static_cast<int>(tree.evaluated_ids[path[i]]);

				// Parent output is at logits[parent_pos + 1]
				int logits_idx = parent_pos + 1;
				if (logits_idx < num_outputs)
				{
					auto probs = softmax(&logits[logits_idx * vocab_size], vocab_size);
					float prob = std::max(probs[child_token], MIN_PROB);
					log_prob += std::log(prob);
				}
				else
				{
					log_prob += std::log(MIN_PROB);
				}
			}

			// Last token (excluded from tree, predicted by leaf output)
			if (path.size() > 0)
			{
				int leaf = path.back();
				int logits_idx = leaf + 1;

				// Get last character of notation
				int last_token = static_cast<int>(notation.back());

				if (logits_idx < num_outputs)
				{
					auto probs = softmax(&logits[logits_idx * vocab_size], vocab_size);
					float prob = std::max(probs[last_token], MIN_PROB);
					if (show_debug)
					{
						std::cout << "Last: logits_idx=" << logits_idx << " token=" << last_token
							<< " ('" << static_cast<char>(last_token)
							<< "') prob=" << prob << " log=" << std::log(prob) << "\n";
						std::cout << "Total log_prob: " << log_prob + std::log(prob) << "\n";
					}
					log_prob += std::log(prob);
				}
				else
				{
					log_prob += std::log(MIN_PROB);
				}
			}

			scored_moves.push_back({notation, log_prob});
		}

		// Sort by score descending
		std::sort(scored_moves.begin(), scored_moves.end(),
			[](const ScoredMove& a, const ScoredMove& b) { return a.log_prob > b.log_prob; });

		// Compute priors
		float max_score = scored_moves[0].log_prob;
		std::vector<float> exp_scores;
		float sum_exp = 0.0f;
		for (const auto& sm : scored_moves)
		{
			float exp_s = std::exp(sm.log_prob - max_score);
			exp_scores.push_back(exp_s);
			sum_exp += exp_s;
		}

		std::cout << "============================================================================\n";
		std::cout << "Policy Priors (C++ Tree Model, sorted by log score):\n";
		std::cout << "============================================================================\n\n";

		std::cout << "| Rank | Move | Log Score | Prior |\n";
		std::cout << "|------|------|-----------|-------|\n";

		for (size_t i = 0; i < std::min((size_t)10, scored_moves.size()); i++)
		{
			float prior = exp_scores[i] / sum_exp;
			std::cout << std::fixed << std::setprecision(6);
			std::cout << "|  " << std::setw(3) << (i + 1) << " | "
				<< std::setw(4) << scored_moves[i].notation << " | "
				<< std::setw(9) << scored_moves[i].log_prob << " | "
				<< prior << " |\n";
		}
		std::cout << "\n";

		std::cout << "For TypeScript comparison (top 5):\n";
		for (size_t i = 0; i < std::min((size_t)5, scored_moves.size()); i++)
		{
			float prior = exp_scores[i] / sum_exp;
			std::cout << std::fixed << std::setprecision(6);
			std::cout << "  " << (i + 1) << ". " << scored_moves[i].notation
				<< " log_score=" << scored_moves[i].log_prob
				<< " prior=" << prior << "\n";
		}
		std::cout << "\n";

		std::cout << "Expected TypeScript tree model output:\n";
		std::cout << "  1. zz log_score=-5.988579 prior=0.108183\n";
		std::cout << "  2. za log_score=-6.066297 prior=0.100094\n";
		std::cout << "  3. zb log_score=-6.180488 prior=0.089293\n";
		std::cout << "\n";

		std::cout << "Expected C++ prefix cache output (uses different prefix!):\n";
		std::cout << "  1. aa log_score=-6.347701 prior=0.079293\n";
		std::cout << "  2. az log_score=-6.392598 prior=0.075812\n";
		std::cout << "  3. zz log_score=-6.454264 prior=0.071278\n";
		std::cout << "\n";

		return 0;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "\n! ONNX Runtime Error: " << e.what() << "\n";
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n! Error: " << e.what() << "\n";
		return 1;
	}
}
