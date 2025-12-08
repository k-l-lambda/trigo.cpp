/**
 * Test Policy and Value Inference Validation
 *
 * Compare C++ inference results with Python/TypeScript baseline
 * Use TGN prefix from actual self-play game
 */

#include "../include/trigo_game.hpp"
#include "../include/trigo_coords.hpp"
#include "../include/tgn_utils.hpp"
#include "../include/tgn_tokenizer.hpp"
#include "../include/shared_model_inferencer.hpp"
#include "../include/prefix_tree_builder.hpp"
#include <iostream>
#include <vector>
#include <string>


using namespace trigo;


int main()
{
	std::cout << "=== Policy & Value Inference Validation ===" << std::endl;
	std::cout << std::endl;

	// Test configuration
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached";
	BoardShape shape{5, 5, 1};

	// Create initial game state (empty board)
	TrigoGame game(shape);
	game.start_game();

	// Get TGN representation
	std::string tgn_text = game_to_tgn(game, false);
	std::cout << "TGN prefix:" << std::endl;
	std::cout << tgn_text << std::endl;
	std::cout << std::endl;

	// Tokenize
	TGNTokenizer tokenizer;
	auto tgn_tokens = tokenizer.encode(tgn_text, 8192, false, false, false, false);

	// Add START token
	std::vector<int64_t> prefix_tokens;
	prefix_tokens.push_back(1);  // START
	prefix_tokens.insert(prefix_tokens.end(), tgn_tokens.begin(), tgn_tokens.end());

	std::cout << "Prefix tokens (" << prefix_tokens.size() << "): ";
	for (size_t i = 0; i < std::min(prefix_tokens.size(), size_t(20)); i++)
	{
		std::cout << prefix_tokens[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	// Get valid moves
	auto valid_moves = game.valid_move_positions();
	std::cout << "Valid moves: " << valid_moves.size() << std::endl;

	// Build candidate sequences for first 5 moves
	std::vector<Position> test_moves;
	for (size_t i = 0; i < std::min(size_t(5), valid_moves.size()); i++)
	{
		test_moves.push_back(valid_moves[i]);
	}
	test_moves.push_back(valid_moves[valid_moves.size() - 1]);  // Last move

	std::vector<std::vector<int64_t>> candidate_sequences;
	std::vector<std::string> move_names;

	for (const auto& move : test_moves)
	{
		std::string coord = encode_ab0yz(move, shape);
		move_names.push_back(coord);

		auto move_tokens = tokenizer.encode(coord, 2048, false, false, false, false);

		std::vector<int64_t> seq = prefix_tokens;
		seq.insert(seq.end(), move_tokens.begin(), move_tokens.end());

		candidate_sequences.push_back(seq);
	}

	// Add PASS candidate
	auto pass_tokens = tokenizer.encode("PASS", 2048, false, false, false, false);
	std::vector<int64_t> pass_seq = prefix_tokens;
	pass_seq.insert(pass_seq.end(), pass_tokens.begin(), pass_tokens.end());
	candidate_sequences.push_back(pass_seq);
	move_names.push_back("PASS");

	std::cout << "Test candidates: ";
	for (const auto& name : move_names)
	{
		std::cout << name << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	// Build prefix tree
	PrefixTreeBuilder tree_builder;
	auto tree = tree_builder.build_tree(candidate_sequences);

	std::cout << "Tree structure:" << std::endl;
	std::cout << "  Num nodes: " << tree.num_nodes << std::endl;
	std::cout << "  Evaluated IDs size: " << tree.evaluated_ids.size() << std::endl;
	std::cout << std::endl;

	// Load model and run inference
	std::cout << "Loading models..." << std::endl;
	SharedModelInferencer inferencer(
		model_dir + "/base_model.onnx",
		model_dir + "/policy_head.onnx",
		model_dir + "/value_head.onnx",
		false  // CPU
	);
	std::cout << "Models loaded" << std::endl;
	std::cout << std::endl;

	// Run policy inference
	std::cout << "[POLICY INFERENCE]" << std::endl;
	int prefix_len = static_cast<int>(prefix_tokens.size());
	int eval_len = tree.num_nodes;

	auto logits = inferencer.policy_inference(
		prefix_tokens,
		tree.evaluated_ids,
		tree.evaluated_mask,
		1,  // batch_size
		prefix_len,
		eval_len
	);

	int vocab_size = 128;  // Known from model

	std::cout << "Policy logits:" << std::endl;
	for (size_t i = 0; i < move_names.size(); i++)
	{
		int leaf_pos = tree.move_to_leaf[i];
		const auto& seq = candidate_sequences[i];
		int64_t last_token = seq.back();

		int logit_idx = leaf_pos * vocab_size + static_cast<int>(last_token);
		float logit = logits[logit_idx];

		std::cout << "  " << move_names[i] << ": leaf_pos=" << leaf_pos
		          << ", last_token=" << last_token
		          << ", logit=" << logit << std::endl;
	}
	std::cout << std::endl;

	// Run value inference
	std::cout << "[VALUE INFERENCE]" << std::endl;

	auto value_result = inferencer.value_inference(
		prefix_tokens,
		1,  // batch_size
		prefix_len,
		3   // VALUE token ID
	);

	std::cout << "Value score: " << value_result[0] << std::endl;
	std::cout << std::endl;

	std::cout << "=== C++ Validation Complete ===" << std::endl;
	std::cout << "Compare these results with Python/TypeScript version" << std::endl;

	return 0;
}
