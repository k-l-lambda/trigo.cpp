/**
 * Test SharedModelInferencer with exported ONNX models
 *
 * This test verifies:
 * - Model loading (base, policy, value heads)
 * - Policy inference (TreeLM mode)
 * - Value inference (EvaluationLM mode)
 * - Output shapes and basic sanity checks
 */

#include "../include/shared_model_inferencer.hpp"
#include "../include/tgn_tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>


using namespace trigo;


void test_model_loading()
{
	std::cout << "\n[TEST] Model Loading\n";
	std::cout << "====================\n";

	std::string base_path = "../models/test_export_shared/base_model.onnx";
	std::string policy_path = "../models/test_export_shared/policy_head.onnx";
	std::string value_path = "../models/test_export_shared/value_head.onnx";

	try
	{
		SharedModelInferencer inferencer(base_path, policy_path, value_path, false);  // CPU mode
		std::cout << "✓ All models loaded successfully\n";

		// Print model info
		inferencer.print_model_info();

		std::cout << "✓ Model loading test passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Model loading failed: " << e.what() << std::endl;
		throw;
	}
}


void test_policy_inference()
{
	std::cout << "\n[TEST] Policy Inference\n";
	std::cout << "=======================\n";

	std::string base_path = "../models/test_export_shared/base_model.onnx";
	std::string policy_path = "../models/test_export_shared/policy_head.onnx";
	std::string value_path = "../models/test_export_shared/value_head.onnx";

	SharedModelInferencer inferencer(base_path, policy_path, value_path, false);

	// Create test inputs matching model dimensions
	int batch_size = 1;  // Model exported with fixed batch_size=1
	int prefix_len = 128;  // Fixed in export
	int eval_len = 64;  // Fixed in export
	int vocab_size = 128;  // From test model

	// Random prefix_ids [batch, prefix_len]
	std::vector<int64_t> prefix_ids;
	for (int b = 0; b < batch_size; b++)
	{
		for (int i = 0; i < prefix_len; i++)
		{
			prefix_ids.push_back(32 + (i % 90));  // ASCII printable range
		}
	}

	// Random evaluated_ids [batch, eval_len]
	std::vector<int64_t> evaluated_ids;
	for (int b = 0; b < batch_size; b++)
	{
		for (int i = 0; i < eval_len; i++)
		{
			evaluated_ids.push_back(40 + (i % 80));
		}
	}

	// Causal mask for evaluated region [batch, eval_len, eval_len]
	std::vector<float> evaluated_mask;
	for (int b = 0; b < batch_size; b++)
	{
		// Lower triangular (causal)
		for (int i = 0; i < eval_len; i++)
		{
			for (int j = 0; j < eval_len; j++)
			{
				evaluated_mask.push_back(j <= i ? 1.0f : 0.0f);
			}
		}
	}

	std::cout << "Input shapes:\n";
	std::cout << "  prefix_ids: [" << batch_size << ", " << prefix_len << "]\n";
	std::cout << "  evaluated_ids: [" << batch_size << ", " << eval_len << "]\n";
	std::cout << "  evaluated_mask: [" << batch_size << ", " << eval_len << ", " << eval_len << "]\n";

	// Run inference
	auto logits = inferencer.policy_inference(
		prefix_ids,
		evaluated_ids,
		evaluated_mask,
		batch_size,
		prefix_len,
		eval_len
	);

	// Check output shape
	int expected_len = eval_len + 1;  // m + 1 (last prefix + all evaluated)
	size_t expected_size = batch_size * expected_len * vocab_size;

	std::cout << "Output shape: [" << batch_size << ", " << expected_len << ", " << vocab_size << "]\n";
	std::cout << "Expected size: " << expected_size << ", Actual size: " << logits.size() << "\n";

	assert(logits.size() == expected_size);

	// Print some sample logits
	std::cout << "\nSample logits (first batch, first position, first 10 tokens):\n";
	for (int i = 0; i < 10; i++)
	{
		std::cout << "  " << logits[i] << "\n";
	}

	// Sanity check: logits should be finite
	bool all_finite = true;
	for (float logit : logits)
	{
		if (!std::isfinite(logit))
		{
			all_finite = false;
			break;
		}
	}
	assert(all_finite);

	std::cout << "✓ All logits are finite\n";
	std::cout << "✓ Policy inference test passed\n";
}


void test_value_inference()
{
	std::cout << "\n[TEST] Value Inference\n";
	std::cout << "======================\n";

	std::string base_path = "../models/test_export_shared/base_model.onnx";
	std::string policy_path = "../models/test_export_shared/policy_head.onnx";
	std::string value_path = "../models/test_export_shared/value_head.onnx";

	SharedModelInferencer inferencer(base_path, policy_path, value_path, false);

	// Create test inputs matching model dimensions
	int batch_size = 1;  // Model exported with fixed batch_size=1
	int seq_len = 128 + 64 - 1;  // prefix_len + eval_len - 1 (VALUE token will make it 192 total)

	// Random input_ids [batch, seq_len]
	std::vector<int64_t> input_ids;
	for (int b = 0; b < batch_size; b++)
	{
		for (int i = 0; i < seq_len; i++)
		{
			input_ids.push_back(32 + (i % 90));
		}
	}

	std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]\n";

	// Run inference
	auto values = inferencer.value_inference(input_ids, batch_size, seq_len, 3);

	// Check output shape
	std::cout << "Output shape: [" << batch_size << "]\n";
	std::cout << "Values: ";
	for (int i = 0; i < batch_size; i++)
	{
		std::cout << values[i];
		if (i < batch_size - 1) std::cout << ", ";
	}
	std::cout << "\n";

	assert(values.size() == static_cast<size_t>(batch_size));

	// Sanity checks
	bool all_finite = true;
	bool reasonable_range = true;
	for (float value : values)
	{
		if (!std::isfinite(value))
		{
			all_finite = false;
		}
		// Values should generally be in range [-2, 2] for untrained model
		if (value < -10.0f || value > 10.0f)
		{
			reasonable_range = false;
		}
	}

	assert(all_finite);
	std::cout << "✓ All values are finite\n";

	if (reasonable_range)
	{
		std::cout << "✓ Values in reasonable range\n";
	}
	else
	{
		std::cout << "⚠ Warning: Some values outside typical range (model may be untrained)\n";
	}

	std::cout << "✓ Value inference test passed\n";
}


void test_tokenizer_integration()
{
	std::cout << "\n[TEST] Tokenizer + Inferencer Integration\n";
	std::cout << "==========================================\n";

	// Tokenize TGN game notation
	TGNTokenizer tokenizer;
	std::string tgn_game = "B3 000\nW5 abc\nB9 xyz";

	auto tokens = tokenizer.encode(tgn_game, 200, false, false, true, false);

	std::cout << "TGN game: \"" << tgn_game << "\"\n";
	std::cout << "Token count: " << tokens.size() << "\n";
	std::cout << "First 20 tokens: ";
	for (size_t i = 0; i < std::min(size_t(20), tokens.size()); i++)
	{
		std::cout << tokens[i] << " ";
	}
	std::cout << "\n";

	// Use tokens for value inference
	std::string base_path = "../models/test_export_shared/base_model.onnx";
	std::string policy_path = "../models/test_export_shared/policy_head.onnx";
	std::string value_path = "../models/test_export_shared/value_head.onnx";

	SharedModelInferencer inferencer(base_path, policy_path, value_path, false);

	// Prepare input (batch_size=1, must be exactly 191 to fit model: 128+64-1)
	int batch_size = 1;
	int seq_len = 128 + 64 - 1;  // 191

	std::vector<int64_t> input_ids;
	if (tokens.size() >= static_cast<size_t>(seq_len))
	{
		input_ids.assign(tokens.begin(), tokens.begin() + seq_len);
	}
	else
	{
		// Pad with spaces if too short
		input_ids.assign(tokens.begin(), tokens.end());
		while (input_ids.size() < static_cast<size_t>(seq_len))
		{
			input_ids.push_back(32);  // SPACE
		}
	}

	// Run value inference
	auto values = inferencer.value_inference(input_ids, batch_size, seq_len, 3);

	std::cout << "Predicted value: " << values[0] << "\n";

	assert(values.size() == 1);
	assert(std::isfinite(values[0]));

	std::cout << "✓ Tokenizer + Inferencer integration test passed\n";
}


int main()
{
	std::cout << "SharedModelInferencer Test Suite\n";
	std::cout << "=================================\n";

	try
	{
		test_model_loading();
		test_policy_inference();
		test_value_inference();
		test_tokenizer_integration();

		std::cout << "\n" << std::string(70, '=') << "\n";
		std::cout << "✅ ALL TESTS PASSED!\n";
		std::cout << "SharedModelInferencer is working correctly.\n";
		std::cout << std::string(70, '=') << "\n";

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "\n❌ TEST FAILED: Unknown error" << std::endl;
		return 1;
	}
}
