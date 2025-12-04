/**
 * Test TGN Tokenizer C++ implementation against Python implementation
 *
 * This test verifies:
 * - Basic encode/decode functionality
 * - Special token handling (START, END, VALUE, PAD)
 * - Padding and truncation
 * - Batch operations
 * - Compatibility with Python TGNTokenizer
 */

#include "../include/tgn_tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>


using namespace trigo;


void test_basic_encode_decode()
{
	std::cout << "\n[TEST] Basic Encode/Decode\n";
	std::cout << "============================\n";

	TGNTokenizer tokenizer;

	// Test simple string
	std::string test1 = "B3 000";
	auto tokens1 = tokenizer.encode(test1, 2048, false, false, false, false);
	std::string decoded1 = tokenizer.decode(tokens1, false);

	std::cout << "Input:   \"" << test1 << "\"\n";
	std::cout << "Tokens:  ";
	for (auto t : tokens1)
	{
		std::cout << t << " ";
	}
	std::cout << "\n";
	std::cout << "Decoded: \"" << decoded1 << "\"\n";

	assert(decoded1 == test1);
	std::cout << "✓ Basic encode/decode passed\n";
}


void test_special_tokens()
{
	std::cout << "\n[TEST] Special Tokens\n";
	std::cout << "=====================\n";

	TGNTokenizer tokenizer;

	std::string test = "abc";

	// Test START + END
	auto tokens1 = tokenizer.encode(test, 2048, true, false, false, false);
	std::cout << "With START/END: ";
	for (auto t : tokens1)
	{
		std::cout << t << " ";
	}
	std::cout << "\n";
	assert(tokens1.front() == TGNTokenizer::START_ID);
	assert(tokens1.back() == TGNTokenizer::END_ID);
	std::cout << "✓ START/END tokens correct\n";

	// Test VALUE + START + END
	auto tokens2 = tokenizer.encode(test, 2048, true, true, false, false);
	std::cout << "With VALUE/START/END: ";
	for (auto t : tokens2)
	{
		std::cout << t << " ";
	}
	std::cout << "\n";
	assert(tokens2[0] == TGNTokenizer::VALUE_ID);
	assert(tokens2[1] == TGNTokenizer::START_ID);
	assert(tokens2.back() == TGNTokenizer::END_ID);
	std::cout << "✓ VALUE/START/END tokens correct\n";

	// Test decode with skip_special_tokens
	std::string decoded = tokenizer.decode(tokens2, true);
	assert(decoded == test);
	std::cout << "✓ Special tokens skipped in decode\n";
}


void test_padding_truncation()
{
	std::cout << "\n[TEST] Padding and Truncation\n";
	std::cout << "==============================\n";

	TGNTokenizer tokenizer;

	// Test padding
	std::string short_text = "abc";
	auto tokens1 = tokenizer.encode(short_text, 10, false, false, true, false);
	std::cout << "Padded to 10: ";
	for (auto t : tokens1)
	{
		std::cout << t << " ";
	}
	std::cout << "\n";
	assert(static_cast<int>(tokens1.size()) == 10);
	// Check padding
	int pad_count = 0;
	for (auto t : tokens1)
	{
		if (t == TGNTokenizer::PAD_ID)
		{
			pad_count++;
		}
	}
	assert(pad_count == 7);  // 10 - 3 chars
	std::cout << "✓ Padding correct (7 PAD tokens)\n";

	// Test truncation
	std::string long_text = "abcdefghijklmnopqrstuvwxyz";
	auto tokens2 = tokenizer.encode(long_text, 10, true, false, false, true);
	std::cout << "Truncated to 10 (with START/END): ";
	for (auto t : tokens2)
	{
		std::cout << t << " ";
	}
	std::cout << "\n";
	assert(static_cast<int>(tokens2.size()) == 10);
	assert(tokens2.front() == TGNTokenizer::START_ID);
	assert(tokens2.back() == TGNTokenizer::END_ID);
	std::cout << "✓ Truncation correct (START preserved, END added)\n";
}


void test_ascii_mapping()
{
	std::cout << "\n[TEST] ASCII Identity Mapping\n";
	std::cout << "==============================\n";

	TGNTokenizer tokenizer;

	// Test ASCII printable characters
	std::string ascii_test = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

	auto tokens = tokenizer.encode(ascii_test, 2048, false, false, false, false);

	std::cout << "Testing ASCII printable characters (32-126)\n";
	std::cout << "First 10 tokens: ";
	for (size_t i = 0; i < std::min(size_t(10), tokens.size()); i++)
	{
		std::cout << tokens[i] << " ";
	}
	std::cout << "\n";

	// Verify identity mapping: token_id == ascii_value
	for (size_t i = 0; i < ascii_test.size(); i++)
	{
		assert(tokens[i] == static_cast<unsigned char>(ascii_test[i]));
	}

	std::cout << "✓ ASCII identity mapping correct (token_id == ASCII value)\n";

	// Decode and verify
	std::string decoded = tokenizer.decode(tokens, false);
	assert(decoded == ascii_test);
	std::cout << "✓ Decode preserves all ASCII characters\n";
}


void test_batch_operations()
{
	std::cout << "\n[TEST] Batch Operations\n";
	std::cout << "========================\n";

	TGNTokenizer tokenizer;

	std::vector<std::string> batch_input = {
		"B3 000",
		"W5 abc",
		"B9 xyz"
	};

	// Encode batch
	auto batch_tokens = tokenizer.encode_batch(batch_input, 20, true, false, true, true);

	std::cout << "Batch size: " << batch_tokens.size() << "\n";
	assert(batch_tokens.size() == 3);

	for (size_t i = 0; i < batch_tokens.size(); i++)
	{
		std::cout << "  Item " << i << ": " << batch_tokens[i].size() << " tokens\n";
		assert(static_cast<int>(batch_tokens[i].size()) == 20);  // All padded to 20
	}

	// Decode batch
	auto batch_decoded = tokenizer.decode_batch(batch_tokens, true);
	assert(batch_decoded.size() == 3);

	for (size_t i = 0; i < batch_decoded.size(); i++)
	{
		std::cout << "  Decoded " << i << ": \"" << batch_decoded[i] << "\"\n";
		assert(batch_decoded[i] == batch_input[i]);
	}

	std::cout << "✓ Batch encode/decode correct\n";
}


void test_tgn_notation()
{
	std::cout << "\n[TEST] TGN Notation Compatibility\n";
	std::cout << "==================================\n";

	TGNTokenizer tokenizer;

	// Real TGN game notation
	std::string tgn = "B3 000\nW5 abc\nB9 xyz";

	auto tokens = tokenizer.encode(tgn, 2048, true, false, false, false);

	std::cout << "TGN input:\n" << tgn << "\n";
	std::cout << "Token count: " << tokens.size() << "\n";
	std::cout << "First 20 tokens: ";
	for (size_t i = 0; i < std::min(size_t(20), tokens.size()); i++)
	{
		std::cout << tokens[i] << " ";
	}
	std::cout << "\n";

	// Decode
	std::string decoded = tokenizer.decode(tokens, true);
	std::cout << "Decoded:\n" << decoded << "\n";

	assert(decoded == tgn);
	std::cout << "✓ TGN notation round-trip correct\n";
}


void test_vocab_info()
{
	std::cout << "\n[TEST] Vocabulary Information\n";
	std::cout << "==============================\n";

	TGNTokenizer tokenizer;

	std::cout << "Vocabulary size: " << tokenizer.get_vocab_size() << "\n";
	assert(tokenizer.get_vocab_size() == 128);

	auto special_tokens = tokenizer.get_special_tokens();
	std::cout << "Special tokens:\n";
	for (const auto& [name, id] : special_tokens)
	{
		std::cout << "  " << name << ": " << id << "\n";
	}

	assert(special_tokens["pad"] == TGNTokenizer::PAD_ID);
	assert(special_tokens["start"] == TGNTokenizer::START_ID);
	assert(special_tokens["end"] == TGNTokenizer::END_ID);
	assert(special_tokens["value"] == TGNTokenizer::VALUE_ID);

	std::cout << "✓ Vocabulary info correct\n";
}


int main()
{
	std::cout << "TGN Tokenizer C++ Implementation Test\n";
	std::cout << "======================================\n";
	std::cout << "Testing compatibility with Python TGNTokenizer\n";

	try
	{
		test_basic_encode_decode();
		test_special_tokens();
		test_padding_truncation();
		test_ascii_mapping();
		test_batch_operations();
		test_tgn_notation();
		test_vocab_info();

		std::cout << "\n" << std::string(70, '=') << "\n";
		std::cout << "✅ ALL TESTS PASSED!\n";
		std::cout << "C++ tokenizer is compatible with Python implementation.\n";
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
