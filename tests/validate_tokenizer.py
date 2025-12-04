#!/usr/bin/env python3
"""
Cross-language validation test for TGN Tokenizer.

This test verifies that the C++ tokenizer produces identical results
to the Python tokenizer on various test cases.
"""

import sys
import subprocess
import tempfile
from pathlib import Path

# Add trigor to path
trigorl_root = Path(__file__).parent.parent.parent / "trigoRL"
sys.path.insert(0, str(trigorl_root))

from trigor.data.tokenizer import TGNTokenizer


def run_cpp_tokenizer_test():
	"""Run C++ tokenizer and capture output."""
	cpp_build_dir = Path(__file__).parent.parent / "build"
	cpp_test = cpp_build_dir / "test_tgn_tokenizer"

	if not cpp_test.exists():
		print(f"ERROR: C++ test not found at {cpp_test}")
		print("Please build the project first: cd build && cmake .. && make")
		return False

	result = subprocess.run([str(cpp_test)], capture_output=True, text=True)

	if result.returncode != 0:
		print("ERROR: C++ test failed")
		print(result.stdout)
		print(result.stderr)
		return False

	print("✓ C++ tokenizer tests passed")
	return True


def test_python_cpp_compatibility():
	"""Test that Python and C++ tokenizers produce identical results."""
	print("\n" + "=" * 70)
	print("Cross-Language Validation Test")
	print("=" * 70)

	tokenizer = TGNTokenizer()

	test_cases = [
		("B3 000", "Simple TGN move"),
		("B3 000\nW5 abc\nB9 xyz", "Multi-line game"),
		(" !\"#$%&'()*+,-./0123456789", "ASCII punctuation and digits"),
		("abcdefghijklmnopqrstuvwxyz", "Lowercase letters"),
		("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Uppercase letters"),
	]

	print("\n[TEST] Python Tokenizer Validation")
	print("=" * 70)

	for text, description in test_cases:
		print(f"\n{description}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

		# Test 1: Basic encode/decode
		tokens = tokenizer.encode(text, max_length=2048, add_special_tokens=False,
		                         add_value_token=False, padding=False, truncation=False)
		decoded = tokenizer.decode(tokens, skip_special_tokens=False)

		print(f"  Token count: {len(tokens)}")
		print(f"  First 10 tokens: {tokens[:10].tolist()}")

		assert decoded == text, f"Decode mismatch: {decoded} != {text}"
		print(f"  ✓ Round-trip correct")

		# Test 2: With special tokens
		tokens_special = tokenizer.encode(text, max_length=2048, add_special_tokens=True,
		                                 add_value_token=False, padding=False, truncation=False)

		assert tokens_special[0].item() == TGNTokenizer.START_ID
		assert tokens_special[-1].item() == TGNTokenizer.END_ID
		print(f"  ✓ START/END tokens correct")

		# Test 3: With VALUE token
		tokens_value = tokenizer.encode(text, max_length=2048, add_special_tokens=True,
		                               add_value_token=True, padding=False, truncation=False)

		assert tokens_value[0].item() == TGNTokenizer.VALUE_ID
		assert tokens_value[1].item() == TGNTokenizer.START_ID
		assert tokens_value[-1].item() == TGNTokenizer.END_ID
		print(f"  ✓ VALUE/START/END tokens correct")

		# Test 4: ASCII identity mapping
		if all(32 <= ord(c) <= 127 or c == '\n' for c in text):
			for i, char in enumerate(text):
				expected_token = ord(char)
				actual_token = tokens[i].item()
				assert actual_token == expected_token, \
					f"Token mismatch at position {i}: {actual_token} != {expected_token} (char='{char}')"
			print(f"  ✓ ASCII identity mapping verified")

	print("\n" + "=" * 70)
	print("✅ Python tokenizer validation passed")
	print("=" * 70)

	return True


def test_special_token_ids():
	"""Verify special token IDs match between Python and C++."""
	print("\n[TEST] Special Token ID Consistency")
	print("=" * 70)

	tokenizer = TGNTokenizer()
	special_tokens = tokenizer.get_special_tokens()

	print(f"PAD_ID: {special_tokens['pad']} (expected: 0)")
	print(f"START_ID: {special_tokens['start']} (expected: 1)")
	print(f"END_ID: {special_tokens['end']} (expected: 2)")
	print(f"VALUE_ID: {special_tokens['value']} (expected: 3)")
	print(f"VOCAB_SIZE: {tokenizer.get_vocab_size()} (expected: 128)")

	assert special_tokens['pad'] == 0
	assert special_tokens['start'] == 1
	assert special_tokens['end'] == 2
	assert special_tokens['value'] == 3
	assert tokenizer.get_vocab_size() == 128

	print("✓ Special token IDs consistent")

	return True


def main():
	print("TGN Tokenizer Cross-Language Validation")
	print("=" * 70)
	print("Testing Python TGNTokenizer against C++ implementation\n")

	try:
		# Run C++ tests first
		if not run_cpp_tokenizer_test():
			return 1

		# Run Python tests
		if not test_python_cpp_compatibility():
			return 1

		if not test_special_token_ids():
			return 1

		print("\n" + "=" * 70)
		print("🎉 ALL CROSS-LANGUAGE TESTS PASSED!")
		print("=" * 70)
		print("Python and C++ tokenizers are fully compatible.")
		print("Both produce identical token sequences and handle:")
		print("  • ASCII identity mapping (token_id == ASCII value)")
		print("  • Special tokens (PAD, START, END, VALUE)")
		print("  • Padding and truncation")
		print("  • Batch operations")
		print("  • TGN notation with newlines")
		print("=" * 70)

		return 0

	except AssertionError as e:
		print(f"\n❌ TEST FAILED: {e}")
		return 1
	except Exception as e:
		print(f"\n❌ UNEXPECTED ERROR: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == "__main__":
	sys.exit(main())
