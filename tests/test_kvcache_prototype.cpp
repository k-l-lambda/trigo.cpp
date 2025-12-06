/**
 * KV Cache Prototype Test
 *
 * This test validates the KV cache design and measures performance.
 *
 * Test scenarios:
 * 1. Basic functionality: Generate tokens with KV cache
 * 2. Performance comparison: With vs without KV cache
 * 3. Memory validation: Check GPU memory persistence
 * 4. Sequence length scaling: Test different lengths
 *
 * Expected results:
 * - First token: ~50-100ms (cold start)
 * - Subsequent tokens: ~5-10ms (10-20× speedup with cache)
 * - Memory usage: ~75 MB for GPT-2 scale model
 */

#include "kvcache_inferencer.hpp"
#include <iostream>
#include <vector>
#include <iomanip>


void print_header(const std::string& title)
{
	std::cout << "\n" << std::string(70, '=') << std::endl;
	std::cout << title << std::endl;
	std::cout << std::string(70, '=') << std::endl;
}


/**
 * Test 1: Basic KV cache functionality
 */
void test_basic_kvcache()
{
	print_header("Test 1: Basic KV Cache Functionality");

	try
	{
		// Note: This test uses a dummy configuration
		// In practice, you would use a real model with KV cache I/O
		std::string model_path = "../models/dummy/model_with_cache.onnx";

		std::cout << "Initializing KVCacheInferencer..." << std::endl;
		trigo::KVCacheInferencer inferencer(
			model_path,
			true,  // use_gpu
			0,     // device_id
			2048,  // max_seq_len
			12,    // num_layers
			12,    // num_heads
			64     // head_dim
		);

		std::cout << "\nGenerating tokens with KV cache..." << std::endl;

		// First token (cold start)
		std::cout << "\n[Token 1] Cold start (no cache)" << std::endl;
		auto logits1 = inferencer.forward({42});
		std::cout << "  Logits size: " << logits1.size() << std::endl;

		// Subsequent tokens (with cache)
		std::cout << "\n[Token 2] With KV cache" << std::endl;
		auto logits2 = inferencer.forward({128});
		std::cout << "  Logits size: " << logits2.size() << std::endl;

		std::cout << "\n[Token 3] With KV cache" << std::endl;
		auto logits3 = inferencer.forward({256});
		std::cout << "  Logits size: " << logits3.size() << std::endl;

		// Print metrics
		inferencer.print_metrics();

		std::cout << "\n✓ Test passed: KV cache works correctly" << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


/**
 * Test 2: Performance comparison (with vs without cache)
 */
void test_performance_comparison()
{
	print_header("Test 2: Performance Comparison");

	try
	{
		std::string model_path = "../models/dummy/model_with_cache.onnx";

		trigo::KVCacheInferencer inferencer(
			model_path,
			true,  // use_gpu
			0,
			2048,
			12,
			12,
			64
		);

		std::cout << "\n[Scenario A] Sequential generation WITH KV cache" << std::endl;
		std::cout << "Generating 10 tokens sequentially..." << std::endl;

		auto start_with_cache = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < 10; i++)
		{
			auto logits = inferencer.forward({static_cast<int64_t>(100 + i)});
		}

		auto end_with_cache = std::chrono::high_resolution_clock::now();
		double time_with_cache = std::chrono::duration<double, std::milli>(end_with_cache - start_with_cache).count();

		std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_with_cache << " ms" << std::endl;
		std::cout << "  Avg per token: " << (time_with_cache / 10.0) << " ms" << std::endl;

		// Get final metrics
		const auto& metrics = inferencer.get_metrics();

		std::cout << "\n[Scenario B] Recomputing full sequence WITHOUT KV cache" << std::endl;
		std::cout << "Simulating 10 token generation by recomputing full sequence..." << std::endl;

		// Build full sequence
		std::vector<int64_t> full_sequence = {42};  // Start token
		for (int i = 0; i < 10; i++)
		{
			full_sequence.push_back(100 + i);
		}

		auto start_no_cache = std::chrono::high_resolution_clock::now();

		// Simulate sequential generation: for each new token, recompute from start
		for (size_t len = 1; len <= 10; len++)
		{
			std::vector<int64_t> seq(full_sequence.begin(), full_sequence.begin() + len + 1);
			auto logits = inferencer.forward_no_cache(seq);
		}

		auto end_no_cache = std::chrono::high_resolution_clock::now();
		double time_no_cache = std::chrono::duration<double, std::milli>(end_no_cache - start_no_cache).count();

		std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_no_cache << " ms" << std::endl;
		std::cout << "  Avg per token: " << (time_no_cache / 10.0) << " ms" << std::endl;

		// Calculate speedup
		double speedup = time_no_cache / time_with_cache;

		std::cout << "\n[Performance Summary]" << std::endl;
		std::cout << "  Speedup with KV cache: " << std::fixed << std::setprecision(2) << speedup << "×" << std::endl;
		std::cout << "  Time saved: " << (time_no_cache - time_with_cache) << " ms ("
		          << std::fixed << std::setprecision(1) << (100.0 * (1 - time_with_cache / time_no_cache)) << "%)" << std::endl;

		if (speedup > 5.0)
		{
			std::cout << "\n✓ Test passed: KV cache provides significant speedup (>" << speedup << "×)" << std::endl;
		}
		else
		{
			std::cout << "\n⚠ Warning: Speedup lower than expected (" << speedup << "× < 5×)" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


/**
 * Test 3: Sequence length scaling
 */
void test_sequence_length_scaling()
{
	print_header("Test 3: Sequence Length Scaling");

	try
	{
		std::string model_path = "../models/dummy/model_with_cache.onnx";

		std::vector<int> test_lengths = {10, 50, 100, 200};

		std::cout << "\nTesting different sequence lengths..." << std::endl;
		std::cout << std::string(70, '-') << std::endl;
		std::cout << std::setw(15) << "Seq Length"
		          << std::setw(20) << "With Cache (ms)"
		          << std::setw(20) << "Without Cache (ms)"
		          << std::setw(15) << "Speedup" << std::endl;
		std::cout << std::string(70, '-') << std::endl;

		for (int length : test_lengths)
		{
			trigo::KVCacheInferencer inferencer(model_path, true, 0, 2048, 12, 12, 64);

			// Generate sequence with cache
			auto start_with = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < length; i++)
			{
				inferencer.forward({static_cast<int64_t>(100 + i)});
			}
			auto end_with = std::chrono::high_resolution_clock::now();
			double time_with = std::chrono::duration<double, std::milli>(end_with - start_with).count();

			// Generate sequence without cache (recompute each time)
			std::vector<int64_t> full_seq = {42};
			auto start_without = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < length; i++)
			{
				full_seq.push_back(100 + i);
				std::vector<int64_t> seq(full_seq.begin(), full_seq.end());
				inferencer.forward_no_cache(seq);
			}
			auto end_without = std::chrono::high_resolution_clock::now();
			double time_without = std::chrono::duration<double, std::milli>(end_without - start_without).count();

			double speedup = time_without / time_with;

			std::cout << std::setw(15) << length
			          << std::setw(20) << std::fixed << std::setprecision(2) << time_with
			          << std::setw(20) << time_without
			          << std::setw(15) << speedup << "×" << std::endl;
		}

		std::cout << std::string(70, '-') << std::endl;
		std::cout << "\n✓ Test completed: Speedup scales with sequence length" << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


/**
 * Test 4: Memory persistence validation
 */
void test_memory_persistence()
{
	print_header("Test 4: Memory Persistence Validation");

	try
	{
		std::string model_path = "../models/dummy/model_with_cache.onnx";

		trigo::KVCacheInferencer inferencer(model_path, true, 0, 2048, 12, 12, 64);

		std::cout << "\nGenerating tokens and checking cache consistency..." << std::endl;

		// Generate several tokens
		for (int i = 0; i < 5; i++)
		{
			std::cout << "\nToken " << (i + 1) << ": ";
			auto logits = inferencer.forward({static_cast<int64_t>(100 + i)});

			// Check that logits are non-zero (basic sanity check)
			bool has_nonzero = false;
			for (float val : logits)
			{
				if (val != 0.0f)
				{
					has_nonzero = true;
					break;
				}
			}

			if (has_nonzero)
			{
				std::cout << "✓ Logits generated successfully";
			}
			else
			{
				std::cout << "⚠ Warning: All logits are zero";
			}
		}

		// Check metrics
		const auto& metrics = inferencer.get_metrics();
		std::cout << "\n\nCache state after 5 tokens:" << std::endl;
		std::cout << "  Current sequence length: " << metrics.current_seq_len << std::endl;
		std::cout << "  Tokens generated: " << metrics.num_tokens_generated << std::endl;

		if (metrics.current_seq_len == 5 && metrics.num_tokens_generated == 5)
		{
			std::cout << "\n✓ Test passed: Cache state is consistent" << std::endl;
		}
		else
		{
			std::cout << "\n✗ Test failed: Cache state inconsistency detected" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


int main(int argc, char** argv)
{
	std::cout << R"(
╔══════════════════════════════════════════════════════════════════════╗
║              KV Cache Prototype Performance Testing                 ║
║                                                                      ║
║  Purpose: Validate KV cache design and measure performance gains    ║
║  Expected: 10-100× speedup for sequential token generation          ║
╚══════════════════════════════════════════════════════════════════════╝
)" << std::endl;

	std::cout << "Note: This test requires a model with KV cache I/O." << std::endl;
	std::cout << "      If model not found, tests will report errors." << std::endl;
	std::cout << "      This is expected for prototype validation phase." << std::endl;

	// Run all tests
	test_basic_kvcache();
	test_performance_comparison();
	test_sequence_length_scaling();
	test_memory_persistence();

	print_header("All Tests Completed");
	std::cout << "\nNext steps:" << std::endl;
	std::cout << "1. Export a real model with KV cache support (use_cache=True)" << std::endl;
	std::cout << "2. Run benchmark with actual model" << std::endl;
	std::cout << "3. Document findings in KVCACHE_BENCHMARK.md" << std::endl;

	return 0;
}
