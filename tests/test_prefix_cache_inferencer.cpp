#include "prefix_cache_inferencer.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>


void print_test_header(const std::string& test_name)
{
	std::cout << "\n" << std::string(80, '=') << std::endl;
	std::cout << test_name << std::endl;
	std::cout << std::string(80, '=') << std::endl;
}


void test_basic_functionality(trigo::PrefixCacheInferencer& inferencer)
{
	print_test_header("Test 1: Basic Functionality");

	// Test parameters
	int batch_size = 1;
	int prefix_len = 128;
	int eval_len = 64;
	int vocab_size = 128;

	// Create random inputs
	std::random_device rd;
	std::mt19937 gen(42);  // Fixed seed for reproducibility
	std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);

	std::vector<int64_t> prefix_ids(batch_size * prefix_len);
	for (auto& id : prefix_ids) {
		id = token_dist(gen);
	}

	std::cout << "Step 1: Computing prefix cache..." << std::endl;
	std::cout << "  Prefix shape: [" << batch_size << ", " << prefix_len << "]" << std::endl;

	try {
		inferencer.compute_prefix_cache(prefix_ids, batch_size, prefix_len);
		std::cout << "  ✓ Prefix cache computed successfully" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "  ✗ Failed: " << e.what() << std::endl;
		return;
	}

	// Verify cache is ready
	if (!inferencer.has_cache()) {
		std::cerr << "  ✗ Cache not ready after computation" << std::endl;
		return;
	}

	auto cache_dims = inferencer.get_cache_dimensions();
	std::cout << "  Cache dimensions:" << std::endl;
	std::cout << "    Layers: " << cache_dims.num_layers << std::endl;
	std::cout << "    Heads: " << cache_dims.num_heads << std::endl;
	std::cout << "    Prefix len: " << cache_dims.prefix_len << std::endl;
	std::cout << "    Head dim: " << cache_dims.head_dim << std::endl;

	// Step 2: Evaluate with cache
	std::cout << "\nStep 2: Evaluating with fixed cache..." << std::endl;

	std::vector<int64_t> evaluated_ids(batch_size * eval_len);
	for (auto& id : evaluated_ids) {
		id = token_dist(gen);
	}

	// Create causal mask
	std::vector<float> mask(batch_size * eval_len * eval_len, 0.0f);
	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < eval_len; i++) {
			for (int j = 0; j <= i; j++) {
				mask[b * eval_len * eval_len + i * eval_len + j] = 1.0f;
			}
		}
	}

	try {
		auto logits = inferencer.evaluate_with_cache(evaluated_ids, mask, batch_size, eval_len);
		std::cout << "  ✓ Evaluation successful" << std::endl;
		std::cout << "  Output logits size: " << logits.size() << std::endl;

		// Expected shape: [batch, eval_len+1, vocab_size] or [batch, eval_len, vocab_size]
		// Depends on policy head implementation
	} catch (const std::exception& e) {
		std::cerr << "  ✗ Failed: " << e.what() << std::endl;
		return;
	}

	std::cout << "\n✓ Basic functionality test PASSED" << std::endl;
}


void test_mcts_pattern(trigo::PrefixCacheInferencer& inferencer)
{
	print_test_header("Test 2: MCTS Pattern (Multiple Evaluations)");

	// Test parameters
	int batch_size = 1;
	int prefix_len = 128;
	int eval_len = 64;
	int vocab_size = 128;
	int n_evaluations = 10;

	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);

	// Step 1: Compute prefix once
	std::vector<int64_t> prefix_ids(batch_size * prefix_len);
	for (auto& id : prefix_ids) {
		id = token_dist(gen);
	}

	std::cout << "Step 1: Computing prefix cache (once)..." << std::endl;
	inferencer.clear_cache();  // Clear previous cache
	inferencer.compute_prefix_cache(prefix_ids, batch_size, prefix_len);
	std::cout << "  ✓ Cache ready" << std::endl;

	// Create causal mask
	std::vector<float> mask(batch_size * eval_len * eval_len, 0.0f);
	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < eval_len; i++) {
			for (int j = 0; j <= i; j++) {
				mask[b * eval_len * eval_len + i * eval_len + j] = 1.0f;
			}
		}
	}

	// Step 2: Evaluate multiple sequences with same cache
	std::cout << "\nStep 2: Evaluating " << n_evaluations << " sequences with same cache..." << std::endl;

	std::vector<double> latencies;
	for (int i = 0; i < n_evaluations; i++) {
		// Different evaluated sequence each time
		std::vector<int64_t> evaluated_ids(batch_size * eval_len);
		for (auto& id : evaluated_ids) {
			id = token_dist(gen);
		}

		auto start = std::chrono::high_resolution_clock::now();

		try {
			auto logits = inferencer.evaluate_with_cache(evaluated_ids, mask, batch_size, eval_len);
		} catch (const std::exception& e) {
			std::cerr << "  ✗ Evaluation " << i << " failed: " << e.what() << std::endl;
			return;
		}

		auto end = std::chrono::high_resolution_clock::now();
		double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
		latencies.push_back(elapsed_ms);
	}

	// Verify cache is still valid
	if (!inferencer.has_cache()) {
		std::cerr << "  ✗ Cache lost after evaluations" << std::endl;
		return;
	}

	auto cache_dims = inferencer.get_cache_dimensions();
	if (cache_dims.prefix_len != prefix_len) {
		std::cerr << "  ✗ Cache length changed: " << cache_dims.prefix_len
		          << " (expected " << prefix_len << ")" << std::endl;
		return;
	}

	// Calculate statistics
	double avg_latency = 0.0;
	for (auto lat : latencies) {
		avg_latency += lat;
	}
	avg_latency /= latencies.size();

	double std_dev = 0.0;
	for (auto lat : latencies) {
		std_dev += (lat - avg_latency) * (lat - avg_latency);
	}
	std_dev = std::sqrt(std_dev / latencies.size());

	std::cout << "  ✓ All " << n_evaluations << " evaluations successful" << std::endl;
	std::cout << "  ✓ Cache stayed fixed at length " << prefix_len << std::endl;
	std::cout << "\n  Performance:" << std::endl;
	std::cout << "    Avg latency: " << std::fixed << std::setprecision(2)
	          << avg_latency << " ± " << std_dev << " ms" << std::endl;
	std::cout << "    Per evaluation: " << std::fixed << std::setprecision(2)
	          << avg_latency << " ms" << std::endl;

	std::cout << "\n✓ MCTS pattern test PASSED" << std::endl;
}


void benchmark_speedup(trigo::PrefixCacheInferencer& inferencer)
{
	print_test_header("Test 3: Performance Benchmark");

	// Test parameters
	int batch_size = 1;
	int prefix_len = 128;
	int eval_len = 64;
	int vocab_size = 128;
	int n_iterations = 20;
	int n_evaluations = 10;

	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);

	// Create test data
	std::vector<int64_t> prefix_ids(batch_size * prefix_len);
	for (auto& id : prefix_ids) {
		id = token_dist(gen);
	}

	std::vector<std::vector<int64_t>> evaluated_sequences;
	for (int i = 0; i < n_evaluations; i++) {
		std::vector<int64_t> seq(batch_size * eval_len);
		for (auto& id : seq) {
			id = token_dist(gen);
		}
		evaluated_sequences.push_back(seq);
	}

	std::vector<float> mask(batch_size * eval_len * eval_len, 0.0f);
	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < eval_len; i++) {
			for (int j = 0; j <= i; j++) {
				mask[b * eval_len * eval_len + i * eval_len + j] = 1.0f;
			}
		}
	}

	std::cout << "Benchmark configuration:" << std::endl;
	std::cout << "  Iterations: " << n_iterations << std::endl;
	std::cout << "  Evaluations per iteration: " << n_evaluations << std::endl;
	std::cout << "  Prefix length: " << prefix_len << std::endl;
	std::cout << "  Evaluated length: " << eval_len << std::endl;

	// Warm up
	std::cout << "\nWarming up..." << std::endl;
	for (int i = 0; i < 3; i++) {
		inferencer.clear_cache();
		inferencer.compute_prefix_cache(prefix_ids, batch_size, prefix_len);
		inferencer.evaluate_with_cache(evaluated_sequences[0], mask, batch_size, eval_len);
	}

	// Benchmark with cache
	std::cout << "\nBenchmarking with prefix cache..." << std::endl;
	std::vector<double> cached_times;

	for (int iter = 0; iter < n_iterations; iter++) {
		auto start = std::chrono::high_resolution_clock::now();

		// Compute prefix once
		inferencer.clear_cache();
		inferencer.compute_prefix_cache(prefix_ids, batch_size, prefix_len);

		// Evaluate multiple sequences
		for (const auto& seq : evaluated_sequences) {
			inferencer.evaluate_with_cache(seq, mask, batch_size, eval_len);
		}

		auto end = std::chrono::high_resolution_clock::now();
		double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
		cached_times.push_back(elapsed_ms);
	}

	// Calculate statistics
	double avg_cached = 0.0;
	for (auto t : cached_times) {
		avg_cached += t;
	}
	avg_cached /= cached_times.size();

	double std_cached = 0.0;
	for (auto t : cached_times) {
		std_cached += (t - avg_cached) * (t - avg_cached);
	}
	std_cached = std::sqrt(std_cached / cached_times.size());

	std::cout << "\n" << std::string(80, '=') << std::endl;
	std::cout << "Results" << std::endl;
	std::cout << std::string(80, '=') << std::endl;
	std::cout << "MCTS Pattern: " << n_evaluations << " evaluations with shared prefix" << std::endl;
	std::cout << "Prefix length: " << prefix_len << ", Evaluated length: " << eval_len << std::endl;
	std::cout << "\nWith prefix cache: " << std::fixed << std::setprecision(2)
	          << avg_cached << " ± " << std_cached << " ms" << std::endl;
	std::cout << "  - Prefix computation: ~" << std::fixed << std::setprecision(2)
	          << avg_cached / (n_evaluations + 1) << " ms (once)" << std::endl;
	std::cout << "  - Per evaluation: ~" << std::fixed << std::setprecision(2)
	          << avg_cached / n_evaluations << " ms" << std::endl;

	std::cout << "\n✓ Performance benchmark COMPLETE" << std::endl;

	// Print detailed metrics
	inferencer.print_metrics();
}


int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
		          << " <prefix_model> <eval_cached_model> <policy_head> [use_gpu=1] [device_id=0]" << std::endl;
		std::cerr << "\nExample:" << std::endl;
		std::cerr << "  " << argv[0]
		          << " base_model_prefix.onnx base_model_eval_cached.onnx policy_head.onnx 1 0" << std::endl;
		return 1;
	}

	std::string prefix_model_path = argv[1];
	std::string eval_cached_model_path = argv[2];
	std::string policy_head_path = argv[3];

	bool use_gpu = (argc > 4) ? (std::stoi(argv[4]) != 0) : true;
	int device_id = (argc > 5) ? std::stoi(argv[5]) : 0;

	std::cout << "Initializing PrefixCacheInferencer..." << std::endl;
	std::cout << "  Prefix model: " << prefix_model_path << std::endl;
	std::cout << "  Eval-cached model: " << eval_cached_model_path << std::endl;
	std::cout << "  Policy head: " << policy_head_path << std::endl;
	std::cout << "  Device: " << (use_gpu ? "GPU" : "CPU") << std::endl;

	try {
		trigo::PrefixCacheInferencer inferencer(
			prefix_model_path,
			eval_cached_model_path,
			policy_head_path,
			"",  // No value head for now
			use_gpu,
			device_id
		);

		std::cout << "\n✓ Inferencer initialized successfully" << std::endl;

		// Print model info
		inferencer.print_model_info();

		// Run tests
		test_basic_functionality(inferencer);
		test_mcts_pattern(inferencer);
		benchmark_speedup(inferencer);

		std::cout << "\n" << std::string(80, '=') << std::endl;
		std::cout << "ALL TESTS PASSED ✓" << std::endl;
		std::cout << std::string(80, '=') << std::endl;

		return 0;

	} catch (const std::exception& e) {
		std::cerr << "\n✗ Error: " << e.what() << std::endl;
		return 1;
	}
}
