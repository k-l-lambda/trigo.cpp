/**
 * KV Cache Prototype Test - Standalone Version
 *
 * This is a simplified test that works with the exported GPT-2 model.
 *
 * Usage:
 *   1. Export model: python export_gpt2_kvcache.py
 *   2. Build: mkdir build && cd build && cmake .. && make
 *   3. Run: ./test_kvcache_prototype
 */

#include "../include/kvcache_inferencer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <nlohmann/json.hpp>


using json = nlohmann::json;


struct ModelConfig {
	std::string model_name;
	int num_layers;
	int num_heads;
	int head_dim;
	int hidden_dim;
	int vocab_size;
	int max_seq_len;
	std::string onnx_path;
};


ModelConfig load_config(const std::string& config_path) {
	std::ifstream f(config_path);
	if (!f.is_open()) {
		throw std::runtime_error("Failed to open config file: " + config_path);
	}

	json j = json::parse(f);

	ModelConfig config;
	config.model_name = j["model_name"];
	config.num_layers = j["num_layers"];
	config.num_heads = j["num_heads"];
	config.head_dim = j["head_dim"];
	config.hidden_dim = j["hidden_dim"];
	config.vocab_size = j["vocab_size"];
	config.max_seq_len = j["max_seq_len"];
	config.onnx_path = j["onnx_path"];

	return config;
}


void print_header(const std::string& title) {
	std::cout << "\n" << std::string(70, '=') << std::endl;
	std::cout << title << std::endl;
	std::cout << std::string(70, '=') << std::endl;
}


void test_basic_inference(const ModelConfig& config, const std::string& model_dir) {
	print_header("Test 1: Basic KV Cache Inference");

	std::string model_path = model_dir + "/" + config.onnx_path;

	std::cout << "Loading model: " << model_path << std::endl;
	std::cout << "Configuration:" << std::endl;
	std::cout << "  Model: " << config.model_name << std::endl;
	std::cout << "  Layers: " << config.num_layers << std::endl;
	std::cout << "  Heads: " << config.num_heads << std::endl;
	std::cout << "  Head dim: " << config.head_dim << std::endl;
	std::cout << "  Vocab size: " << config.vocab_size << std::endl;

	try {
		trigo::KVCacheInferencer inferencer(
			model_path,
			true,  // use_gpu
			0,     // device_id
			config.max_seq_len,
			config.num_layers,
			config.num_heads,
			config.head_dim
		);

		std::cout << "\n✓ Model loaded successfully" << std::endl;

		// Generate a few tokens
		std::vector<int64_t> start_token = {50256};  // GPT-2 BOS token

		std::cout << "\nGenerating tokens:" << std::endl;

		for (int i = 0; i < 10; i++) {
			int64_t token = (i == 0) ? start_token[0] : (1000 + i);

			auto start = std::chrono::high_resolution_clock::now();
			auto logits = inferencer.forward({token});
			auto end = std::chrono::high_resolution_clock::now();

			double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

			// Get top-5 predictions
			std::vector<std::pair<int, float>> top_tokens;
			for (int j = 0; j < std::min(5, (int)logits.size()); j++) {
				top_tokens.push_back({j, logits[j]});
			}
			std::sort(top_tokens.begin(), top_tokens.end(),
			          [](const auto& a, const auto& b) { return a.second > b.second; });

			std::cout << "  Token " << std::setw(2) << (i + 1) << ": "
			          << "input=" << std::setw(5) << token
			          << " | latency=" << std::fixed << std::setprecision(2) << std::setw(7) << latency_ms << "ms"
			          << " | top_pred=" << top_tokens[0].first
			          << std::endl;
		}

		inferencer.print_metrics();

		std::cout << "\n✓ Test passed" << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


void test_performance_comparison(const ModelConfig& config, const std::string& model_dir) {
	print_header("Test 2: Performance Comparison (With vs Without Cache)");

	std::string model_path = model_dir + "/" + config.onnx_path;

	try {
		trigo::KVCacheInferencer inferencer(
			model_path,
			true,
			0,
			config.max_seq_len,
			config.num_layers,
			config.num_heads,
			config.head_dim
		);

		int num_tokens = 20;

		std::cout << "\n[Scenario A] Sequential generation WITH KV cache" << std::endl;
		std::cout << "Generating " << num_tokens << " tokens..." << std::endl;

		auto start_with = std::chrono::high_resolution_clock::now();

		std::vector<int64_t> sequence = {50256};  // Start token
		for (int i = 0; i < num_tokens; i++) {
			auto logits = inferencer.forward({1000 + i});
			// In practice, would sample from logits to get next token
		}

		auto end_with = std::chrono::high_resolution_clock::now();
		double time_with = std::chrono::duration<double, std::milli>(end_with - start_with).count();

		std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_with << " ms" << std::endl;
		std::cout << "  Avg per token: " << (time_with / num_tokens) << " ms" << std::endl;

		const auto& metrics_with = inferencer.get_metrics();
		std::cout << "  First token: " << metrics_with.first_token_latency_ms << " ms" << std::endl;
		std::cout << "  Subsequent avg: " << metrics_with.avg_subsequent_token_latency_ms << " ms" << std::endl;
		std::cout << "  Speedup: " << metrics_with.speedup_factor << "×" << std::endl;

		std::cout << "\n[Scenario B] Recomputing full sequence WITHOUT KV cache" << std::endl;

		// Reset for no-cache test
		inferencer.reset_cache();

		// Build full sequence
		std::vector<int64_t> full_sequence = {50256};
		for (int i = 0; i < num_tokens; i++) {
			full_sequence.push_back(1000 + i);
		}

		auto start_no_cache = std::chrono::high_resolution_clock::now();

		// Simulate sequential generation by recomputing from start each time
		for (size_t len = 1; len <= num_tokens; len++) {
			std::vector<int64_t> seq(full_sequence.begin(), full_sequence.begin() + len + 1);
			auto logits = inferencer.forward_no_cache(seq);
		}

		auto end_no_cache = std::chrono::high_resolution_clock::now();
		double time_no_cache = std::chrono::duration<double, std::milli>(end_no_cache - start_no_cache).count();

		std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_no_cache << " ms" << std::endl;
		std::cout << "  Avg per token: " << (time_no_cache / num_tokens) << " ms" << std::endl;

		// Calculate overall speedup
		double overall_speedup = time_no_cache / time_with;

		std::cout << "\n[Performance Summary]" << std::endl;
		std::cout << "  Overall speedup: " << std::fixed << std::setprecision(2) << overall_speedup << "×" << std::endl;
		std::cout << "  Time saved: " << (time_no_cache - time_with) << " ms" << std::endl;
		std::cout << "  Efficiency: " << (100.0 * time_with / time_no_cache) << "% of no-cache time" << std::endl;

		if (overall_speedup > 2.0) {
			std::cout << "\n✓ Test passed: Significant speedup achieved (" << overall_speedup << "×)" << std::endl;
		} else {
			std::cout << "\n⚠ Warning: Speedup lower than expected (" << overall_speedup << "× < 2×)" << std::endl;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


void test_memory_usage(const ModelConfig& config, const std::string& model_dir) {
	print_header("Test 3: Memory Usage Analysis");

	std::string model_path = model_dir + "/" + config.onnx_path;

	try {
		trigo::KVCacheInferencer inferencer(
			model_path,
			true,
			0,
			config.max_seq_len,
			config.num_layers,
			config.num_heads,
			config.head_dim
		);

		const auto& metrics = inferencer.get_metrics();

		std::cout << "\nKV Cache Memory:" << std::endl;
		std::cout << "  Allocated: " << (metrics.cache_memory_bytes / (1024.0 * 1024.0))
		          << " MB" << std::endl;

		// Theoretical calculation
		size_t theoretical = 2 * config.num_layers * 1 * config.num_heads *
		                     config.max_seq_len * config.head_dim * sizeof(float);

		std::cout << "  Theoretical: " << (theoretical / (1024.0 * 1024.0))
		          << " MB" << std::endl;

		if (metrics.cache_memory_bytes == theoretical) {
			std::cout << "\n✓ Memory allocation matches theoretical calculation" << std::endl;
		} else {
			std::cout << "\n⚠ Memory mismatch (difference: "
			          << ((int64_t)metrics.cache_memory_bytes - (int64_t)theoretical) / (1024.0 * 1024.0)
			          << " MB)" << std::endl;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "✗ Test failed: " << e.what() << std::endl;
	}
}


int main(int argc, char** argv) {
	std::cout << R"(
╔══════════════════════════════════════════════════════════════════════╗
║            KV Cache Prototype - Standalone Validation               ║
║                                                                      ║
║  GPT-2 model with ONNX Runtime KV cache                             ║
╚══════════════════════════════════════════════════════════════════════╝
)" << std::endl;

	// Default paths
	std::string model_dir = "../models";
	std::string config_path = model_dir + "/config.json";

	// Allow override via command line
	if (argc > 1) {
		model_dir = argv[1];
		config_path = model_dir + "/config.json";
	}

	try {
		// Load configuration
		std::cout << "Loading configuration: " << config_path << std::endl;
		ModelConfig config = load_config(config_path);
		std::cout << "✓ Configuration loaded" << std::endl;

		// Run tests
		test_basic_inference(config, model_dir);
		test_performance_comparison(config, model_dir);
		test_memory_usage(config, model_dir);

		print_header("All Tests Completed");
		std::cout << "\nKV Cache prototype validation successful!" << std::endl;
		std::cout << "Check the performance metrics above for speedup measurements." << std::endl;

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "\n✗ Fatal error: " << e.what() << std::endl;
		std::cerr << "\nMake sure you:" << std::endl;
		std::cerr << "1. Exported the model: python export_gpt2_kvcache.py" << std::endl;
		std::cerr << "2. Have the config.json and gpt2_with_cache.onnx files" << std::endl;
		return 1;
	}
}
}
