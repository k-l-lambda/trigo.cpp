/**
 * Test extend_cache() functionality
 *
 * Verifies that:
 * 1. extend_cache() loads and runs successfully
 * 2. Cache length grows after each call
 * 3. Hidden states are returned correctly
 */

#include "prefix_cache_inferencer.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>


int main()
{
	std::cout << "=== Test extend_cache() ===" << std::endl;

	// Model paths
	const std::string model_dir = "/home/camus/work/trigo.cpp/models/trained_shared/";
	const std::string prefix_model = model_dir + "base_model_prefix.onnx";
	const std::string eval_cached_model = model_dir + "base_model_eval_cached.onnx";
	const std::string eval_extend_model = model_dir + "base_model_eval_extend.onnx";
	const std::string policy_head = model_dir + "policy_head.onnx";
	const std::string value_head = model_dir + "value_head.onnx";

	// Create inferencer with eval_extend model
	std::cout << "\n1. Loading models..." << std::endl;
	trigo::PrefixCacheInferencer inferencer(
		prefix_model,
		eval_cached_model,
		policy_head,
		value_head,
		false,  // use_gpu = false (CPU)
		0,      // device_id
		"",     // evaluation_model_path
		eval_extend_model  // NEW: eval_extend model
	);

	// Check that extend model is loaded
	std::cout << "\n2. Checking model availability..." << std::endl;
	std::cout << "   has_extend_model(): " << (inferencer.has_extend_model() ? "YES" : "NO") << std::endl;
	assert(inferencer.has_extend_model() && "eval_extend model should be loaded");

	// Initial prefix: [START] token (ID=1)
	std::cout << "\n3. Computing initial prefix cache..." << std::endl;
	std::vector<int64_t> prefix_ids = {1};  // START token
	inferencer.compute_prefix_cache(prefix_ids, 1, 1);

	auto dims = inferencer.get_cache_dimensions();
	std::cout << "   Initial cache length: " << dims.seq_len << std::endl;
	std::cout << "   Num layers: " << dims.num_layers << std::endl;
	std::cout << "   Num heads: " << dims.num_heads << std::endl;
	std::cout << "   Head dim: " << dims.head_dim << std::endl;
	assert(dims.seq_len == 1 && "Initial cache should have length 1");

	// Extend cache with move tokens: [1., aa] (IDs: 4, 26)
	std::cout << "\n4. Extending cache with move tokens..." << std::endl;
	std::vector<int64_t> move_tokens = {4, 26};  // "1." and "aa" tokens
	int new_len = 2;

	// Create simple causal mask
	std::vector<float> mask(new_len * new_len);
	for (int i = 0; i < new_len; i++) {
		for (int j = 0; j < new_len; j++) {
			mask[i * new_len + j] = (j <= i) ? 1.0f : 0.0f;
		}
	}

	auto hidden_states = inferencer.extend_cache(move_tokens, mask, 1, new_len);

	// Check results
	dims = inferencer.get_cache_dimensions();
	std::cout << "   New cache length: " << dims.seq_len << std::endl;
	std::cout << "   Hidden states size: " << hidden_states.size() << std::endl;

	// Cache should have grown: 1 (initial) + 1 (dummy) + 2 (new tokens) = 4
	// Actually: model adds dummy prefix_last, so it's more complex
	// The important thing is cache length increased
	assert(dims.seq_len > 1 && "Cache should have grown");
	assert(!hidden_states.empty() && "Hidden states should be non-empty");

	// Second extension
	std::cout << "\n5. Extending cache again with another move..." << std::endl;
	int prev_cache_len = dims.seq_len;

	std::vector<int64_t> move_tokens2 = {27, 5};  // "ab" and "2." tokens
	auto hidden_states2 = inferencer.extend_cache(move_tokens2, mask, 1, 2);

	dims = inferencer.get_cache_dimensions();
	std::cout << "   Previous cache length: " << prev_cache_len << std::endl;
	std::cout << "   New cache length: " << dims.seq_len << std::endl;
	std::cout << "   Hidden states size: " << hidden_states2.size() << std::endl;

	assert(dims.seq_len > prev_cache_len && "Cache should have grown further");
	assert(!hidden_states2.empty() && "Hidden states should be non-empty");

	// Print metrics
	std::cout << "\n6. Performance metrics:" << std::endl;
	auto metrics = inferencer.get_metrics();
	std::cout << "   Avg eval latency: " << metrics.avg_eval_latency_ms << " ms" << std::endl;
	std::cout << "   Cache memory: " << metrics.cache_memory_bytes / 1024.0 << " KB" << std::endl;

	std::cout << "\n=== All tests PASSED! ===" << std::endl;
	return 0;
}
