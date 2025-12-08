#include "prefix_cache_inferencer.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>


namespace trigo
{


PrefixCacheInferencer::PrefixCacheInferencer(
	const std::string& prefix_model_path,
	const std::string& eval_cached_model_path,
	const std::string& policy_head_path,
	const std::string& value_head_path,
	bool use_gpu,
	int device_id
)
	: env_(ORT_LOGGING_LEVEL_WARNING, "PrefixCacheInferencer")
	, use_gpu_(use_gpu)
	, device_id_(device_id)
	, cache_ready_(false)
{
	// Configure session options
	session_options_.SetIntraOpNumThreads(4);
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	// Add CUDA provider if requested
	if (use_gpu_) {
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = device_id_;
		cuda_options.arena_extend_strategy = 0;
		cuda_options.gpu_mem_limit = SIZE_MAX;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
		cuda_options.do_copy_in_default_stream = 1;

		session_options_.AppendExecutionProvider_CUDA(cuda_options);
	}

	// Load prefix model
	try {
		#ifdef _WIN32
		std::wstring prefix_path_w(prefix_model_path.begin(), prefix_model_path.end());
		prefix_session_ = std::make_unique<Ort::Session>(env_, prefix_path_w.c_str(), session_options_);
		#else
		prefix_session_ = std::make_unique<Ort::Session>(env_, prefix_model_path.c_str(), session_options_);
		#endif
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to load prefix model: " + std::string(e.what()));
	}

	// Load eval_cached model
	try {
		#ifdef _WIN32
		std::wstring eval_path_w(eval_cached_model_path.begin(), eval_cached_model_path.end());
		eval_cached_session_ = std::make_unique<Ort::Session>(env_, eval_path_w.c_str(), session_options_);
		#else
		eval_cached_session_ = std::make_unique<Ort::Session>(env_, eval_cached_model_path.c_str(), session_options_);
		#endif
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to load eval_cached model: " + std::string(e.what()));
	}

	// Load policy head
	try {
		#ifdef _WIN32
		std::wstring policy_path_w(policy_head_path.begin(), policy_head_path.end());
		policy_session_ = std::make_unique<Ort::Session>(env_, policy_path_w.c_str(), session_options_);
		#else
		policy_session_ = std::make_unique<Ort::Session>(env_, policy_head_path.c_str(), session_options_);
		#endif
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to load policy head: " + std::string(e.what()));
	}

	// Load value head (optional)
	if (!value_head_path.empty()) {
		try {
			#ifdef _WIN32
			std::wstring value_path_w(value_head_path.begin(), value_head_path.end());
			value_session_ = std::make_unique<Ort::Session>(env_, value_path_w.c_str(), session_options_);
			#else
			value_session_ = std::make_unique<Ort::Session>(env_, value_head_path.c_str(), session_options_);
			#endif
		} catch (const Ort::Exception& e) {
			std::cerr << "Warning: Failed to load value head: " << e.what() << std::endl;
		}
	}

	// Get cache dimensions from eval_cached model inputs
	// Expected inputs: evaluated_ids, evaluated_mask, past_key_0, past_value_0, past_key_1, ...
	size_t num_inputs = eval_cached_session_->GetInputCount();

	// Find cache tensors to determine dimensions
	for (size_t i = 0; i < num_inputs; i++) {
		auto input_name_ptr = eval_cached_session_->GetInputNameAllocated(i, allocator_);
		std::string input_name(input_name_ptr.get());

		if (input_name.find("past_key_0") != std::string::npos) {
			// Get shape of first cache tensor
			auto type_info = eval_cached_session_->GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto shape = tensor_info.GetShape();

			// Shape: [batch, num_heads, prefix_len, head_dim]
			cache_dims_.batch_size = static_cast<int>(shape[0]);
			cache_dims_.num_heads = static_cast<int>(shape[1]);
			cache_dims_.prefix_len = static_cast<int>(shape[2]);
			cache_dims_.head_dim = static_cast<int>(shape[3]);

			// Count number of layers
			cache_dims_.num_layers = 0;
			for (size_t j = 0; j < num_inputs; j++) {
				auto name_ptr = eval_cached_session_->GetInputNameAllocated(j, allocator_);
				std::string name(name_ptr.get());
				if (name.find("past_key_") != std::string::npos) {
					cache_dims_.num_layers++;
				}
			}

			break;
		}
	}

	std::cout << "PrefixCacheInferencer initialized:" << std::endl;
	std::cout << "  Prefix model: " << prefix_model_path << std::endl;
	std::cout << "  Eval-cached model: " << eval_cached_model_path << std::endl;
	std::cout << "  Policy head: " << policy_head_path << std::endl;
	if (value_session_) {
		std::cout << "  Value head: " << value_head_path << std::endl;
	}
	std::cout << "  Device: " << (use_gpu_ ? "GPU" : "CPU") << std::endl;
	std::cout << "  Cache dimensions: " << cache_dims_.num_layers << " layers, "
	          << cache_dims_.num_heads << " heads, "
	          << cache_dims_.head_dim << " head_dim" << std::endl;
}


void PrefixCacheInferencer::compute_prefix_cache(
	const std::vector<int64_t>& prefix_ids,
	int batch_size,
	int prefix_len
)
{
	auto start = std::chrono::high_resolution_clock::now();

	// Prepare input tensor
	std::vector<int64_t> input_shape = {batch_size, prefix_len};

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);

	auto input_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		const_cast<int64_t*>(prefix_ids.data()),
		prefix_ids.size(),
		input_shape.data(),
		input_shape.size()
	);

	// Get input name
	auto input_name_ptr = prefix_session_->GetInputNameAllocated(0, allocator_);
	const char* input_names[] = {input_name_ptr.get()};

	// Get output names (cache tensors)
	size_t num_outputs = prefix_session_->GetOutputCount();
	std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
	std::vector<const char*> output_names;

	for (size_t i = 0; i < num_outputs; i++) {
		output_name_ptrs.push_back(prefix_session_->GetOutputNameAllocated(i, allocator_));
		output_names.push_back(output_name_ptrs.back().get());
	}

	// Run inference
	auto output_tensors = prefix_session_->Run(
		Ort::RunOptions{nullptr},
		input_names,
		&input_tensor,
		1,
		output_names.data(),
		output_names.size()
	);

	// Extract cache tensors
	// Output format: cache_key_0, cache_value_0, cache_key_1, cache_value_1, ...
	cached_keys_.clear();
	cached_values_.clear();

	for (size_t i = 0; i < output_tensors.size(); i += 2) {
		// Key tensor
		auto& key_tensor = output_tensors[i];
		auto key_data = key_tensor.GetTensorData<float>();
		auto key_shape = key_tensor.GetTensorTypeAndShapeInfo().GetShape();
		size_t key_size = std::accumulate(key_shape.begin(), key_shape.end(), 1LL, std::multiplies<int64_t>());

		std::vector<float> key_vec(key_data, key_data + key_size);
		cached_keys_.push_back(std::move(key_vec));

		// Value tensor
		auto& value_tensor = output_tensors[i + 1];
		auto value_data = value_tensor.GetTensorData<float>();
		auto value_shape = value_tensor.GetTensorTypeAndShapeInfo().GetShape();
		size_t value_size = std::accumulate(value_shape.begin(), value_shape.end(), 1LL, std::multiplies<int64_t>());

		std::vector<float> value_vec(value_data, value_data + value_size);
		cached_values_.push_back(std::move(value_vec));
	}

	cache_ready_ = true;
	cache_dims_.batch_size = batch_size;
	cache_dims_.prefix_len = prefix_len;

	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

	metrics_.prefix_computation_ms = elapsed_ms;
	metrics_.num_prefix_computations++;

	// Calculate cache memory
	size_t total_cache_elements = 0;
	for (const auto& k : cached_keys_) {
		total_cache_elements += k.size();
	}
	for (const auto& v : cached_values_) {
		total_cache_elements += v.size();
	}
	metrics_.cache_memory_bytes = total_cache_elements * sizeof(float);
	metrics_.cache_length = prefix_len;
}


std::vector<float> PrefixCacheInferencer::evaluate_with_cache(
	const std::vector<int64_t>& evaluated_ids,
	const std::vector<float>& evaluated_mask,
	int batch_size,
	int eval_len
)
{
	if (!cache_ready_) {
		throw std::runtime_error("Cache not ready. Call compute_prefix_cache() first.");
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Prepare input tensors
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);

	// 1. evaluated_ids [batch, eval_len]
	std::vector<int64_t> eval_ids_shape = {batch_size, eval_len};
	auto eval_ids_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		const_cast<int64_t*>(evaluated_ids.data()),
		evaluated_ids.size(),
		eval_ids_shape.data(),
		eval_ids_shape.size()
	);

	// 2. evaluated_mask [batch, eval_len, eval_len]
	std::vector<int64_t> mask_shape = {batch_size, eval_len, eval_len};
	auto mask_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		const_cast<float*>(evaluated_mask.data()),
		evaluated_mask.size(),
		mask_shape.data(),
		mask_shape.size()
	);

	// 3. Cache tensors (past_key_i, past_value_i for each layer)
	std::vector<Ort::Value> cache_tensors;
	std::vector<int64_t> cache_shape = {
		cache_dims_.batch_size,
		cache_dims_.num_heads,
		cache_dims_.prefix_len,
		cache_dims_.head_dim
	};

	for (int i = 0; i < cache_dims_.num_layers; i++) {
		// past_key_i
		cache_tensors.push_back(Ort::Value::CreateTensor<float>(
			memory_info,
			const_cast<float*>(cached_keys_[i].data()),
			cached_keys_[i].size(),
			cache_shape.data(),
			cache_shape.size()
		));

		// past_value_i
		cache_tensors.push_back(Ort::Value::CreateTensor<float>(
			memory_info,
			const_cast<float*>(cached_values_[i].data()),
			cached_values_[i].size(),
			cache_shape.data(),
			cache_shape.size()
		));
	}

	// Prepare all input tensors
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(eval_ids_tensor));
	input_tensors.push_back(std::move(mask_tensor));
	for (auto& cache_tensor : cache_tensors) {
		input_tensors.push_back(std::move(cache_tensor));
	}

	// Get input names
	size_t num_inputs = eval_cached_session_->GetInputCount();
	std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
	std::vector<const char*> input_names;

	for (size_t i = 0; i < num_inputs; i++) {
		input_name_ptrs.push_back(eval_cached_session_->GetInputNameAllocated(i, allocator_));
		input_names.push_back(input_name_ptrs.back().get());
	}

	// Get output names
	auto output_name_ptr = eval_cached_session_->GetOutputNameAllocated(0, allocator_);
	const char* output_names[] = {output_name_ptr.get()};

	// Run eval_cached model
	auto output_tensors = eval_cached_session_->Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		input_tensors.data(),
		input_tensors.size(),
		output_names,
		1
	);

	// Get hidden states output
	auto& hidden_states_tensor = output_tensors[0];
	auto hidden_states_shape = hidden_states_tensor.GetTensorTypeAndShapeInfo().GetShape();
	// Shape: [batch, eval_len, hidden_dim]

	int hidden_dim = static_cast<int>(hidden_states_shape[2]);
	auto hidden_data = hidden_states_tensor.GetTensorData<float>();
	size_t hidden_states_size = batch_size * eval_len * hidden_dim;
	std::vector<float> hidden_states(hidden_data, hidden_data + hidden_states_size);

	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

	// Update metrics
	metrics_.num_evaluations++;
	double prev_avg = metrics_.avg_eval_latency_ms;
	metrics_.avg_eval_latency_ms = (prev_avg * (metrics_.num_evaluations - 1) + elapsed_ms) / metrics_.num_evaluations;

	// NOTE: For MCTS prefix-cache pattern, we return raw hidden states [batch, eval_len, hidden_dim]
	// The policy head expects [batch, prefix_len+eval_len, hidden_dim] which is incompatible
	// with cached mode. Caller should use standard base_model if they need policy head integration.
	// For MCTS, we typically only need the final token's logits, which can be extracted separately.
	return hidden_states;
}


float PrefixCacheInferencer::value_inference_with_cache(int value_token_id)
{
	if (!cache_ready_) {
		throw std::runtime_error("Cache not ready. Call compute_prefix_cache() first.");
	}

	if (!value_session_) {
		throw std::runtime_error("Value head not loaded. Provide value_head_path in constructor.");
	}

	// 1. Create VALUE token input [1, 1]
	std::vector<int64_t> value_ids = {value_token_id};
	std::vector<float> value_mask = {1.0f};  // Single token, trivial mask

	// 2. Get hidden states using cached inference
	auto hidden_states = evaluate_with_cache(value_ids, value_mask, 1, 1);

	// hidden_states shape: [1, 1, hidden_dim]
	int hidden_dim = static_cast<int>(hidden_states.size());

	// 3. Run value head: hidden_states → value
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);

	// Value head expects [batch, hidden_dim] (rank 2), not [batch, seq_len, hidden_dim]
	// We only have one token, so just reshape to [1, hidden_dim]
	std::vector<int64_t> hidden_shape = {1, hidden_dim};
	auto hidden_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		hidden_states.data(),
		hidden_states.size(),
		hidden_shape.data(),
		hidden_shape.size()
	);

	// Get input/output names
	auto input_name_ptr = value_session_->GetInputNameAllocated(0, allocator_);
	const char* input_names[] = {input_name_ptr.get()};

	auto output_name_ptr = value_session_->GetOutputNameAllocated(0, allocator_);
	const char* output_names[] = {output_name_ptr.get()};

	// Run value head
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(hidden_tensor));

	auto output_tensors = value_session_->Run(
		Ort::RunOptions{nullptr},
		input_names,
		input_tensors.data(),
		input_tensors.size(),
		output_names,
		1
	);

	// Get value output [1, 1]
	auto& value_tensor = output_tensors[0];
	auto value_data = value_tensor.GetTensorData<float>();

	return value_data[0];  // Single value prediction
}


std::vector<float> PrefixCacheInferencer::evaluate_standard(
	const std::vector<int64_t>& prefix_ids,
	const std::vector<int64_t>& evaluated_ids,
	const std::vector<float>& evaluated_mask,
	int batch_size,
	int prefix_len,
	int eval_len
)
{
	// This would use a standard base_model.onnx (without cache)
	// For now, throw as not implemented
	// In practice, you would load a third model (base_model.onnx) for this
	throw std::runtime_error("Standard evaluation not implemented yet. Use SharedModelInferencer for baseline.");
}


void PrefixCacheInferencer::clear_cache()
{
	cached_keys_.clear();
	cached_values_.clear();
	cache_ready_ = false;
}


void PrefixCacheInferencer::print_metrics() const
{
	std::cout << "\n========================================" << std::endl;
	std::cout << "Prefix Cache Performance Metrics" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Prefix computations: " << metrics_.num_prefix_computations << std::endl;
	std::cout << "  Avg latency: " << std::fixed << std::setprecision(2)
	          << metrics_.prefix_computation_ms << " ms" << std::endl;
	std::cout << "\nEvaluations with cache: " << metrics_.num_evaluations << std::endl;
	std::cout << "  Avg latency: " << std::fixed << std::setprecision(2)
	          << metrics_.avg_eval_latency_ms << " ms" << std::endl;
	std::cout << "\nCache info:" << std::endl;
	std::cout << "  Length: " << metrics_.cache_length << " tokens" << std::endl;
	std::cout << "  Memory: " << std::fixed << std::setprecision(2)
	          << metrics_.cache_memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

	if (metrics_.speedup_vs_standard > 0.0) {
		std::cout << "\nSpeedup vs standard: " << std::fixed << std::setprecision(2)
		          << metrics_.speedup_vs_standard << "×" << std::endl;
	}
	std::cout << "========================================\n" << std::endl;
}


void PrefixCacheInferencer::print_model_info() const
{
	std::cout << "\n========================================" << std::endl;
	std::cout << "Model Information" << std::endl;
	std::cout << "========================================" << std::endl;

	std::cout << "\nPrefix Model:" << std::endl;
	std::cout << "  Inputs: " << prefix_session_->GetInputCount() << std::endl;
	for (size_t i = 0; i < prefix_session_->GetInputCount(); i++) {
		auto name = prefix_session_->GetInputNameAllocated(i, allocator_);
		auto type_info = prefix_session_->GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto shape = tensor_info.GetShape();

		std::cout << "    " << name.get() << ": [";
		for (size_t j = 0; j < shape.size(); j++) {
			std::cout << shape[j];
			if (j < shape.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}

	std::cout << "  Outputs: " << prefix_session_->GetOutputCount() << " (cache tensors)" << std::endl;

	std::cout << "\nEval-Cached Model:" << std::endl;
	std::cout << "  Inputs: " << eval_cached_session_->GetInputCount() << std::endl;
	std::cout << "  Outputs: " << eval_cached_session_->GetOutputCount() << std::endl;

	std::cout << "\nCache dimensions:" << std::endl;
	std::cout << "  Layers: " << cache_dims_.num_layers << std::endl;
	std::cout << "  Heads: " << cache_dims_.num_heads << std::endl;
	std::cout << "  Head dim: " << cache_dims_.head_dim << std::endl;
	std::cout << "  Prefix len: " << cache_dims_.prefix_len << std::endl;

	std::cout << "========================================\n" << std::endl;
}


std::vector<float> PrefixCacheInferencer::create_causal_mask(int eval_len)
{
	std::vector<float> mask(eval_len * eval_len, 0.0f);

	for (int i = 0; i < eval_len; i++) {
		for (int j = 0; j <= i; j++) {
			mask[i * eval_len + j] = 1.0f;
		}
	}

	return mask;
}


std::vector<float> PrefixCacheInferencer::expand_mask_to_batch(
	const std::vector<float>& mask,
	int batch_size,
	int eval_len
)
{
	std::vector<float> batched_mask;
	batched_mask.reserve(batch_size * eval_len * eval_len);

	for (int b = 0; b < batch_size; b++) {
		batched_mask.insert(batched_mask.end(), mask.begin(), mask.end());
	}

	return batched_mask;
}


std::vector<float> PrefixCacheInferencer::policy_inference_from_hidden(
	const std::vector<float>& hidden_states,
	int batch_size,
	int seq_len,
	int hidden_dim
)
{
	// Create input tensor for policy head
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	std::vector<int64_t> hidden_shape = {batch_size, seq_len, hidden_dim};

	// Make a mutable copy of hidden_states for CreateTensor
	std::vector<float> hidden_copy = hidden_states;

	Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		hidden_copy.data(),
		hidden_copy.size(),
		hidden_shape.data(),
		hidden_shape.size()
	);

	// Prepare input/output names
	std::vector<const char*> input_names = {"hidden_states"};
	std::vector<const char*> output_names = {"logits"};

	// Prepare inputs
	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(hidden_tensor));

	// Run policy head
	auto outputs = policy_session_->Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		inputs.data(),
		inputs.size(),
		output_names.data(),
		output_names.size()
	);

	// Extract logits
	float* logits_ptr = outputs[0].GetTensorMutableData<float>();
	auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int output_seq_len = logits_shape[1];
	int vocab_size = logits_shape[2];

	size_t logits_size = batch_size * output_seq_len * vocab_size;
	std::vector<float> logits(logits_ptr, logits_ptr + logits_size);

	return logits;
}


}  // namespace trigo
