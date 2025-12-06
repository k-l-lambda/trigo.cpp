/**
 * KV Cache Inferencer - Implementation
 */

#include "kvcache_inferencer.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>


namespace trigo
{


KVCacheInferencer::KVCacheInferencer(
	const std::string& model_path,
	bool use_gpu,
	int device_id,
	int max_seq_len,
	int num_layers,
	int num_heads,
	int head_dim
)
	: env_(ORT_LOGGING_LEVEL_WARNING, "KVCacheInferencer"),
	  use_gpu_(use_gpu),
	  device_id_(device_id),
	  max_seq_len_(max_seq_len),
	  num_layers_(num_layers),
	  num_heads_(num_heads),
	  head_dim_(head_dim),
	  hidden_dim_(num_heads * head_dim),
	  current_seq_len_(0)
{
	// Check environment variable to force CPU
	const char* force_cpu = std::getenv("TRIGO_FORCE_CPU");
	if (force_cpu != nullptr && std::string(force_cpu) == "1")
	{
		use_gpu_ = false;
		std::cout << "✓ TRIGO_FORCE_CPU=1 detected, using CPU execution provider" << std::endl;
	}

	// Configure session options
	session_options_.SetIntraOpNumThreads(4);
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// Enable CUDA if requested
	if (use_gpu_)
	{
		try
		{
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = device_id_;
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
			cuda_options.do_copy_in_default_stream = 1;

			session_options_.AppendExecutionProvider_CUDA(cuda_options);
			std::cout << "✓ CUDA execution provider enabled (device " << device_id_ << ")" << std::endl;
		}
		catch (const Ort::Exception& e)
		{
			std::cerr << "⚠ Warning: Could not enable CUDA provider: " << e.what() << std::endl;
			std::cerr << "  Falling back to CPU" << std::endl;
			use_gpu_ = false;
		}
	}

	// Load ONNX model
	try
	{
		std::cout << "Loading ONNX model: " << model_path << std::endl;

		#ifdef _WIN32
		std::wstring path_w(model_path.begin(), model_path.end());
		session_ = std::make_unique<Ort::Session>(env_, path_w.c_str(), session_options_);
		#else
		session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
		#endif

		std::cout << "✓ Model loaded successfully" << std::endl;
	}
	catch (const Ort::Exception& e)
	{
		throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
	}

	// Create memory info objects
	memory_info_cpu_ = std::make_unique<Ort::MemoryInfo>(
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
	);

	if (use_gpu_)
	{
		memory_info_gpu_ = std::make_unique<Ort::MemoryInfo>(
			"Cuda", OrtDeviceAllocator, device_id_, OrtMemTypeDefault
		);
	}

	// Initialize KV cache
	init_kv_cache();

	// Initialize metrics
	metrics_.max_seq_len = max_seq_len_;
	metrics_.cache_memory_bytes = 2 * num_layers_ * 1 * num_heads_ * max_seq_len_ * head_dim_ * sizeof(float);

	std::cout << "✓ KVCacheInferencer initialized" << std::endl;
	std::cout << "  Max sequence length: " << max_seq_len_ << std::endl;
	std::cout << "  KV cache memory: " << (metrics_.cache_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
}


KVCacheInferencer::~KVCacheInferencer()
{
	// Ort::Value has RAII, will clean up GPU memory automatically
}


void KVCacheInferencer::init_kv_cache()
{
	std::cout << "Initializing KV cache tensors..." << std::endl;

	// Cache shape: [batch=1, num_heads, max_seq_len, head_dim]
	std::vector<int64_t> cache_shape = {1, static_cast<int64_t>(num_heads_),
	                                     static_cast<int64_t>(max_seq_len_),
	                                     static_cast<int64_t>(head_dim_)};

	// Get allocator for the target device
	OrtAllocator* target_allocator = nullptr;
	if (use_gpu_)
	{
		// Get CUDA allocator
		Ort::ThrowOnError(Ort::GetApi().GetAllocatorWithDefaultOptions(&target_allocator));
	}
	else
	{
		// Get CPU allocator
		Ort::ThrowOnError(Ort::GetApi().GetAllocatorWithDefaultOptions(&target_allocator));
	}

	// Allocate KV cache tensors for each layer
	for (int i = 0; i < num_layers_; i++)
	{
		// Create key cache tensor
		Ort::Value key_cache = Ort::Value::CreateTensor<float>(
			target_allocator,
			cache_shape.data(),
			cache_shape.size()
		);

		// Create value cache tensor
		Ort::Value value_cache = Ort::Value::CreateTensor<float>(
			target_allocator,
			cache_shape.data(),
			cache_shape.size()
		);

		// Initialize to zero (important for first token)
		float* key_ptr = key_cache.GetTensorMutableData<float>();
		float* value_ptr = value_cache.GetTensorMutableData<float>();
		size_t cache_size = num_heads_ * max_seq_len_ * head_dim_;

		if (use_gpu_)
		{
			#ifdef USE_CUDA
			cudaMemset(key_ptr, 0, cache_size * sizeof(float));
			cudaMemset(value_ptr, 0, cache_size * sizeof(float));
			#else
			std::cerr << "⚠ Warning: CUDA requested but not compiled with USE_CUDA" << std::endl;
			std::memset(key_ptr, 0, cache_size * sizeof(float));
			std::memset(value_ptr, 0, cache_size * sizeof(float));
			#endif
		}
		else
		{
			std::memset(key_ptr, 0, cache_size * sizeof(float));
			std::memset(value_ptr, 0, cache_size * sizeof(float));
		}

		past_key_cache_.push_back(std::move(key_cache));
		past_value_cache_.push_back(std::move(value_cache));
	}

	std::cout << "✓ Initialized " << num_layers_ << " layers of KV cache" << std::endl;
}


std::vector<float> KVCacheInferencer::forward(const std::vector<int64_t>& input_ids)
{
	auto start = std::chrono::high_resolution_clock::now();

	bool is_first_token = (current_seq_len_ == 0);

	// Create IOBinding for zero-copy GPU operations
	Ort::IoBinding io_binding(*session_);

	// 1. Bind input_ids
	std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
	Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
		*memory_info_cpu_,
		const_cast<int64_t*>(input_ids.data()),
		input_ids.size(),
		input_shape.data(),
		input_shape.size()
	);
	io_binding.BindInput("input_ids", input_tensor);

	// 2. Bind position_ids (current_seq_len, current_seq_len+1, ...)
	std::vector<int64_t> position_ids;
	for (size_t i = 0; i < input_ids.size(); i++)
	{
		position_ids.push_back(current_seq_len_ + i);
	}
	std::vector<int64_t> position_shape = {1, static_cast<int64_t>(position_ids.size())};
	Ort::Value position_tensor = Ort::Value::CreateTensor<int64_t>(
		*memory_info_cpu_,
		position_ids.data(),
		position_ids.size(),
		position_shape.data(),
		position_shape.size()
	);
	io_binding.BindInput("position_ids", position_tensor);

	// 3. Bind KV cache inputs (from previous step)
	for (int i = 0; i < num_layers_; i++)
	{
		std::string key_name = "past_key_" + std::to_string(i);
		std::string value_name = "past_value_" + std::to_string(i);

		io_binding.BindInput(key_name.c_str(), past_key_cache_[i]);
		io_binding.BindInput(value_name.c_str(), past_value_cache_[i]);
	}

	// 4. Bind outputs (logits + updated KV cache)
	auto* output_memory_info = use_gpu_ ? memory_info_gpu_.get() : memory_info_cpu_.get();

	io_binding.BindOutput("logits", *output_memory_info);
	for (int i = 0; i < num_layers_; i++)
	{
		std::string key_name = "present_key_" + std::to_string(i);
		std::string value_name = "present_value_" + std::to_string(i);

		io_binding.BindOutput(key_name.c_str(), *output_memory_info);
		io_binding.BindOutput(value_name.c_str(), *output_memory_info);
	}

	// 5. Run inference
	session_->Run(Ort::RunOptions{nullptr}, io_binding);

	// 6. Get outputs
	std::vector<Ort::Value> outputs = io_binding.GetOutputValues();

	// 7. Update KV cache (move semantics, no copy)
	for (int i = 0; i < num_layers_; i++)
	{
		past_key_cache_[i] = std::move(outputs[1 + i * 2]);
		past_value_cache_[i] = std::move(outputs[2 + i * 2]);
	}

	// 8. Extract logits (may need CPU copy if on GPU)
	Ort::Value& logits_tensor = outputs[0];
	float* logits_ptr = logits_tensor.GetTensorMutableData<float>();
	size_t logits_size = logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

	std::vector<float> logits;
	if (use_gpu_)
	{
		#ifdef USE_CUDA
		logits.resize(logits_size);
		cudaMemcpy(logits.data(), logits_ptr, logits_size * sizeof(float), cudaMemcpyDeviceToHost);
		#else
		// Fallback if CUDA not available
		logits.assign(logits_ptr, logits_ptr + logits_size);
		#endif
	}
	else
	{
		logits.assign(logits_ptr, logits_ptr + logits_size);
	}

	// Update sequence state
	current_seq_len_ += input_ids.size();
	full_sequence_.insert(full_sequence_.end(), input_ids.begin(), input_ids.end());

	// Update metrics
	auto end = std::chrono::high_resolution_clock::now();
	double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
	update_metrics(latency_ms, is_first_token);

	return logits;
}


std::vector<float> KVCacheInferencer::forward_no_cache(const std::vector<int64_t>& input_ids)
{
	auto start = std::chrono::high_resolution_clock::now();

	// Simple forward pass without KV cache (for baseline comparison)
	// Input: full sequence
	std::vector<const char*> input_names = {"input_ids"};
	std::vector<const char*> output_names = {"logits"};

	std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

	Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
		*memory_info_cpu_,
		const_cast<int64_t*>(input_ids.data()),
		input_ids.size(),
		input_shape.data(),
		input_shape.size()
	);

	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(input_tensor));

	auto outputs = session_->Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		input_tensors.data(),
		input_tensors.size(),
		output_names.data(),
		output_names.size()
	);

	float* logits_ptr = outputs[0].GetTensorMutableData<float>();
	size_t logits_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

	std::vector<float> logits(logits_ptr, logits_ptr + logits_size);

	auto end = std::chrono::high_resolution_clock::now();
	double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "[NO CACHE] Latency: " << latency_ms << " ms (seq_len=" << input_ids.size() << ")" << std::endl;

	return logits;
}


void KVCacheInferencer::reset_cache()
{
	std::cout << "Resetting KV cache..." << std::endl;

	current_seq_len_ = 0;
	full_sequence_.clear();

	// Clear metrics for new sequence
	metrics_.num_tokens_generated = 0;
	metrics_.first_token_latency_ms = 0.0;
	metrics_.avg_subsequent_token_latency_ms = 0.0;
	metrics_.speedup_factor = 0.0;

	// Re-initialize cache to zero
	init_kv_cache();
}


void KVCacheInferencer::update_metrics(double latency_ms, bool is_first_token)
{
	metrics_.num_tokens_generated++;
	metrics_.current_seq_len = current_seq_len_;

	if (is_first_token)
	{
		metrics_.first_token_latency_ms = latency_ms;
	}
	else
	{
		// Update running average
		int subsequent_tokens = metrics_.num_tokens_generated - 1;
		double total = metrics_.avg_subsequent_token_latency_ms * (subsequent_tokens - 1) + latency_ms;
		metrics_.avg_subsequent_token_latency_ms = total / subsequent_tokens;

		// Calculate speedup factor (comparing to first token as baseline)
		if (metrics_.first_token_latency_ms > 0)
		{
			metrics_.speedup_factor = metrics_.first_token_latency_ms / metrics_.avg_subsequent_token_latency_ms;
		}
	}
}


void KVCacheInferencer::print_metrics() const
{
	std::cout << "\n" << std::string(70, '=') << std::endl;
	std::cout << "KV Cache Performance Metrics" << std::endl;
	std::cout << std::string(70, '=') << std::endl;

	std::cout << "\n[Configuration]" << std::endl;
	std::cout << "  Execution: " << (use_gpu_ ? "GPU (CUDA)" : "CPU") << std::endl;
	std::cout << "  Max sequence length: " << metrics_.max_seq_len << std::endl;
	std::cout << "  Layers: " << num_layers_ << std::endl;
	std::cout << "  Heads: " << num_heads_ << std::endl;
	std::cout << "  Head dimension: " << head_dim_ << std::endl;

	std::cout << "\n[Memory Usage]" << std::endl;
	std::cout << "  KV cache: " << std::fixed << std::setprecision(2)
	          << (metrics_.cache_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

	std::cout << "\n[Performance]" << std::endl;
	std::cout << "  Tokens generated: " << metrics_.num_tokens_generated << std::endl;
	std::cout << "  Current sequence length: " << metrics_.current_seq_len << std::endl;
	std::cout << "  First token latency: " << std::fixed << std::setprecision(2)
	          << metrics_.first_token_latency_ms << " ms" << std::endl;

	if (metrics_.num_tokens_generated > 1)
	{
		std::cout << "  Avg subsequent token latency: " << std::fixed << std::setprecision(2)
		          << metrics_.avg_subsequent_token_latency_ms << " ms" << std::endl;
		std::cout << "  Speedup factor: " << std::fixed << std::setprecision(2)
		          << metrics_.speedup_factor << "×" << std::endl;
	}

	std::cout << std::string(70, '=') << std::endl;
}


}  // namespace trigo
