/**
 * Shared Model Inferencer - Implementation
 */

#include "shared_model_inferencer.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>


namespace trigo
{


SharedModelInferencer::SharedModelInferencer(
	const std::string& base_model_path,
	const std::string& policy_head_path,
	const std::string& value_head_path,
	bool use_gpu,
	int device_id
)
	: env_(ORT_LOGGING_LEVEL_WARNING, "SharedModelInferencer"),
	  use_gpu_(use_gpu),
	  device_id_(device_id)
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
		}
	}

	// Load ONNX models
	try
	{
		std::cout << "Loading ONNX models..." << std::endl;

		// Convert paths to wide strings for Windows compatibility
		#ifdef _WIN32
		std::wstring base_path_w(base_model_path.begin(), base_model_path.end());
		std::wstring policy_path_w(policy_head_path.begin(), policy_head_path.end());
		std::wstring value_path_w(value_head_path.begin(), value_head_path.end());

		base_session_ = std::make_unique<Ort::Session>(env_, base_path_w.c_str(), session_options_);
		policy_session_ = std::make_unique<Ort::Session>(env_, policy_path_w.c_str(), session_options_);
		value_session_ = std::make_unique<Ort::Session>(env_, value_path_w.c_str(), session_options_);
		#else
		base_session_ = std::make_unique<Ort::Session>(env_, base_model_path.c_str(), session_options_);
		policy_session_ = std::make_unique<Ort::Session>(env_, policy_head_path.c_str(), session_options_);
		value_session_ = std::make_unique<Ort::Session>(env_, value_head_path.c_str(), session_options_);
		#endif

		std::cout << "✓ Base model loaded: " << base_model_path << std::endl;
		std::cout << "✓ Policy head loaded: " << policy_head_path << std::endl;
		std::cout << "✓ Value head loaded: " << value_head_path << std::endl;
	}
	catch (const Ort::Exception& e)
	{
		throw std::runtime_error(std::string("Failed to load ONNX models: ") + e.what());
	}
}


std::vector<float> SharedModelInferencer::policy_inference(
	const std::vector<int64_t>& prefix_ids,
	const std::vector<int64_t>& evaluated_ids,
	const std::vector<float>& evaluated_mask,
	int batch_size,
	int prefix_len,
	int eval_len
)
{
	// Step 1: Run base model to get hidden states
	// Input shapes: prefix_ids [batch, n], evaluated_ids [batch, m], evaluated_mask [batch, m, m]
	// Output shape: hidden_states [batch, n+m, hidden_dim]

	std::vector<const char*> input_names = {"prefix_ids", "evaluated_ids", "evaluated_mask"};
	std::vector<const char*> output_names = {"hidden_states"};

	// Create input tensors
	std::vector<int64_t> prefix_shape = {batch_size, prefix_len};
	std::vector<int64_t> evaluated_shape = {batch_size, eval_len};
	std::vector<int64_t> mask_shape = {batch_size, eval_len, eval_len};

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	Ort::Value prefix_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		const_cast<int64_t*>(prefix_ids.data()),
		prefix_ids.size(),
		prefix_shape.data(),
		prefix_shape.size()
	);

	Ort::Value evaluated_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		const_cast<int64_t*>(evaluated_ids.data()),
		evaluated_ids.size(),
		evaluated_shape.data(),
		evaluated_shape.size()
	);

	Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		const_cast<float*>(evaluated_mask.data()),
		evaluated_mask.size(),
		mask_shape.data(),
		mask_shape.size()
	);

	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(prefix_tensor));
	input_tensors.push_back(std::move(evaluated_tensor));
	input_tensors.push_back(std::move(mask_tensor));

	// Run base model
	auto base_outputs = base_session_->Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		input_tensors.data(),
		input_tensors.size(),
		output_names.data(),
		output_names.size()
	);

	// Extract hidden states
	float* hidden_states_ptr = base_outputs[0].GetTensorMutableData<float>();
	auto hidden_shape = base_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int total_len = hidden_shape[1];  // n + m
	int hidden_dim = hidden_shape[2];

	// Step 2: For the test model, policy_head takes full hidden_states [batch, n+m, hidden_dim]
	// and internally extracts positions n-1 onward for logits
	std::vector<float> hidden_for_policy(
		hidden_states_ptr,
		hidden_states_ptr + (batch_size * total_len * hidden_dim)
	);

	// Step 3: Run policy head
	// Input: hidden_states [batch, n+m, hidden_dim]
	// Output: logits [batch, m+1, vocab_size] (for positions n-1 onward)

	std::vector<const char*> policy_input_names = {"hidden_states"};
	std::vector<const char*> policy_output_names = {"logits"};

	std::vector<int64_t> policy_hidden_shape = {batch_size, total_len, hidden_dim};

	Ort::Value policy_hidden_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		hidden_for_policy.data(),
		hidden_for_policy.size(),
		policy_hidden_shape.data(),
		policy_hidden_shape.size()
	);

	std::vector<Ort::Value> policy_inputs;
	policy_inputs.push_back(std::move(policy_hidden_tensor));

	auto policy_outputs = policy_session_->Run(
		Ort::RunOptions{nullptr},
		policy_input_names.data(),
		policy_inputs.data(),
		policy_inputs.size(),
		policy_output_names.data(),
		policy_output_names.size()
	);

	// Extract logits
	float* logits_ptr = policy_outputs[0].GetTensorMutableData<float>();
	auto logits_shape = policy_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int output_seq_len = logits_shape[1];  // Should be m+1
	int vocab_size = logits_shape[2];

	size_t logits_size = batch_size * output_seq_len * vocab_size;
	std::vector<float> logits(logits_ptr, logits_ptr + logits_size);

	return logits;
}


std::vector<float> SharedModelInferencer::value_inference(
	const std::vector<int64_t>& input_ids,
	int batch_size,
	int seq_len,
	int value_token_id
)
{
	// Step 1: Append VALUE token to input_ids
	// EvaluationLM internally appends VALUE token before inference
	std::vector<int64_t> input_ids_with_value;
	input_ids_with_value.reserve(batch_size * (seq_len + 1));

	for (int b = 0; b < batch_size; b++)
	{
		// Copy original sequence
		for (int i = 0; i < seq_len; i++)
		{
			input_ids_with_value.push_back(input_ids[b * seq_len + i]);
		}
		// Append VALUE token
		input_ids_with_value.push_back(value_token_id);
	}

	int total_seq_len = seq_len + 1;

	// Step 2: Split into prefix and evaluated for base model
	// Use same split as training: prefix_len = 128
	int prefix_len = 128;
	int eval_len = total_seq_len - prefix_len;

	std::vector<int64_t> prefix_ids;
	std::vector<int64_t> evaluated_ids;
	prefix_ids.reserve(batch_size * prefix_len);
	evaluated_ids.reserve(batch_size * eval_len);

	for (int b = 0; b < batch_size; b++)
	{
		for (int i = 0; i < prefix_len; i++)
		{
			prefix_ids.push_back(input_ids_with_value[b * total_seq_len + i]);
		}
		for (int i = prefix_len; i < total_seq_len; i++)
		{
			evaluated_ids.push_back(input_ids_with_value[b * total_seq_len + i]);
		}
	}

	// Step 3: Create causal mask for evaluated region
	std::vector<float> evaluated_mask = create_causal_mask(eval_len);
	std::vector<float> evaluated_mask_batch = expand_mask_to_batch(evaluated_mask, batch_size, eval_len);

	// Step 4: Run base model
	std::vector<const char*> input_names = {"prefix_ids", "evaluated_ids", "evaluated_mask"};
	std::vector<const char*> output_names = {"hidden_states"};

	std::vector<int64_t> prefix_shape = {batch_size, prefix_len};
	std::vector<int64_t> evaluated_shape = {batch_size, eval_len};
	std::vector<int64_t> mask_shape = {batch_size, eval_len, eval_len};

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	Ort::Value prefix_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		prefix_ids.data(),
		prefix_ids.size(),
		prefix_shape.data(),
		prefix_shape.size()
	);

	Ort::Value evaluated_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		evaluated_ids.data(),
		evaluated_ids.size(),
		evaluated_shape.data(),
		evaluated_shape.size()
	);

	Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		evaluated_mask_batch.data(),
		evaluated_mask_batch.size(),
		mask_shape.data(),
		mask_shape.size()
	);

	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(prefix_tensor));
	input_tensors.push_back(std::move(evaluated_tensor));
	input_tensors.push_back(std::move(mask_tensor));

	auto base_outputs = base_session_->Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		input_tensors.data(),
		input_tensors.size(),
		output_names.data(),
		output_names.size()
	);

	// Step 5: Extract hidden state at VALUE token position (last position)
	float* hidden_states_ptr = base_outputs[0].GetTensorMutableData<float>();
	auto hidden_shape = base_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int total_len = hidden_shape[1];  // n + m
	int hidden_dim = hidden_shape[2];

	std::vector<float> value_hidden;
	value_hidden.reserve(batch_size * hidden_dim);

	for (int b = 0; b < batch_size; b++)
	{
		int last_pos_offset = b * total_len * hidden_dim + (total_len - 1) * hidden_dim;
		for (int h = 0; h < hidden_dim; h++)
		{
			value_hidden.push_back(hidden_states_ptr[last_pos_offset + h]);
		}
	}

	// Step 6: Run value head
	// Input: hidden_states [batch, hidden_dim]
	// Output: values [batch]

	std::vector<const char*> value_input_names = {"hidden_states"};
	std::vector<const char*> value_output_names = {"values"};

	std::vector<int64_t> value_hidden_shape = {batch_size, hidden_dim};

	Ort::Value value_hidden_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		value_hidden.data(),
		value_hidden.size(),
		value_hidden_shape.data(),
		value_hidden_shape.size()
	);

	std::vector<Ort::Value> value_inputs;
	value_inputs.push_back(std::move(value_hidden_tensor));

	auto value_outputs = value_session_->Run(
		Ort::RunOptions{nullptr},
		value_input_names.data(),
		value_inputs.data(),
		value_inputs.size(),
		value_output_names.data(),
		value_output_names.size()
	);

	// Extract values
	float* values_ptr = value_outputs[0].GetTensorMutableData<float>();
	std::vector<float> values(values_ptr, values_ptr + batch_size);

	return values;
}


void SharedModelInferencer::print_model_info() const
{
	std::cout << "\n" << std::string(70, '=') << std::endl;
	std::cout << "Shared Model Inferencer - Model Information" << std::endl;
	std::cout << std::string(70, '=') << std::endl;

	// Base model info
	std::cout << "\n[Base Model]" << std::endl;
	size_t num_inputs = base_session_->GetInputCount();
	size_t num_outputs = base_session_->GetOutputCount();
	std::cout << "  Inputs: " << num_inputs << std::endl;
	for (size_t i = 0; i < num_inputs; i++)
	{
		auto input_name = base_session_->GetInputNameAllocated(i, allocator_);
		auto input_shape = base_session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << "    " << input_name.get() << ": [";
		for (size_t j = 0; j < input_shape.size(); j++)
		{
			std::cout << input_shape[j];
			if (j < input_shape.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}
	std::cout << "  Outputs: " << num_outputs << std::endl;
	for (size_t i = 0; i < num_outputs; i++)
	{
		auto output_name = base_session_->GetOutputNameAllocated(i, allocator_);
		auto output_shape = base_session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << "    " << output_name.get() << ": [";
		for (size_t j = 0; j < output_shape.size(); j++)
		{
			std::cout << output_shape[j];
			if (j < output_shape.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}

	// Policy head info
	std::cout << "\n[Policy Head]" << std::endl;
	num_inputs = policy_session_->GetInputCount();
	num_outputs = policy_session_->GetOutputCount();
	std::cout << "  Inputs: " << num_inputs << std::endl;
	for (size_t i = 0; i < num_inputs; i++)
	{
		auto input_name = policy_session_->GetInputNameAllocated(i, allocator_);
		std::cout << "    " << input_name.get() << std::endl;
	}
	std::cout << "  Outputs: " << num_outputs << std::endl;
	for (size_t i = 0; i < num_outputs; i++)
	{
		auto output_name = policy_session_->GetOutputNameAllocated(i, allocator_);
		std::cout << "    " << output_name.get() << std::endl;
	}

	// Value head info
	std::cout << "\n[Value Head]" << std::endl;
	num_inputs = value_session_->GetInputCount();
	num_outputs = value_session_->GetOutputCount();
	std::cout << "  Inputs: " << num_inputs << std::endl;
	for (size_t i = 0; i < num_inputs; i++)
	{
		auto input_name = value_session_->GetInputNameAllocated(i, allocator_);
		std::cout << "    " << input_name.get() << std::endl;
	}
	std::cout << "  Outputs: " << num_outputs << std::endl;
	for (size_t i = 0; i < num_outputs; i++)
	{
		auto output_name = value_session_->GetOutputNameAllocated(i, allocator_);
		std::cout << "    " << output_name.get() << std::endl;
	}

	std::cout << std::string(70, '=') << std::endl;
}


std::vector<float> SharedModelInferencer::create_causal_mask(int eval_len)
{
	// Create lower triangular matrix (causal attention)
	std::vector<float> mask(eval_len * eval_len, 0.0f);

	for (int i = 0; i < eval_len; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			mask[i * eval_len + j] = 1.0f;
		}
	}

	return mask;
}


std::vector<float> SharedModelInferencer::expand_mask_to_batch(
	const std::vector<float>& mask,
	int batch_size,
	int eval_len
)
{
	std::vector<float> batch_mask;
	batch_mask.reserve(batch_size * eval_len * eval_len);

	for (int b = 0; b < batch_size; b++)
	{
		batch_mask.insert(batch_mask.end(), mask.begin(), mask.end());
	}

	return batch_mask;
}


}  // namespace trigo
