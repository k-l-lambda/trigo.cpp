/**
 * Test loading trained ONNX models
 *
 * This test verifies that the trained models from trigoRL can be loaded
 * by SharedModelInferencer and checks their input/output shapes.
 */

#include "../include/shared_model_inferencer.hpp"
#include <iostream>
#include <stdexcept>


using namespace trigo;


void test_monolithic_models()
{
	std::cout << "\n==========================================================\n";
	std::cout << "Test: Loading Trained Monolithic Models\n";
	std::cout << "==========================================================\n\n";

	// Path to trained models
	std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/";
	std::string tree_model = model_dir + "GPT2CausalLM_ep0019_tree.onnx";
	std::string eval_model = model_dir + "GPT2CausalLM_ep0019_evaluation.onnx";

	std::cout << "Model paths:\n";
	std::cout << "  Tree (policy): " << tree_model << "\n";
	std::cout << "  Eval (value):  " << eval_model << "\n\n";

	try
	{
		std::cout << "Creating ONNX Runtime environment...\n";
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_trained_models");
		Ort::SessionOptions session_options;

		// Try CUDA, fallback to CPU if not available
		try
		{
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;
			session_options.AppendExecutionProvider_CUDA(cuda_options);
			std::cout << "Using CUDA execution provider\n";
		}
		catch (...)
		{
			std::cout << "CUDA not available, using CPU\n";
		}

		// Load tree model (policy)
		std::cout << "\n[1] Loading Tree Model (Policy)\n";
		std::cout << "================================\n";
		Ort::Session tree_session(env, tree_model.c_str(), session_options);

		// Get tree model metadata
		Ort::AllocatorWithDefaultOptions allocator;

		size_t num_tree_inputs = tree_session.GetInputCount();
		size_t num_tree_outputs = tree_session.GetOutputCount();

		std::cout << "Number of inputs: " << num_tree_inputs << "\n";
		std::cout << "Number of outputs: " << num_tree_outputs << "\n\n";

		std::cout << "Input details:\n";
		for (size_t i = 0; i < num_tree_inputs; i++)
		{
			auto input_name = tree_session.GetInputNameAllocated(i, allocator);
			auto type_info = tree_session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto shape = tensor_info.GetShape();

			std::cout << "  [" << i << "] " << input_name.get() << ": [";
			for (size_t j = 0; j < shape.size(); j++)
			{
				if (j > 0) std::cout << ", ";
				std::cout << shape[j];
			}
			std::cout << "]\n";
		}

		std::cout << "\nOutput details:\n";
		for (size_t i = 0; i < num_tree_outputs; i++)
		{
			auto output_name = tree_session.GetOutputNameAllocated(i, allocator);
			auto type_info = tree_session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto shape = tensor_info.GetShape();

			std::cout << "  [" << i << "] " << output_name.get() << ": [";
			for (size_t j = 0; j < shape.size(); j++)
			{
				if (j > 0) std::cout << ", ";
				std::cout << shape[j];
			}
			std::cout << "]\n";
		}

		// Load evaluation model (value)
		std::cout << "\n[2] Loading Evaluation Model (Value)\n";
		std::cout << "====================================\n";
		Ort::Session eval_session(env, eval_model.c_str(), session_options);

		size_t num_eval_inputs = eval_session.GetInputCount();
		size_t num_eval_outputs = eval_session.GetOutputCount();

		std::cout << "Number of inputs: " << num_eval_inputs << "\n";
		std::cout << "Number of outputs: " << num_eval_outputs << "\n\n";

		std::cout << "Input details:\n";
		for (size_t i = 0; i < num_eval_inputs; i++)
		{
			auto input_name = eval_session.GetInputNameAllocated(i, allocator);
			auto type_info = eval_session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto shape = tensor_info.GetShape();

			std::cout << "  [" << i << "] " << input_name.get() << ": [";
			for (size_t j = 0; j < shape.size(); j++)
			{
				if (j > 0) std::cout << ", ";
				std::cout << shape[j];
			}
			std::cout << "]\n";
		}

		std::cout << "\nOutput details:\n";
		for (size_t i = 0; i < num_eval_outputs; i++)
		{
			auto output_name = eval_session.GetOutputNameAllocated(i, allocator);
			auto type_info = eval_session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto shape = tensor_info.GetShape();

			std::cout << "  [" << i << "] " << output_name.get() << ": [";
			for (size_t j = 0; j < shape.size(); j++)
			{
				if (j > 0) std::cout << ", ";
				std::cout << shape[j];
			}
			std::cout << "]\n";
		}

		std::cout << "\n✓ Both models loaded successfully!\n";
		std::cout << "\n==========================================================\n";
		std::cout << "RESULT: Models are compatible with ONNX Runtime\n";
		std::cout << "==========================================================\n";

		return;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "\n✗ ONNX Runtime Error: " << e.what() << "\n";
		std::cerr << "\n==========================================================\n";
		std::cerr << "RESULT: Failed to load models\n";
		std::cerr << "==========================================================\n";
		throw;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n✗ Error: " << e.what() << "\n";
		throw;
	}
}


int main()
{
	try
	{
		std::cout << "\n";
		std::cout << "======================================================================\n";
		std::cout << "Trained Model Loading Test\n";
		std::cout << "======================================================================\n";

		test_monolithic_models();

		std::cout << "\n✓ All tests passed!\n\n";
		return 0;
	}
	catch (...)
	{
		std::cerr << "\n✗ Test failed!\n\n";
		return 1;
	}
}
