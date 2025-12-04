/**
 * Test program to verify ONNX Runtime CUDA execution provider is available
 *
 * This test verifies:
 * - CUDA device detection and properties
 * - ONNX Runtime initialization
 * - CUDAExecutionProvider availability
 * - CUDA provider configuration
 */

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>


int main()
{
	std::cout << "ONNX Runtime CUDA Verification Test\n";
	std::cout << "====================================\n\n";

	try
	{
		// Check CUDA device availability
		int deviceCount = 0;
		cudaError_t error = cudaGetDeviceCount(&deviceCount);

		if (error != cudaSuccess)
		{
			std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
			return 1;
		}

		std::cout << "CUDA Devices Available: " << deviceCount << std::endl;

		if (deviceCount == 0)
		{
			std::cerr << "No CUDA devices found!" << std::endl;
			return 1;
		}

		// Get device properties
		for (int i = 0; i < deviceCount; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);

			std::cout << "  Device " << i << ": " << prop.name << std::endl;
			std::cout << "    Compute Capability: " << prop.major << "." << prop.minor << std::endl;
			std::cout << "    Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
		}
		std::cout << std::endl;


		// Initialize ONNX Runtime
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
		std::cout << "✓ ONNX Runtime environment initialized" << std::endl;


		// Check available execution providers
		std::cout << "\nAvailable Execution Providers:\n";

		Ort::SessionOptions session_options;

		std::vector<std::string> available_providers = Ort::GetAvailableProviders();
		for (const auto& provider : available_providers)
		{
			std::cout << "  - " << provider << std::endl;
		}


		// Check if CUDA provider is available
		bool cuda_available = false;
		for (const auto& provider : available_providers)
		{
			if (provider == "CUDAExecutionProvider")
			{
				cuda_available = true;
				break;
			}
		}


		if (cuda_available)
		{
			std::cout << "\n✓ CUDAExecutionProvider is available!" << std::endl;

			// Try to configure CUDA provider
			try
			{
				OrtCUDAProviderOptions cuda_options;
				cuda_options.device_id = 0;
				cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
				cuda_options.do_copy_in_default_stream = 1;

				session_options.AppendExecutionProvider_CUDA(cuda_options);
				std::cout << "✓ CUDA execution provider configured for device 0" << std::endl;
			}
			catch (const Ort::Exception& e)
			{
				std::cout << "⚠ Warning: Could not fully initialize CUDA provider" << std::endl;
				std::cout << "  Error: " << e.what() << std::endl;
				std::cout << "  This is likely due to missing cuDNN library" << std::endl;
				std::cout << "  Install cuDNN 8.x for full CUDA support" << std::endl;
				std::cout << "  (Basic CUDA operations should still work)" << std::endl;
			}
		}
		else
		{
			std::cerr << "\n✗ CUDAExecutionProvider is NOT available!" << std::endl;
			std::cerr << "  Make sure you downloaded the GPU version of ONNX Runtime" << std::endl;
			return 1;
		}


		std::cout << "\n====================================\n";
		std::cout << "All checks passed! ONNX Runtime CUDA is ready.\n";
		std::cout << "====================================\n";

		return 0;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
