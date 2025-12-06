/**
 * Minimal KV Cache Test - Just verify model loads and runs
 */

#include "../include/kvcache_inferencer.hpp"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "KV Cache Prototype - Minimal Test" << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        // Load config
        std::string config_path = "../models/config.json";
        std::ifstream f(config_path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        json config_json = json::parse(f);

        std::string model_name = config_json["model_name"];
        int num_layers = config_json["num_layers"];
        int num_heads = config_json["num_heads"];
        int head_dim = config_json["head_dim"];
        int vocab_size = config_json["vocab_size"];
        int max_seq_len = config_json["max_seq_len"];
        std::string onnx_path = config_json["onnx_path"];

        std::cout << "\nModel Configuration:" << std::endl;
        std::cout << "  Name: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << std::endl;
        std::cout << "  Heads: " << num_heads << std::endl;
        std::cout << "  Vocab size: " << vocab_size << std::endl;

        // Load model
        std::string model_path = "../models/" + onnx_path;
        std::cout << "\nLoading model: " << model_path << std::endl;

        trigo::KVCacheInferencer inferencer(
            model_path,
            true,  // use_gpu
            0,
            max_seq_len,
            num_layers,
            num_heads,
            head_dim
        );

        std::cout << "✓ Model loaded successfully" << std::endl;

        // Generate a few tokens
        std::cout << "\nGenerating 5 tokens..." << std::endl;

        for (int i = 0; i < 5; i++) {
            int64_t token = 100 + i;

            auto start = std::chrono::high_resolution_clock::now();
            auto logits = inferencer.forward({token});
            auto end = std::chrono::high_resolution_clock::now();

            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "  Token " << (i + 1) << ": "
                      << "latency=" << latency_ms << "ms, "
                      << "logits_size=" << logits.size()
                      << std::endl;
        }

        // Print metrics
        inferencer.print_metrics();

        std::cout << "\n✓ Test passed!" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
