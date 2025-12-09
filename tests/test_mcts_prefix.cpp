/**
 * MCTS Search from Specific Prefix
 *
 * Test MCTS search from prefix: "[Board 5x5]\n\n1. a0 "
 * Print detailed statistics for comparison with TypeScript
 */

#include "trigo_game.hpp"
#include "prefix_cache_inferencer.hpp"
#include "cached_mcts.hpp"
#include <iostream>
#include <iomanip>

using namespace trigo;

int main() {
    std::cout << "=== C++ MCTS Search from Prefix ===" << std::endl;
    std::cout << std::endl;

    // Setup
    std::string model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached";

    BoardShape shape{5, 5, 1};
    TrigoGame game(shape);
    game.start_game();

    // Play a0 (2,0,0)
    game.drop(Position{2, 0, 0});

    std::cout << "Game state: " << game_to_tgn(game, false) << std::endl;
    std::cout << std::endl;

    // Initialize inferencer
    std::cout << "Loading models from: " << model_dir << std::endl;
    auto inferencer = std::make_shared<PrefixCacheInferencer>(
        model_dir + "/base_model_prefix.onnx",
        model_dir + "/base_model_eval_cached.onnx",
        model_dir + "/policy_head.onnx",
        model_dir + "/value_head.onnx",
        false,  // CPU
        0
    );
    std::cout << std::endl;

    // Create MCTS with same config as tests
    int num_simulations = 50;
    float c_puct = 1.0f;
    int seed = 42;

    std::cout << "MCTS Configuration:" << std::endl;
    std::cout << "  Simulations: " << num_simulations << std::endl;
    std::cout << "  c_puct: " << c_puct << std::endl;
    std::cout << "  Seed: " << seed << std::endl;
    std::cout << std::endl;

    auto mcts = std::make_shared<CachedMCTS>(inferencer, num_simulations, c_puct, seed);

    // Run search
    std::cout << "Running MCTS search (White to move)..." << std::endl;
    auto start = std::chrono::steady_clock::now();

    auto action = mcts->search(game);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    std::cout << "Search completed in " << elapsed << "ms" << std::endl;
    std::cout << std::endl;

    // Get root statistics
    std::cout << "=== MCTS Search Results ===" << std::endl;
    std::cout << std::endl;

    if (action.is_pass) {
        std::cout << "Selected move: PASS" << std::endl;
    } else {
        std::string move_notation = encode_ab0yz(action.position, shape);
        std::cout << "Selected move: " << move_notation
                  << " (" << action.position.x << "," << action.position.y << "," << action.position.z << ")"
                  << std::endl;
    }
    std::cout << "Confidence: " << action.confidence << std::endl;
    std::cout << std::endl;

    std::cout << "Note: Detailed visit counts, priors, and Q-values are printed by CachedMCTS when MCTS_ENABLE_PROFILING is ON" << std::endl;
    std::cout << std::endl;

    std::cout << "To see full statistics, run with profiling enabled:" << std::endl;
    std::cout << "  cmake -DENABLE_MCTS_PROFILING=ON -B build" << std::endl;
    std::cout << "  make -C build test_mcts_prefix" << std::endl;
    std::cout << "  ./build/test_mcts_prefix" << std::endl;

    return 0;
}
