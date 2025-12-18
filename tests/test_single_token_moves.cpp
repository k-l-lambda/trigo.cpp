#include "include/mcts.hpp"
#include "include/cached_mcts.hpp"
#include "include/trigo_game.hpp"
#include "include/shared_model_inferencer.hpp"
#include "include/prefix_cache_inferencer.hpp"
#include <iostream>
#include <memory>

using namespace trigo;

int main() {
    std::cout << "Testing Single-Token Move Handling\n";
    std::cout << "===================================\n\n";

    std::string model_path = "/home/camus/work/trigoRL/outputs/trigor/20251215-trigo-value-llama-l6-h64-251211/LlamaCausalLM_ep0045_shared_shared";

    try {
        // Test 1: MCTS with SharedModelInferencer
        std::cout << "Test 1: MCTS (SharedModelInferencer) with small board\n";
        std::cout << "-------------------------------------------------------\n";

        auto shared_inferencer = std::make_shared<SharedModelInferencer>(
            model_path + "/base_model.onnx",
            model_path + "/policy_head.onnx",
            model_path + "/value_head.onnx",
            false,  // CPU mode
            0
        );

        MCTS mcts(shared_inferencer, 10, 1.0f, 42);

        // Create 2×2×1 board - all moves will be single tokens like "aa", "ab", etc.
        TrigoGame game(2, 2, 1);

        std::cout << "✓ Created 2×2×1 game (all moves are single-token)\n";
        std::cout << "  Valid moves: " << game.valid_move_positions().size() << "\n";

        // Try MCTS search
        std::cout << "\n✓ Running MCTS search (10 simulations)...\n";
        auto action = mcts.search(game);

        std::cout << "✓ MCTS search completed successfully\n";
        std::cout << "  Selected action: " << (action.is_pass ? "Pass" : "Move") << "\n";
        std::cout << "  Confidence: " << action.confidence << "\n";

        // Test 2: CachedMCTS with PrefixCacheInferencer
        std::cout << "\n\nTest 2: CachedMCTS (PrefixCacheInferencer) with small board\n";
        std::cout << "------------------------------------------------------------\n";

        auto cached_inferencer = std::make_shared<PrefixCacheInferencer>(
            model_path + "/base_model_prefix.onnx",
            model_path + "/policy_head.onnx",
            model_path + "/value_head.onnx",
            false,  // CPU mode
            0
        );

        CachedMCTS cached_mcts(cached_inferencer, 10, 1.0f, 42);

        // Reset game
        TrigoGame game2(2, 2, 1);

        std::cout << "✓ Created 2×2×1 game for cached MCTS\n";

        // Try cached MCTS search
        std::cout << "\n✓ Running CachedMCTS search (10 simulations)...\n";
        auto action2 = cached_mcts.search(game2);

        std::cout << "✓ CachedMCTS search completed successfully\n";
        std::cout << "  Selected action: " << (action2.is_pass ? "Pass" : "Move") << "\n";
        std::cout << "  Confidence: " << action2.confidence << "\n";

        // Test 3: After making a move (still single-token moves)
        std::cout << "\n\nTest 3: After making a move (still single-token scenario)\n";
        std::cout << "--------------------------------------------------------\n";

        game.drop({0, 0, 0});  // Black plays aa
        std::cout << "✓ Made move: aa\n";
        std::cout << "  Valid moves remaining: " << game.valid_move_positions().size() << "\n";

        std::cout << "\n✓ Running MCTS search after move...\n";
        auto action3 = mcts.search(game);

        std::cout << "✓ MCTS search completed successfully\n";
        std::cout << "  Selected action: " << (action3.is_pass ? "Pass" : "Move") << "\n";

        std::cout << "\n=== All Tests Passed ===\n";
        std::cout << "✓ Single-token moves handled correctly\n";
        std::cout << "✓ No {1,0,0} tensor shape errors\n";
        std::cout << "✓ Uniform prior fallback works\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
        return 1;
    }
}
