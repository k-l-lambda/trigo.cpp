/**
 * Test IncrementalCachedMCTSPolicy GPU vs CPU performance
 */
#include <iostream>
#include <chrono>
#include "self_play_policy.hpp"
#include "trigo_game.hpp"

using namespace trigo;

void benchmark_policy(const std::string& name, IncrementalCachedMCTSPolicy& policy, int num_games) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    int total_moves = 0;
    
    for (int game = 0; game < num_games; game++) {
        TrigoGame g(BoardShape{5, 5, 1});
        policy.reset_cache();
        
        int moves = 0;
        while (g.get_game_status() != GameStatus::FINISHED && moves < 30) {
            auto action = policy.select_action(g);
            if (action.is_pass) {
                g.pass();
            } else {
                g.drop(action.position);
            }
            moves++;
        }
        total_moves += moves;
        std::cout << "  Game " << (game + 1) << ": " << moves << " moves" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Total time: " << total_ms / 1000.0 << " s" << std::endl;
    std::cout << "  Total moves: " << total_moves << std::endl;
    std::cout << "  Avg time per move: " << total_ms / total_moves << " ms" << std::endl;
    std::cout << "  Avg time per game: " << total_ms / num_games / 1000.0 << " s" << std::endl;
}

int main() {
    std::string model_path = "/home/camus/work/trigo.cpp/models/trained_shared";
    int num_simulations = 20;
    float c_puct = 1.0f;
    int seed = 42;
    int num_games = 2;
    
    std::cout << "=== IncrementalCachedMCTSPolicy GPU vs CPU Benchmark ===" << std::endl;
    std::cout << "Simulations: " << num_simulations << std::endl;
    std::cout << "Games: " << num_games << std::endl;
    
    // CPU benchmark
    {
        std::cout << "\n--- Creating CPU policy ---" << std::endl;
        IncrementalCachedMCTSPolicy cpu_policy(model_path, num_simulations, c_puct, seed, false, 0);
        benchmark_policy("CPU", cpu_policy, num_games);
    }
    
    // GPU benchmark
    {
        std::cout << "\n--- Creating GPU policy ---" << std::endl;
        IncrementalCachedMCTSPolicy gpu_policy(model_path, num_simulations, c_puct, seed, true, 0);
        benchmark_policy("GPU", gpu_policy, num_games);
    }
    
    return 0;
}
