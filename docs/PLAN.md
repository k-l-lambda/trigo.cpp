# Implementation Plan - Trigo.cpp CUDA MCTS Engine

**Goal**: Build a C++/CUDA MCTS self-play engine achieving **300-1000 games/hour** on single GPU.

**Timeline**: 10 weeks to production-ready system

**Status**: Phase 0 Complete - Architecture designed, feasibility confirmed

---

## Executive Summary

This document outlines the implementation plan for a high-performance CUDA-accelerated MCTS engine for Trigo (3D Go). The system will generate training data **100-1200× faster** than the TypeScript baseline by leveraging:

1. GPU-parallel tree traversal (10-20×)
2. Batched neural network inference (5-10×)
3. C++ vs JavaScript performance (2-3×)
4. Memory and algorithm optimizations (2×)

The implementation validates against the existing TypeScript golden reference to ensure correctness.

---

## Phase 1: Core Infrastructure (Weeks 1-2)

### Goal
Working C++ game engine and CPU-only MCTS that generates legal games.

### Tasks

#### 1.1 Build System Setup (Days 1-2)
**Deliverable**: CMake build with all dependencies

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(trigo_cpp CUDA CXX)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

add_library(trigo_game src/game_state.cpp)
add_library(trigo_mcts src/mcts_engine.cu)

pybind11_add_module(cuda_mcts src/bindings.cpp)
target_link_libraries(cuda_mcts PRIVATE
    trigo_game
    trigo_mcts
    ${TORCH_LIBRARIES}
)
```

**Files to create**:
- `CMakeLists.txt` (root)
- `setup.py` (Python packaging)
- `.gitignore`

#### 1.2 TrigoGame Class (Days 3-8)
**Deliverable**: Complete Trigo game engine in C++

**Port from TypeScript**:
- Source: `trigoRL/third_party/trigo/trigo-web/inc/trigo/game.ts` (800 lines)
- Target: `include/game_state.hpp` + `src/game_state.cpp`

**Core data structures**:
```cpp
struct Position {
    int16_t x, y, z;
};

struct Move {
    int16_t x, y, z;
    int8_t type;    // 0=place, 1=pass
    int8_t player;  // 1=black, 2=white
};

class TrigoGame {
public:
    // Constructor
    TrigoGame(int x, int y, int z);

    // Core operations
    bool is_legal_move(const Move& move) const;
    bool apply_move(const Move& move);
    bool undo_move();

    // Queries
    std::vector<Move> get_legal_moves() const;
    std::vector<Position> get_valid_positions() const;
    int8_t get_stone(const Position& pos) const;
    int8_t current_player() const;
    bool is_terminal() const;

    // Scoring
    struct Territory {
        int black, white, neutral;
    };
    Territory get_territory() const;

    // Serialization
    std::string to_tgn() const;
    static TrigoGame from_tgn(const std::string& tgn);

private:
    // Board state
    static constexpr int MAX_SIZE = 9;
    int8_t board_[MAX_SIZE][MAX_SIZE][MAX_SIZE];
    int8_t shape_x_, shape_y_, shape_z_;

    // Game state
    int8_t current_player_;
    int pass_count_;
    std::vector<Move> history_;

    // Ko detection
    std::vector<Position> last_captured_;

    // Internal methods
    std::vector<Position> find_captured_groups(const Position& pos);
    std::vector<Position> flood_fill_group(const Position& start, int8_t color);
    int count_liberties(const std::vector<Position>& group);
    bool is_ko_violation(const Move& move) const;
    bool is_suicide(const Move& move) const;
    std::vector<Position> get_neighbors(const Position& pos) const;
};
```

**Critical algorithms to port**:

1. **Capture Detection** (from `gameUtils.ts:124-235`):
```cpp
std::vector<Position> TrigoGame::find_captured_groups(const Position& pos) {
    std::vector<Position> captured;
    int8_t enemy = (board_[pos.x][pos.y][pos.z] == 1) ? 2 : 1;

    // Check all 6 neighbors
    for (const auto& neighbor : get_neighbors(pos)) {
        if (board_[neighbor.x][neighbor.y][neighbor.z] != enemy) {
            continue;
        }

        // BFS to find connected group
        auto group = flood_fill_group(neighbor, enemy);

        // Count liberties (empty neighbors)
        if (count_liberties(group) == 0) {
            captured.insert(captured.end(), group.begin(), group.end());
        }
    }

    return captured;
}

std::vector<Position> TrigoGame::flood_fill_group(
    const Position& start, int8_t color
) {
    std::vector<Position> group;
    std::queue<Position> queue;
    std::set<std::tuple<int, int, int>> visited;

    queue.push(start);
    visited.insert({start.x, start.y, start.z});

    while (!queue.empty()) {
        Position pos = queue.front();
        queue.pop();
        group.push_back(pos);

        for (const auto& neighbor : get_neighbors(pos)) {
            if (board_[neighbor.x][neighbor.y][neighbor.z] == color &&
                !visited.count({neighbor.x, neighbor.y, neighbor.z})) {
                queue.push(neighbor);
                visited.insert({neighbor.x, neighbor.y, neighbor.z});
            }
        }
    }

    return group;
}
```

2. **Ko Rule** (from `gameUtils.ts:300-350`):
```cpp
bool TrigoGame::is_ko_violation(const Move& move) const {
    // Ko: Can't recapture immediately if it recreates previous position
    if (last_captured_.size() != 1) {
        return false;  // Ko only applies to single stone captures
    }

    Position last_cap = last_captured_[0];
    return (move.x == last_cap.x &&
            move.y == last_cap.y &&
            move.z == last_cap.z);
}
```

3. **TGN Conversion** (from `game.ts:500-650`):
```cpp
std::string TrigoGame::to_tgn() const {
    std::ostringstream ss;

    // Header: board shape
    ss << shape_x_ << " " << shape_y_ << " " << shape_z_ << "\n";

    // Moves in ab0yz notation
    for (const auto& move : history_) {
        if (move.type == 1) {  // Pass
            ss << "pass ";
        } else {
            ss << encode_ab0yz(move.x, move.y, move.z,
                              shape_x_, shape_y_, shape_z_) << " ";
        }
    }

    // Score (if game ended)
    if (is_terminal()) {
        auto territory = get_territory();
        int score_diff = territory.white - territory.black;
        ss << "; " << score_diff;
    }

    return ss.str();
}
```

#### 1.3 Basic MCTS (CPU) (Days 9-12)
**Deliverable**: CPU-only MCTS that generates games

```cpp
struct MCTSNode {
    int32_t parent_idx;
    int32_t first_child_idx;
    int32_t next_sibling_idx;
    int32_t num_children;

    Move move;
    int32_t game_id;

    float visit_count;
    float total_value;
    float prior_prob;
    float mean_value;  // Q(s,a) = W/N
};

class MCTSEngine {
public:
    MCTSEngine(int num_games, int max_nodes_per_game);

    // MCTS phases
    int select_leaf(int root_idx);
    void expand_node(int node_idx, const TrigoGame& game,
                     const std::vector<float>& policy_probs);
    void backup(int leaf_idx, float value);

    // Move selection
    Move get_best_move(int root_idx, float temperature);

private:
    std::vector<MCTSNode> nodes_;
    int next_node_idx_;

    float compute_ucb1(const MCTSNode& node, float parent_visits);
};
```

**UCB1 Formula**:
```cpp
float MCTSEngine::compute_ucb1(const MCTSNode& node, float parent_visits) {
    float Q = node.mean_value;
    float U = c_puct_ * node.prior_prob *
              sqrtf(parent_visits) / (1.0f + node.visit_count);
    return Q + U;
}
```

#### 1.4 Python Bindings (Days 13-14)
**Deliverable**: Python-callable interface

```cpp
// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cuda_mcts, m) {
    py::class_<TrigoGame>(m, "TrigoGame")
        .def(py::init<int, int, int>())
        .def("apply_move", &TrigoGame::apply_move)
        .def("get_legal_moves", &TrigoGame::get_legal_moves)
        .def("to_tgn", &TrigoGame::to_tgn)
        .def_static("from_tgn", &TrigoGame::from_tgn);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, int>())
        .def("select_leaf", &MCTSEngine::select_leaf)
        .def("expand_node", &MCTSEngine::expand_node)
        .def("backup", &MCTSEngine::backup)
        .def("get_best_move", &MCTSEngine::get_best_move);
}
```

**Python usage**:
```python
import cuda_mcts

game = cuda_mcts.TrigoGame(3, 3, 1)
moves = game.get_legal_moves()
game.apply_move(moves[0])
tgn = game.to_tgn()
```

### Validation (Phase 1)

```python
# tests/test_game_state.py
def test_capture_detection():
    """Test capture matches TypeScript"""
    # Create identical board state in both
    ts_game = run_typescript_game(moves)
    cpp_game = cuda_mcts.TrigoGame(3, 3, 1)
    for move in moves:
        cpp_game.apply_move(move)

    assert cpp_game.to_tgn() == ts_game.to_tgn()

def test_ko_rule():
    """Test Ko prevention"""
    game = cuda_mcts.TrigoGame(5, 5, 1)
    # Set up Ko situation
    # Attempt recapture - should fail
    assert not game.apply_move(ko_move)
```

**Phase 1 Complete**: Python-callable C++ MCTS generates legal games (CPU-only)

---

## Phase 2: CUDA Kernels (Weeks 3-4)

### Goal
GPU-accelerated tree operations, 10-20× faster than CPU.

### Tasks

#### 2.1 GPU Memory Layout (Days 15-17)
**Deliverable**: Efficient GPU memory structures

```cpp
class TreeMemoryPool {
public:
    static constexpr int MAX_NODES_PER_GAME = 1'000'000;

    struct GPUPool {
        MCTSNode* nodes;           // [num_games * MAX_NODES]
        int* alloc_counters;       // [num_games]
        GameState* game_states;    // [num_games]
    };

    TreeMemoryPool(int num_games);
    ~TreeMemoryPool();

    // Allocate node (device function)
    __device__ int alloc_node(int game_id);

    GPUPool* get_device_pool() { return d_pool_; }

private:
    GPUPool* h_pool_;  // Host
    GPUPool* d_pool_;  // Device
};
```

**Board State Cache** (delta encoding):
```cpp
struct BoardStateDelta {
    int16_t x, y, z;
    int8_t stone_type;
    int8_t action;  // 0=place, 1=remove
};

class GameStateCache {
    // Root boards (full state)
    int8_t* d_root_boards;  // [num_games][9][9][9]

    // Delta history
    BoardStateDelta* d_deltas;  // [MAX_NODES][max_deltas]
    int* d_delta_counts;        // [MAX_NODES]

    // Reconstruct board at node
    __device__ void reconstruct(int node_idx, int8_t* out_board);
};
```

#### 2.2 UCB1 Selection Kernel (Days 18-20)
**Deliverable**: GPU kernel for tree traversal

```cuda
__global__ void select_leaf_kernel(
    const MCTSNode* nodes,
    const int* root_indices,
    int num_games,
    float c_puct,
    int* output_leaves
) {
    int game_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (game_id >= num_games) return;

    int current = root_indices[game_id];

    // Traverse until leaf
    while (nodes[current].first_child_idx != -1) {
        int child_idx = nodes[current].first_child_idx;
        float best_ucb = -INFINITY;
        int best_child = -1;

        float parent_visits = nodes[current].visit_count;

        // Find child with max UCB1
        while (child_idx != -1) {
            float Q = nodes[child_idx].mean_value;
            float U = c_puct * nodes[child_idx].prior_prob *
                      sqrtf(parent_visits) /
                      (1.0f + nodes[child_idx].visit_count);
            float ucb = Q + U;

            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child_idx;
            }

            child_idx = nodes[child_idx].next_sibling_idx;
        }

        current = best_child;
    }

    output_leaves[game_id] = current;
}
```

**Launch configuration**:
```cpp
int block_size = 256;
int num_blocks = (num_games + block_size - 1) / block_size;
select_leaf_kernel<<<num_blocks, block_size>>>(
    d_nodes, d_roots, num_games, c_puct, d_leaves
);
```

#### 2.3 Expansion Kernel (Days 21-23)
**Deliverable**: GPU kernel for child creation

```cuda
__global__ void expand_leaves_kernel(
    MCTSNode* nodes,
    GameState* games,
    TreeMemoryPool::GPUPool* pool,
    const int* leaf_indices,
    int leaf_count,
    const float* policy_probs,
    int num_legal_moves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leaf_count) return;

    int leaf_idx = leaf_indices[idx];
    int game_id = nodes[leaf_idx].game_id;

    // Allocate children
    int first_child = -1;
    int prev_child = -1;

    for (int i = 0; i < num_legal_moves; i++) {
        int child_idx = atomicAdd(&pool->alloc_counters[game_id], 1);
        if (child_idx >= MAX_NODES_PER_GAME) break;

        int global_idx = game_id * MAX_NODES_PER_GAME + child_idx;

        // Initialize child
        nodes[global_idx].parent_idx = leaf_idx;
        nodes[global_idx].first_child_idx = -1;
        nodes[global_idx].next_sibling_idx = -1;
        nodes[global_idx].visit_count = 0;
        nodes[global_idx].total_value = 0;
        nodes[global_idx].prior_prob = policy_probs[i];
        nodes[global_idx].game_id = game_id;

        // Link siblings
        if (first_child == -1) {
            first_child = global_idx;
        } else {
            nodes[prev_child].next_sibling_idx = global_idx;
        }
        prev_child = global_idx;
    }

    // Update parent
    nodes[leaf_idx].first_child_idx = first_child;
    nodes[leaf_idx].num_children = num_legal_moves;
}
```

#### 2.4 Backpropagation Kernel (Days 24-26)
**Deliverable**: GPU kernel for value updates

```cuda
__global__ void backup_values_kernel(
    MCTSNode* nodes,
    const int* leaf_indices,
    const float* values,
    int leaf_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leaf_count) return;

    int current = leaf_indices[idx];
    float value = values[idx];

    // Backpropagate to root
    while (current != -1) {
        // Atomic updates
        atomicAdd(&nodes[current].visit_count, 1.0f);
        atomicAdd(&nodes[current].total_value, value);

        // Update mean (non-atomic, eventual consistency OK)
        nodes[current].mean_value =
            nodes[current].total_value / nodes[current].visit_count;

        // Move to parent, flip value for opponent
        current = nodes[current].parent_idx;
        value = -value;
    }
}
```

#### 2.5 Kernel Profiling (Days 27-28)
**Deliverable**: Performance analysis and tuning

```bash
# Profile with nvprof
nvprof --print-gpu-trace ./build/trigo_mcts_test

# Analyze occupancy
nvprof --analysis-metrics -o profile.nvprof ./build/trigo_mcts_test
nvvp profile.nvprof

# Check memory transfers
nvprof --print-api-trace ./build/trigo_mcts_test
```

**Optimization targets**:
- Coalesced memory access (align structs, sequential reads)
- Minimize warp divergence (balanced tree traversal)
- Use shared memory for hot data (current node)
- Reduce atomic contention (backprop batching)

### Validation (Phase 2)

```python
def test_gpu_cpu_equivalence():
    """GPU and CPU produce same results"""
    game = cuda_mcts.TrigoGame(3, 3, 1)

    # Run MCTS on CPU
    cpu_engine = MCTSEngine(num_games=1, device='cpu')
    cpu_move = cpu_engine.search(game, num_sims=100)

    # Run MCTS on GPU
    gpu_engine = MCTSEngine(num_games=1, device='cuda')
    gpu_move = gpu_engine.search(game, num_sims=100)

    # Visit counts should be identical
    assert cpu_move == gpu_move
```

**Phase 2 Complete**: MCTS runs on GPU, 10-20× faster than CPU

---

## Phase 3: Neural Network Integration (Weeks 5-6)

### Goal
Batched policy/value inference with LibTorch.

### Tasks

#### 3.1 Prefix Tree Builder (Days 29-33)
**Deliverable**: Port from TypeScript `trigoTreeAgent.ts:77-168`

```cpp
class PrefixTreeBuilder {
public:
    struct TreeStructure {
        std::vector<int32_t> evaluated_ids;  // [m] tokens
        std::vector<float> evaluated_mask;   // [m][m] attention
        std::vector<int> move_to_leaf;       // Move → leaf index
        int num_nodes;
    };

    TreeStructure build_tree(
        const std::string& prefix_tgn,
        const std::vector<Move>& moves
    );

private:
    // Recursive grouping (port from TypeScript)
    struct TokenSeq {
        std::vector<int32_t> tokens;
        int move_idx;
    };

    TreeStructure build_recursive(
        const std::vector<TokenSeq>& seqs,
        int depth = 0
    );

    // Token conversion
    std::vector<int32_t> move_to_tokens(const Move& move);
};
```

**Algorithm** (from TypeScript):
```cpp
TreeStructure PrefixTreeBuilder::build_recursive(
    const std::vector<TokenSeq>& seqs, int depth
) {
    if (seqs.empty()) {
        return TreeStructure();
    }

    // Group by first token
    std::map<int32_t, std::vector<TokenSeq>> groups;
    for (const auto& seq : seqs) {
        if (seq.tokens.empty()) continue;
        int32_t first = seq.tokens[0];
        TokenSeq residue = seq;
        residue.tokens.erase(residue.tokens.begin());
        groups[first].push_back(residue);
    }

    // Build nodes
    TreeStructure result;
    for (const auto& [token, group] : groups) {
        result.evaluated_ids.push_back(token);

        if (!group[0].tokens.empty()) {
            // Recurse on residues
            auto subtree = build_recursive(group, depth + 1);
            result.evaluated_ids.insert(
                result.evaluated_ids.end(),
                subtree.evaluated_ids.begin(),
                subtree.evaluated_ids.end()
            );
        }
    }

    // Build ancestor mask
    int m = result.evaluated_ids.size();
    result.evaluated_mask.resize(m * m, 0.0f);
    // ... (mask construction logic)

    return result;
}
```

#### 3.2 Policy Model Adapter (Days 34-37)
**Deliverable**: TreeLM inference via LibTorch

```cpp
class PolicyModelAdapter {
public:
    PolicyModelAdapter(torch::jit::script::Module model);

    // Batch inference
    std::vector<std::vector<float>> infer_batch(
        const std::vector<std::string>& prefix_tgns,
        const std::vector<std::vector<Move>>& move_batches
    );

private:
    torch::jit::script::Module model_;
    PrefixTreeBuilder tree_builder_;

    // Convert to tensors
    torch::Tensor tokenize(const std::string& tgn);
    torch::Tensor create_prefix_tensor(const std::vector<std::string>& tgns);
    torch::Tensor create_evaluated_tensor(const std::vector<TreeStructure>& trees);
    torch::Tensor create_mask_tensor(const std::vector<TreeStructure>& trees);

    // Extract probabilities
    std::vector<float> extract_move_probs(
        const torch::Tensor& logits,
        const TreeStructure& tree
    );
};
```

**Inference**:
```cpp
std::vector<std::vector<float>> PolicyModelAdapter::infer_batch(
    const std::vector<std::string>& prefix_tgns,
    const std::vector<std::vector<Move>>& move_batches
) {
    int batch_size = prefix_tgns.size();

    // Build prefix trees
    std::vector<TreeStructure> trees;
    for (int i = 0; i < batch_size; i++) {
        trees.push_back(tree_builder_.build_tree(
            prefix_tgns[i], move_batches[i]
        ));
    }

    // Create input tensors
    auto prefix_ids = create_prefix_tensor(prefix_tgns);
    auto evaluated_ids = create_evaluated_tensor(trees);
    auto evaluated_mask = create_mask_tensor(trees);

    // Forward pass
    std::vector<torch::jit::IValue> inputs = {
        prefix_ids, evaluated_ids, evaluated_mask
    };
    auto output = model_.forward(inputs).toTensor();

    // Extract probabilities
    std::vector<std::vector<float>> all_probs;
    for (int i = 0; i < batch_size; i++) {
        auto logits = output[i];  // [m+1, vocab_size]
        all_probs.push_back(extract_move_probs(logits, trees[i]));
    }

    return all_probs;
}
```

#### 3.3 Value Model Adapter (Days 38-40)
**Deliverable**: EvaluationLM inference

```cpp
class ValueModelAdapter {
public:
    ValueModelAdapter(torch::jit::script::Module model);

    // Batch inference
    std::vector<float> infer_batch(
        const std::vector<std::string>& state_tgns
    );

private:
    torch::jit::script::Module model_;

    torch::Tensor tokenize_batch(const std::vector<std::string>& tgns);
};
```

**Simpler than policy** (no prefix tree):
```cpp
std::vector<float> ValueModelAdapter::infer_batch(
    const std::vector<std::string>& state_tgns
) {
    // Tokenize all states
    auto input_ids = tokenize_batch(state_tgns);

    // Forward (model appends VALUE token internally)
    auto output = model_.forward({input_ids}).toTensor();

    // Extract values
    std::vector<float> values;
    for (int i = 0; i < output.size(0); i++) {
        values.push_back(output[i].item<float>());
    }

    return values;
}
```

#### 3.4 Dynamic Batching (Days 41-42)
**Deliverable**: Timeout-based batch aggregation

```cpp
class DynamicBatcher {
public:
    static constexpr int MAX_BATCH = 64;
    static constexpr int TIMEOUT_MS = 10;

    struct EvalRequest {
        int leaf_idx;
        std::string state_tgn;
        std::vector<Move> legal_moves;
        std::promise<EvalResult> promise;
    };

    // Submit request (called by MCTS threads)
    std::future<EvalResult> submit(EvalRequest req);

    // Process batch (inference thread)
    void process_batch();

private:
    std::queue<EvalRequest> pending_;
    std::mutex mutex_;
    std::condition_variable cv_;

    PolicyModelAdapter policy_;
    ValueModelAdapter value_;
};
```

**Usage**:
```cpp
// MCTS thread
auto future = batcher.submit({leaf_idx, tgn, moves});

// Wait for result
auto result = future.get();  // Blocks until batch processed
expand_node(leaf_idx, result.policy_probs);
backup(leaf_idx, result.value);
```

### Validation (Phase 3)

```python
def test_policy_inference():
    """Test TreeLM integration"""
    policy = torch.jit.load("models/policy_tree.pt")
    adapter = cuda_mcts.PolicyModelAdapter(policy)

    game = cuda_mcts.TrigoGame(3, 3, 1)
    moves = game.get_legal_moves()
    probs = adapter.infer_batch([game.to_tgn()], [moves])

    # Check probabilities sum to 1
    assert abs(sum(probs[0]) - 1.0) < 1e-5
    assert len(probs[0]) == len(moves)

def test_value_inference():
    """Test EvaluationLM integration"""
    value = torch.jit.load("models/value_eval.pt")
    adapter = cuda_mcts.ValueModelAdapter(value)

    game = cuda_mcts.TrigoGame(3, 3, 1)
    values = adapter.infer_batch([game.to_tgn()])

    # Check value in range [-1, 1]
    assert -1.0 <= values[0] <= 1.0
```

**Phase 3 Complete**: Full MCTS with neural network guidance

---

## Phase 4: Validation & Optimization (Weeks 7-8)

### Goal
Correctness verified, performance optimized, 300+ games/hour achieved.

### Tasks

#### 4.1 Comprehensive Validation Suite (Days 43-47)
**Deliverable**: Validation against TypeScript golden reference

```python
# tools/validate_mcts.py
import subprocess
import cuda_mcts
import torch
from pathlib import Path

def run_typescript_selfplay(board_shape, seed, num_games):
    """Generate games with TypeScript"""
    cmd = [
        "npx", "tsx",
        "../trigoRL/third_party/trigo/trigo-web/tools/selfPlayGames.ts",
        "--games", str(num_games),
        "--board", f"{board_shape[0]}*{board_shape[1]}*{board_shape[2]}",
        "--seed", str(seed),
        "--output", "validation/ts_output"
    ]
    subprocess.run(cmd, check=True, cwd=".")
    return list(Path("validation/ts_output").glob("*.tgn"))

def run_cuda_selfplay(board_shape, seed, num_games):
    """Generate games with C++/CUDA"""
    torch.manual_seed(seed)

    policy = torch.jit.load("models/policy_tree.pt")
    value = torch.jit.load("models/value_eval.pt")

    engine = cuda_mcts.CudaMCTSSelfPlay(
        policy, value,
        num_parallel_games=1,  # Deterministic
        mcts_simulations=800
    )

    return engine.generate_games(num_games, [board_shape])

def validate_game(tgn):
    """Validate single game"""
    from trigor.data import TGNValueDataset

    # Parse with TGNValueDataset
    dataset = TGNValueDataset.from_string(tgn)

    # Check all moves legal
    game = cuda_mcts.TrigoGame.from_tgn(tgn)
    assert game.is_valid(), "Game contains illegal moves"

    # Check territory score
    territory = game.get_territory()
    score_diff = territory.white - territory.black
    assert tgn.endswith(f"; {score_diff}"), "Score mismatch"

    return True

def main():
    test_cases = [
        {"board": (3, 3, 1), "seed": 42, "games": 10},
        {"board": (5, 5, 1), "seed": 123, "games": 10},
        {"board": (2, 2, 2), "seed": 456, "games": 10},
    ]

    for tc in test_cases:
        print(f"\nTesting {tc['board']} with seed {tc['seed']}...")

        # Generate with both implementations
        ts_games = run_typescript_selfplay(**tc)
        cuda_games = run_cuda_selfplay(**tc)

        # Validate CUDA games
        for i, tgn in enumerate(cuda_games):
            print(f"  Validating game {i+1}/{len(cuda_games)}...", end="")
            assert validate_game(tgn), f"Game {i+1} invalid"
            print(" ✓")

        print(f"✓ All {len(cuda_games)} games valid")
```

#### 4.2 Performance Profiling (Days 48-50)
**Deliverable**: Bottleneck analysis

```bash
# Profile MCTS
nsys profile --stats=true -o profile.qdrep \
    python scripts/benchmark_mcts.py

# Analyze
nsys-ui profile.qdrep

# Check kernel execution
ncu --set full -o kernel_profile \
    python scripts/benchmark_mcts.py

# Memory bandwidth
ncu --metrics dram__bytes.sum.per_second \
    python scripts/benchmark_mcts.py
```

**Metrics to collect**:
- Total games/hour
- MCTS simulations/second
- Kernel execution time (select, expand, backup)
- Inference time (policy, value)
- Memory transfer overhead
- GPU utilization

#### 4.3 Kernel Optimization (Days 51-54)
**Deliverable**: Tuned CUDA kernels

**Optimizations**:

1. **Coalesced Memory Access**:
```cuda
// Bad: Strided access
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    output[i * stride] = input[i * stride];  // Uncoalesced
}

// Good: Sequential access
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    output[i] = input[i];  // Coalesced
}
```

2. **Shared Memory for Hot Data**:
```cuda
__global__ void select_leaf_optimized(...) {
    __shared__ MCTSNode current_node;
    __shared__ float ucb_scores[MAX_CHILDREN];

    // Load current node to shared memory
    if (threadIdx.x == 0) {
        current_node = nodes[current_idx];
    }
    __syncthreads();

    // Parallel UCB computation
    int child_idx = current_node.first_child_idx + threadIdx.x;
    if (threadIdx.x < current_node.num_children) {
        ucb_scores[threadIdx.x] = compute_ucb1(nodes[child_idx], ...);
    }
    __syncthreads();

    // Reduction to find max
    // ...
}
```

3. **Warp-Level Primitives**:
```cuda
// Find max UCB across warp
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
```

#### 4.4 TGN Writer (Days 55-56)
**Deliverable**: Output formatting with hash-based filenames

```cpp
class TGNWriter {
public:
    // Write games to directory
    void write_games(
        const std::vector<std::string>& tgn_strings,
        const std::string& output_dir
    );

private:
    // Generate filename from content hash
    std::string hash_filename(const std::string& tgn);
};
```

**Implementation**:
```cpp
std::string TGNWriter::hash_filename(const std::string& tgn) {
    // SHA-256 hash
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((unsigned char*)tgn.c_str(), tgn.length(), hash);

    // Convert to hex (first 16 chars)
    std::ostringstream ss;
    ss << "game_";
    for (int i = 0; i < 8; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << (int)hash[i];
    }
    ss << ".tgn";

    return ss.str();
}
```

### Validation (Phase 4)

```bash
# Run full validation suite
python tools/validate_mcts.py

# Expected output:
# Testing (3, 3, 1) with seed 42...
#   Validating game 1/10... ✓
#   ...
# ✓ All 10 games valid
#
# Testing (5, 5, 1) with seed 123...
#   ...
#
# ✓ ALL TESTS PASSED
```

**Phase 4 Complete**: Validated system hitting 300+ games/hour

---

## Phase 5: Production Integration (Weeks 9-10)

### Goal
Full AlphaZero training loop integrated with TrigoRL.

### Tasks

#### 5.1 Python Scripts (Days 57-59)
**Deliverable**: User-facing scripts

```python
# scripts/generate_selfplay_cuda.py
import argparse
import torch
import cuda_mcts
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--output", default="data/selfplay")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=800)
    args = parser.parse_args()

    # Load models
    policy = torch.jit.load(args.policy)
    value = torch.jit.load(args.value)

    # Create engine
    engine = cuda_mcts.CudaMCTSSelfPlay(
        policy, value,
        num_parallel_games=args.parallel,
        mcts_simulations=args.simulations
    )

    # Generate games
    board_shapes = [(3,3,1), (5,5,1), (2,2,2), (3,3,3)]
    tgn_games = engine.generate_games(
        args.games, board_shapes,
        progress_callback=lambda cur, tot, rate:
            print(f"\r[{cur}/{tot}] {rate:.1f} games/sec", end="")
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tgn in tgn_games:
        filename = hash_filename(tgn)
        (output_dir / filename).write_text(tgn)

    print(f"\n✓ Saved {len(tgn_games)} games to {output_dir}")

if __name__ == "__main__":
    main()
```

#### 5.2 AlphaZero Training Loop (Days 60-62)
**Deliverable**: End-to-end training script

```python
# scripts/alphazero_training_loop.py
import torch
import cuda_mcts
from pathlib import Path
from trigor.training import LMTrainer
from trigor.data import TGNValueDataset, get_tokenizer
from trigor.models import ValueCausalLoss
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="alphazero")
def main(cfg: DictConfig):
    for iteration in range(cfg.num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}")
        print(f"{'='*80}")

        # Step 1: Self-play with current model
        print("\n[1/4] Generating self-play games...")
        policy = torch.jit.load(
            f"checkpoints/iter{iteration}_policy.pt"
        )
        value = torch.jit.load(
            f"checkpoints/iter{iteration}_value.pt"
        )

        engine = cuda_mcts.CudaMCTSSelfPlay(
            policy, value,
            num_parallel_games=cfg.num_parallel_games,
            mcts_simulations=cfg.mcts_simulations,
            c_puct=cfg.c_puct,
            temperature=cfg.temperature
        )

        tgn_games = engine.generate_games(
            num_games=cfg.games_per_iteration,
            board_shapes=cfg.board_shapes
        )

        # Step 2: Save TGN files
        print(f"\n[2/4] Saving {len(tgn_games)} games...")
        output_dir = Path(f"data/iter{iteration}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, tgn in enumerate(tgn_games):
            (output_dir / f"game_{i:06d}.tgn").write_text(tgn)

        # Step 3: Train on new data
        print(f"\n[3/4] Training on new data...")
        tokenizer = get_tokenizer()
        dataset = TGNValueDataset(
            data_dir=str(output_dir),
            tokenizer=tokenizer,
            max_length=cfg.max_seq_len
        )

        model = ValueCausalLoss.from_config(cfg.model)
        trainer = LMTrainer(cfg, model, dataset)
        trainer.train(num_epochs=cfg.epochs_per_iteration)

        # Step 4: Export for next iteration
        print(f"\n[4/4] Exporting models...")
        export_to_torchscript(
            trainer.model,
            f"checkpoints/iter{iteration+1}_policy.pt",
            f"checkpoints/iter{iteration+1}_value.pt"
        )

        print(f"\n✓ Iteration {iteration} complete")

    print(f"\n{'='*80}")
    print(f"AlphaZero training complete!")
    print(f"{'='*80}")

def export_to_torchscript(model, policy_path, value_path):
    """Export TreeLM and EvaluationLM to TorchScript"""
    from trigor.models import TreeLM, EvaluationLM

    # Export TreeLM (policy)
    tree_lm = TreeLM(model.model)
    tree_lm.eval()
    traced_policy = torch.jit.trace(tree_lm, ...)
    traced_policy.save(policy_path)

    # Export EvaluationLM (value)
    eval_lm = EvaluationLM(model)
    eval_lm.eval()
    traced_value = torch.jit.trace(eval_lm, ...)
    traced_value.save(value_path)

if __name__ == "__main__":
    main()
```

#### 5.3 Documentation (Days 63-66)
**Deliverable**: Complete documentation

Files to create:
- `docs/API.md` - Python and C++ API reference
- `docs/ARCHITECTURE.md` - Technical design details
- `docs/PORTING.md` - TypeScript → C++ notes
- `docs/PERFORMANCE.md` - Optimization guide
- `docs/TROUBLESHOOTING.md` - Common issues

#### 5.4 Benchmarking (Days 67-68)
**Deliverable**: Performance report

```python
# scripts/benchmark.py
def benchmark_throughput():
    """Measure games/hour"""
    engine = cuda_mcts.CudaMCTSSelfPlay(...)

    start = time.time()
    tgn_games = engine.generate_games(num_games=100, ...)
    elapsed = time.time() - start

    throughput = len(tgn_games) / elapsed * 3600
    print(f"Throughput: {throughput:.1f} games/hour")

def benchmark_scaling():
    """Test parallel scaling"""
    for num_parallel in [1, 2, 4, 8]:
        engine = cuda_mcts.CudaMCTSSelfPlay(
            ..., num_parallel_games=num_parallel
        )
        # Measure throughput
```

#### 5.5 Production Deployment (Days 69-70)
**Deliverable**: Production-ready system

- Error handling and logging
- Configuration management
- Resource monitoring
- Graceful shutdown
- Documentation updates

### Validation (Phase 5)

```bash
# Run full AlphaZero iteration
python scripts/alphazero_training_loop.py \
    --config configs/alphazero.yaml

# Expected: Complete iteration in < 2 hours
```

**Phase 5 Complete**: Full AlphaZero training loop operational

---

## Success Criteria

### Correctness
- ✅ All validation tests pass vs TypeScript
- ✅ 10,000+ games generated without crashes
- ✅ TGN output parses with TGNValueDataset
- ✅ No illegal moves, captures match, Ko works

### Performance
- ✅ Achieve 300-1000 games/hour on single GPU
- ✅ MCTS 1000-5000 simulations/second
- ✅ GPU utilization > 80%
- ✅ Memory usage < 2GB

### Integration
- ✅ Works in AlphaZero training loop
- ✅ Python API clean and documented
- ✅ Models export/import correctly
- ✅ Compatible with TrigoRL pipeline

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Game logic bugs | HIGH | Extensive validation vs TypeScript, unit tests |
| GPU memory overflow | MEDIUM | Dynamic node pools, monitoring, degradation |
| Performance < target | MEDIUM | Profiling, kernel optimization, batching |
| Prefix tree bugs | HIGH | Test with TypeScript golden outputs |
| Training instability | LOW | Use proven model architectures |

---

## Development Guidelines

### Code Style
- C++17 standard
- Google C++ Style Guide
- Comments for complex algorithms
- Document TypeScript source locations

### Testing
- Unit tests for each component
- Integration tests for end-to-end flow
- Validation against TypeScript golden
- Performance regression tests

### Version Control
- Feature branches for each phase
- Code review before merge
- Tag releases (v0.1, v0.2, ...)
- Document breaking changes

---

## Current Status

**Phase 0: Planning** ✅ COMPLETE
- Architecture designed
- Feasibility confirmed
- Critical files identified
- Performance targets set

**Next**: Begin Phase 1 - Core Infrastructure

---

## References

### TypeScript Files to Port
1. `trigoRL/third_party/trigo/trigo-web/inc/trigo/game.ts` (800 lines)
2. `trigoRL/third_party/trigo/trigo-web/inc/trigo/gameUtils.ts` (500 lines)
3. `trigoRL/third_party/trigo/trigo-web/inc/trigoTreeAgent.ts` (lines 77-168)

### Python Integration Points
4. `trigoRL/trigor/models/treeLM.py`
5. `trigoRL/trigor/models/evaluationLM.py`
6. `trigoRL/trigor/data/tgn_value_dataset.py`

### Validation Reference
7. `trigoRL/third_party/trigo/trigo-web/tools/selfPlayGames.ts`

---

**Estimated Timeline**: 10 weeks to production-ready system
**Team**: 1-2 developers
**Hardware**: CUDA-capable GPU (RTX 2060+)
