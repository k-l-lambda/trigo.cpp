# CUDA MCTS Feasibility Research

## Overview

This document evaluates the feasibility of rewriting MCTS algorithm using CUDA for GPU acceleration in the Trigo project.

## Current Implementation Analysis

### Existing MCTS Architecture

**Files**: `/home/camus/work/trigo.cpp/include/`
- `mcts.hpp`: AlphaZero-style MCTS with SharedModelInferencer
- `cached_mcts.hpp`: MCTS with KV-cache using PrefixCacheInferencer

**Current Flow** (per simulation):
```
1. Select: Traverse tree using PUCT (serial)
2. Expand: Create children with policy priors (NN inference)
3. Evaluate: Value network inference
4. Backpropagate: Update statistics (serial)
```

### Profiling Summary

**Model Specifications**:
- Architecture: GPT2-style transformer
- Size: 6 layers, 64 hidden dim, 8 attention heads
- Parameters: ~500K
- Input: Variable-length TGN token sequence (typically 10-50 tokens)

**Per-Simulation Breakdown** (measured on RTX 2060):

| Component | Time | Details |
|-----------|------|---------|
| Selection | <0.5ms | Tree traversal, PUCT calculation |
| Policy Inference | ~2ms | Single forward pass, batch=1 |
| Value Inference | ~1ms | Single forward pass, batch=1 |
| Tree Building | ~0.5ms | Prefix tree construction, node allocation |
| Backprop | <0.1ms | Statistics update |
| **Total per sim** | ~4ms | |

**Per-Move Breakdown** (50 simulations):

| Stage | Time | Percentage |
|-------|------|------------|
| NN Inference (50×) | ~140ms | ~93% |
| Tree Operations | ~10ms | ~7% |
| **Total** | ~150ms | 100% |

**Current Benchmarks** (5×5×1 board, 50 simulations):

| Policy | Device | Time/Move | Games/sec |
|--------|--------|-----------|-----------|
| AlphaZero | GPU | 150ms | 0.222 |
| AlphaZero | CPU | 179ms | 0.185 |
| Cached-MCTS | CPU | 611ms | 0.037 |

**Key Observation**: Neural network inference dominates execution time (~93%). Each move requires 50 sequential NN calls (one per simulation), which is highly inefficient for GPU utilization.

## GPU Parallelization Approaches

### Approach 1: MCTS-NC (Pure GPU MCTS with Random Rollouts)

**Source**: [Klesk et al., 2024 - MCTS-NC](https://github.com/pklesk/mcts_numba_cuda)

**Key Techniques**:
- Three-level parallelization: Leaf / Root / Tree
- All MCTS stages parallelized: selection, expansion, playouts, backup
- Lock-free design (but uses reduction patterns, not truly atomic-free)
- Minimal device-host memory transfers

**Performance** (Connect 4, 5s/move, single-thread CPU baseline):
- Vanilla CPU MCTS: 17.1k playouts
- GPU MCTS-NC: 2.3M playouts (**135× speedup**)

**Important Context**:
- Speedup is vs **single-threaded** CPU baseline
- Uses random rollouts, not neural network evaluation
- Game-specific (Connect 4 has small branching factor)
- Not directly comparable to AlphaZero-style MCTS

**Pros**:
- Massive speedup for pure MCTS
- Well-documented implementation
- Proven for Connect 4, Gomoku

**Cons**:
- Uses random rollouts, NOT neural network evaluation
- Python/Numba, not native C++/CUDA
- **Not applicable to AlphaZero-style MCTS**

### Approach 2: Full GPU MCTS (AlphaGPU/Boardlaw)

**Source**: [AlphaGPU - Julia](https://discourse.julialang.org/t/a-simple-full-gpu-implementation-of-alphazero/103726)

**Key Idea**:
- Entire MCTS (tree search + NN inference) runs on GPU
- Tree structure stored in GPU memory
- Eliminates CPU-GPU data transfer overhead

**Challenges**:
1. **Dynamic memory allocation**: Tree grows unpredictably during search
2. **Warp divergence**: Different threads traverse different path lengths, causing severe underutilization
3. **Memory layout**: Need Structure-of-Arrays (SoA) for coalesced access
4. **Synchronization**: Atomic operations needed for visit count updates
5. **Irregular control flow**: PUCT selection varies per node

**Why it's difficult**:
- GPU excels at regular, data-parallel workloads
- MCTS tree traversal is inherently irregular
- Trade-off between parallelism and synchronization overhead

### Approach 3: Tree-Parallel DNN-MCTS (Batch Leaf Evaluation)

**Source**: [arxiv:2310.05313](https://arxiv.org/pdf/2310.05313)

**Key Idea**:
1. Run N parallel simulations to leaf nodes (CPU threads)
2. Collect N leaf states into a batch
3. Single batched neural network inference (batch=N)
4. Parallel backpropagation

**Virtual Loss Technique**:
- Prevents multiple simulations from selecting same path
- Temporarily add a "virtual loss" to visited nodes
- Formula: Q_virtual = (W - n_virtual) / (N + n_virtual)
- Corrected during backpropagation

**Performance Expectations** (from literature):
- NN throughput improvement: 3-10× with batch=32-128
- End-to-end speedup depends on NN% of runtime

**Speedup Analysis for Trigo**:
```
Current: NN = 93% of runtime
If NN throughput improves by K×:
  End-to-end speedup S ≈ 1 / (0.07 + 0.93/K)

K=4:  S ≈ 1 / (0.07 + 0.23) = 3.3×
K=8:  S ≈ 1 / (0.07 + 0.12) = 5.3×
K=16: S ≈ 1 / (0.07 + 0.06) = 7.7×
```

**Realistic expectation**: 3-5× end-to-end speedup

### Approach 4: MuZero Architecture

**Source**: [Schrittwieser et al., 2020](https://arxiv.org/abs/1911.08265)

**Key Changes**:
- Fixed-size hidden state (avoids variable-length TGN)
- Dynamics network predicts next latent state
- No tokenization overhead

**Components**:
1. Representation network: board → hidden state (h)
2. Dynamics network: (h, action) → h' (constant size)
3. Prediction network: h → (policy, value)

**Advantages for GPU**:
- Fixed tensor sizes enable efficient batching
- No prefix tree construction
- Dynamics network can be very lightweight

**Cons**:
- Requires complete model retraining
- Different training paradigm (model-based RL)
- Significant implementation effort

## Additional Approaches (from GPT-5.1 review)

### Approach 5: Hybrid CPU-GPU Parallelization

**Key Idea**:
- CPU threads handle tree traversal (8-32 threads)
- GPU handles batched NN inference
- Dynamic batching with request queue

**Implementation**:
```
CPU Thread Pool (8 threads)
    │
    ├── Thread 1: Select → Push leaf to queue
    ├── Thread 2: Select → Push leaf to queue
    ├── ...
    └── Thread 8: Select → Push leaf to queue
                      │
                      ▼
              Inference Queue
                      │
                      ▼ (when batch full or timeout)
              GPU Batch Inference
                      │
                      ▼
              Results Distribution
                      │
                      ▼
              Parallel Backprop
```

**Pros**:
- No CUDA kernel development needed
- Works with existing ONNX Runtime
- Good CPU-GPU overlap potential

### Approach 6: NN-Level Optimizations

Before algorithmic changes, consider:
1. **Model quantization**: FP16 or INT8 inference
2. **TensorRT optimization**: Fused kernels, layer optimization
3. **Smaller model**: Trade accuracy for speed
4. **Batched inference API**: Even without parallel MCTS

**Potential gains**: 2-3× without algorithmic changes

### Approach 7: Root Parallelization

**Key Idea**:
- Run K independent MCTS trees with different seeds
- Aggregate root statistics at the end

**Pros**:
- Simple to implement
- Perfect parallelism
- Works with existing code

**Cons**:
- K× memory usage
- May not be as statistically efficient as single-tree

## Trigo-Specific Considerations

### Current Architecture Constraints

1. **Neural Network**: ONNX Runtime (already GPU-enabled)
2. **Policy Network**: Requires prefix tree construction (complex token structure)
3. **Value Network**: Simple position evaluation
4. **TGN Format**: Game state needs tokenization (variable length)

### Why KV-Cache Doesn't Help MCTS

The prefix cache optimization was designed for:
- Linear sequence extension (autoregressive generation)
- Sharing computation across similar prefixes

**Problem with MCTS**:
- MCTS explores a tree, not a linear sequence
- Each simulation path diverges at different points
- Cache must be recomputed for each divergent path
- Result: 6× slowdown instead of expected 3-4× speedup

## Recommended Approach: Batch Leaf Evaluation

### Implementation Sketch

```cpp
class BatchMCTS {
private:
    int batch_size = 32;
    int num_threads = 8;
    ThreadPool thread_pool;
    ConcurrentQueue<LeafRequest> leaf_queue;

public:
    PolicyAction search(const TrigoGame& game) {
        // Initialize tree
        root = create_root(game);

        int total_sims = num_simulations;
        int completed = 0;

        while (completed < total_sims) {
            // 1. Parallel selection with virtual loss
            vector<future<LeafNode*>> futures;
            for (int i = 0; i < batch_size && completed + i < total_sims; i++) {
                futures.push_back(thread_pool.submit([&]() {
                    return select_with_virtual_loss(root, game);
                }));
            }

            // 2. Collect leaf nodes
            vector<LeafNode*> leaves;
            for (auto& f : futures) {
                leaves.push_back(f.get());
            }

            // 3. Batch NN inference
            auto [policies, values] = batch_inference(leaves);

            // 4. Expand and backpropagate
            for (int i = 0; i < leaves.size(); i++) {
                expand(leaves[i], policies[i]);
                backpropagate(leaves[i], values[i]);
            }

            completed += leaves.size();
        }

        return select_best_child(root);
    }
};
```

### Virtual Loss Implementation

```cpp
MCTSNode* select_with_virtual_loss(MCTSNode* node, const TrigoGame& game) {
    while (!node->is_leaf()) {
        // Apply virtual loss before selection
        node->virtual_losses++;

        // Select best child with virtual loss in Q calculation
        float best_score = -INF;
        MCTSNode* best = nullptr;
        for (auto& child : node->children) {
            float q = (child->total_value - child->virtual_losses)
                    / (child->visit_count + child->virtual_losses);
            float u = c_puct * child->prior * sqrt(node->visit_count)
                    / (1 + child->visit_count + child->virtual_losses);
            float score = q + u;
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        node = best;
    }
    return node;
}

void backpropagate(MCTSNode* node, float value) {
    while (node != nullptr) {
        node->virtual_losses--;  // Remove virtual loss
        node->visit_count++;
        node->total_value += value;
        value = -value;
        node = node->parent;
    }
}
```

### Required Changes

1. **SharedModelInferencer**: Add batch inference API
   - `batch_value_inference(vector<vector<int64_t>>& token_batches)`
   - `batch_policy_inference(vector<TreeStructure>& tree_batches)`

2. **MCTSNode**: Add virtual loss tracking
   - `atomic<int> virtual_losses`

3. **New BatchMCTS class**: Implements parallel selection and batched evaluation

4. **Thread pool**: For parallel CPU operations

## Feasibility Matrix (Revised)

| Approach | Complexity | Expected Speedup | Fits Trigo | Priority |
|----------|------------|------------------|------------|----------|
| Batch Leaf Evaluation | Medium | 3-5× | ✓ | **1** |
| NN Optimization (FP16/TensorRT) | Low | 2-3× | ✓ | **2** |
| Root Parallelization | Low | ~K× (memory cost) | ✓ | 3 |
| CUDA Kernel MCTS | High | 5-20× | Uncertain | 4 |
| MuZero Architecture | Very High | Significant | ✓ | 5 (long-term) |

## Implementation Roadmap

### Phase 1: NN Optimization (1 week)
1. Enable FP16 inference in ONNX Runtime
2. Profile and identify remaining bottlenecks
3. Consider TensorRT conversion

### Phase 2: Batch Inference Infrastructure (1-2 weeks)
1. Add batch value inference to SharedModelInferencer
2. Add batch policy inference support
3. Unit tests for batch correctness

### Phase 3: Batch MCTS Implementation (2-3 weeks)
1. Implement `BatchMCTS` class
2. Implement Virtual Loss mechanism
3. Thread pool for parallel select/backprop
4. Benchmark against current AlphaZero policy

### Phase 4: Optimization (1-2 weeks)
1. Tune batch size for optimal throughput
2. Profile CPU-GPU overlap
3. Memory optimization (node pooling)

## Conclusion

**Recommendation**: Start with Batch Leaf Evaluation

**Revised Rationale**:
1. Current bottleneck is NN inference (93% of runtime)
2. Batching can improve NN throughput 4-8×
3. Expected **end-to-end speedup: 3-5×** (not 5-10×)
4. Moderate implementation complexity
5. Foundation for future optimizations

**Secondary Recommendations**:
1. Enable FP16 inference (easy win)
2. Consider TensorRT for further NN optimization
3. Root parallelization as simple alternative

**Not recommended**: Full CUDA rewrite of MCTS tree traversal

**Reasons**:
1. NN inference dominates; tree operations are only 7%
2. Irregular control flow causes GPU underutilization
3. High implementation cost with uncertain benefits
4. Batch evaluation achieves similar benefits with less effort

## References

1. MCTS-NC: https://github.com/pklesk/mcts_numba_cuda
2. AlphaGPU: https://discourse.julialang.org/t/a-simple-full-gpu-implementation-of-alphazero/103726
3. Tree-Parallel DNN-MCTS: https://arxiv.org/pdf/2310.05313
4. MuZero: https://arxiv.org/abs/1911.08265
5. AlphaZero: https://arxiv.org/abs/1712.01815

---

*Document reviewed by GPT-5.1 on December 11, 2025. Key feedback incorporated: refined speedup estimates, added profiling details, expanded alternative approaches, clarified implementation sketch.*
