# Full MCTS Integration with Shared Cache - Remaining Work

## Current Status

### What's Already Done ✅

1. **PrefixCacheInferencer** - Complete
   - `compute_prefix_cache()` - Compute KV cache once
   - `policy_inference_with_cache()` - Policy evaluation using cache
   - `value_inference_with_cache()` - Value evaluation using cache
   - Both inference methods share the same prefix cache

2. **CachedAlphaZeroPolicy** - Simplified Implementation
   - Loads all 4 models (prefix, eval_cached, policy_head, value_head)
   - Demonstrates cache sharing concept
   - Simple value-based move selection (not full MCTS)
   - Integrated with PolicyFactory

3. **MCTS Class** - Standard Implementation (without cache)
   - Uses `SharedModelInferencer` (standard inference, no cache)
   - Full AlphaZero MCTS algorithm:
     * Selection (PUCT formula)
     * Expansion (creates child nodes)
     * Evaluation (value network)
     * Backpropagation (updates statistics)
   - Works correctly but doesn't use prefix cache optimization

---

## What's Missing for Full Integration

### 1. Policy Prior Integration in Expansion

**Current State** (mcts.hpp:343-345):
```cpp
// Select random unexpanded move (could use policy network here)
MCTSNode* new_child = nullptr;
float prior = 1.0f;  // Uniform prior for now (could use policy network)
```

**What's Needed**:
- Modify `expand()` to call `policy_inference_with_cache()` to get move priors
- Use policy network logits to set proper `prior_prob` for each child node
- This guides tree exploration using policy priors (key AlphaZero feature)

**Implementation**:
```cpp
// In expand() method:
// 1. Compute prefix cache once for current game state
inferencer->compute_prefix_cache(game_tokens, 1, seq_len);

// 2. Build tree structure for all valid moves
PrefixTreeStructure tree = build_tree_for_moves(game, valid_moves);

// 3. Get policy logits for all moves using cache
auto hidden_states = inferencer->evaluate_with_cache(
    tree.evaluated_ids, tree.evaluated_mask, 1, tree.evaluated_len
);
auto policy_logits = inferencer->policy_inference_from_hidden(hidden_states, ...);

// 4. Extract prior probabilities for each move
// 5. Create child nodes with proper priors
```

---

### 2. Cached MCTS Class

**What's Needed**:
Create new `CachedMCTS` class that uses `PrefixCacheInferencer` instead of `SharedModelInferencer`.

**Key Differences from Current MCTS**:
```cpp
class CachedMCTS
{
private:
    std::unique_ptr<PrefixCacheInferencer> inferencer;  // ← Use cached inferencer
    // ... rest of MCTS infrastructure

public:
    PolicyAction search(const TrigoGame& game)
    {
        // CRITICAL: Compute prefix cache ONCE at root
        auto game_tokens = game_to_tokens(game);
        inferencer->compute_prefix_cache(game_tokens, 1, seq_len);

        // Run simulations (all use the same cache)
        for (int sim = 0; sim < num_simulations; sim++)
        {
            // Selection, expansion, evaluation, backprop
            // All value evaluations reuse the root cache
        }
    }

    float evaluate(TrigoGame& game)
    {
        // Use value_inference_with_cache() instead of standard inference
        return inferencer->value_inference_with_cache(3);  // VALUE token
    }
};
```

**Challenge**: Current MCTS modifies game state during tree traversal
- Each node represents a different game state
- Cache is computed for root position, but child positions are different
- Need to decide: Re-compute cache for each node, or share root cache?

**Two Approaches**:

**Approach A: Per-Node Cache** (Most Accurate)
- Compute separate cache for each MCTS node
- Pro: Most accurate, each position has optimal cache
- Con: More memory, cache computation overhead

**Approach B: Root Cache Only** (Faster)
- Share root cache across all simulations
- Only works if tree doesn't go too deep from root
- Pro: Maximum cache reuse, minimal overhead
- Con: Less accurate for deep nodes (cache mismatch)

**Recommended: Hybrid Approach**
- Compute cache at root and key decision points
- Reuse cache for shallow nodes (depth < 5)
- Re-compute for deeper nodes or major branches

---

### 3. Batch Inference for Leaf Evaluation

**Current State**: Sequential evaluation
```cpp
// Each simulation evaluates one leaf position
for (int sim = 0; sim < num_simulations; sim++)
{
    float value = evaluate(game_copy);  // One at a time
}
```

**Optimization Opportunity**:
- Collect multiple leaf nodes
- Batch evaluate them together
- Reuse shared cache across batch

**Implementation**:
```cpp
// Collect leaf positions from multiple simulations
std::vector<TrigoGame> leaf_positions;

// Phase 1: Run simulations to leaf nodes (no evaluation)
for (int sim = 0; sim < num_simulations; sim += batch_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        MCTSNode* leaf = select_and_expand(root.get(), game_copy);
        leaf_positions.push_back(game_copy);
    }

    // Phase 2: Batch evaluate all leaves
    auto values = batch_value_inference(leaf_positions);

    // Phase 3: Backpropagate
    for (int b = 0; b < batch_size; b++)
    {
        backpropagate(leaves[b], values[b]);
    }
}
```

---

### 4. Integration with PolicyFactory

**Current State**:
- `PolicyFactory` has "alphazero" type → uses standard MCTS
- `PolicyFactory` has "cached-alphazero" type → uses simplified CachedAlphaZeroPolicy

**What's Needed**:
- Update "cached-alphazero" to use full CachedMCTS implementation
- Or add new "cached-mcts" type for full integration

```cpp
else if (type == "cached-mcts")
{
    // Full AlphaZero MCTS with shared cache
    if (model_path.empty())
    {
        throw std::runtime_error("Cached MCTS requires model_path");
    }

    auto inferencer = std::make_shared<PrefixCacheInferencer>(...);
    return std::make_unique<CachedMCTS>(inferencer, num_sims, c_puct, seed);
}
```

---

## Implementation Plan

### Phase A: Basic Integration (Low Complexity)

1. **Create CachedMCTS class** (1-2 hours)
   - Copy MCTS class structure
   - Replace `SharedModelInferencer` with `PrefixCacheInferencer`
   - Update `evaluate()` to use `value_inference_with_cache()`
   - Keep root-only cache for simplicity

2. **Update expand() with policy priors** (2-3 hours)
   - Call `policy_inference_with_cache()` during expansion
   - Extract move priors from policy logits
   - Set proper `prior_prob` for child nodes

3. **Integration testing** (1 hour)
   - Test CachedMCTS on sample games
   - Verify correctness vs standard MCTS
   - Measure performance improvement

4. **PolicyFactory integration** (30 mins)
   - Add "cached-mcts" type
   - Update documentation

**Expected Result**:
- Full AlphaZero MCTS with cache support
- 2-3× speedup over standard MCTS
- Combined: ~35-40× faster than TypeScript

---

### Phase B: Advanced Optimization (Medium Complexity)

1. **Hybrid cache strategy** (2-3 hours)
   - Cache at root + key decision points
   - Heuristic for when to recompute cache
   - Balance accuracy vs performance

2. **Batch leaf evaluation** (3-4 hours)
   - Collect multiple leaf nodes
   - Batch inference API
   - Synchronization and backpropagation

3. **Performance tuning** (2-3 hours)
   - Profile cache hit rates
   - Optimize memory usage
   - Tune batch sizes

**Expected Result**:
- 5-10× additional speedup for batch inference
- Combined: ~100× faster than TypeScript
- Production-ready for large-scale self-play

---

## Current Workaround

**For immediate use**, the current `CachedAlphaZeroPolicy` provides:
- Value cache optimization (2.4× speedup)
- Simple move selection based on position value
- Good enough for:
  * Initial self-play data generation
  * Baseline performance benchmarking
  * Testing cache infrastructure

**Limitations**:
- No tree search (just greedy value-based selection)
- No policy priors (random move selection)
- Not using full AlphaZero algorithm

---

## Estimated Effort

| Task | Complexity | Time | Priority |
|------|------------|------|----------|
| Phase A: Basic CachedMCTS | Low | 4-6 hours | High |
| Phase B: Advanced Optimization | Medium | 7-10 hours | Medium |
| Testing & Documentation | Low | 2-3 hours | High |
| **Total** | - | **13-19 hours** | - |

---

## Performance Expectations

### Current (Phase 5.7)
- CachedAlphaZeroPolicy: ~5ms per move (GPU)
- Value inference: 0.42-0.85ms (CPU)
- Simple greedy selection (no tree search)

### After Phase A (Basic CachedMCTS)
- Full MCTS with 50 simulations: ~50-70ms per move
- 3-4× faster than standard MCTS (~280ms)
- Proper tree search with value guidance

### After Phase B (Optimized + Batch)
- Batch evaluation (16-32 positions): ~10-20ms per batch
- Parallel self-play (8 games): ~200-300ms per move (all games)
- 10-20× faster than standard MCTS

---

## Recommendation

**For Production Deployment Now**:
- Use current `CachedAlphaZeroPolicy` for simple tasks
- Works well for initial data generation
- ~12× faster than TypeScript already

**For Full AlphaZero Training**:
- Implement Phase A (Basic CachedMCTS)
- Essential for proper tree search and policy priors
- 4-6 hours of work, high value

**For Scale-Up**:
- Implement Phase B (Optimization + Batch)
- Required for large-scale self-play (1M+ games)
- Medium priority, can wait for initial results

---

**Status**: Phase 5.7 complete, ready for Phase A implementation or production deployment with current simplified policy.
