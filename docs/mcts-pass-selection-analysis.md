# MCTS Pass Selection Analysis

## Overview

This document describes the investigation into premature double-pass behavior in self-play games, specifically analyzing why games end with "Pass Pass" when the board still has significant neutral territory.

**Problem**: Generated games sometimes end with double pass at move 13 when:
- Board has 48 valid moves available
- Territory: Black=6, White=6, Neutral=48
- Board is only 20% filled (12/60 positions)

**Example game**:
```
[Board 5x3x4]

1. a0z b0z
2. aza azz
3. azy azb
4. a0y a0b
5. 0az yzb
6. 00y yzz
7. Pass Pass
; 0
```

## Test Tool: `test_mcts_from_prefix`

### Purpose

Analyze MCTS behavior at specific game positions by:
1. Replaying a sequence of moves (prefix)
2. Running full MCTS search from that position
3. Reporting detailed statistics including Pass selection probability

### Usage

```bash
./test_mcts_from_prefix <board_shape> <moves> <num_simulations> [model_dir] [seed]
```

**Parameters**:
- `board_shape`: Board dimensions (e.g., "5x3x4" or "19x19")
- `moves`: Space-separated move notation (e.g., "a0z b0z aza azz")
- `num_simulations`: Number of MCTS simulations (e.g., 200)
- `model_dir`: Directory containing ONNX models (default: models/policy_value_model)
- `seed`: Random seed for MCTS (default: 42)

**Model files required**:
- `base_model_prefix.onnx`
- `base_model_eval_cached.onnx`
- `policy_head.onnx`
- `value_head.onnx`

### Example

```bash
./test_mcts_from_prefix "5x3x4" "a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz" 200 ../models/trained_shared_shared_cached 42
```

### Output

The tool provides:
1. **Root cache computation time** - Initial KV-cache setup
2. **Per-simulation breakdown** - Selection, expansion, evaluation times
3. **Policy priors** - Top 5 moves with log scores and prior probabilities
4. **Final visit counts** - All children with visits, priors, and Q-values
5. **Selected action** - Best move with confidence

Example output:
```
[CachedMCTS] Child visit counts after search:
  aaa: visits=9, prior=0.052414, Q=-0.082888
  zaz: visits=21, prior=0.140252, Q=-0.060767
  bzy: visits=13, prior=0.092790, Q=-0.057332
  PASS: visits=0, prior=0.000000, Q=0.000000

=== Selected Action ===
Action: zaz
Confidence: 0.105000
```

## Test Results

### Single Position Analysis (Seed=42)

**Position**: After 12 moves "a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz"

**MCTS Statistics** (200 simulations):
- **Pass visits**: 0/200 (0%)
- **Pass prior**: 0.000000
- **Pass Q-value**: 0.000000

**Top selected moves**:
1. `zaz`: 21 visits (10.5%) - prior=0.140252
2. `bzy`: 13 visits (6.5%) - prior=0.092790
3. `00z`: 11 visits (5.5%) - prior=0.067156
4. `0zz`: 10 visits (5.0%) - prior=0.054853
5. `aaa`, `aaz`, `0aa`, `0za`: 9 visits each (4.5%)

**Key finding**: Pass selection probability is **0%** at this position with standard MCTS parameters.

### Multi-Seed Analysis

**Test protocol**:
- 50 different random seeds (1-50)
- 200 MCTS simulations per seed
- Same position and model

**Results**:

```
Pass selected: 0 times (0/50 tests)
  -> Pass was NEVER selected

Action distribution (top 10):
  baz       :   4 times (  8.0%)
  0zz       :   4 times (  8.0%)
  zaz       :   3 times (  6.0%)
  y0y       :   3 times (  6.0%)
  y0a       :   3 times (  6.0%)
  bzz       :   3 times (  6.0%)
  zay       :   2 times (  4.0%)
  z0y       :   2 times (  4.0%)
  yab       :   2 times (  4.0%)
  bzb       :   2 times (  4.0%)
```

**Total unique actions selected**: 29 different moves across 50 tests

### Test Script

The automated testing script `test_pass_selection.sh`:

```bash
#!/bin/bash
BOARD="5x3x4"
MOVES="a0z b0z aza azz azy azb a0y a0b 0az yzb 00y yzz"
SIMULATIONS=200
MODEL="../models/trained_shared_shared_cached"
NUM_TESTS=50

for seed in $(seq 1 $NUM_TESTS); do
    result=$(./test_mcts_from_prefix "$BOARD" "$MOVES" $SIMULATIONS $MODEL $seed 2>&1 | grep "^Action:" | head -1)
    action=$(echo "$result" | awk '{print $2}')
    echo "Test $seed (seed=$seed): $action"
    # ... count actions
done
```

## Analysis

### Why Pass Is Never Selected

**1. Extremely Low Prior Probability**

Pass move has a minimal prior probability configured at `1e-10` (cached_mcts.hpp:74):
```cpp
CachedMCTS(
    std::shared_ptr<PrefixCacheInferencer> inf,
    int num_sims = 50,
    float exploration = 1.0f,
    int seed = 42,
    float dir_alpha = 0.03f,
    float dir_epsilon = 0.25f,
    float pass_prob = 1e-10f  // Default minimal prior for Pass
)
```

After renormalization with 48 valid moves (priors ranging 0.05-0.14), Pass prior becomes effectively **0.000000**.

**2. PUCT Formula**

The PUCT selection formula is:
```
PUCT(a) = Q(a) + c_puct × P(a) × sqrt(N_parent + 1) / (1 + N(a))
```

With P(Pass) ≈ 0:
- Pass exploration term: `1.414 × 0.0 × sqrt(201) / 1 ≈ 0`
- Other moves exploration term: `1.414 × 0.1 × sqrt(201) / 1 ≈ 2.0`

Even with Dirichlet noise, Pass is excluded from noise mixing (cached_mcts.hpp:1164):
```cpp
if (child->is_pass)
{
    // Keep Pass prior unchanged (no noise applied)
    continue;
}
```

**3. Deterministic Behavior**

The test results show:
- 50 different random seeds → 0 Pass selections
- 29 different moves selected → Pass is never chosen
- Dirichlet noise affects regular moves only

### Contradiction with Actual Games

**Observed**: Games end with "Pass Pass" at move 13

**MCTS analysis**: Pass should NEVER be selected at move 13

**Possible explanations**:

1. **Different game state reached**
   - Replay sequence might not match actual game history
   - Move encoding/decoding mismatch

2. **Different policy used during generation**
   - Not using `CachedMCTSPolicy`
   - Using `CachedAlphaZeroPolicy` (stub implementation)
   - Using random or neural-only policy

3. **Different MCTS parameters**
   - Different simulation count
   - Different pass prior value
   - Different temperature setting

4. **Bug in self-play generation loop**
   - Incorrect terminal detection
   - Pass selection logic error
   - State synchronization issue

## Recommendations

### 1. Verify Actual Policy Used

Check `self_play_generator.cpp` to confirm which policy class is instantiated:

```cpp
// What policy is actually used?
auto black_policy = create_policy(black_policy_type, ...);
auto white_policy = create_policy(white_policy_type, ...);
```

Ensure it uses `CachedMCTSPolicy`, not `CachedAlphaZeroPolicy` (which is just a stub).

### 2. Add Logging to Self-Play Generator

Instrument the main loop to log:
- Policy type for each move
- MCTS statistics (if available)
- Terminal detection decisions
- Pass selection circumstances

### 3. Reproduce Exact Game Generation

Run the generator with:
- Same random seed as the problematic game
- Same parameters (--black-policy alphazero --white-policy alphazero --mcts-simulations 200)
- Verbose logging enabled

### 4. Consider Adjusting Pass Prior

If Pass should be playable in certain situations:

**Option A**: Increase base pass prior from `1e-10` to `0.01` or `0.001`

**Option B**: Dynamic pass prior based on game state:
- High prior when board is >80% filled
- Low prior in opening/midgame

**Option C**: Include Pass in Dirichlet noise at root

### 5. Terminal Detection Review

Review the terminal detection logic:
```cpp
bool check_natural_terminal(TrigoGame& game, int stone_count)
{
    int minStones = (totalPositions - 1) / 2;
    if (stone_count < minStones) return false;

    auto territory = game.get_territory();
    if (territory.neutral != 0) return false;

    return !game.has_any_capturing_move();
}
```

Ensure it correctly prevents premature termination.

## Files Modified

### Source Code

1. **tests/test_mcts_from_prefix.cpp** (new file)
   - Complete MCTS analysis tool
   - Supports custom seeds
   - Reports detailed statistics

2. **CMakeLists.txt**
   - Added `test_mcts_from_prefix` target
   - Linked with trigo_game and trigo_inference

### Test Scripts

1. **build/test_pass_selection.sh** (new file)
   - Automated multi-seed testing
   - Action distribution analysis
   - Pass selection rate reporting

## Build Instructions

```bash
cd /home/camus/work/trigo.cpp/build
make test_mcts_from_prefix -j$(nproc)
```

Enable profiling for detailed MCTS statistics:
```bash
cmake .. -DMCTS_ENABLE_PROFILING=ON
make test_mcts_from_prefix -j$(nproc)
```

## Conclusion

The MCTS analysis tool successfully demonstrates that:

1. **Pass is never selected** at the analyzed position across 50 different random seeds
2. **Pass prior is effectively zero** due to renormalization with many valid moves
3. **Dirichlet noise excludes Pass**, preventing exploration
4. **PUCT formula** ensures Pass nodes are never visited when P(Pass) ≈ 0

The contradiction between test results (0% Pass) and actual games (Pass at move 13) indicates a discrepancy between the test environment and the actual self-play generation, which requires further investigation into the policy selection and game state management in `self_play_generator.cpp`.

---

**Author**: Claude Code
**Date**: December 19, 2025
**Related Issue**: Premature double-pass in generated games
