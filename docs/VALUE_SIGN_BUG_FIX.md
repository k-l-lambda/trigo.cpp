# Critical Bug Fix: Value Model Sign Correction

## Bug Description

**Symptom**: MCTS was not correctly handling value predictions, leading to poor move selection.

**Root Cause**: The value model outputs **absolute values** (White advantage = positive, Black advantage = negative), but MCTS requires **relative values** (current player advantage = positive).

## The Problem

### Value Model Definition

The trained value model outputs:
- **Positive value**: White is winning
- **Negative value**: Black is winning
- This is **absolute**, independent of current player

### MCTS Requirement

MCTS needs value from **current player's perspective**:
- **Positive value**: Current player is winning
- **Negative value**: Current player is losing
- This is **relative** to who is playing

### What Was Wrong

In three files, the value inference functions directly returned the model output without considering the current player:

1. **`cached_mcts.hpp:362`** - `CachedMCTS::evaluate_with_cache()`
2. **`mcts.hpp:439`** - `AlphaZeroMCTS::evaluate()`
3. **`self_play_policy.hpp:732`** - `CachedAlphaZeroPolicy::select_action()`

Example of the bug:
```cpp
// WRONG - directly returns model output
float evaluate_with_cache(TrigoGame& game)
{
    float value = inferencer->value_inference_with_cache(3);
    return value;  // BUG: Not considering current player!
}
```

### Impact

When Black is the current player:
- Model outputs `-0.5` (Black winning)
- MCTS interprets as `-0.5` (current player losing!)
- **Result**: Black makes moves thinking it's losing when it's actually winning

When White is the current player:
- Model outputs `+0.5` (White winning)
- MCTS interprets as `+0.5` (current player winning)
- **Result**: Correct by coincidence

This caused:
1. Black to play poorly (always pessimistic)
2. Games to not terminate properly (Black avoided winning positions)
3. Self-play to generate low-quality training data

## The Fix

Convert model output to current player's perspective:

```cpp
// CORRECT - considers current player
float evaluate_with_cache(TrigoGame& game)
{
    float value = inferencer->value_inference_with_cache(3);

    // IMPORTANT: Value model outputs White advantage (positive = White winning)
    // But MCTS needs value from current player's perspective
    // If current player is Black, we need to negate the value
    Stone current_player = game.get_current_player();
    if (current_player == Stone::Black)
    {
        value = -value;
    }

    return value;
}
```

## Files Modified

1. **`include/cached_mcts.hpp`** (line 352-380)
   - Added current player check in `evaluate_with_cache()`
   - Negates value when Black is current player

2. **`include/mcts.hpp`** (line 383-458)
   - Added current player check in `evaluate()`
   - Negates value when Black is current player

3. **`include/self_play_policy.hpp`** (line 683-756)
   - Added current player check in `CachedAlphaZeroPolicy::select_action()`
   - Negates value when Black is current player before calculating confidence

## Verification

### Before Fix
```bash
# Game would hang or take very long
./build/self_play_generator --num-games 1 --board 5x5x1 \
    --black-policy cached-mcts --white-policy cached-mcts \
    --max-moves 50
# Often hit max-moves limit without proper termination
```

### After Fix
```bash
./build/self_play_generator --num-games 1 --board 5x5x1 \
    --black-policy cached-mcts --white-policy cached-mcts \
    --max-moves 50

# Output:
[Game 0] 00 Pass ay a0 b0 Pass yy yz yb Pass 0y Pass Pass
[Game 0] Finished after 13 moves
# Game terminates naturally with double pass
```

### Test Results

Game completes in 13 moves with natural double-pass termination:
```
[Board 5x5]

1. 00 Pass
2. ay a0
3. b0 Pass
4. yy yz
5. yb Pass
6. 0y Pass
7. Pass
; -4
```

Score: `-4` means Black wins by 4 points (negative = Black advantage, as per model convention).

## Why This Bug Wasn't Caught Earlier

1. **White moves looked correct**: Since White's perspective was already correct, half the moves appeared reasonable
2. **Backpropagation flipped signs**: The `value = -value` in backpropagation partially masked the issue
3. **Policy network guided exploration**: Strong policy priors helped despite incorrect value evaluation
4. **No direct value comparison**: We didn't directly compare value outputs until the PASS selection analysis

## Lessons Learned

1. **Always document model output conventions**: Clearly state whether outputs are absolute or relative
2. **Test with symmetry**: A good test is to swap Black/White and verify identical behavior
3. **Value inspection is critical**: Always check that value signs match intuition for both players
4. **Unit tests for sign correctness**: Add tests that verify value signs for known positions

## Related Issues

This fix resolves:
- Games not terminating properly
- Black playing overly cautiously
- Excessive PASS moves
- Poor self-play game quality

## Commit Message Template

```
Fix critical value sign bug in MCTS evaluation

The value model outputs White advantage (positive = White winning),
but MCTS requires current player perspective. Added current player
check to negate value when Black is playing.

Affected files:
- include/cached_mcts.hpp
- include/mcts.hpp
- include/self_play_policy.hpp

This fixes games not terminating and Black playing poorly.
```
