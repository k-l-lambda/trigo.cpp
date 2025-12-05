# Trigo.cpp - High-Performance MCTS Tools

**Project Scope**: C++/CUDA tools for Trigo game engine and MCTS self-play generation

**Goal**: Provide high-performance tools for TrigoRL training pipeline

---

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Python Training Pipeline (TrigoRL) - SEPARATE PROJECT      │
│  ├─ PyTorch Model Training                                   │
│  ├─ ONNX Model Export (exportOnnx.py)                        │
│  ├─ Training Data Loading (.tgn files)                       │
│  └─ Weights & Biases Integration                             │
└─────────────────────────────────────────────────────────────┘
                           ↓ exports
                    ONNX Models (.onnx)
                           ↓ uses
┌─────────────────────────────────────────────────────────────┐
│  C++ Inference & Generation Tools (trigo.cpp) - THIS PROJECT│
│  ├─ SharedModelInferencer (ONNX Runtime + CUDA)             │
│  │   ├─ Policy Network Inference                             │
│  │   ├─ Value Network Inference                              │
│  │   └─ Prefix Tree Attention Builder                        │
│  ├─ TrigoGame (3D Go rules engine)                           │
│  │   ├─ Board State Management                               │
│  │   ├─ Move Validation                                      │
│  │   ├─ Capture & Ko Detection                               │
│  │   └─ Territory Calculation                                │
│  ├─ MCTS (Monte Carlo Tree Search)                           │
│  │   ├─ Pure MCTS (UCB1, random rollouts)                    │
│  │   └─ AlphaZero MCTS (PUCT, value network) [planned]      │
│  ├─ Self-Play Generator (data generation tool)               │
│  │   ├─ RandomPolicy                                         │
│  │   ├─ NeuralPolicy (ONNX inference)                        │
│  │   ├─ MCTSPolicy                                           │
│  │   └─ TGN File Export                                      │
│  ├─ CUDA Kernels [future]                                    │
│  │   ├─ Parallel MCTS Tree Operations                        │
│  │   └─ Batched Game State Evaluation                        │
│  └─ Python Bindings (pybind11) [future]                     │
└─────────────────────────────────────────────────────────────┘
                           ↓ generates
                    Training Data (.tgn)
                           ↓ feeds back to
                      TrigoRL Pipeline
```

### Project Roles

**TrigoRL** (separate repository):
- PyTorch model training (policy + value networks)
- ONNX model export via `exportOnnx.py`
- Training data loading from .tgn files
- Python-based inference for development

**trigo.cpp** (this repository):
- High-performance C++/CUDA tools for production
- ONNX Runtime inference (CPU + GPU)
- Game engine implementation
- MCTS algorithms (pure + AlphaZero style)
- Self-play data generation (.tgn files)

**Data Flow**:
1. trigoRL trains models → exports .onnx files
2. trigo.cpp loads .onnx → runs self-play → generates .tgn files
3. trigoRL loads .tgn files → continues training (iterative improvement)

---

## Implementation Status

### Phase 1: Model Inference ✅ COMPLETE

**Components**:
- ✅ `SharedModelInferencer` - ONNX Runtime with shared base model
- ✅ `TGNTokenizer` - Compatible with Python training tokenizer
- ✅ `PrefixTreeBuilder` - Tree attention support
- ✅ ONNX models can be loaded and run
- ✅ Model format: 3-model architecture (base + policy_head + value_head)

**Trained Models**:
- Location: `/home/camus/work/trigo.cpp/models/trained_shared/`
- Files: `base_model.onnx`, `policy_head.onnx`, `value_head.onnx`

**Tests**:
- ✅ `test_neural_policy_inference.cpp` - Full inference pipeline
- ✅ `test_shared_model_inferencer.cpp` - Model loading
- ✅ `test_tgn_consistency.cpp` - Format validation

---

### Phase 2: Game Engine ✅ COMPLETE

**Components**:
- ✅ `TrigoGame` - Complete 3D Go engine
- ✅ `trigo_coords.hpp` - ab0yz coordinate encoding
- ✅ `trigo_game_utils.hpp` - Capture, Ko, territory
- ✅ `tgn_utils.hpp` - Shared TGN generation
- ✅ Cross-language validation (100/100 games vs TypeScript)

**Self-Play Generation**:
- ✅ `RandomPolicy` - Baseline
- ✅ `NeuralPolicy` - ONNX inference with correct TGN format
- ✅ `MCTSPolicy` - Basic MCTS (CPU, performance limited)
- ✅ `self_play_generator` - Command-line tool

**Performance**:
- Random vs Random: ~3 games/sec (CPU)
- Neural vs Random: ~1 game/sec (CPU)
- MCTS vs Random: Too slow (<0.1 games/sec, needs optimization)

**Tests**:
- ✅ `test_trigo_coords.cpp`
- ✅ `test_trigo_game_utils.cpp`
- ✅ `test_trigo_game.cpp`
- ✅ `test_game_replay.cpp`
- ✅ `test_tgn_consistency.cpp`

---

### Phase 3: MCTS Algorithm ⚠️ PARTIALLY COMPLETE

**Status**:
- ✅ Basic MCTS structure implemented (`include/mcts.hpp`)
- ✅ UCB1 selection working
- ✅ Tree expansion working
- ✅ Backpropagation working
- ✅ Unit test passes (`test_mcts.cpp`)
- ❌ **Performance issue**: Too slow for practical use (>10sec per move)

**Performance Analysis** (Confirmed):
- **Simulation/rollout phase**: 923ms per simulation (99.9% of time)
- Selection: ~0μs (negligible)
- Expansion: ~0.45ms (0.05% of time)
- 5 simulations = 4.2 seconds
- 800 simulations = ~12 minutes per move (impractical)

**Bottleneck Root Cause**:
- Random playouts to completion on 5x5x5 boards are extremely expensive
- Each rollout requires hundreds of game state copies and move validations
- CPU-only implementation without value network

**Solutions**:
1. **Replace rollouts with value network** (AlphaZero style) - most practical
   - Use existing `SharedModelInferencer::value_inference()`
   - Eliminates expensive rollouts
   - Dramatically faster (~1-10ms per evaluation)
2. GPU acceleration (Phase 4) - future work
3. Heavy CPU optimization - diminishing returns

---

### Phase 4: GPU Acceleration - FUTURE

**Planned Components**:
- CUDA MCTS kernels for parallel tree operations
- Batched neural network inference
- Target: 50-100 games/sec on GPU

**Not Started**.

---

## Current Tasks

### Immediate: Value Network Integration (Recommended)

**Problem**: MCTS rollouts take 923ms each, making it impractical for real-time play.

**Solution**: Replace random rollouts with value network evaluation (AlphaZero-style MCTS):
1. Create `AlphaZeroMCTS` variant in `include/mcts.hpp`
2. Replace `simulate()` with `SharedModelInferencer::value_inference()`
3. Use PUCT formula instead of UCB1 (already implemented in `MCTSNode::puct_score()`)
4. Expected performance: ~1-10ms per evaluation (100× faster)

**Tasks**:
- [ ] Implement `AlphaZeroMCTS` class with value network
- [ ] Test performance improvement
- [ ] Integrate into `self_play_generator`

### Alternative: Optimize Rollouts (Lower Priority)

If value network integration is deferred:
- Profile `simulate()` function for optimization opportunities
- Consider early termination heuristics
- Likely diminishing returns (rollouts are inherently expensive)

---

## Development Guidelines

### Code Style
- C++17 standard
- Modern C++ (curly braces on standalone lines, tab indentation)
- Comprehensive comments
- DRY principle (avoid code duplication)

### Testing
- Unit tests for each component
- Cross-language validation where applicable
- Performance regression tests

### Focus
- **This project**: Tools and infrastructure
- **TrigoRL project**: Training, model export, Python training pipeline
- No training code in trigo.cpp

---

## References

### TypeScript Source (for validation)
- `trigoRL/third_party/trigo/trigo-web/inc/trigo/game.ts`
- `trigoRL/third_party/trigo/trigo-web/inc/trigo/gameUtils.ts`

### Python Integration
- `trigoRL/trigor/data/tgn_dataset.py` - Loads .tgn files
- `trigoRL/trigor/data/tokenizer.py` - TGN tokenization
- `trigoRL/exportOnnx.py` - ONNX model export

---

**Last Updated**: December 5, 2025
**Current Focus**: MCTS performance bottleneck identified - simulation phase takes 923ms per iteration (99.9% of total time)
**Next Step**: Consider value network integration to replace expensive rollouts
