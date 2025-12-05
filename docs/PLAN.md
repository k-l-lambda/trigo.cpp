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

### Phase 3: MCTS Algorithm ✅ COMPLETE

**Status**:
- ✅ PureMCTS with random rollouts (`include/mcts_moc.hpp`)
  - UCB1 selection, tree expansion, backpropagation working
  - Reference implementation for validation
  - Performance: ~923ms per simulation (limited to testing)
- ✅ AlphaZero-style MCTS with value network (`include/mcts.hpp`)
  - Uses `SharedModelInferencer::value_inference()` for evaluation
  - PUCT formula for exploration
  - **Performance: 255× speedup** (~3.6ms per simulation vs 923ms)
  - Production-ready implementation

**Performance Comparison**:
| Implementation | Time per simulation | 50 simulations | 800 simulations |
|----------------|---------------------|----------------|-----------------|
| PureMCTS (rollouts) | 923ms | 46 seconds | 12+ minutes |
| MCTS (value network) | 3.6ms | 180ms | 2.9 seconds |
| **Speedup** | **255×** | **255×** | **255×** |

**File Organization**:
- `include/mcts.hpp` - Production MCTS with value network (MCTS class)
- `include/mcts_moc.hpp` - Reference pure MCTS (PureMCTS class)
- `include/self_play_policy.hpp` - Policy interfaces using both implementations

---

### Phase 4: GPU Acceleration - FUTURE

**Planned Components**:
- CUDA MCTS kernels for parallel tree operations
- Batched neural network inference
- Target: 50-100 games/sec on GPU

**Not Started**.

---

## Current Tasks

### Next: HybridPolicy Implementation (Optional Enhancement)

**Status**: Currently a placeholder in `self_play_policy.hpp:344`

**Purpose**: Combine neural policy priors with MCTS search (full AlphaZero algorithm)

**Current Implementation**:
- HybridPolicy exists but only wraps NeuralPolicy
- MCTS class supports PUCT formula and value network
- Need to integrate policy network priors into MCTS tree search

**Tasks**:
- [ ] Add policy prior support to MCTSNode
- [ ] Integrate `policy_inference()` into MCTS expansion
- [ ] Use priors to guide tree exploration
- [ ] Test performance vs pure neural policy
- [ ] Compare with pure MCTS approach

**Priority**: Low (current MCTS and NeuralPolicy work well independently)

---

### Alternative: Python Bindings (Integration)

**Goal**: Expose C++ tools to Python for easier integration with TrigoRL training pipeline

**Tasks**:
- [ ] Set up pybind11
- [ ] Expose TrigoGame class
- [ ] Expose policy classes (Random, Neural, MCTS)
- [ ] Expose self-play generation functions
- [ ] Create Python package

**Priority**: Medium (improves integration but not blocking)

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
**Current Status**: Phase 3 MCTS complete - AlphaZero-style MCTS with value network achieves 255× speedup
**Next Step**: Optional enhancements (HybridPolicy, Python bindings) or GPU acceleration (Phase 4)
