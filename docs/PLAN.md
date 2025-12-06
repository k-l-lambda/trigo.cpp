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

### Phase 4: MCTS Performance Benchmarking ✅ COMPLETE

**Completed**: Comprehensive three-way performance comparison (December 5, 2025)

**Test Configuration**:
- 10 games with AlphaZero MCTS (50 simulations per move)
- Board: 5×5×1
- Model: Dynamic ONNX shared architecture
- Hardware: RTX 3090 (24GB), Multi-core CPU

**Performance Results**:

| Implementation | Time per Move | Total Duration | Speedup vs TypeScript |
|----------------|---------------|----------------|----------------------|
| **C++ CPU** | 280ms | 117s | **6.59×** |
| **C++ GPU** | 335ms | 178s | 5.51× |
| **TypeScript** | 1846ms | 641s | 1× (baseline) |

**Key Findings**:

1. **C++ is 5.47× faster than TypeScript** for MCTS self-play
   - Production-ready for large-scale data generation
   - Can generate 10K games in 32.5 hours (CPU)

2. **GPU is SLOWER than CPU for batch=1 MCTS** (counter-intuitive but expected)
   - GPU: 335ms vs CPU: 280ms per move (0.66× performance)
   - **1.52× slower overall** (178s vs 117s)
   - Root cause: Small batch size (batch=1) underutilizes GPU parallelism
   - CUDA kernel launch overhead (~100-150μs) dominates small model inference
   - 7 Memcpy nodes added for GPU, some operators fall back to CPU
   - GPU cores 99% idle with batch=1 workload

3. **GPU advantage depends on workload characteristics**:
   - ❌ Self-play with batch=1 MCTS: CPU wins (1.52× faster)
   - ✅ Training with batch=256+: GPU wins (10-50× faster expected)
   - ✅ Batch inference with 64+ positions: GPU wins (5-20× faster expected)

**Recommendations**:
- **Use CPU for MCTS self-play** (default: `TRIGO_FORCE_CPU=1`)
- Use GPU only for training (where large batches are natural)
- Future optimization: Implement batch MCTS leaf evaluation for GPU
- Future optimization: Parallel self-play (multiple games simultaneously)

**Documentation**: See `docs/PERFORMANCE_ANALYSIS-1205.md` for detailed analysis

---

### Phase 5: KV Cache Optimization ✅ Phase 5.1 Complete, Phase 5.2 In Progress

**Goal**: Implement Prefix KV Cache for BaseModelWithTreeAttention to accelerate MCTS inference

**Status**: Python core implementation complete (December 6, 2025)

**Research Findings** (documented in `docs/KVCACHE_DESIGN.md`):

✅ **ONNX Runtime C++ API supports PyTorch-like GPU memory management**
- `IOBinding` API for zero-copy GPU tensor binding
- `Value::CreateTensor()` with CUDA memory for persistent GPU tensors
- Full support for cross-inference tensor reuse

**Implementation Approaches**:

1. **IOBinding + Persistent GPU Tensors** (Recommended)
   - Similar to PyTorch KV cache pattern
   - Zero CPU-GPU copy overhead
   - Highest performance (10-100× speedup for sequential generation)

2. **Manual CUDA Memory Management** (Advanced)
   - Lower-level control with `cudaMalloc`/`cudaFree`
   - Wrap CUDA buffers as `Ort::Value` tensors
   - Suitable for special requirements

**Prototype Validation Results** (documented in `prototype/kvcache/KVCACHE_BENCHMARK.md`):
- ✅ **4.78× speedup** achieved with Python ONNX Runtime
- First 10 tokens: 24.83ms (no cache) → 5.20ms (with cache)
- Average per token: 2.48ms → 0.52ms
- Memory overhead: ~8 KB per token (4-layer model)

**Expected Performance**:
- First token latency: No change
- **Subsequent token latency: 10-100× reduction** (vs recomputing full sequence)
- Memory overhead: ~75 MB per batch (GPT-2 scale, 2048 max length)

**Key APIs**:
```cpp
// Create CUDA memory info
auto memory_info = Ort::MemoryInfo::CreateCuda(device_id, OrtMemTypeDefault);

// Create persistent GPU tensor
Ort::Value cache = Ort::Value::CreateTensor<float>(cuda_allocator, shape);

// Bind to inference
Ort::IoBinding io_binding(session);
io_binding.BindInput("past_key_cache", cache);
io_binding.BindOutput("present_key_cache", memory_info);
```

**Completed Tasks**:

- ✅ **Phase 5.1: Python Core Implementation** (COMPLETE - December 6, 2025)
  - ✅ Modified `BaseModelWithTreeAttention` in `trigoRL/exportOnnx.py`
  - ✅ Implemented two execution modes (cache vs no-cache)
  - ✅ Added cache helper methods (`_get_cache_length`, `_tuple_to_cache`, `_cache_to_tuple`)
  - ✅ Built attention mask builders for both modes
  - ✅ Comprehensive unit tests (12/12 passing)
  - ✅ Integration test with real GPT2 model (all tests passing)
  - ✅ Documentation: See implementation details in `trigoRL/tests/test_kvcache.py`

**Current Tasks**:

- [ ] **Phase 5.2: ONNX Export Implementation** (IN PROGRESS)
  - [ ] Implement `export_shared_architecture_with_cache()` method
  - [ ] Create `CachedONNXWrapper` for flat cache I/O
  - [ ] Add `--with-cache` CLI flag to exportOnnx.py
  - [ ] Export two ONNX models (base_model.onnx + base_model_cached.onnx)
  - [ ] Validate exported models with onnxruntime

- [ ] **Phase 5.3: Performance Benchmarking**
  - [ ] Measure speedup in Python (target: 2-5× for MCTS use case)
  - [ ] Validate numerical accuracy
  - [ ] Test different prefix/evaluated lengths
  - [ ] Document results in `docs/KVCACHE_BENCHMARK.md`

- [ ] **Phase 5.4: C++ Integration** (FUTURE)
  - [ ] Update `SharedModelInferencer` to support cached models
  - [ ] Add KV cache management with IOBinding
  - [ ] Integrate with MCTS policy evaluation
  - [ ] End-to-end testing with self-play

**Implementation Details**:

**Python Core** (`trigoRL/exportOnnx.py:742-985`):
- `BaseModelWithTreeAttention` now supports `use_cache=True` parameter
- Two execution modes:
  - **No cache**: Computes full sequence (prefix + evaluated)
  - **Cache mode**: Skips prefix, only computes evaluated tokens
- Cache format: Uses transformers `DynamicCache` internally, converts to/from tuple for ONNX
- Position IDs: In cache mode, evaluated tokens get positions `prefix_length + mask_row_sums - 1`
- Attention mask: Cache mode allows evaluated tokens to attend to full cached prefix

**Test Coverage** (`trigoRL/tests/test_kvcache.py`):
- ✅ Cache vs no-cache correctness (3 tests)
- ✅ Position IDs calculation (2 tests)
- ✅ Attention mask construction (3 tests)
- ✅ Cache format conversion (4 tests)
- ✅ Integration with GPT2 model (3 tests)

**Limitations**:
- Model must be exported with `use_cache=True` (past/present key-value inputs/outputs)
- Static shape models may need fixed `max_seq_len`
- Memory scales with: `2 * num_layers * batch * num_heads * max_seq_len * head_dim * sizeof(float)`

**Priority**: Medium-High (significant inference speedup for MCTS sequential evaluation)

---

### Phase 6: Batched GPU Acceleration - FUTURE

**Planned Components**:
- Batch MCTS leaf evaluation (evaluate 64-256 positions simultaneously)
- Parallel self-play generation (8-16 games concurrently)
- CUDA MCTS kernels for parallel tree operations
- Target: 10-20× speedup with proper batching

**Priority**: Low (current CPU implementation is sufficient for production use)

**Not Started**.

---

## Current Tasks

### Next: Phase 5.2 - ONNX Export Implementation

**Goal**: Export BaseModelWithTreeAttention with KV cache support to ONNX format

**Tasks**:
1. Implement `export_shared_architecture_with_cache()` method in `trigoRL/exportOnnx.py`
2. Create `CachedONNXWrapper` class to flatten cache I/O for ONNX
3. Add `--with-cache` CLI flag to export script
4. Export two ONNX models:
   - `base_model.onnx` (no cache) - existing functionality
   - `base_model_cached.onnx` (with cache) - new cached version
5. Validate exported models with onnxruntime:
   - Test cache mode correctness
   - Verify cache tensor shapes
   - Measure inference speedup

**Success Criteria**:
- ✅ Both ONNX models load successfully in onnxruntime
- ✅ Cached model produces numerically equivalent results
- ✅ Cache tensors have correct shapes (per layer)
- ✅ Measured speedup >2× for MCTS use case (prefix reuse)

**Priority**: High (unlocks C++ integration)

---

### Alternative: HybridPolicy Implementation (Optional Enhancement)

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

**Last Updated**: December 6, 2025
**Current Status**:
- Phase 4 MCTS Benchmarking complete - C++ CPU is 5.47× faster than TypeScript
- Phase 5.1 KV Cache Python core complete - All tests passing (12/12)
- Phase 5.2 ONNX export in progress
**Production Ready**: C++ MCTS with CPU execution is production-ready for large-scale self-play data generation
**Next Step**: Phase 5.2 - ONNX export implementation with KV cache support
