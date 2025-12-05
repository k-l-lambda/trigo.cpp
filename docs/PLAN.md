# Implementation Plan - Trigo.cpp CUDA MCTS Engine

**Goal**: Build a C++/CUDA MCTS self-play engine achieving **300-1000 games/hour** on single GPU.

**Timeline**: 10 weeks to production-ready system

**Status**: Phase 0 Complete - Architecture designed, feasibility confirmed

**Updated**: December 2025 - Revised to use ONNX Runtime with shared model architecture

---

## Executive Summary

This document outlines the implementation plan for a high-performance CUDA-accelerated MCTS engine for Trigo (3D Go). The system will generate training data **100-1200× faster** than the TypeScript baseline by leveraging:

1. GPU-parallel tree traversal (10-20×)
2. Batched neural network inference (5-10×) with shared base model
3. C++ vs JavaScript performance (2-3×)
4. Memory and algorithm optimizations (2×)

**Key Architecture Decision**: Use **ONNX Runtime with shared base model** (not LibTorch, not llama.cpp):
- Export 3 ONNX models: `base_model.onnx` (400MB), `policy_head.onnx` (10MB), `value_head.onnx` (1MB)
- Share base transformer between policy and value inference → **48% memory savings**
- Run base model once per batch → **50% inference speedup**
- Leverage existing `exportOnnx.py` infrastructure
- Custom tree attention patterns work out-of-the-box

The implementation validates against the existing TypeScript golden reference to ensure correctness.

---

## Research Documents

See `docs/research/` for detailed analysis:
- **LLAMA_CPP_ANALYSIS.md**: Comparison of ONNX Runtime vs llama.cpp, justification for chosen approach
- **MODEL_INFERENCE.md**: Existing ONNX export infrastructure analysis
- **CUDA_INFERENCE.md**: ONNX Runtime CUDA execution provider guide

---

## Revised Architecture

### Component Stack

```
Python Training Pipeline (TrigoRL)
    ↓ exportOnnx.py (EXISTING + MODIFICATIONS)
ONNX Models (3 separate files)
    ├─ base_model.onnx (400MB) - Shared GPT-2 transformer with tree attention
    ├─ policy_head.onnx (10MB) - TreeLM output projection
    └─ value_head.onnx (1MB)   - EvaluationLM value head
    ↓
Python Bindings (pybind11)
    ↓
C++ Orchestration Layer
    ├─ SharedModelInferencer (ONNX Runtime + manual composition)
    ├─ GameBatchManager (multi-game parallel execution)
    ├─ PrefixTreeBuilder (port from TypeScript trigoTreeAgent.ts)
    └─ TGNWriter (output formatting)
        ↓
CUDA MCTS Kernels + Trigo Game Engine
    ├─ select_leaf_kernel (UCB1 traversal)
    ├─ expand_leaves_kernel (node creation)
    ├─ backup_values_kernel (value backpropagation)
    └─ TrigoGame (3D Go rules)
```

### Memory Budget (Revised)

| Component | Size | Notes |
|-----------|------|-------|
| Base model (ONNX) | 400MB | Shared between policy/value |
| Policy head (ONNX) | 10MB | Output projection only |
| Value head (ONNX) | 1MB | Small MLP |
| MCTS tree nodes | 640MB | 8 games × 1M nodes × 80 bytes |
| Board state cache | 17MB | Delta encoding |
| Inference buffers | 100MB | Pinned memory |
| ONNX Runtime | 50MB | Library overhead |
| **Total** | **~1.2GB** | Fits on 8GB+ GPUs |

**Previous plan**: 1.8GB (duplicate models)
**Savings**: 600MB (33% reduction)

---

## Phase 0: Planning ✅ COMPLETE

**Status**: Architecture designed, feasibility confirmed

**Key Decisions**:
- ✅ Use ONNX Runtime (not LibTorch, not llama.cpp)
- ✅ Implement manual model sharing (3-model architecture)
- ✅ Start with model inference (Phase 3 first)
- ✅ Validate against TypeScript golden reference

**Research Completed**:
- Analyzed existing `exportOnnx.py` capabilities
- Confirmed ONNX Runtime CUDA support
- Evaluated llama.cpp vs ONNX Runtime
- Identified shared model optimization opportunity

---

## Phase 1: Model Inference (Weeks 1-2) ⬅️ START HERE

**Goal**: Working ONNX inference with shared model architecture

**Why Start Here**:
- Most critical component for MCTS performance
- Can validate against TypeScript immediately
- Establishes foundation for rest of system

### Tasks

#### 1.1 Build System Setup (Days 1-2)
**Deliverable**: CMake build with ONNX Runtime

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(trigo_cpp CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ONNX Runtime (GPU build)
set(ONNXRUNTIME_ROOT_DIR "/path/to/onnxruntime-linux-x64-gpu-1.17.0")
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)

# pybind11
find_package(pybind11 REQUIRED)

# Inference library
add_library(trigo_inference
    src/shared_model_inferencer.cpp
    src/prefix_tree_builder.cpp
    src/tgn_tokenizer.cpp
)

target_include_directories(trigo_inference PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(trigo_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    CUDA::cudart
)

# Python bindings
pybind11_add_module(cuda_mcts_inference src/bindings.cpp)
target_link_libraries(cuda_mcts_inference PRIVATE trigo_inference)
```

**Files to create**:
- `CMakeLists.txt` (root)
- `setup.py` (Python packaging)
- `.gitignore`
- `README.md` (build instructions)

**Installation**:
```bash
# Download ONNX Runtime GPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.17.0

# Build
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=$ONNXRUNTIME_ROOT
make -j$(nproc)
```

#### 1.2 Export Shared Model Architecture (Days 3-4)
**Deliverable**: Modified `exportOnnx.py` with `--shared-architecture` mode

**Location**: `/home/camus/work/trigoRL/exportOnnx.py`

**New functionality**:
```python
def export_shared_architecture(self, training_dir: str):
    """
    Export base model and heads separately for shared inference.

    Outputs:
        - base_model.onnx: Shared transformer with tree attention
            Input: input_ids [batch, seq_len]
                   attention_mask [batch, seq_len, seq_len]
                   position_ids [batch, seq_len]
            Output: hidden_states [batch, seq_len, hidden_dim]

        - policy_head.onnx: TreeLM output projection
            Input: hidden_states [batch, m+1, hidden_dim]
            Output: logits [batch, m+1, vocab_size]

        - value_head.onnx: EvaluationLM value prediction
            Input: hidden_states [batch, hidden_dim]
            Output: values [batch]
    """
    # Implementation details in research/LLAMA_CPP_ANALYSIS.md
```

**Testing**:
```bash
# Export from existing checkpoint
cd /home/camus/work/trigoRL
python exportOnnx.py training_output/run_xyz \
    --shared-architecture \
    --output-dir models/shared/

# Should produce:
# models/shared/base_model.onnx (~400MB)
# models/shared/policy_head.onnx (~10MB)
# models/shared/value_head.onnx (~1MB)
```

#### 1.3 TGN Tokenizer (Days 5-6)
**Deliverable**: C++ TGN tokenization compatible with Python

**Header**: `include/tgn_tokenizer.hpp`

```cpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>

class TGNTokenizer {
public:
    // Vocabulary constants (from trigoRL tokenizer)
    static constexpr int PAD_TOKEN = 0;
    static constexpr int START_TOKEN = 1;
    static constexpr int END_TOKEN = 2;
    static constexpr int VALUE_TOKEN = 3;
    static constexpr int VOCAB_SIZE = 128;

    // Tokenize TGN string to token IDs
    static std::vector<int64_t> tokenize(const std::string& tgn);

    // Detokenize token IDs back to string
    static std::string detokenize(const std::vector<int64_t>& tokens);

    // Convert token ID to character (simple ASCII mapping)
    static char token_to_char(int64_t token_id);

    // Convert character to token ID
    static int64_t char_to_token(char c);
};
```

**Implementation**: `src/tgn_tokenizer.cpp`

```cpp
#include "tgn_tokenizer.hpp"

std::vector<int64_t> TGNTokenizer::tokenize(const std::string& tgn) {
    std::vector<int64_t> tokens;
    tokens.reserve(tgn.length() + 2);

    // Prepend START token
    tokens.push_back(START_TOKEN);

    // Convert each character to token (direct ASCII mapping)
    for (char c : tgn) {
        tokens.push_back(char_to_token(c));
    }

    // Append END token
    tokens.push_back(END_TOKEN);

    return tokens;
}

int64_t TGNTokenizer::char_to_token(char c) {
    // Direct identity mapping: token_id = ascii_value
    // Special tokens: 0-3 (PAD, START, END, VALUE)
    // 10 = LF (newline)
    // 32-127 = ASCII printable characters
    return static_cast<int64_t>(static_cast<unsigned char>(c));
}

char TGNTokenizer::token_to_char(int64_t token_id) {
    if (token_id < 0 || token_id > 127) {
        return '?';  // Unknown token
    }
    return static_cast<char>(token_id);
}

std::string TGNTokenizer::detokenize(const std::vector<int64_t>& tokens) {
    std::string result;
    result.reserve(tokens.size());

    for (int64_t token : tokens) {
        // Skip special tokens
        if (token == START_TOKEN || token == END_TOKEN ||
            token == PAD_TOKEN || token == VALUE_TOKEN) {
            continue;
        }
        result += token_to_char(token);
    }

    return result;
}
```

**Validation**:
```cpp
// Test against Python tokenizer
auto tokens = TGNTokenizer::tokenize("3 3 1\na00 b00");
// Should match: [1, 51, 32, 51, 32, 49, 10, 97, 48, 48, 32, 98, 48, 48, 2]
```

#### 1.4 Prefix Tree Builder (Days 7-10)
**Deliverable**: C++ port of TypeScript prefix tree algorithm

**Source**: `/home/camus/work/trigoRL/third_party/trigo/trigo-web/inc/trigoTreeAgent.ts` lines 77-168

**Header**: `include/prefix_tree_builder.hpp`

```cpp
#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct Move {
    int16_t x, y, z;
    int8_t type;  // 0=place, 1=pass
};

struct TreeStructure {
    // Token sequences
    std::vector<int64_t> prefix_ids;      // [n] prefix tokens
    std::vector<int64_t> evaluated_ids;   // [m] evaluated tokens (flattened tree)

    // Attention mask for evaluated region
    std::vector<float> evaluated_mask;    // [m, m] attention pattern (0/1)

    // Mapping from move index to leaf position in evaluated_ids
    std::vector<int> move_to_leaf;        // [num_moves] → leaf index

    // Token sequences for each move (for debugging)
    std::vector<std::vector<int64_t>> move_tokens;  // [num_moves][tokens]

    // Dimensions
    int prefix_len;  // n
    int eval_len;    // m
    int num_moves;
};

class PrefixTreeBuilder {
public:
    // Build prefix tree from game state and legal moves
    TreeStructure build_tree(
        const std::string& prefix_tgn,
        const std::vector<Move>& moves
    );

private:
    // Recursive tree building (port from TypeScript)
    struct TokenSequence {
        std::vector<int64_t> tokens;
        int move_index;
    };

    struct TreeNode {
        int64_t token;
        std::vector<TreeNode> children;
        std::vector<int> leaf_moves;  // Move indices for leaves
    };

    TreeNode build_recursive(
        const std::vector<TokenSequence>& sequences,
        int depth = 0
    );

    // Convert move to token sequence
    std::vector<int64_t> move_to_tokens(const Move& move);

    // Flatten tree to evaluated_ids and build mask
    void flatten_tree(
        const TreeNode& root,
        std::vector<int64_t>& evaluated_ids,
        std::vector<float>& evaluated_mask,
        std::vector<int>& move_to_leaf
    );

    // Build ancestor attention mask
    void build_ancestor_mask(
        const TreeNode& root,
        std::vector<float>& mask,
        int num_nodes
    );
};
```

**Key Algorithm** (recursive grouping):

```cpp
TreeNode PrefixTreeBuilder::build_recursive(
    const std::vector<TokenSequence>& sequences,
    int depth
) {
    if (sequences.empty()) {
        return TreeNode{};
    }

    // Group sequences by first token
    std::map<int64_t, std::vector<TokenSequence>> groups;
    for (const auto& seq : sequences) {
        if (seq.tokens.empty()) {
            // Leaf node - this is a complete move
            continue;
        }

        int64_t first_token = seq.tokens[0];
        TokenSequence residue = seq;
        residue.tokens.erase(residue.tokens.begin());

        groups[first_token].push_back(residue);
    }

    // Build tree nodes for each unique first token
    TreeNode root;
    for (const auto& [token, group] : groups) {
        TreeNode child;
        child.token = token;

        // Recurse on remaining tokens
        if (!group.empty() && !group[0].tokens.empty()) {
            child = build_recursive(group, depth + 1);
            child.token = token;  // Set token at this level
        }

        root.children.push_back(child);
    }

    return root;
}
```

**Testing**: Compare outputs with TypeScript implementation

#### 1.5 Shared Model Inferencer (Days 11-13)
**Deliverable**: C++ class that loads 3 ONNX models and composes inference

**Header**: `include/shared_model_inferencer.hpp`

```cpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include "prefix_tree_builder.hpp"

class SharedModelInferencer {
public:
    SharedModelInferencer(
        const std::string& base_model_path,
        const std::string& policy_head_path,
        const std::string& value_head_path,
        int gpu_device_id = 0
    );

    ~SharedModelInferencer();

    // Combined inference (policy + value)
    struct InferenceResult {
        std::vector<std::vector<float>> policy_probs;  // [batch][num_moves]
        std::vector<float> values;                      // [batch]
        float inference_time_ms;
    };

    InferenceResult infer_batch(
        const std::vector<std::string>& state_tgns,
        const std::vector<std::vector<Move>>& legal_moves
    );

    // Policy-only inference (faster when value not needed)
    std::vector<std::vector<float>> infer_policy_batch(
        const std::vector<std::string>& state_tgns,
        const std::vector<std::vector<Move>>& legal_moves
    );

    // Value-only inference
    std::vector<float> infer_value_batch(
        const std::vector<std::string>& state_tgns
    );

private:
    // ONNX Runtime components
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    Ort::Session base_session_;
    Ort::Session policy_session_;
    Ort::Session value_session_;

    Ort::MemoryInfo memory_info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Helper components
    PrefixTreeBuilder tree_builder_;

    // Model metadata
    std::vector<const char*> base_input_names_;
    std::vector<const char*> base_output_names_;
    std::vector<const char*> policy_input_names_;
    std::vector<const char*> policy_output_names_;
    std::vector<const char*> value_input_names_;
    std::vector<const char*> value_output_names_;

    int vocab_size_;
    int hidden_dim_;

    // Core inference methods
    Ort::Value run_base_model(
        const std::vector<int64_t>& input_ids,
        const std::vector<float>& attention_mask,
        const std::vector<int64_t>& position_ids,
        const std::vector<int64_t>& input_shape
    );

    std::vector<std::vector<float>> run_policy_head(
        const Ort::Value& hidden_states,
        const std::vector<TreeStructure>& trees
    );

    std::vector<float> run_value_head(
        const Ort::Value& hidden_states
    );

    // Utility methods
    std::vector<int64_t> build_position_ids(
        const TreeStructure& tree
    );

    std::vector<float> apply_softmax(
        const float* logits,
        int vocab_size
    );
};
```

**Key Implementation**:

```cpp
SharedModelInferencer::InferenceResult
SharedModelInferencer::infer_batch(
    const std::vector<std::string>& state_tgns,
    const std::vector<std::vector<Move>>& legal_moves
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int batch_size = state_tgns.size();

    // Step 1: Build prefix trees for all games
    std::vector<TreeStructure> trees;
    for (int i = 0; i < batch_size; i++) {
        trees.push_back(tree_builder_.build_tree(
            state_tgns[i], legal_moves[i]
        ));
    }

    // Step 2: Prepare batched inputs
    // ... (pad to max lengths, construct tensors)

    // Step 3: Run base model ONCE
    auto hidden_states = run_base_model(
        input_ids, attention_mask, position_ids, input_shape
    );

    // Step 4: Run policy head
    auto policy_probs = run_policy_head(hidden_states, trees);

    // Step 5: Run value head
    auto values = run_value_head(hidden_states);

    auto end_time = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time
    ).count();

    return InferenceResult{policy_probs, values, elapsed_ms};
}
```

#### 1.6 Python Bindings (Day 14)
**Deliverable**: pybind11 interface for inference

**File**: `src/bindings.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "shared_model_inferencer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_mcts_inference, m) {
    m.doc() = "CUDA-accelerated MCTS inference for Trigo";

    // Move structure
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("x", &Move::x)
        .def_readwrite("y", &Move::y)
        .def_readwrite("z", &Move::z)
        .def_readwrite("type", &Move::type);

    // Inference result
    py::class_<SharedModelInferencer::InferenceResult>(m, "InferenceResult")
        .def_readonly("policy_probs", &SharedModelInferencer::InferenceResult::policy_probs)
        .def_readonly("values", &SharedModelInferencer::InferenceResult::values)
        .def_readonly("inference_time_ms", &SharedModelInferencer::InferenceResult::inference_time_ms);

    // Main inferencer class
    py::class_<SharedModelInferencer>(m, "SharedModelInferencer")
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("base_model_path"),
             py::arg("policy_head_path"),
             py::arg("value_head_path"),
             py::arg("gpu_device_id") = 0)
        .def("infer_batch", &SharedModelInferencer::infer_batch)
        .def("infer_policy_batch", &SharedModelInferencer::infer_policy_batch)
        .def("infer_value_batch", &SharedModelInferencer::infer_value_batch);
}
```

**Python Usage**:

```python
import cuda_mcts_inference as cmi

# Load models
inferencer = cmi.SharedModelInferencer(
    "models/shared/base_model.onnx",
    "models/shared/policy_head.onnx",
    "models/shared/value_head.onnx",
    gpu_device_id=0
)

# Prepare input
state_tgns = ["3 3 1\na00 b00", "3 3 1\na00"]
legal_moves = [
    [cmi.Move(x=0, y=1, z=0, type=0), cmi.Move(x=1, y=0, z=0, type=0)],
    [cmi.Move(x=0, y=0, z=0, type=0), cmi.Move(x=1, y=0, z=0, type=0)]
]

# Run inference
result = inferencer.infer_batch(state_tgns, legal_moves)

print(f"Policy probs: {result.policy_probs}")
print(f"Values: {result.values}")
print(f"Time: {result.inference_time_ms:.2f}ms")
```

### Validation (Phase 1)

```python
# test_inference.py - Validate against TypeScript

import cuda_mcts_inference as cmi
import subprocess
import json
import numpy as np

def test_policy_inference():
    """Compare C++ policy inference with TypeScript"""

    # Test case
    test_tgn = "3 3 1\na00 b00 a10"
    test_moves = [
        cmi.Move(x=1, y=1, z=0, type=0),  # b10
        cmi.Move(x=2, y=1, z=0, type=0),  # c10
    ]

    # C++ inference
    inferencer = cmi.SharedModelInferencer(
        "models/shared/base_model.onnx",
        "models/shared/policy_head.onnx",
        "models/shared/value_head.onnx"
    )
    cpp_result = inferencer.infer_batch([test_tgn], [test_moves])
    cpp_probs = cpp_result.policy_probs[0]

    # TypeScript inference (via Node.js)
    ts_result = run_typescript_inference(test_tgn, test_moves, "policy")
    ts_probs = ts_result['policy_probs']

    # Compare (should be very close, allow small numerical differences)
    assert np.allclose(cpp_probs, ts_probs, rtol=1e-4, atol=1e-5)
    print("✓ Policy inference matches TypeScript")

def test_value_inference():
    """Compare C++ value inference with TypeScript"""

    test_tgn = "3 3 1\na00 b00 a10 b10"

    inferencer = cmi.SharedModelInferencer(...)
    cpp_result = inferencer.infer_batch([test_tgn], [[]])
    cpp_value = cpp_result.values[0]

    ts_result = run_typescript_inference(test_tgn, None, "value")
    ts_value = ts_result['value']

    assert abs(cpp_value - ts_value) < 0.001
    print("✓ Value inference matches TypeScript")

def run_typescript_inference(tgn, moves, mode):
    """Run TypeScript inference via subprocess"""
    # Call into TypeScript codebase
    # ...
    pass

if __name__ == "__main__":
    test_policy_inference()
    test_value_inference()
    print("\n✓ All inference tests passed!")
```

**Phase 1 Complete**: Working ONNX inference with shared model, validated against TypeScript

---

## Phase 2: Core Game Engine (Weeks 3-4) ✅ COMPLETE

**Goal**: Complete Trigo game engine in C++

**Status**: Core components implemented and validated

### Completed Tasks

#### 2.1 TrigoGame Class ✅ COMPLETE
**Deliverable**: Complete Trigo game engine in C++

**Port from TypeScript**:
- Source: `trigoRL/third_party/trigo/trigo-web/inc/trigo/game.ts` (800 lines)
- Target: `include/trigo_game.hpp` + related files

**Implemented Components**:
- ✅ `include/trigo_types.hpp` (214 lines) - Core types (Stone, Position, Step, GameResult, etc.)
- ✅ `include/trigo_coords.hpp` (194 lines) - ab0yz coordinate encoding/decoding
- ✅ `include/trigo_game_utils.hpp` (706 lines) - Game rules (capture, Ko, suicide, territory)
- ✅ `include/trigo_game.hpp` - Complete TrigoGame class
- ✅ Cross-language validation: 100/100 games validated against TypeScript
- ✅ Python bindings via pybind11
- ✅ OpenAI Gym environment wrapper

#### 2.2 Self-Play Data Generation ✅ COMPLETE
**Deliverable**: Offline training data generation system

**Implemented Components**:
- ✅ `include/self_play_policy.hpp` (230 lines) - Extensible policy interface
  - RandomPolicy (baseline with 5% pass probability)
  - NeuralPolicy placeholder (ONNX + online IPC support planned)
  - MCTSPolicy placeholder
  - HybridPolicy placeholder
- ✅ `include/game_recorder.hpp` (276 lines) - TGN export with minimal headers
  - Only `[Board 5x5x5]` header (no Event/Site/Date/Black/White to avoid training data leakage)
  - Proper ab0yz coordinate encoding
  - Score comment at end
- ✅ `src/self_play_generator.cpp` (254 lines) - Command-line self-play generator
  - Configurable board size, policies, output directory
  - Progress logging and statistics
  - Performance: 3.33 games/sec baseline (random policy)

**Command-Line Interface**:
```bash
./self_play_generator --num-games 1000 --board 5x5x5 \
    --black-policy random --white-policy random \
    --output /path/to/data --seed 42
```

**TGN Output Format** (minimal headers for training):
```tgn
[Board 5x5x5]

1. yby 0yy
2. 0aa zyz
3. 00z Pass
...
; -1
```

#### 2.3 Python Integration ✅ COMPLETE
**Deliverable**: Python training pipeline integration

**Validated Components**:
- ✅ TGNDataset loads C++ generated .tgn files directly
- ✅ TGNByteTokenizer (128-token vocabulary) compatible
- ✅ Deterministic train/val splitting (hash-based)
- ✅ End-to-end training tested (loss decreasing normally)
- ✅ Configuration system ready (Hydra + wandb)

**Test Results**:
```
✓ 20 TGN files loaded successfully
✓ Tokenization: 872 tokens from 870 chars
✓ Train/val split: 17/3 files, no overlap
✓ Training: 544K params, loss 4.80 → 4.10
✓ Validation: loss ~4.01, no NaN
```

#### 2.4 Validation Tests ✅ COMPLETE
**Deliverable**: Cross-language validation and testing

**Tests Implemented**:
- ✅ `tests/test_trigo_coords.cpp` - ab0yz coordinate encoding
- ✅ `tests/test_trigo_game_utils.cpp` - Capture, Ko, territory calculation
- ✅ `tests/test_trigo_game.cpp` - Complete game engine
- ✅ `tests/test_game_replay.cpp` - JSON game replay (100/100 validation)
- ✅ Python integration tests (dataset loading, training pipeline)

**Python Validation**:
- ✅ `trigoRL/tests/test_dataset_loading.py` - TGN loading from C++
- ✅ `trigoRL/tests/test_training_pipeline.py` - End-to-end training

**Phase 2 Complete**: C++ game engine + self-play data generation + Python training integration

**Performance Summary**:
- Game engine: Validated against TypeScript (100/100 games)
- Self-play generation: 3.33 games/sec (random policy baseline)
- Training pipeline: Working end-to-end with decreasing loss
- Architecture: Extensible for future ONNX/MCTS policies

---

## Phase 2 (Original): Basic MCTS (CPU) - NOT STARTED

**Note**: Original Phase 2 Tasks 2.2-2.4 moved here for future implementation

### Tasks (Deferred)

#### 2.2 Basic MCTS (CPU) (Days 21-24) - DEFERRED
**Deliverable**: CPU-only MCTS that generates games

```cpp
struct MCTSNode {
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

**Critical algorithms** (from TypeScript):
1. Capture detection via flood fill (gameUtils.ts:124-235)
2. Ko rule enforcement (gameUtils.ts:300-350)
3. TGN conversion (game.ts:500-650)

#### 2.2 Basic MCTS (CPU) (Days 21-24)
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

#### 2.3 Python Bindings for Game Engine (Days 25-26)
**Deliverable**: Python-callable game interface

```cpp
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
        .def("backup", &MCTSEngine::backup);
}
```

#### 2.4 Validation Tests (Days 27-28)
**Deliverable**: Test suite comparing against TypeScript

```python
def test_capture_detection():
    """Test capture matches TypeScript"""
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

**Phase 2 Complete**: Python-callable C++ game engine generating legal games (CPU-only)

---

## Phase 3: CUDA MCTS Kernels (Weeks 5-6)

**Goal**: GPU-accelerated tree operations, 10-20× faster than CPU

### Tasks

#### 3.1 GPU Memory Layout (Days 29-31)
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

#### 3.2 UCB1 Selection Kernel (Days 32-34)
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
        // Find child with max UCB1
        // ...
        current = best_child;
    }

    output_leaves[game_id] = current;
}
```

#### 3.3 Expansion Kernel (Days 35-37)
**Deliverable**: GPU kernel for child creation

#### 3.4 Backpropagation Kernel (Days 38-40)
**Deliverable**: GPU kernel for value updates

#### 3.5 Kernel Profiling (Days 41-42)
**Deliverable**: Performance analysis and tuning

**Phase 3 Complete**: MCTS runs on GPU, 10-20× faster than CPU

---

## Phase 4: Integration (Weeks 7-8)

**Goal**: Full system with inference + MCTS + validation

### Tasks

#### 4.1 Integrate Inference with MCTS (Days 43-45)
**Deliverable**: MCTS calls SharedModelInferencer for evaluations

#### 4.2 Dynamic Batching System (Days 46-48)
**Deliverable**: Timeout-based batch aggregation

#### 4.3 TGN Writer (Days 49-50)
**Deliverable**: Output formatting with hash-based filenames

#### 4.4 Comprehensive Validation Suite (Days 51-54)
**Deliverable**: Validation against TypeScript golden reference

#### 4.5 End-to-End Testing (Days 55-56)
**Deliverable**: Generate 1000 games, validate all

**Phase 4 Complete**: Full system generating validated games

---

## Phase 5: Optimization & Production (Weeks 9-10)

**Goal**: Performance optimization and production deployment

### Tasks

#### 5.1 Performance Profiling (Days 57-59)
**Deliverable**: Bottleneck analysis with nvprof/nsys

#### 5.2 Kernel Optimization (Days 60-63)
**Deliverable**: Tuned CUDA kernels

#### 5.3 Python Scripts (Days 64-66)
**Deliverable**: User-facing scripts

```python
# scripts/generate_selfplay_cuda.py
import cuda_mcts

engine = cuda_mcts.CudaMCTSSelfPlay(
    base_model="models/shared/base_model.onnx",
    policy_head="models/shared/policy_head.onnx",
    value_head="models/shared/value_head.onnx",
    num_parallel_games=8,
    mcts_simulations=800
)

tgn_games = engine.generate_games(
    num_games=1000,
    board_shapes=[(3,3,1), (5,5,1), (2,2,2), (3,3,3)]
)
```

#### 5.4 AlphaZero Training Loop (Days 67-68)
**Deliverable**: End-to-end training script

#### 5.5 Documentation & Deployment (Days 69-70)
**Deliverable**: Production-ready system

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
- ✅ Memory usage < 1.5GB

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
| ONNX model export issues | MEDIUM | Test early, validate against Python inference |
| GPU memory overflow | MEDIUM | Dynamic node pools, monitoring, degradation |
| Performance < target | MEDIUM | Profiling, kernel optimization, shared model helps |
| Inference bottleneck | HIGH | Batching, CUDA streams, shared base model |

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

## Current Status (Updated December 5, 2025)

**Project Scope Clarification**:
- **trigo.cpp**: Data generation tools (game engine + self-play generator)
- **trigoRL**: Training, ONNX export, Python inference (separate project)
- **Focus**: MCTS-based data generation, not training infrastructure

---

**Phase 0: Planning** ✅ COMPLETE
- Architecture designed
- Feasibility confirmed
- Research completed (ONNX Runtime vs llama.cpp)
- Shared model approach validated

**Phase 1: Model Inference** ⚠️ **PARTIALLY COMPLETE**

**Status**:
- ✅ SharedModelInferencer implemented (shared_model_inferencer.cpp, 14KB)
- ✅ ONNX Runtime integration working
- ✅ Test models can be loaded and run
- ✅ Trained ONNX models available:
  - `/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/`
  - `GPT2CausalLM_ep0019_tree.onnx` (3.5MB)
  - `GPT2CausalLM_ep0019_evaluation.onnx` (3.5MB)

**Blockers for MCTS Development**:
1. ❌ **NeuralPolicy not implemented** (HIGHEST PRIORITY)
   - Need to connect SharedModelInferencer to IPolicy interface
   - Need to convert TrigoGame state → model input tensors
   - Need to convert model output logits → action probabilities

2. ⚠️ **Model format mismatch** (MEDIUM PRIORITY)
   - Test uses: base_model.onnx + policy_head.onnx + value_head.onnx (3-model split)
   - Available: tree.onnx + evaluation.onnx (2-model monolithic)
   - Need to verify which format is correct and update code accordingly

3. ❓ **PrefixTreeBuilder may be needed** (UNKNOWN)
   - If tree.onnx requires tree attention mask, must implement PrefixTreeBuilder
   - Current status: header exists but implementation incomplete

**Next Steps**:
1. Verify trained ONNX models can be loaded by SharedModelInferencer
2. Implement NeuralPolicy class (self_play_policy.hpp)
3. Test NeuralPolicy with self_play_generator
4. Implement MCTS algorithm (once NeuralPolicy works)

**Phase 2: Core Game Engine** ✅ COMPLETE (December 2025)
- ✅ TrigoGame class (types, coords, game utils, full engine)
- ✅ Self-play data generation (policy interface, game recorder, generator)
- ✅ Python integration (dataset loading, training pipeline)
- ✅ Cross-language validation (100/100 games)
- ✅ Performance baseline: 3.33 games/sec (random policy)
- ✅ TGN export with minimal headers (no training data leakage)

**Phase 3: MCTS Algorithm** ⬅️ **NEXT MAJOR MILESTONE**

**Dependencies**:
- Requires NeuralPolicy (Phase 1 blocker #1)
- Requires working model inference

**Components to Implement**:
- ❌ MCTSNode structure (tree representation)
- ❌ UCB1 selection (tree traversal)
- ❌ Tree expansion (add children)
- ❌ Backpropagation (update values)
- ❌ Root move selection (with temperature)

**Expected Performance**:
- CPU MCTS: 5-10 games/sec (vs 3.33 random)
- GPU MCTS (future): 50+ games/sec

**Phase 4: GPU Acceleration** - FUTURE
- CUDA kernels for parallel tree operations
- Batched neural network inference
- Target: 50-100 games/sec

**Timeline**:
- Phase 2: ✅ Complete
- Phase 1: ⚠️ Blocked on NeuralPolicy
- Phase 3 (MCTS): Waiting for Phase 1
- Phase 4 (GPU): Future work

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
7. `trigoRL/exportOnnx.py` (to be modified)

### Validation Reference
8. `trigoRL/third_party/trigo/trigo-web/tools/selfPlayGames.ts`

### Research Documents
9. `docs/research/LLAMA_CPP_ANALYSIS.md` - Detailed comparison and architecture decisions
10. `docs/research/MODEL_INFERENCE.md` - Existing infrastructure analysis
11. `docs/research/CUDA_INFERENCE.md` - ONNX Runtime CUDA guide

---

**Updated**: December 2025
**Next Phase**: Model Inference Implementation (Phase 1)
**Estimated Timeline**: 10 weeks to production-ready system
