# Trigo.cpp - High-Performance C++ Tools for Trigo AI

C++/CUDA inference and self-play tools for [Trigo](https://github.com/k-l-lambda/trigo) (3D Go). Provides ONNX Runtime-based neural network inference, AlphaZero-style MCTS, and high-performance self-play data generation for the [TrigoRL training pipeline](../trigoRL).

## Overview

This project implements production-ready tools for Trigo AI development:

**Key Features**:
- 🚀 **ONNX Runtime Integration**: CPU and GPU inference with trained models
- 🎯 **AlphaZero MCTS**: Value network evaluation (255× faster than random rollouts)
- 🔧 **Self-Play Generator**: Command-line tool for training data generation
- ✅ **Cross-Language Validation**: 100% compatibility with TypeScript reference
- 📦 **Multiple Policies**: Random, Neural, Pure MCTS, AlphaZero MCTS
- 📊 **TGN Format**: Compatible with TrigoRL training pipeline

## Quick Start

### Prerequisites

- CMake 3.18+
- GCC 9+ or Clang 10+
- CUDA Toolkit 11.0+ (optional, for GPU inference)
- ONNX Runtime 1.17.0+ (provided in repository)

### Build

```bash
# Clone repository
cd /path/to/trigo.cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
./test_trigo_game
./test_alphazero_mcts
```

### Usage

#### Self-Play Data Generation

```bash
# Generate 1000 games using random policies
./self_play_generator \
    --num-games 1000 \
    --board 5x5x5 \
    --black-policy random \
    --white-policy random \
    --output /path/to/data/games \
    --seed 42

# Generate games with neural policy
./self_play_generator \
    --num-games 100 \
    --board 5x5x5 \
    --black-policy neural \
    --white-policy neural \
    --model-path ../models/trained_shared \
    --output /path/to/data/neural_games

# Generate games with MCTS (AlphaZero-style)
./self_play_generator \
    --num-games 10 \
    --board 5x5x5 \
    --black-policy neural \
    --white-policy neural \
    --model-path ../models/trained_shared \
    --output /path/to/data/mcts_games \
    --mcts-simulations 50
```

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
│  │   ├─ AlphaZero MCTS (PUCT, value network) - Production   │
│  │   └─ Pure MCTS (UCB1, random rollouts) - Reference       │
│  ├─ Self-Play Generator (data generation tool)               │
│  │   ├─ RandomPolicy                                         │
│  │   ├─ NeuralPolicy (ONNX inference)                        │
│  │   ├─ MCTSPolicy (Pure MCTS)                               │
│  │   └─ TGN File Export                                      │
│  └─ Python Bindings (pybind11) [future]                     │
└─────────────────────────────────────────────────────────────┘
                           ↓ generates
                    Training Data (.tgn)
                           ↓ feeds back to
                      TrigoRL Pipeline
```

### Directory Structure

```
trigo.cpp/
├── include/              # Public C++ headers
│   ├── trigo_game.hpp               # 3D Go game engine
│   ├── trigo_coords.hpp             # ab0yz coordinate system
│   ├── trigo_game_utils.hpp         # Capture, Ko, territory
│   ├── mcts.hpp                     # AlphaZero MCTS (value network)
│   ├── mcts_moc.hpp                 # Pure MCTS (random rollouts)
│   ├── self_play_policy.hpp         # Policy interfaces
│   ├── shared_model_inferencer.hpp  # ONNX Runtime wrapper
│   ├── prefix_tree_builder.hpp      # Tree attention
│   ├── tgn_tokenizer.hpp            # TGN tokenization
│   └── tgn_utils.hpp                # TGN generation utilities
├── src/                  # Implementation
│   ├── trigo_game.cpp
│   ├── shared_model_inferencer.cpp
│   ├── tgn_tokenizer.cpp
│   ├── prefix_tree_builder.cpp
│   └── self_play_generator.cpp      # Main CLI tool
├── tests/                # Unit tests
│   ├── test_trigo_game.cpp
│   ├── test_mcts.cpp
│   ├── test_alphazero_mcts.cpp
│   ├── test_neural_policy_inference.cpp
│   └── ...
├── models/               # Trained ONNX models
│   └── trained_shared/
│       ├── base_model.onnx
│       ├── policy_head.onnx
│       └── value_head.onnx
├── docs/                 # Documentation
│   └── PLAN.md           # Development roadmap
├── CMakeLists.txt
└── README.md
```

## Performance

### MCTS Performance Comparison

| Implementation | Time per simulation | 50 simulations | 800 simulations |
|----------------|---------------------|----------------|-----------------|
| PureMCTS (rollouts) | 923ms | 46 seconds | 12+ minutes |
| MCTS (value network) | 3.6ms | 180ms | 2.9 seconds |
| **Speedup** | **255×** | **255×** | **255×** |

**Test System**: Intel CPU, NVIDIA GPU, ONNX Runtime 1.17.0

### Self-Play Generation Speed

| Policy Combination | Games per second (5×5×5) |
|-------------------|--------------------------|
| Random vs Random | ~3 games/sec |
| Neural vs Random | ~1 game/sec |
| Neural vs Neural | ~0.5 games/sec |
| MCTS vs Random | ~0.3 games/sec |

## Implementation Status

### ✅ Phase 1: Model Inference - COMPLETE

- ✅ `SharedModelInferencer` - ONNX Runtime with shared base model
- ✅ `TGNTokenizer` - Compatible with Python training tokenizer
- ✅ `PrefixTreeBuilder` - Tree attention support
- ✅ ONNX models can be loaded and run
- ✅ Model format: 3-model architecture (base + policy_head + value_head)

### ✅ Phase 2: Game Engine - COMPLETE

- ✅ `TrigoGame` - Complete 3D Go engine
- ✅ `trigo_coords.hpp` - ab0yz coordinate encoding
- ✅ `trigo_game_utils.hpp` - Capture, Ko, territory
- ✅ `tgn_utils.hpp` - Shared TGN generation
- ✅ Cross-language validation (100/100 games vs TypeScript)

### ✅ Phase 3: MCTS Algorithm - COMPLETE

- ✅ PureMCTS with random rollouts (`include/mcts_moc.hpp`)
  - UCB1 selection, tree expansion, backpropagation working
  - Reference implementation for validation
  - Performance: ~923ms per simulation
- ✅ AlphaZero-style MCTS with value network (`include/mcts.hpp`)
  - Uses `SharedModelInferencer::value_inference()` for evaluation
  - PUCT formula for exploration
  - **Performance: 255× speedup** (~3.6ms per simulation)
  - Production-ready implementation

### 🚧 Phase 4: GPU Acceleration - FUTURE

- Planned: CUDA MCTS kernels for parallel tree operations
- Planned: Batched neural network inference
- Target: 50-100 games/sec on GPU

## Validation

The implementation is validated against the TypeScript golden reference at `trigoRL/third_party/trigo/trigo-web/`.

**Validation Results**:
- ✅ 100/100 games match TypeScript implementation
- ✅ All moves legal (capture, Ko, suicide rules)
- ✅ Territory scoring matches
- ✅ TGN format parseable by TGNValueDataset
- ✅ Games terminate correctly

## Integration with TrigoRL Training

### Data Flow

1. **TrigoRL** trains models → exports `.onnx` files
2. **trigo.cpp** loads `.onnx` → runs self-play → generates `.tgn` files
3. **TrigoRL** loads `.tgn` files → continues training (iterative improvement)

### Model Format

The project uses a 3-model architecture:
- `base_model.onnx` - Shared transformer base
- `policy_head.onnx` - Policy network (move prediction)
- `value_head.onnx` - Value network (position evaluation)

Models are exported from TrigoRL using `exportOnnx.py`.

## Development

### Building Tests

```bash
cd build

# Build specific test
make test_trigo_game

# Run test
./test_trigo_game
```

### Available Tests

- `test_trigo_game` - Game engine validation
- `test_trigo_coords` - Coordinate system
- `test_trigo_game_utils` - Go rules (capture, Ko)
- `test_mcts` - Pure MCTS implementation
- `test_alphazero_mcts` - AlphaZero MCTS performance
- `test_neural_policy_inference` - Neural policy
- `test_tgn_consistency` - TGN format validation
- `test_game_replay` - Cross-language validation

### Code Style

- C++17 standard
- Modern C++ (curly braces on standalone lines, tab indentation)
- Comprehensive comments
- DRY principle (avoid code duplication)

## Documentation

- [Development Plan](docs/PLAN.md) - Roadmap and implementation status
- [Model Inference](docs/research/MODEL_INFERENCE.md) - ONNX Runtime integration
- [CUDA Inference](docs/research/CUDA_INFERENCE.md) - GPU acceleration research
- [Validation Report](docs/research/VALIDATION_REPORT.md) - Cross-language validation

## References

- [Trigo Game Rules](https://github.com/k-l-lambda/trigo)
- [TrigoRL Training Pipeline](../trigoRL)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [ONNX Runtime](https://onnxruntime.ai/)

## License

[Specify license]

---

**Project Scope**: C++/CUDA tools for Trigo game engine and MCTS self-play generation

**Goal**: Provide high-performance tools for TrigoRL training pipeline

**Status**: Phases 1-3 Complete - Production-ready self-play generation with AlphaZero MCTS
