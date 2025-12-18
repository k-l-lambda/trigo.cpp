# Trigo.cpp - High-Performance C++ Tools for Trigo AI

C++/CUDA inference and self-play tools for [Trigo](https://github.com/k-l-lambda/trigo) (3D Go). Provides ONNX Runtime-based neural network inference, AlphaZero-style MCTS, and high-performance self-play data generation for the [TrigoRL training pipeline](../trigoRL).

## Overview

This project implements production-ready tools for Trigo AI development:

**Key Features**:
- 🚀 **ONNX Runtime Integration**: CPU and GPU inference with trained models
- 🎯 **AlphaZero MCTS**: Value network evaluation (255× faster than random rollouts)
- 🔧 **Self-Play Generator**: Command-line tool for training data generation
- 🎲 **Random Board Selection**: 220 candidate shapes (2D and 3D) for diverse training
- ✅ **Cross-Language Validation**: 100% compatibility with TypeScript reference
- 📦 **Multiple Policies**: Random, Neural, Pure MCTS, AlphaZero MCTS
- 📊 **TGN Format**: Compatible with TrigoRL training pipeline

## Quick Start

### Prerequisites

- **CMake 3.18+**
- **GCC 9+ or Clang 10+** (C++17 support required)
- **ONNX Runtime 1.17.0+** (see installation instructions below)
- **CUDA Toolkit 11.0+** (optional, for GPU inference)

### Installing ONNX Runtime

ONNX Runtime is required for neural network inference. This project supports **ONNX Runtime 1.17.0+**.

**Version Selection Guide:**
- **H100/H200 GPUs**: Use version 1.20.0+ (required)
- **RTX 30xx/40xx, A100 GPUs**: Use version 1.17.0 or 1.20.0
- **Older GPUs (V100, P100)**: Use version 1.17.0
- **CPU-only**: Use version 1.17.0 or newer

**TL;DR - Quick Setup:**

```bash
cd /path/to/trigo.cpp

# For H100/H200 GPUs (use ONNX Runtime 1.20.0):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.0.tgz && rm onnxruntime-linux-x64-gpu-1.20.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-gpu-1.20.0/lib:$LD_LIBRARY_PATH

# OR for RTX 30xx/40xx/A100 GPUs (use ONNX Runtime 1.17.0):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz && rm onnxruntime-linux-x64-gpu-1.17.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-gpu-1.17.0/lib:$LD_LIBRARY_PATH

# Build
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**Quick Check:** The CMake configuration will automatically search for ONNX Runtime in:
1. `trigo.cpp/onnxruntime-linux-x64-gpu-1.20.0/` (bundled GPU version - H100/H200)
2. `trigo.cpp/onnxruntime-linux-x64-gpu-1.17.0/` (bundled GPU version - RTX/A100)
3. `trigo.cpp/onnxruntime-linux-x64-1.17.0/` (bundled CPU version)
4. `/opt/onnxruntime/` (system installation)
5. `/usr/local/` (system installation)

If you have ONNX Runtime pre-installed in the project directory, you can skip to the [Build](#build) section.

Choose one of the following installation methods:

#### Method 1: Download Pre-built Binaries (Recommended)

**Option A: Install to project directory (no sudo required):**
```bash
# Navigate to trigo.cpp project root
cd /path/to/trigo.cpp

# Choose based on your GPU:

# For H100/H200 (Hopper architecture):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.0.tgz
rm onnxruntime-linux-x64-gpu-1.20.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-gpu-1.20.0/lib:$LD_LIBRARY_PATH

# OR for RTX 30xx/40xx, A100 (Ampere/Ada):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz
rm onnxruntime-linux-x64-gpu-1.17.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-gpu-1.17.0/lib:$LD_LIBRARY_PATH

# OR for CPU-only inference:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-1.17.0.tgz
rm onnxruntime-linux-x64-1.17.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH
```

**Option B: Install system-wide (requires sudo):**
```bash
# Download ONNX Runtime (choose based on your GPU)
cd /tmp

# For H100/H200:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.0.tgz
sudo mv onnxruntime-linux-x64-gpu-1.20.0 /opt/onnxruntime

# OR for RTX 30xx/40xx, A100:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz
sudo mv onnxruntime-linux-x64-gpu-1.17.0 /opt/onnxruntime

# Add to library path permanently
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Version Compatibility:**

| GPU Architecture | ONNX Runtime Version | CUDA Version | Notes |
|-----------------|---------------------|--------------|-------|
| **RTX 30xx/40xx** (Ampere/Ada) | 1.17.0+ | CUDA 11.8+ | Recommended for most users |
| **H100/H200** (Hopper) | 1.20.0+ | CUDA 12.x | Required for Hopper architecture |
| **A100** (Ampere) | 1.17.0+ | CUDA 11.8+ | Fully supported |
| **V100** (Volta) | 1.17.0+ | CUDA 11.0+ | Legacy support |

**Important Notes:**
- **H100/H200 users**: MUST use ONNX Runtime 1.20.0+ with CUDA 12.x
- **RTX 30xx/40xx users**: ONNX Runtime 1.17.0 works, but 1.20.0+ recommended
- **Older GPUs (V100, P100)**: ONNX Runtime 1.17.0 is sufficient

**Download URLs for different versions:**
```bash
# ONNX Runtime 1.20.0 (for H100/H200)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz

# ONNX Runtime 1.17.0 (for RTX 30xx/40xx, A100)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
```

Check available versions at: https://github.com/microsoft/onnxruntime/releases

#### Method 2: Install via Package Manager (Ubuntu/Debian)

```bash
# Add Microsoft package repository
wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update

# Install ONNX Runtime
sudo apt-get install -y libonnxruntime
```

#### Method 3: Build from Source (Advanced)

```bash
# Clone ONNX Runtime repository
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Build CPU version
./build.sh --config Release --build_shared_lib --parallel

# Or build GPU version with CUDA
./build.sh --config Release --build_shared_lib --parallel --use_cuda \
    --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu

# Install
sudo cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/
sudo cp include/onnxruntime/core/session/*.h /usr/local/include/
sudo ldconfig
```

#### Verify Installation

```bash
# Check if library is found
ldconfig -p | grep onnxruntime

# Expected output:
# libonnxruntime.so.1.17.0 (libc6,x86-64) => /opt/onnxruntime/lib/libonnxruntime.so.1.17.0
```

### Build

```bash
# Clone repository (if not already cloned)
cd /path/to/trigo.cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# If ONNX Runtime is in a custom location, specify the path:
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime

# Build (use all CPU cores)
make -j$(nproc)

# Verify build
ls -lh self_play_generator test_trigo_game test_alphazero_mcts
```

**Expected output:**
```
✓ 49 targets built successfully
✓ Binaries created in build/:
  - self_play_generator (main CLI tool)
  - test_trigo_game (game engine tests)
  - test_alphazero_mcts (MCTS tests)
  - test_*.cpp (other unit tests)
```

### Run Tests

```bash
# Run game engine tests
./test_trigo_game

# Run MCTS tests
./test_alphazero_mcts

# Run all tests
ctest --output-on-failure
```

### Common Build Issues

**Issue 1: ONNX Runtime not found**
```
CMake Error: Could not find onnxruntime library
```
**Solution:**
- Verify ONNX Runtime is installed: `ldconfig -p | grep onnxruntime`
- Set `ONNXRUNTIME_ROOT` environment variable or CMake option
- Check `LD_LIBRARY_PATH` includes ONNX Runtime lib directory

**Issue 2: CUDA not found (GPU builds)**
```
CMake Error: Could not find CUDA
```
**Solution:**
- Install CUDA Toolkit: `sudo apt-get install nvidia-cuda-toolkit`
- Set `CUDA_HOME`: `export CUDA_HOME=/usr/local/cuda`
- Verify CUDA installation: `nvcc --version`

**Issue 3: Linker errors at runtime**
```
error while loading shared libraries: libonnxruntime.so.1.17.0: cannot open shared object file
```
**Solution:**
```bash
# Add ONNX Runtime to library path
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
# Or permanently:
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
sudo ldconfig
```

**Issue 4: CMake version too old**
```
CMake Error: CMake 3.18 or higher is required
```
**Solution:**
```bash
# Install newer CMake from Kitware repository
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt-get update && sudo apt-get install cmake
```

### Usage

#### Self-Play Data Generation

**Generate games with random board shapes (recommended for training):**
```bash
# Random board selection from 220 candidates (2D: 2-13×1-13×1, 3D: 2-5×2-5×2-5)
# This creates a diverse dataset covering various board sizes
export TRIGO_FORCE_CPU=1

./self_play_generator \
    --num-games 100 \
    --random-board \
    --black-policy mcts \
    --white-policy mcts \
    --model ../models/trained_shared \
    --output /path/to/data/mcts_games \
    --seed 42

# With custom board ranges (e.g., small 2D boards only)
./self_play_generator \
    --num-games 100 \
    --random-board \
    --board-ranges "3-9x3-9x1-1,2-3x2-3x2-2" \
    --black-policy mcts \
    --white-policy mcts \
    --model ../models/trained_shared \
    --output /path/to/data/mcts_games
```

**Generate games with fixed board size:**
```bash
# AlphaZero-style MCTS with value network on 5×5×5 board
# Force CPU for best performance (1.52× faster than GPU for batch=1 MCTS)
export TRIGO_FORCE_CPU=1

./self_play_generator \
    --num-games 100 \
    --board 5x5x5 \
    --black-policy mcts \
    --white-policy mcts \
    --model ../models/trained_shared \
    --output /path/to/data/mcts_games \
    --seed 42

# With custom MCTS parameters
./self_play_generator \
    --num-games 100 \
    --board 5x5x5 \
    --black-policy mcts \
    --white-policy mcts \
    --model ../models/trained_shared \
    --mcts-simulations 50 \
    --mcts-c-puct 1.5 \
    --output /path/to/data/mcts_games
```

**Generate games with neural policy (faster, less exploration):**
```bash
./self_play_generator \
    --num-games 1000 \
    --board 5x5x5 \
    --black-policy neural \
    --white-policy neural \
    --model ../models/trained_shared \
    --output /path/to/data/neural_games
```

**Generate baseline games with random policy:**
```bash
# Random policy with random board shapes
./self_play_generator \
    --num-games 10000 \
    --random-board \
    --black-policy random \
    --white-policy random \
    --output /path/to/data/random_games \
    --seed 42

# Random policy with fixed board
./self_play_generator \
    --num-games 10000 \
    --board 5x5x5 \
    --black-policy random \
    --white-policy random \
    --output /path/to/data/random_games \
    --seed 42
```

#### Board Shape Options

The generator supports two modes for board shape selection:

**Fixed Board (--board):**
```bash
--board 5x5x5    # Fixed 5×5×5 board for all games
--board 9x9x1    # Fixed 9×9×1 (2D) board for all games
--board 13x13x1  # Fixed 13×13 (traditional Go size)
```

**Random Board (--random-board):**
```bash
--random-board   # Randomly select from 220 candidate shapes per game
```

The random board mode uses default ranges:
- **2D boards**: 2-13×1-13×1 (156 shapes)
- **3D boards**: 2-5×2-5×2-5 (64 shapes)
- **Total**: 220 candidate shapes

**Custom Board Ranges (--board-ranges):**

You can specify custom ranges with `--board-ranges` (requires `--random-board`):

```bash
# Format: "minX-maxXxminY-maxYxminZ-maxZ,..."
--random-board --board-ranges "2-13x1-13x1-1,2-5x2-5x2-5"  # Default (220 shapes)
--random-board --board-ranges "3-9x3-9x1-1"                 # Small 2D boards only
--random-board --board-ranges "2-3x2-3x2-3"                 # Tiny 3D boards only
--random-board --board-ranges "5-5x5-5x5-5,9-9x9-9x1-1"    # Mix of 5×5×5 and 9×9
```

**Range Format**: `minX-maxXxminY-maxYxminZ-maxZ`
- Multiple ranges can be comma-separated
- Each range generates all combinations within bounds
- Example: `2-3x2-3x1-1` generates: 2×2×1, 2×3×1, 3×2×1, 3×3×1 (4 shapes)

Random board selection is recommended for training diverse models that generalize across board sizes.

**Parameter Rules**:
- `--board` and `--random-board` are mutually exclusive
- `--board-ranges` requires `--random-board`

#### Policy Options

Available policy types:
- `random` - Random valid moves (fast, no model required)
- `neural` - Direct neural network inference (requires `--model`)
- `mcts` - AlphaZero MCTS with value network (requires `--model`)

#### MCTS Parameters

- `--mcts-simulations N` - Number of MCTS simulations per move (default: 50)
- `--mcts-c-puct F` - Exploration constant for PUCT formula (default: 1.5)
- `--mcts-temperature F` - Temperature for move selection (default: 1.0)
- `--mcts-dirichlet-alpha F` - Dirichlet noise alpha for root exploration (default: 0.3)

#### Model Path

The `--model` parameter should point to a directory containing the 3-model ONNX architecture:
```
models/trained_shared/
├── base_model.onnx       # Shared transformer base
├── policy_head.onnx      # Policy network
└── value_head.onnx       # Value network
```

Models are exported from TrigoRL using `exportOnnx.py`.

#### Performance Tips

**For Self-Play Generation:**
- Use `TRIGO_FORCE_CPU=1` for MCTS (CPU is 1.52× faster than GPU)
- MCTS with 50 simulations: ~280ms per move on CPU
- Can generate 10,000 games in 32.5 hours on a single CPU

**For GPU Inference:**
- GPU is recommended only for training with large batches (256+)
- Small batch sizes (batch=1) underutilize GPU parallelism
- GPU shows ~1.52× performance penalty for MCTS due to kernel launch overhead

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
│  │   ├─ Random Board Selection (220 candidates)              │
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
│   ├── board_shape_candidates.hpp   # Random board shape generation
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

### C++ vs TypeScript MCTS Performance

Comprehensive benchmarking (December 2025) shows significant performance advantages:

| Implementation | Time per Move | Games per Minute | Speedup vs TypeScript |
|----------------|---------------|------------------|----------------------|
| **C++ CPU (MCTS)** | 280ms | 3.6 games/min | **6.59×** |
| **C++ GPU (MCTS)** | 335ms | 3.0 games/min | 5.51× |
| TypeScript (MCTS) | 1846ms | 0.65 games/min | 1× (baseline) |

**Key Findings:**
- **C++ is 5.47× faster** than TypeScript for MCTS self-play
- **CPU outperforms GPU by 1.52×** for batch=1 MCTS workloads
- Can generate **10,000 games in 32.5 hours** on a single CPU

### Value Network vs Random Rollouts

AlphaZero-style MCTS with value network provides massive speedup over traditional rollouts:

| Implementation | Time per simulation | 50 simulations | 800 simulations |
|----------------|---------------------|----------------|-----------------|
| PureMCTS (rollouts) | 923ms | 46 seconds | 12+ minutes |
| MCTS (value network) | 3.6ms | 180ms | 2.9 seconds |
| **Speedup** | **255×** | **255×** | **255×** |

**Test Configuration:**
- Board: 5×5×1
- MCTS simulations: 50 per move
- Model: Dynamic ONNX shared architecture
- Hardware: Multi-core CPU + RTX 3090 (24GB)

### Why CPU is Faster Than GPU for MCTS

For batch=1 MCTS workloads, CPU shows better performance due to:
- **Kernel launch overhead**: ~100-150μs per GPU call dominates small inference
- **Memory transfers**: 7 additional Memcpy operations for GPU
- **Underutilization**: GPU cores 99% idle with batch=1
- **Operator fallback**: Some operators fall back to CPU

**Recommendation:**
- ✅ Use CPU for MCTS self-play (batch=1)
- ✅ Use GPU for training (batch=256+)
- ✅ Future: Batch MCTS leaf evaluation for GPU (64-256 positions simultaneously)

### Production Capacity

**Single CPU Performance:**
- 7.7 games per minute (MCTS, 50 simulations/move)
- 10,000 games in 32.5 hours
- Ready for large-scale RL training pipelines

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
