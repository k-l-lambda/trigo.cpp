# MCTS Performance Benchmark

Automated script for comparing C++ and TypeScript MCTS self-play performance.

## Quick Start

### Basic Usage (with default parameters)

```bash
cd /home/camus/work/trigo.cpp/tools
./benchmark_mcts.sh
```

Default parameters:
- Board: 5x5x1
- Games: 10
- MCTS Simulations: 50
- C++ Model: `../models/trained_shared`
- TypeScript Dir: `/home/camus/work/trigo/trigo-web`

### Custom Parameters

```bash
# Test 5x5x5 board, 20 games, 100 simulations per move
./benchmark_mcts.sh --board 5x5x5 --games 20 --simulations 100

# Specify custom model paths
./benchmark_mcts.sh \
  --board 5x5x1 \
  --games 10 \
  --simulations 50 \
  --cpp-model /path/to/model \
  --ts-dir /path/to/trigo-web

# Specify output directory
./benchmark_mcts.sh --output-dir /tmp/my_benchmark
```

### Show Help

```bash
./benchmark_mcts.sh --help
```


## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--board SHAPE` | Board shape (format: 5x5x1) | 5x5x1 |
| `--games N` | Number of games | 10 |
| `--simulations N` | MCTS simulations per move | 50 |
| `--cpp-model PATH` | C++ model path | ../models/trained_shared |
| `--ts-dir PATH` | TypeScript project directory | /home/camus/work/trigo/trigo-web |
| `--output-dir PATH` | Output directory | /tmp/mcts_benchmark_TIMESTAMP |


## Output Files

The script generates the following files:

```
/tmp/mcts_benchmark_TIMESTAMP/
├── benchmark_report.txt     # Test report summary
├── cpp_log.txt              # C++ full log
├── ts_log.txt               # TypeScript full log
├── cpp_output/              # C++ generated game files
│   └── game_*.tgn
└── ts_output/               # TypeScript generated game files
    └── game_*.tgn
```


## Sample Output

```
============================================================
MCTS Performance Benchmark
============================================================
Configuration:
  Board Shape:        5x5x1
  Number of Games:    10
  MCTS Simulations:   50
  Output Directory:   /tmp/mcts_benchmark_20251205_170000

C++ Configuration:
  Build Directory:    /home/camus/work/trigo.cpp/build
  Model Path:         ../models/trained_shared

TypeScript Configuration:
  Project Directory:  /home/camus/work/trigo/trigo-web
  Board Format:       5*5*1
============================================================

[1/2] Running C++ AlphaZero MCTS...
--------------------------------------------------------------
✓ C++ test completed successfully
Duration: 162.45s

[2/2] Running TypeScript MCTS...
--------------------------------------------------------------
✓ TypeScript test completed successfully
Duration: 624.12s

============================================================
Results Analysis
============================================================

Performance Summary:
--------------------------------------------------------------
Metric                    | C++ MCTS        | TypeScript MCTS
--------------------------------------------------------------
Total Duration            | 162.45s         | 624.12s
Games Completed           | 10              | 10
Total Moves               | 508             | 311
Avg Moves per Game        | 50.8            | 31.1
Time per Game             | 16.25s          | 62.41s
--------------------------------------------------------------

Speedup: C++ is 3.84× faster than TypeScript
```


## Recommended Test Scenarios

### Quick Test (~15 minutes)
```bash
./benchmark_mcts.sh --board 5x5x1 --games 5 --simulations 25
```

### Standard Test (~15 minutes)
```bash
./benchmark_mcts.sh --board 5x5x1 --games 10 --simulations 50
```

### Large-Scale Test (~1 hour)
```bash
./benchmark_mcts.sh --board 5x5x5 --games 20 --simulations 100
```

### High-Precision MCTS (~2 hours)
```bash
./benchmark_mcts.sh --board 5x5x1 --games 10 --simulations 200
```


## Important Notes

1. **C++ Model Path**: Ensure `--cpp-model` points to a directory containing:
   - `base_model.onnx`
   - `policy_head.onnx`
   - `value_head.onnx`

2. **TypeScript Configuration**: Ensure the TypeScript project's `.env` file has the correct model path configured

3. **GPU Acceleration Issue** ⚠️:
   - **Current Status**: C++ version may crash when attempting to load CUDA
   - **Cause**: System has CUDA 11.8, but ONNX Runtime 1.17.0 requires CUDA 12.x
   - **Solutions**:
     1. Download ONNX Runtime 1.16.3 (supports CUDA 11.8)
     2. Or upgrade CUDA to 12.x
     3. Or use previous test data as reference (see below)
   - **Reference Data**: Even on CPU, C++ is 3.85× faster

4. **Known Test Results** (5x5x1, 10 games, 50 simulations, CPU mode):
   - C++ MCTS: 162s (16.2s/game, 50.8 moves/game)
   - TypeScript MCTS: 624s (62.4s/game, 31.1 moves/game)
   - Speedup: 3.85×

4. **Time Estimates**:
   - 5x5x1, 10 games, 50 simulations: ~13 minutes
   - 5x5x5, 10 games, 50 simulations: ~30 minutes
   - Time scales with board size and simulation count


## Troubleshooting

### C++ Test Failed
1. Check if model files exist
2. Review `cpp_log.txt` for error details
3. Verify C++ project is correctly compiled

### TypeScript Test Failed
1. Verify Node.js version is correct (v21.7.1)
2. Check `.env` file configuration
3. Review `ts_log.txt` for error details
4. Verify onnxruntime-node is correctly installed

### Performance Anomalies
1. Ensure no other high-load programs are running during tests
2. Check CPU/memory usage
3. Run multiple times and take the average


## Related Documentation

- [Performance Analysis Report](../docs/PERFORMANCE_ANALYSIS.md)
- [C++ MCTS Implementation](../include/mcts.hpp)
- [TypeScript MCTS Implementation](../../trigo/trigo-web/inc/mctsAgent.ts)
