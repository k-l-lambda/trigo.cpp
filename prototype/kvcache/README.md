# KV Cache Prototype - Standalone Validation

This is an isolated prototype for validating ONNX Runtime KV cache implementation with GPU memory management.

## Overview

**Purpose**: Validate KV cache design with a real transformer model (GPT-2) before integrating into the main TrigoRL project.

**Components**:
- **Python export script**: Exports GPT-2 from Hugging Face with KV cache I/O
- **C++ inferencer**: `KVCacheInferencer` class using ONNX Runtime IOBinding API
- **Benchmark tests**: Measure latency, memory, and speedup

## Quick Start

### 1. Prerequisites

```bash
# Python dependencies
pip install torch transformers onnx

# System dependencies (if not already installed)
# - CUDA 11.8+ (for GPU support)
# - ONNX Runtime 1.17.0
# - CMake 3.18+
```

### 2. Export Model

```bash
cd /home/camus/work/trigo.cpp/prototype/kvcache

# Export GPT-2 with KV cache
python export_gpt2_kvcache.py

# Output:
#   models/gpt2_with_cache.onnx (~500 MB)
#   models/config.json
```

**Expected output**:
```
Loading gpt2 model...
Model configuration:
  Layers: 12
  Heads: 12
  Head dimension: 64
  Vocabulary size: 50257
  Max sequence length: 1024

Exporting to ONNX: ./models/gpt2_with_cache.onnx
✓ Model exported successfully
✓ Configuration saved: ./models/config.json
✓ ONNX model is valid

Model size: 500.23 MB
```

### 3. Build C++ Test

```bash
mkdir build
cd build

# With GPU support (recommended)
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ..

# Or CPU-only
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF ..

make -j$(nproc)
```

### 4. Run Benchmark

```bash
# GPU mode (recommended)
./test_kvcache_prototype

# CPU mode
export TRIGO_FORCE_CPU=1
./test_kvcache_prototype
```

## Expected Results

### Test 1: Basic Inference

```
✓ Model loaded successfully

Generating tokens:
  Token  1: input=50256 | latency=  52.34ms | top_pred=123
  Token  2: input= 1001 | latency=   6.78ms | top_pred=456
  Token  3: input= 1002 | latency=   6.45ms | top_pred=789
  ...
```

**Expected**:
- First token: 50-100ms (cold start)
- Subsequent tokens: 5-10ms (**10-20× faster**)

### Test 2: Performance Comparison

```
[Scenario A] Sequential generation WITH KV cache
  Total time: 150.32 ms
  Avg per token: 7.52 ms
  First token: 52.34 ms
  Subsequent avg: 6.45 ms
  Speedup: 8.11×

[Scenario B] Recomputing full sequence WITHOUT KV cache
  Total time: 2847.56 ms
  Avg per token: 142.38 ms

[Performance Summary]
  Overall speedup: 18.94×
  Time saved: 2697.24 ms
  Efficiency: 5.3% of no-cache time

✓ Test passed: Significant speedup achieved (18.94×)
```

**Expected**:
- Overall speedup: **10-50×** (scales with sequence length)
- Efficiency: **95% time saved**

### Test 3: Memory Usage

```
KV Cache Memory:
  Allocated: 75.50 MB
  Theoretical: 75.50 MB

✓ Memory allocation matches theoretical calculation
```

**Formula**:
```
Cache size = 2 × num_layers × batch × num_heads × max_seq_len × head_dim × sizeof(float)
           = 2 × 12 × 1 × 12 × 1024 × 64 × 4
           = 75.5 MB
```

## Benchmark Targets

| Metric | Target | Measured |
|--------|--------|----------|
| First token latency | 50-100ms | _TBD_ |
| Subsequent token latency | 5-10ms | _TBD_ |
| Speedup factor | >10× | _TBD_ |
| Memory overhead | ~75 MB | _TBD_ |
| GPU memory copies | 0 (IOBinding) | _TBD_ |

## Architecture

### KVCacheInferencer Design

```cpp
class KVCacheInferencer {
    // Persistent GPU tensors (RAII)
    std::vector<Ort::Value> past_key_cache_;   // [num_layers]
    std::vector<Ort::Value> past_value_cache_;

    // IOBinding for zero-copy GPU operations
    Ort::Value forward(const std::vector<int64_t>& input_ids) {
        Ort::IoBinding io_binding(*session_);

        // Bind inputs (CPU)
        io_binding.BindInput("input_ids", input_tensor);

        // Bind cache (GPU, persistent)
        for (int i = 0; i < num_layers_; i++) {
            io_binding.BindInput(f"past_key_{i}", past_key_cache_[i]);
            io_binding.BindInput(f"past_value_{i}", past_value_cache_[i]);
        }

        // Bind outputs (GPU, auto-allocate)
        io_binding.BindOutput("logits", *memory_info_gpu_);
        for (int i = 0; i < num_layers_; i++) {
            io_binding.BindOutput(f"present_key_{i}", *memory_info_gpu_);
            io_binding.BindOutput(f"present_value_{i}", *memory_info_gpu_);
        }

        // Run inference (all GPU, zero CPU-GPU copy)
        session_->Run(Ort::RunOptions{nullptr}, io_binding);

        // Update cache (move semantics, no copy)
        auto outputs = io_binding.GetOutputValues();
        for (int i = 0; i < num_layers_; i++) {
            past_key_cache_[i] = std::move(outputs[1 + i * 2]);
            past_value_cache_[i] = std::move(outputs[2 + i * 2]);
        }

        return outputs[0];  // logits
    }
};
```

**Key features**:
- **Zero-copy**: Cache stays in GPU memory across calls
- **RAII**: Automatic cleanup via `Ort::Value` destructor
- **Move semantics**: No GPU memory copies between steps

### Model Export

The Python script exports GPT-2 with explicit KV cache I/O:

**Inputs**:
- `input_ids`: [batch, seq_len]
- `position_ids`: [batch, seq_len]
- `past_key_0` to `past_key_11`: [batch, num_heads, past_len, head_dim]
- `past_value_0` to `past_value_11`: [batch, num_heads, past_len, head_dim]

**Outputs**:
- `logits`: [batch, seq_len, vocab_size]
- `present_key_0` to `present_key_11`: [batch, num_heads, total_len, head_dim]
- `present_value_0` to `present_value_11`: [batch, num_heads, total_len, head_dim]

**Dynamic axes**: All sequence dimensions support dynamic lengths.

## Troubleshooting

### Model export fails

**Issue**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip install torch transformers onnx
```

### CMake can't find ONNX Runtime

**Issue**: `ONNX Runtime not found`

**Solution**: Set explicit path:
```bash
cmake -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime ..
```

### CUDA version mismatch

**Issue**: ONNX Runtime requires CUDA 12.x but system has 11.8

**Solution**: Use CPU mode:
```bash
export TRIGO_FORCE_CPU=1
./test_kvcache_prototype
```

Or use CPU-only ONNX Runtime.

### Model inference fails

**Issue**: `Failed to load ONNX model` or wrong I/O names

**Solution**: Verify model was exported correctly:
```bash
python -c "import onnx; m = onnx.load('models/gpt2_with_cache.onnx'); print([i.name for i in m.graph.input[:5]])"
```

Expected: `['input_ids', 'position_ids', 'past_key_0', 'past_value_0', 'past_key_1']`

## Next Steps

After successful validation:

1. **Document findings** in `docs/KVCACHE_BENCHMARK.md`:
   - Measured latencies
   - Actual speedup factors
   - Memory usage
   - GPU profiling data

2. **Integrate into main project**:
   - Update `exportOnnx.py` in TrigoRL to support KV cache
   - Modify `SharedModelInferencer` to use IOBinding
   - Add KV cache mode to `NeuralPolicy`

3. **Production optimization**:
   - Batch multiple positions for GPU efficiency
   - Implement sliding window for long sequences
   - Add mixed precision (FP16) support

## File Structure

```
prototype/kvcache/
├── README.md                    # This file
├── export_gpt2_kvcache.py       # Model export script
├── CMakeLists.txt               # Build configuration
├── test_kvcache.cpp             # Test program
├── include/
│   └── kvcache_inferencer.hpp   # Inferencer header
├── src/
│   └── kvcache_inferencer.cpp   # Inferencer implementation
├── models/                      # Generated (not in git)
│   ├── gpt2_with_cache.onnx
│   └── config.json
└── build/                       # Generated (not in git)
    └── test_kvcache_prototype
```

## Performance Expectations

### Theoretical Analysis

Without KV cache, generating N tokens requires:
```
Time = Sum(T_inference(i)) for i in 1..N
     = T_model * (1 + 2 + 3 + ... + N)
     = T_model * N * (N + 1) / 2
     = O(N^2)
```

With KV cache:
```
Time = T_first + (N - 1) * T_cached
     = O(N)
```

**Speedup** (for N=100 tokens):
```
Speedup = (100 * 101 / 2) / (1 + 99)
        = 5050 / 100
        = 50.5×
```

### Practical Considerations

- **GPU kernel launch overhead**: ~100μs per call
- **Memory bandwidth**: Limited by PCIe for small batches
- **Tensor allocation**: Amortized over sequence length
- **Attention computation**: Quadratic in sequence length

**Expected real-world speedup**: 10-50× (depending on sequence length and hardware)

## References

- **Main design doc**: `../../docs/KVCACHE_DESIGN.md`
- **ONNX Runtime C++ API**: https://onnxruntime.ai/docs/api/c/
- **Transformers KV cache**: https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin.generate
- **IOBinding tutorial**: https://onnxruntime.ai/docs/performance/tune-performance.html#io-binding

---

**Status**: ✅ Ready for testing
**Last updated**: December 5, 2025
