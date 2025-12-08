#!/bin/bash
# Comprehensive Prefix Cache Performance Benchmark
# December 8, 2025

set -e

MODEL_DIR="../trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached"
OUTPUT_DIR="/tmp/prefix_cache_benchmark_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "===================================================================="
echo "Prefix Cache Performance Benchmark"
echo "===================================================================="
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Test 1: Cached inference with real game
echo "[Test 1/3] Running cached inference with real game..."
./build/test_cached_inference_game "$MODEL_DIR" 2>&1 | tee "$OUTPUT_DIR/test1_cached_inference.log"

echo ""
echo "[Test 2/3] Running dynamic shape benchmark..."
./build/benchmark_dynamic_shapes "$MODEL_DIR" 2>&1 | tee "$OUTPUT_DIR/test2_dynamic_shapes.log"

echo ""
echo "[Test 3/3] Running CachedNeuralPolicy integration test..."
./build/test_cached_neural_policy "$MODEL_DIR" 2>&1 | tee "$OUTPUT_DIR/test3_policy_integration.log"

echo ""
echo "===================================================================="
echo "Benchmark Complete"
echo "===================================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Extract key metrics
echo "Extracting performance metrics..."

echo "=== Test 1: Cached Inference with Real Game ===" > "$OUTPUT_DIR/summary.txt"
grep -A 2 "Prefix computation:" "$OUTPUT_DIR/test1_cached_inference.log" >> "$OUTPUT_DIR/summary.txt" || true
echo "" >> "$OUTPUT_DIR/summary.txt"

echo "=== Test 2: Dynamic Shape Performance ===" >> "$OUTPUT_DIR/summary.txt"
grep "Moves.*Prefix Len.*Prefix Time" "$OUTPUT_DIR/test2_dynamic_shapes.log" -A 10 >> "$OUTPUT_DIR/summary.txt" || true
echo "" >> "$OUTPUT_DIR/summary.txt"

echo "=== Test 3: Policy Integration ===" >> "$OUTPUT_DIR/summary.txt"
grep -E "(Time:|Average time:|Selected:)" "$OUTPUT_DIR/test3_policy_integration.log" >> "$OUTPUT_DIR/summary.txt" || true

echo ""
cat "$OUTPUT_DIR/summary.txt"
