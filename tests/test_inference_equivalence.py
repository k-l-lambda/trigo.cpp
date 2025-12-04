#!/usr/bin/env python3
"""
Cross-language inference equivalence test.

This test verifies that C++ ONNX Runtime inference produces identical results
to Python ONNX Runtime inference for both policy and value predictions.
"""

import sys
import subprocess
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def create_test_inputs(batch_size=1, prefix_len=128, eval_len=64, seq_len=191, seed=42):
	"""Create reproducible test inputs."""
	np.random.seed(seed)

	# Generate random token IDs in valid ASCII range
	prefix_ids = np.random.randint(32, 127, size=(batch_size, prefix_len), dtype=np.int64)
	evaluated_ids = np.random.randint(32, 127, size=(batch_size, eval_len), dtype=np.int64)

	# Create causal mask for evaluated region (lower triangular)
	evaluated_mask = np.tril(np.ones((eval_len, eval_len), dtype=np.float32))
	evaluated_mask = np.expand_dims(evaluated_mask, 0).repeat(batch_size, axis=0)

	# Input for value inference
	input_ids = np.random.randint(32, 127, size=(batch_size, seq_len), dtype=np.int64)

	return {
		'prefix_ids': prefix_ids,
		'evaluated_ids': evaluated_ids,
		'evaluated_mask': evaluated_mask,
		'input_ids': input_ids,
	}


def run_python_policy_inference(base_path, policy_path, inputs):
	"""Run policy inference using Python ONNX Runtime."""
	base_session = ort.InferenceSession(base_path, providers=['CPUExecutionProvider'])
	policy_session = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])

	# Run base model
	base_outputs = base_session.run(None, {
		'prefix_ids': inputs['prefix_ids'],
		'evaluated_ids': inputs['evaluated_ids'],
		'evaluated_mask': inputs['evaluated_mask']
	})
	hidden_states = base_outputs[0]  # [batch, n+m, hidden_dim]

	# Run policy head
	policy_outputs = policy_session.run(None, {'hidden_states': hidden_states})
	logits = policy_outputs[0]  # [batch, m+1, vocab_size]

	return logits


def run_python_value_inference(base_path, value_path, inputs, value_token_id=3):
	"""Run value inference using Python ONNX Runtime."""
	base_session = ort.InferenceSession(base_path, providers=['CPUExecutionProvider'])
	value_session = ort.InferenceSession(value_path, providers=['CPUExecutionProvider'])

	batch_size, seq_len = inputs['input_ids'].shape

	# Append VALUE token
	value_token = np.full((batch_size, 1), value_token_id, dtype=np.int64)
	input_ids_with_value = np.concatenate([inputs['input_ids'], value_token], axis=1)

	# Split into prefix and evaluated
	prefix_len = 128
	eval_len = input_ids_with_value.shape[1] - prefix_len

	prefix_ids = input_ids_with_value[:, :prefix_len]
	evaluated_ids = input_ids_with_value[:, prefix_len:]

	# Create causal mask
	evaluated_mask = np.tril(np.ones((eval_len, eval_len), dtype=np.float32))
	evaluated_mask = np.expand_dims(evaluated_mask, 0).repeat(batch_size, axis=0)

	# Run base model
	base_outputs = base_session.run(None, {
		'prefix_ids': prefix_ids,
		'evaluated_ids': evaluated_ids,
		'evaluated_mask': evaluated_mask
	})
	hidden_states = base_outputs[0]  # [batch, n+m, hidden_dim]

	# Extract last position
	value_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]

	# Run value head
	value_outputs = value_session.run(None, {'hidden_states': value_hidden})
	values = value_outputs[0]  # [batch]

	return values


def run_cpp_inference(test_data_path):
	"""Run C++ inference test that loads saved test data."""
	cpp_build_dir = Path(__file__).parent.parent / "build"
	cpp_test = cpp_build_dir / "test_inference_equivalence_cpp"

	if not cpp_test.exists():
		raise FileNotFoundError(f"C++ test not found: {cpp_test}")

	result = subprocess.run(
		[str(cpp_test), test_data_path],
		capture_output=True,
		text=True
	)

	if result.returncode != 0:
		print("C++ test stderr:", result.stderr)
		print("C++ test stdout:", result.stdout)
		raise RuntimeError(f"C++ test failed with code {result.returncode}")

	# Parse output to extract results
	lines = result.stdout.strip().split('\n')
	cpp_results = {}
	for line in lines:
		if line.startswith("POLICY_LOGITS:"):
			cpp_results['policy_logits'] = np.array(
				[float(x) for x in line.split(":")[1].strip().split()]
			)
		elif line.startswith("VALUE:"):
			cpp_results['value'] = float(line.split(":")[1].strip())

	return cpp_results


def compare_arrays(name, python_arr, cpp_arr, rtol=1e-5, atol=1e-5):
	"""Compare two arrays and print statistics."""
	if python_arr.shape != cpp_arr.shape:
		print(f"❌ {name}: Shape mismatch!")
		print(f"  Python: {python_arr.shape}")
		print(f"  C++: {cpp_arr.shape}")
		return False

	abs_diff = np.abs(python_arr - cpp_arr)
	rel_diff = abs_diff / (np.abs(python_arr) + 1e-10)

	max_abs_diff = np.max(abs_diff)
	max_rel_diff = np.max(rel_diff)
	mean_abs_diff = np.mean(abs_diff)

	print(f"\n{name} Comparison:")
	print(f"  Shape: {python_arr.shape}")
	print(f"  Max absolute difference: {max_abs_diff:.6e}")
	print(f"  Max relative difference: {max_rel_diff:.6e}")
	print(f"  Mean absolute difference: {mean_abs_diff:.6e}")

	# Sample values
	print(f"  Python sample (first 5): {python_arr.flat[:5]}")
	print(f"  C++ sample (first 5):    {cpp_arr.flat[:5]}")

	if np.allclose(python_arr, cpp_arr, rtol=rtol, atol=atol):
		print(f"  ✅ PASS (within tolerance: rtol={rtol}, atol={atol})")
		return True
	else:
		print(f"  ❌ FAIL (exceeds tolerance)")
		# Find worst offenders
		worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
		print(f"  Worst diff at index {worst_idx}:")
		print(f"    Python: {python_arr[worst_idx]}")
		print(f"    C++: {cpp_arr[worst_idx]}")
		print(f"    Diff: {abs_diff[worst_idx]}")
		return False


def main():
	print("Cross-Language Inference Equivalence Test")
	print("=" * 70)
	print("Comparing Python ONNX Runtime vs C++ ONNX Runtime\n")

	# Model paths
	base_path = str(Path(__file__).parent.parent / "models/test_export_shared/base_model.onnx")
	policy_path = str(Path(__file__).parent.parent / "models/test_export_shared/policy_head.onnx")
	value_path = str(Path(__file__).parent.parent / "models/test_export_shared/value_head.onnx")

	# Create test inputs
	print("Creating test inputs...")
	inputs = create_test_inputs(batch_size=1, prefix_len=128, eval_len=64, seq_len=191, seed=42)
	print(f"  prefix_ids: {inputs['prefix_ids'].shape}")
	print(f"  evaluated_ids: {inputs['evaluated_ids'].shape}")
	print(f"  evaluated_mask: {inputs['evaluated_mask'].shape}")
	print(f"  input_ids: {inputs['input_ids'].shape}")

	# Save test inputs to temporary file for C++ to load
	with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
		test_data_path = f.name
		json.dump({
			'prefix_ids': inputs['prefix_ids'].tolist(),
			'evaluated_ids': inputs['evaluated_ids'].tolist(),
			'evaluated_mask': inputs['evaluated_mask'].tolist(),
			'input_ids': inputs['input_ids'].tolist(),
			'base_path': base_path,
			'policy_path': policy_path,
			'value_path': value_path,
		}, f)

	try:
		# Run Python inference
		print("\n[Python ONNX Runtime]")
		print("Running policy inference...")
		python_policy_logits = run_python_policy_inference(base_path, policy_path, inputs)
		print(f"  Output shape: {python_policy_logits.shape}")
		print(f"  Sample logits: {python_policy_logits.flat[:5]}")

		print("\nRunning value inference...")
		python_values = run_python_value_inference(base_path, value_path, inputs)
		print(f"  Output shape: {python_values.shape}")
		print(f"  Value: {python_values.item() if python_values.ndim == 0 else python_values[0]}")

		# Run C++ inference
		print("\n[C++ ONNX Runtime]")
		print("Running C++ inference test...")
		cpp_results = run_cpp_inference(test_data_path)

		# Compare results
		print("\n" + "=" * 70)
		print("NUMERICAL EQUIVALENCE VALIDATION")
		print("=" * 70)

		# Policy logits comparison
		policy_pass = compare_arrays(
			"Policy Logits",
			python_policy_logits.flatten(),
			cpp_results['policy_logits'],
			rtol=1e-5,
			atol=1e-5
		)

		# Value comparison
		python_value_arr = np.array([python_values.item() if python_values.ndim == 0 else python_values[0]])
		value_pass = compare_arrays(
			"Value Prediction",
			python_value_arr,
			np.array([cpp_results['value']]),
			rtol=1e-5,
			atol=1e-5
		)

		# Final result
		print("\n" + "=" * 70)
		if policy_pass and value_pass:
			print("🎉 ALL TESTS PASSED!")
			print("C++ and Python ONNX Runtime produce identical results.")
			print("=" * 70)
			return 0
		else:
			print("❌ TESTS FAILED!")
			print("C++ and Python ONNX Runtime results differ.")
			print("=" * 70)
			return 1

	finally:
		# Cleanup
		Path(test_data_path).unlink(missing_ok=True)


if __name__ == "__main__":
	sys.exit(main())
