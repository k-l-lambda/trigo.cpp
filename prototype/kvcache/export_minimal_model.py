"""
Export minimal transformer model with KV cache (no LayerNorm)
This version uses only basic operations that work with older ONNX opsets
"""

import torch
import torch.nn as nn
import os
import json


class MinimalTransformerLayer(nn.Module):
	"""
	Minimal transformer layer with KV cache, no LayerNorm
	"""

	def __init__(self, hidden_dim, num_heads):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.head_dim = hidden_dim // num_heads

		# Attention projections
		self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
		self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

		# FFN
		self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
		self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

	def forward(self, x, past_key=None, past_value=None):
		batch_size, seq_len, _ = x.shape

		# QKV projection
		qkv = self.qkv_proj(x)
		q, k, v = torch.chunk(qkv, 3, dim=-1)

		# Reshape for multi-head attention
		q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		# Concat past K, V
		if past_key is not None:
			k = torch.cat([past_key, k], dim=2)
			v = torch.cat([past_value, v], dim=2)

		# Attention
		attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
		attn = torch.softmax(attn, dim=-1)
		out = torch.matmul(attn, v)

		# Reshape back
		out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
		out = self.out_proj(out)

		# Residual connection
		x = x + out

		# FFN
		ffn_out = self.fc2(torch.relu(self.fc1(x)))
		x = x + ffn_out

		return x, k, v


class MinimalTransformerWithKVCache(nn.Module):
	"""
	Minimal transformer model with KV cache
	"""

	def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len):
		super().__init__()
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.max_seq_len = max_seq_len

		self.embedding = nn.Embedding(vocab_size, hidden_dim)
		self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

		self.layers = nn.ModuleList([
			MinimalTransformerLayer(hidden_dim, num_heads)
			for _ in range(num_layers)
		])

		self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

	def forward(
		self,
		input_ids,
		position_ids,
		past_key_0=None, past_value_0=None,
		past_key_1=None, past_value_1=None,
		past_key_2=None, past_value_2=None,
		past_key_3=None, past_value_3=None,
	):
		# Gather past key/values
		past_kv = [
			(past_key_0, past_value_0),
			(past_key_1, past_value_1),
			(past_key_2, past_value_2),
			(past_key_3, past_value_3),
		]

		# Embeddings
		x = self.embedding(input_ids) + self.pos_embedding(position_ids)

		# Process through layers
		present_keys = []
		present_values = []

		for i, layer in enumerate(self.layers):
			past_key, past_value = past_kv[i]
			x, present_key, present_value = layer(x, past_key, past_value)
			present_keys.append(present_key)
			present_values.append(present_value)

		# Output
		logits = self.lm_head(x)

		# Return logits and all present key/values
		outputs = [logits]
		for key, value in zip(present_keys, present_values):
			outputs.extend([key, value])

		return tuple(outputs)


def export_model(
	output_dir="models",
	vocab_size=1000,
	hidden_dim=256,
	num_layers=4,
	num_heads=4,
	max_seq_len=512,
	opset_version=18  # Use PyTorch's native opset to avoid conversion issues
):
	"""
	Export minimal transformer with KV cache (compatible with ONNX Runtime 1.17)
	"""

	print("=" * 70)
	print("Minimal Transformer KV Cache ONNX Export")
	print("=" * 70)

	print(f"Creating minimal transformer model...")
	model = MinimalTransformerWithKVCache(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		num_heads=num_heads,
		max_seq_len=max_seq_len
	)
	model.eval()

	print(f"Model configuration:")
	print(f"  Layers: {num_layers}")
	print(f"  Heads: {num_heads}")
	print(f"  Head dimension: {hidden_dim // num_heads}")
	print(f"  Vocabulary size: {vocab_size}")
	print(f"  Max sequence length: {max_seq_len}")

	# Create output directory
	os.makedirs(output_dir, exist_ok=True)

	# Dummy inputs
	batch_size = 1
	seq_len = 1
	past_seq_len = 10

	dummy_inputs = (
		torch.randint(0, vocab_size, (batch_size, seq_len)),  # input_ids
		torch.arange(past_seq_len, past_seq_len + seq_len).unsqueeze(0),  # position_ids
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_key_0
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_value_0
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_key_1
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_value_1
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_key_2
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_value_2
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_key_3
		torch.randn(batch_size, num_heads, past_seq_len, hidden_dim // num_heads),  # past_value_3
	)

	# Input/output names
	input_names = ["input_ids", "position_ids"]
	output_names = ["logits"]

	for i in range(num_layers):
		input_names.extend([f"past_key_{i}", f"past_value_{i}"])
		output_names.extend([f"present_key_{i}", f"present_value_{i}"])

	# Dynamic axes
	dynamic_axes = {
		"input_ids": {0: "batch", 1: "seq"},
		"position_ids": {0: "batch", 1: "seq"},
		"logits": {0: "batch", 1: "seq"},
	}

	for i in range(num_layers):
		dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq"}
		dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq"}
		dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq"}
		dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq"}

	# Export
	onnx_path = os.path.join(output_dir, "minimal_model_with_cache.onnx")
	print(f"\nExporting to ONNX: {onnx_path}")
	print(f"Using opset version: {opset_version}")

	torch.onnx.export(
		model,
		dummy_inputs,
		onnx_path,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		opset_version=opset_version,
		do_constant_folding=False,  # Disable to avoid optimization errors
		export_params=True,
	)

	print(f"✓ Model exported successfully")

	# Save config
	config_path = os.path.join(output_dir, "config.json")
	config_data = {
		"model_name": "minimal_transformer",
		"num_layers": num_layers,
		"num_heads": num_heads,
		"head_dim": hidden_dim // num_heads,
		"hidden_dim": hidden_dim,
		"vocab_size": vocab_size,
		"max_seq_len": max_seq_len,
		"onnx_path": "minimal_model_with_cache.onnx"
	}

	with open(config_path, "w") as f:
		json.dump(config_data, f, indent=2)

	print(f"✓ Configuration saved: {config_path}")

	# Verify
	print("\nVerifying exported model...")
	import onnx
	onnx_model = onnx.load(onnx_path)
	onnx.checker.check_model(onnx_model)
	print("✓ ONNX model is valid")

	file_size = os.path.getsize(onnx_path) / (1024 * 1024)
	print(f"\nModel size: {file_size:.2f} MB")

	print("\n" + "=" * 70)
	print("Export complete!")
	print("=" * 70)
	print("\nNext steps:")
	print("1. Build C++ test: mkdir -p build && cd build && cmake -DUSE_CUDA=ON .. && make")
	print("2. Run benchmark: ./test_kvcache_prototype")


if __name__ == "__main__":
	export_model()
