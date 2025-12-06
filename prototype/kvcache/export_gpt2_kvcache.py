#!/usr/bin/env python3
"""
Export GPT-2 model with KV cache to ONNX format

This script exports a simple GPT-2 model from transformers with KV cache I/O,
suitable for validating the C++ KVCacheInferencer implementation.

Model characteristics:
- GPT-2 small (117M parameters)
- 12 layers, 12 attention heads, 768 hidden dim (64 per head)
- Max sequence length: 1024 (configurable)

Output:
- gpt2_with_cache.onnx: Model with KV cache inputs/outputs
- config.json: Model configuration for C++ inference
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import json
import os


class GPT2WithKVCache(nn.Module):
    """
    Wrapper around GPT2LMHeadModel that exposes KV cache I/O for ONNX export.

    This wrapper explicitly handles past_key_values inputs and outputs to ensure
    they are properly traced by ONNX export.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2LMHeadModel(config).transformer
        self.lm_head = GPT2LMHeadModel(config).lm_head

        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_0=None, past_value_0=None,
        past_key_1=None, past_value_1=None,
        past_key_2=None, past_value_2=None,
        past_key_3=None, past_value_3=None,
        past_key_4=None, past_value_4=None,
        past_key_5=None, past_value_5=None,
        past_key_6=None, past_value_6=None,
        past_key_7=None, past_value_7=None,
        past_key_8=None, past_value_8=None,
        past_key_9=None, past_value_9=None,
        past_key_10=None, past_value_10=None,
        past_key_11=None, past_value_11=None,
    ):
        """
        Forward pass with explicit KV cache parameters.

        Args:
            input_ids: [batch, seq_len] - Input token IDs
            position_ids: [batch, seq_len] - Position indices
            past_key_i: [batch, num_heads, past_seq_len, head_dim] - Previous key states
            past_value_i: [batch, num_heads, past_seq_len, head_dim] - Previous value states

        Returns:
            logits: [batch, seq_len, vocab_size]
            present_key_i: [batch, num_heads, total_seq_len, head_dim]
            present_value_i: [batch, num_heads, total_seq_len, head_dim]
        """

        # Reconstruct past_key_values tuple
        past_key_values = []
        past_keys = [
            past_key_0, past_key_1, past_key_2, past_key_3,
            past_key_4, past_key_5, past_key_6, past_key_7,
            past_key_8, past_key_9, past_key_10, past_key_11
        ]
        past_values = [
            past_value_0, past_value_1, past_value_2, past_value_3,
            past_value_4, past_value_5, past_value_6, past_value_7,
            past_value_8, past_value_9, past_value_10, past_value_11
        ]

        for key, value in zip(past_keys, past_values):
            if key is not None and value is not None:
                past_key_values.append((key, value))
            else:
                past_key_values.append(None)

        # Convert to tuple or None
        past_key_values = tuple(past_key_values) if any(p is not None for p in past_key_values) else None

        # Forward through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )

        # Get hidden states and present key values
        hidden_states = outputs.last_hidden_state
        present_key_values = outputs.past_key_values

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Unpack present_key_values into separate tensors for ONNX
        present_outputs = [logits]
        for key, value in present_key_values:
            present_outputs.append(key)
            present_outputs.append(value)

        return tuple(present_outputs)


def export_gpt2_with_kvcache(
    output_dir="./models",
    model_name="gpt2",
    max_seq_len=1024,
    opset_version=14
):
    """
    Export GPT-2 model with KV cache to ONNX.

    Args:
        output_dir: Directory to save the ONNX model
        model_name: Hugging Face model name (default: "gpt2")
        max_seq_len: Maximum sequence length for cache allocation
        opset_version: ONNX opset version (14 recommended for better operator support)
    """

    print(f"Loading {model_name} model...")
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2WithKVCache(config)
    model.eval()

    # Extract configuration
    num_layers = config.n_layer
    num_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    vocab_size = config.vocab_size

    print(f"Model configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create dummy inputs for ONNX export
    batch_size = 1
    seq_len = 1  # Single token input (typical for generation)
    past_seq_len = 10  # Example past sequence length

    dummy_inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "position_ids": torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long).unsqueeze(0),
    }

    # Add dummy past key/values
    for i in range(num_layers):
        dummy_inputs[f"past_key_{i}"] = torch.randn(batch_size, num_heads, past_seq_len, head_dim)
        dummy_inputs[f"past_value_{i}"] = torch.randn(batch_size, num_heads, past_seq_len, head_dim)

    # Define input names
    input_names = ["input_ids", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")

    # Define output names
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")

    # Define dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }

    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq_len"}

    # Export to ONNX
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "gpt2_with_cache.onnx")

    print(f"\nExporting to ONNX: {onnx_path}")
    print(f"  Input names: {len(input_names)} ({input_names[:3]}...)")
    print(f"  Output names: {len(output_names)} ({output_names[:3]}...)")

    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )

    print(f"✓ Model exported successfully")

    # Save configuration for C++ inference
    config_path = os.path.join(output_dir, "config.json")
    config_data = {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": config.n_embd,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "onnx_path": "gpt2_with_cache.onnx"
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"✓ Configuration saved: {config_path}")

    # Verify the exported model
    print("\nVerifying exported model...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    print("\nModel inputs:")
    for input_tensor in onnx_model.graph.input[:5]:  # Show first 5
        print(f"  {input_tensor.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]}")

    print("\nModel outputs:")
    for output_tensor in onnx_model.graph.output[:5]:  # Show first 5
        print(f"  {output_tensor.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]}")

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nModel size: {file_size_mb:.2f} MB")

    return onnx_path, config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export GPT-2 with KV cache to ONNX")
    parser.add_argument("--output-dir", default="./models", help="Output directory")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face model name")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()

    print("=" * 70)
    print("GPT-2 KV Cache ONNX Export")
    print("=" * 70)

    onnx_path, config_path = export_gpt2_with_kvcache(
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        opset_version=args.opset_version
    )

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Build the C++ test: cd .. && mkdir -p build && cd build && cmake .. && make")
    print(f"2. Run benchmark: ./test_kvcache_prototype")
    print(f"3. Check results in KVCACHE_BENCHMARK.md")
