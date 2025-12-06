#!/usr/bin/env python3
"""
Export simpler model with KV cache for prototype validation

This uses a simpler custom model instead of full GPT-2 to avoid export issues.
Implements the core KV cache mechanism for testing IOBinding.
"""

import torch
import torch.nn as nn
import json
import os


class SimpleTransformerWithKVCache(nn.Module):
    """
    Simplified transformer model with KV cache for prototype testing.

    This avoids complex GPT-2 export issues while maintaining all KV cache
    features needed for validation.
    """

    def __init__(self, vocab_size=1000, hidden_dim=256, num_layers=4, num_heads=4, max_seq_len=512):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        # Simple embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Simplified transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

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
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Return logits and all present key/values
        outputs = [logits]
        for key, value in zip(present_keys, present_values):
            outputs.extend([key, value])

        return tuple(outputs)


class SimpleTransformerLayer(nn.Module):
    """
    Simplified transformer layer with KV cache support
    """

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        # Attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # FFN
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, past_key=None, past_value=None):
        batch_size, seq_len, _ = x.shape

        # Self-attention with KV cache
        residual = x
        x = self.ln1(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate with past if available
        if past_key is not None and past_value is not None:
            k = torch.cat([past_key, k], dim=2)  # [batch, heads, past_len+seq_len, head_dim]
            v = torch.cat([past_value, v], dim=2)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)

        x = residual + attn_output

        # FFN
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)

        # Return output and present key/value for cache
        return x, k, v


def export_simple_model(
    output_dir="./models",
    vocab_size=1000,
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    max_seq_len=512,
    opset_version=14
):
    """
    Export simplified transformer with KV cache
    """

    print(f"Creating simple transformer model...")
    model = SimpleTransformerWithKVCache(
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
    for i in range(num_layers):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])

    output_names = ["logits"]
    for i in range(num_layers):
        output_names.extend([f"present_key_{i}", f"present_value_{i}"])

    # Dynamic axes
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

    # Export
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "simple_model_with_cache.onnx")

    print(f"\nExporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✓ Model exported successfully")

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    config_data = {
        "model_name": "simple_transformer",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": hidden_dim // num_heads,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "onnx_path": "simple_model_with_cache.onnx"
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

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nModel size: {file_size_mb:.2f} MB")

    return onnx_path, config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)

    args = parser.parse_args()

    print("=" * 70)
    print("Simple Transformer KV Cache ONNX Export")
    print("=" * 70)

    onnx_path, config_path = export_simple_model(
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len
    )

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Build C++ test: mkdir -p build && cd build && cmake -DUSE_CUDA=ON .. && make")
    print(f"2. Run benchmark: ./test_kvcache_prototype")
