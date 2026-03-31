#!/usr/bin/env python3
"""
Hugging Face Model Quantizer using TurboQuant.

Quantizes LLM weight matrices (Q, K, V, O projections, gate/up/down MLP layers)
using TurboQuant per-vector quantization. Replaces nn.Linear modules with
QuantizedLinear wrappers that dequantize on-the-fly during forward pass.

Usage:
    # Quantize model
    python hf_quantizer.py --model Jackrong/Qwen3.5-27B --bits 4 --output quantized_qwen35_4bit
    
    # Inference with quantized model
    python hf_quantizer.py --model quantized_qwen35_4bit --inference --prompt "Hello, how are you?"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import turboquant
from turboquant import TurboQuantMSE, TurboQuantProd, get_bits_info, get_bits_info


class TurboQuantConfig:
    """Konfigurasi quantization untuk model HF."""
    
    def __init__(
        self,
        bits: int = 4,
        mode: str = "mse",  # "mse" atau "prod"
        quantize_modules: list[str] = None,
        skip_modules: list[str] = None,
        layer_bits: dict = None,  # {"model.layers.0": 2, "model.layers.5": 4, ...}
    ):
        self.bits = bits
        self.mode = mode
        self.quantize_modules = quantize_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                      "gate_proj", "up_proj", "down_proj"]
        self.skip_modules = skip_modules or ["lm_head", "embed_tokens"]
        self.layer_bits = layer_bits or {}
        
    def get_bits_for_layer(self, layer_name: str) -> int:
        """Dapatkan bit-width untuk layer tertentu, support per-layer."""
        for prefix, bits in self.layer_bits.items():
            if prefix in layer_name:
                return bits
        return self.bits


class QuantizedLinear(torch.nn.Module):
    """Linear layer wrapper yang menyimpan weight dalam format ter-quantize."""
    
    def __init__(self, original_linear: torch.nn.Linear, config: TurboQuantConfig, layer_idx: int = None):
        super().__init__()
        self.original_shape = original_linear.weight.shape
        self.config = config
        self.bias = original_linear.bias
        
        # Dapatkan bit-width untuk layer ini
        bits = config.get_bits_for_layer(f"layer_{layer_idx}" if layer_idx is not None else "")
        
        # Flatten weight matrix ke vectors
        weight = original_linear.weight.data.cpu().numpy()
        out_features, in_features = weight.shape
        
        # Reshape: setiap row adalah vektor yang akan di-quantize
        d = in_features
        num_vectors = out_features
        weight_flat = weight  # (out_features, in_features)
        
        # Buat quantizer
        quantizer_cls = TurboQuantMSE if config.mode == "mse" else TurboQuantProd
        self.quantizer = quantizer_cls(d=d, b=bits)
        self.bits = bits
        self.d = d
        self.num_vectors = num_vectors
        
        # Quantize (satu per satu karena weight shapes berbeda)
        indices_list = []
        qjl_list = []
        gamma_list = []
        
        for i in range(num_vectors):
            vec = weight_flat[i]
            if config.mode == "mse":
                idx = self.quantizer.quantize(vec)
            else:
                idx, qjl, gamma = self.quantizer.quantize(vec)
                qjl_list.append(qjl)
                gamma_list.append(gamma)
            indices_list.append(idx)
        
        # Simpan sebagai tensor
        self.register_buffer("indices", torch.tensor(np.array(indices_list), dtype=torch.int32))
        
        if config.mode == "prod":
            self.register_buffer("qjl", torch.tensor(np.array(qjl_list), dtype=torch.int8))
            self.register_buffer("gamma", torch.tensor(np.array(gamma_list), dtype=torch.float32))
        
        # Rotation matrix (shared atau per-vector)
        self.register_buffer("rotation_matrix", torch.tensor(self.quantizer.R, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on-the-fly
        if self.config.mode == "mse":
            weight = self.quantizer.dequantize(self.indices.numpy())
        else:
            weight = self.quantizer.dequantize(
                self.indices.numpy(),
                self.qjl.numpy(),
                self.gamma.numpy()
            )
        
        weight_tensor = torch.tensor(weight, dtype=x.dtype, device=x.device)
        return torch.nn.functional.linear(x, weight_tensor, self.bias)
    
    def get_compressed_size_bytes(self) -> int:
        """Hitung ukuran storage untuk quantized weight."""
        # indices: int32 per element, tapi dalam implementasi nyata bisa packed
        if self.config.mode == "mse":
            return self.num_vectors * self.d * (self.bits // 8) if self.bits >= 8 else self.num_vectors * self.d // (8 // self.bits)
        else:
            qjl_size = self.num_vectors * self.d // 8  # 1-bit packed
            gamma_size = self.num_vectors * 4  # float32
            return qjl_size + gamma_size


def should_quantize_module(module_name: str, config: TurboQuantConfig) -> bool:
    """Tentukan apakah module harus di-quantize."""
    # Skip modules tertentu
    for skip in config.skip_modules:
        if skip in module_name:
            return False
    # Quantize hanya modules target
    for target in config.quantize_modules:
        if target in module_name:
            return True
    return False


def quantize_model(model: torch.nn.Module, config: TurboQuantConfig, verbose: bool = True) -> dict:
    """Quantize model in-place, return statistics."""
    stats = {
        "original_size_bytes": 0,
        "quantized_size_bytes": 0,
        "quantized_modules": 0,
        "skipped_modules": 0,
        "layers": {},
    }
    
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        
        if not should_quantize_module(name, config):
            stats["skipped_modules"] += 1
            original_bytes = module.weight.numel() * 4  # float32
            stats["original_size_bytes"] += original_bytes
            stats["quantized_size_bytes"] += original_bytes  # skip = tidak compressed
            continue
        
        # Quantize module ini
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break
        
        # Dapatkan bit-width untuk layer ini
        bits = config.get_bits_for_layer(name)
        
        original_bytes = module.weight.numel() * 4
        stats["original_size_bytes"] += original_bytes
        
        # Get parent module dan attribute name
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        
        # Buat quantized wrapper
        quantized_layer = QuantizedLinear(module, config, layer_idx)
        quantized_bytes = quantized_layer.get_compressed_size_bytes()
        
        # Replace module
        setattr(parent, attr_name, quantized_layer)
        
        stats["quantized_modules"] += 1
        stats["quantized_size_bytes"] += quantized_bytes
        stats["layers"][name] = {
            "original_bytes": original_bytes,
            "quantized_bytes": quantized_bytes,
            "bits": bits,
            "compression_ratio": original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0,
        }
    
    if verbose:
        print(f"\n=== Quantization Statistics ===")
        orig_mb = stats["original_size_bytes"] / (1024**2)
        quant_mb = stats["quantized_size_bytes"] / (1024**2)
        print(f"Original size: {orig_mb:.2f} MB")
        print(f"Quantized size: {quant_mb:.2f} MB")
        print(f"Compression ratio: {orig_mb / quant_mb:.1f}x")
        print(f"Quantized modules: {stats['quantized_modules']}")
        print(f"Skipped modules: {stats['skipped_modules']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Hugging Face Model Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Model name/ID atau path ke model ter-quantize")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4, 8], help="Bit-width")
    parser.add_argument("--mode", type=str, default="mse", choices=["mse", "prod"])
    parser.add_argument("--output", type=str, default="quantized_model", help="Output directory")
    parser.add_argument("--inference", action="store_true", help="Jalankan inference dengan model ter-quantize")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt untuk inference")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--layer-bits", type=str, default=None,
                       help='JSON dict: {"layer_0-3": 2, "layer_4-8": 4, "layer_9+": 8}')
    
    args = parser.parse_args()
    
    # Parse layer bits
    layer_bits = {}
    if args.layer_bits:
        layer_bits = json.loads(args.layer_bits)
    
    # Buat config
    config = TurboQuantConfig(bits=args.bits, mode=args.mode, layer_bits=layer_bits)
    
    # Load model
    print(f"Loading model: {args.model}")
    print(f"Bit-width: {args.bits}, Mode: {args.mode}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )
    
    # Quantize
    print("\nQuantizing model...")
    start = time.time()
    stats = quantize_model(model, config)
    quant_time = time.time() - start
    print(f"Quantization time: {quant_time:.1f}s")
    
    # Simpan model
    if not args.inference:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simpan quantized state
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        
        # Simpan config dan stats
        with open(output_dir / "quantization_config.json", "w") as f:
            json.dump({
                "bits": args.bits,
                "mode": args.mode,
                "stats": stats,
                "quantization_time": quant_time,
            }, f, indent=2)
        
        print(f"\nSaved to: {output_dir}")
    
    # Inference (opsional)
    if args.inference:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"\n=== Inference ===")
        print(f"Prompt: {args.prompt}")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nOutput:\n{output_text}")


if __name__ == "__main__":
    main()

