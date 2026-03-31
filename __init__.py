"""
TurboQuant: Fast LLM Quantization via Turbo Product Quantization

Reference: arXiv:2504.19874v1 [cs.LG] 28 Apr 2025
Authors: Amir Zandieh (Google Research), Majid Daliri (NYU),
         Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
"""

from .core import (
    TurboQuantConfig,
    TurboQuantMSE,
    TurboQuantProd,
    generate_rotation_matrix,
    compute_lloyd_max_centroids,
    get_centroids,
    precompute_all_centroids,
    qjl_quantize,
    qjl_dequantize,
    get_bits_info,
    get_compressed_size_bits,
    mse_theoretical_bound,
    inner_prod_theoretical_bound,
    run_demo,
)

__all__ = [
    "TurboQuantConfig",
    "TurboQuantMSE",
    "TurboQuantProd",
    "generate_rotation_matrix",
    "compute_lloyd_max_centroids",
    "get_centroids",
    "precompute_all_centroids",
    "qjl_quantize",
    "qjl_dequantize",
    "get_bits_info",
    "get_compressed_size_bits",
    "mse_theoretical_bound",
    "inner_prod_theoretical_bound",
    "run_demo",
]

__version__ = "0.1.0"
__author__ = "Kiald (ajul8866)"
__license__ = "MIT"
