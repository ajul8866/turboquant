# TurboQuant: Fast LLM Quantization via Turbo Product Quantization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

TurboQuant implements **Turbo Product Quantization** — a fast, theoretically-grounded vector quantization method for compressing Large Language Model (LLM) weights with near-optimal distortion rates. Based on the paper:

> **Online Vector Quantization with Near-optimal Distortion Rate**  
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni  
> *arXiv:2504.19874v1* [cs.LG]

## Overview

TurboQuant randomly rotates input vectors so coordinates become nearly i.i.d., then applies optimal Lloyd-Max scalar quantization per coordinate. This achieves distortion bounds that are provably close to optimal.

### Key Features
- **MSE-optimized quantizer** (`TurboQuantMSE`) -- minimizes reconstruction error
- **Inner-product optimized quantizer** (`TurboQuantProd`) -- preserves dot-product similarity
- **Lloyd-Max centroid computation** with Gaussian and exact Beta distribution support
- **Quantized Johnson-Lindenstrauss (QJL)** for residual quantization
- **Hugging Face integration** -- quantize LLMs via CLI with `hf_quantizer.py`
- **Batch processing** for efficient inference
- **Bit-widths**: 1, 2, 3, 4, 8 bits per coordinate

### Theoretical Bounds
- MSE distortion: `D_mse <= sqrt(3*pi)/2 * 4^(-b)`
- Inner-product distortion: `D_prod <= sqrt(3*pi^2)/2 * ||y||^2/d * 4^(-b)`

## Installation

```bash
git clone https://github.com/ajul8866/turboquant.git
cd turboquant
pip install -r requirements.txt
```

## Usage

### Core API

```python
import numpy as np
from turboquant import TurboQuantMSE, TurboQuantProd

# MSE-optimized quantization
d, b = 128, 4
q = TurboQuantMSE(d=d, b=b, seed=42)
x = np.random.randn(d)
x /= np.linalg.norm(x)
idx = q.quantize(x)
x_tilde = q.dequantize(idx)
print(f"MSE: {np.sum((x - x_tilde)**2):.6f}")

# Inner-product optimized (TurboQuant_prod)
q_prod = TurboQuantProd(d=d, b=b, seed=42)
idx, qjl, gamma = q_prod.quantize(x)
x_tilde_prod = q_prod.dequantize(idx, qjl, gamma)
```

### Batch Processing

```python
X = np.random.randn(1000, 128)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
idx_batch = q.quantize_batch(X)
X_tilde_batch = q.dequantize_batch(idx_batch)
```

### Hugging Face Model Quantization

```bash
# Quantize a model
python hf_quantizer.py --model Jackrong/Qwen3.5-27B --bits 4 --output quantized_model
python hf_quantizer.py --model quantized_model --inference --prompt "Hello, how are you?"
```

## Project Structure

```
turboquant/
  turboquant.py          Core quantization algorithms (Lloyd-Max, MSE, Prod, QJL)
  hf_quantizer.py        Hugging Face model quantizer CLI
  requirements.txt       Dependencies (numpy, scipy, torch, transformers)
```

## Dependencies

- Python 3.10+
- numpy >= 1.24.0
- scipy >= 1.10.0
- torch >= 2.0.0
- transformers >= 4.36.0

## Citation

If you use TurboQuant in your research, please cite:

```bibtex
@article{zandieh2025online,
  title={Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

MIT
