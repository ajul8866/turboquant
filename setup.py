#!/usr/bin/env python3
"""
TurboQuant: Fast LLM Quantization via Turbo Product Quantization

Installation:
    pip install -e .

Usage:
    from turboquant import TurboQuantMSE, TurboQuantProd
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="turboquant",
    version="0.1.0",
    author="Kiald (ajul8866)",
    author_email="bizgudboy@gmail.com",
    description="Fast LLM Quantization via Turbo Product Quantization (arXiv:2504.19874)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajul8866/turboquant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "turboquant-quantize=turboquant.hf_quantizer:main",
        ],
    },
    license="MIT",
    keywords="quantization llm compression turbo product quantization lloyd-max",
)
