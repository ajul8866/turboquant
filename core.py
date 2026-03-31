"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
=========================================================================

Reference: arXiv:2504.19874v1 [cs.LG] 28 Apr 2025
Authors: Amir Zandieh (Google Research), Majid Daliri (NYU),
         Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)

Implementation of:
  - Algorithm 1: TurboQuant_mse (MSE-optimized vector quantizer)
  - Algorithm 2: TurboQuant_prod (Inner-product optimized vector quantizer)
  - Quantized Johnson-Lindenstrauss (QJL) transform

Key idea: Randomly rotate input vectors so coordinates become nearly i.i.d.
with Beta (approx. N(0, 1/d)) distribution, then apply optimal Lloyd-Max
scalar quantization per coordinate. For inner products, add a 1-bit QJL
stage on the residual to produce an unbiased estimator.

Distortion bounds (Theorems 1 & 2):
  D_mse  <= sqrt(3*pi)/2 * 4^(-b)                        (MSE-optimal)
  D_prod <= sqrt(3*pi^2)/2 * ||y||^2/d * 4^(-b)          (Inner-product optimal)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm as stdnorm


# ============================================================================
# Configuration constants
# ============================================================================

@dataclass(frozen=True)
class TurboQuantConfig:
    """Supported precompute dimensions and bit-widths."""
    PRECOMPUTE_DIMENSIONS: Tuple[int, ...] = (128, 256, 512, 1024, 1536, 2048, 4096)
    PRECOMPUTE_BITS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


CONFIG = TurboQuantConfig()


# ============================================================================
# Bit-width validation helpers
# ============================================================================

VALID_MSE_BITS: set[int] = {1, 2, 3, 4, 5, 6, 7, 8}
VALID_PROD_BITS: set[int] = {2, 3, 4, 8}


def _validate_bits(b: int, quantizer_type: str) -> None:
    """Validate bit-width parameter. Raises ValueError if invalid."""
    valid = VALID_MSE_BITS if quantizer_type == "mse" else VALID_PROD_BITS
    if b not in valid:
        raise ValueError(
            f"Invalid bit-width b={b} for TurboQuant{quantizer_type}. "
            f"Valid values: {sorted(valid)}"
        )



# ============================================================================
# 1. Precompute scalar quantizer centroids (Lloyd-Max algorithm)
# ============================================================================

def compute_lloyd_max_centroids(
    d: int,
    b: int,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Compute optimal Lloyd-Max quantizer centroids for the marginal
    distribution of one coordinate of a randomly rotated unit vector in R^d.

    The coordinate distribution of a uniform vector on S^{d-1} is:
      f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    which converges to N(0, 1/d) as d grows large.

    Uses Gaussian N(0, sigma^2) with sigma = 1/sqrt(d) -- highly accurate
    for d >= 64.

    Lloyd-Max algorithm (continuous k-means in 1D):
      1. Initialize centroids from probability quantiles of N(0, sigma^2).
      2. Cell boundaries: (-inf, mid_0, mid_1, ..., mid_{K-2}, +inf).
      3. Centroid update: c_k = E[X | cell_k] (conditional mean).
      4. Repeat until convergence.

    Args:
        d: Dimension of input vectors.
        b: Bit-width per coordinate (2^b centroid levels).
        max_iter: Maximum iterations.
        tol: Convergence tolerance (max centroid shift).

    Returns:
        Sorted array of shape (2^b,) with optimal centroids.

    Examples:
        For large d, b=1 gives +/- sqrt(2/pi)/sqrt(d).
        b=2 gives approximately [-1.51, -0.453, 0.453, 1.51]/sqrt(d).
    """
    sigma = 1.0 / math.sqrt(d)
    n_levels = 1 << b  # 2^b

    # Initialize from uniform probability quantiles of N(0, sigma^2).
    quantiles = np.linspace(0.0, 1.0, n_levels + 2)[1:-1]
    centroids = sigma * stdnorm.ppf(quantiles)

    for iteration in range(max_iter):
        # Voronoi cell boundaries: midpoints, with outer boundaries at +/- inf.
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        boundaries = np.concatenate([[-np.inf], midpoints, [np.inf]])

        # Update each centroid to the conditional mean within its cell.
        new_centroids = np.empty(n_levels)
        for k in range(n_levels):
            a_std = boundaries[k] / sigma
            b_std = boundaries[k + 1] / sigma

            # Standard normal pdf/cdf at boundaries.
            # Handle +/- infinity: phi(+/- inf) = 0.
            phi_a = 0.0 if a_std == -np.inf else stdnorm.pdf(a_std)
            phi_b = 0.0 if b_std == np.inf else stdnorm.pdf(b_std)
            Phi_diff = stdnorm.cdf(b_std) - stdnorm.cdf(a_std)

            if Phi_diff > 1e-30:
                # E[X | a < X <= b] = sigma * (phi(a/s) - phi(b/s)) / Phi_diff
                # Note: phi at -inf = 0, phi at +inf = 0 per standard convention.
                new_centroids[k] = sigma * (phi_a - phi_b) / Phi_diff
            else:
                new_centroids[k] = centroids[k]  # frozen cell

        # Check convergence
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    return centroids


def _compute_lloyd_max_exact_beta(
    d: int,
    b: int,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Compute Lloyd-Max centroids using exact Beta distribution via numerical
    integration. Slower but accurate for small d (d < 64).

    Args:
        d: Dimension.
        b: Bit-width.

    Returns:
        Sorted centroids array of shape (2^b,).
    """
    from scipy.integrate import quad

    sigma = 1.0 / math.sqrt(d)
    n_levels = 1 << b

    # Init from Gaussian approx
    quantiles = np.linspace(0.0, 1.0, n_levels + 2)[1:-1]
    centroids = sigma * stdnorm.ppf(quantiles)

    # PDF normalization: C = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
    log_norm = (math.lgamma(d / 2.0)
                - 0.5 * math.log(math.pi)
                - math.lgamma((d - 1.0) / 2.0))
    norm_const = math.exp(log_norm)

    def pdf_beta(x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0
        return norm_const * (1.0 - x * x) ** ((d - 3.0) / 2.0)

    def pdf_beta_x(x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0
        return x * norm_const * (1.0 - x * x) ** ((d - 3.0) / 2.0)

    for _ in range(max_iter):
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        boundaries = np.concatenate([[-1.0], midpoints, [1.0]])

        new = np.empty(n_levels)
        for k in range(n_levels):
            a = max(boundaries[k], -1.0)
            bnd = min(boundaries[k + 1], 1.0)
            num, _ = quad(pdf_beta_x, a, bnd, limit=200)
            den, _ = quad(pdf_beta, a, bnd, limit=200)
            new[k] = num / den if abs(den) > 1e-30 else centroids[k]

        if np.max(np.abs(new - centroids)) < tol:
            break
        centroids = new

    return centroids


# ============================================================================
# Centroid cache
# ============================================================================

_centroid_cache: Dict[Tuple[int, int], np.ndarray] = {}


def get_centroids(d: int, b: int) -> np.ndarray:
    """Get precomputed or compute-on-demand centroids."""
    key = (d, b)
    if key not in _centroid_cache:
        _centroid_cache[key] = compute_lloyd_max_centroids(d, b)
    return _centroid_cache[key]


def precompute_all_centroids(config: TurboQuantConfig | None = None
                             ) -> Dict[Tuple[int, int], np.ndarray]:
    """Precompute and cache centroids for all standard configurations."""
    if config is None:
        config = CONFIG
    for d in config.PRECOMPUTE_DIMENSIONS:
        for b in config.PRECOMPUTE_BITS:
            get_centroids(d, b)
    print(f"[TurboQuant] Precomputed centroids for "
          f"{len(config.PRECOMPUTE_DIMENSIONS)} dims x "
          f"{len(config.PRECOMPUTE_BITS)} bits = "
          f"{len(_centroid_cache)} configurations cached.")
    return _centroid_cache.copy()


# ============================================================================
# 2. Random Rotation Matrix
# ============================================================================

def generate_rotation_matrix(d: int, seed: int | None = None) -> np.ndarray:
    """
    Generate random d x d orthogonal matrix via QR decomposition.

    1. Sample A with i.i.d. N(0, 1) entries.
    2. QR = A (thin QR decomposition).
    3. Enforce positive diagonal of R: Q = Q * diag(sign(diag(R))).

    Result is uniform over O(d), sufficient for TurboQuant.

    Args:
        d: Matrix dimension.
        seed: Optional random seed.

    Returns:
        Orthogonal matrix Pi of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs[np.newaxis, :]


# ============================================================================
# 3. QJL - Quantized Johnson-Lindenstrauss
# ============================================================================

def qjl_quantize(S: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    QJL quantizer: sign(S @ r) -> {-1, +1}^d.

    Args:
        S: Projection matrix (d, d) with i.i.d. N(0,1).
        r: Residual vector (d,).

    Returns:
        Sign vector qjl of shape (d,).
    """
    return np.sign(S @ r).astype(np.float64)


def qjl_dequantize(S: np.ndarray, qjl: np.ndarray,
                   gamma: float, d: int) -> np.ndarray:
    """
    QJL dequantizer: sqrt(pi/2)/d * gamma * S^T @ qjl.

    Unbiased: E[qjl_dequantize | r] = r.

    Args:
        S: Projection matrix (d, d).
        qjl: Sign vector (d,).
        gamma: Residual norm ||r||_2.
        d: Dimension.

    Returns:
        Reconstructed vector of shape (d,).
    """
    scale = math.sqrt(math.pi / 2.0) / d * gamma
    return scale * (S.T @ qjl)


# ============================================================================
# 4. TurboQuant_mse (Algorithm 1)
# ============================================================================

class TurboQuantMSE:
    """
    TurboQuant MSE-optimized vector quantizer (Algorithm 1).

    Pipeline:
      1. Rotate: y = Pi @ x
      2. Quantize each y[j] to nearest of 2^b Lloyd-Max centroids
      3. Reconstruct: x_tilde = Pi^T @ y_tilde

    For unit-norm x, D_mse = E[||x - x_tilde||^2] <= sqrt(3*pi)/2 * 4^(-b).
    Approximate actual values: b=1->0.36, b=2->0.117, b=3->0.03, b=4->0.009.

    Usage:
        >>> q = TurboQuantMSE(d=128, b=3, seed=42)
        >>> idx = q.quantize(x)          # b-bit indices per coord
        >>> x_tilde = q.dequantize(idx)   # reconstruction

    Args:
        d: Vector dimension.
        b: Bits per coordinate.
        seed: Optional random seed for rotation matrix.
        precompute_centroids: Use cached centroids (recommended).
    """

    def __init__(self, d: int, b: int, seed: int | None = None,
                 precompute_centroids: bool = True, internal: bool = False):
        if not internal:
            _validate_bits(b, "mse")
        self.d = d
        self.b = b
        self.n_levels = 1 << b

        self.Pi = generate_rotation_matrix(d, seed=seed)

        if precompute_centroids:
            self.centroids = get_centroids(d, b)
        else:
            self.centroids = compute_lloyd_max_centroids(d, b)
            _centroid_cache[(d, b)] = self.centroids

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize unit vector x. Returns int index array of shape (d,)."""
        y = self.Pi @ x
        # Nearest centroid: O(d * 2^b) via broadcasting
        return np.argmin(np.abs(y[:, np.newaxis] - self.centroids), axis=1)

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        """Reconstruct from indices. Returns float vector of shape (d,)."""
        return self.Pi.T @ self.centroids[idx]


    def quantize_batch(self, X: np.ndarray) -> np.ndarray:
        """Quantize batch of vectors. X: (num_vectors, d) -> indices: (num_vectors, d)."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if X.shape[1] != self.d:
            raise ValueError(f"X.shape[1]={X.shape[1]} != d={self.d}")
        # Rotate all vectors: Y = X @ Pi^T gives (N, d) since y = Pi @ x -> Y = (X @ Pi^T)
        # Actually: for each row x_i, y_i = Pi @ x_i -> Y_i = (Pi @ x_i) -> Y = X @ Pi^T
        Y = X @ self.Pi.T
        # Nearest centroid per coordinate: O(N * d * 2^b)
        # Y shape (N, d), centroids shape (K,) -> diff shape (N, d, K)
        diff = np.abs(Y[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
        return np.argmin(diff, axis=2).astype(np.int32)

    def dequantize_batch(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct batch from indices. indices: (num_vectors, d) -> X_tilde: (num_vectors, d)."""
        if indices.ndim != 2:
            raise ValueError(f"indices must be 2D array, got {indices.ndim}D")
        # Look up centroids: (N, d, K)[indices] -> (N, d)
        Y_quantized = self.centroids[indices]
        # Inverse rotation: x_tilde = Pi^T @ y_tilde -> X_tilde = Y_quantized @ Pi
        return Y_quantized @ self.Pi

    def __repr__(self) -> str:
        return f"TurboQuantMSE(d={self.d}, b={self.b}, levels={self.n_levels})"


# ============================================================================
# 5. TurboQuant_prod (Algorithm 2)
# ============================================================================

class TurboQuantProd:
    """
    TurboQuant inner-product optimized quantizer (Algorithm 2).

    Two stages:
      Stage 1 (MSE): TurboQuantMSE with b-1 bits
      Stage 2 (QJL): 1-bit Quantized JL on residual
      Total = b bits per coordinate.

    Properties (Theorem 2):
      E[<y, x_tilde>] = <y, x>        (unbiased)
      D_prod <= sqrt(3*pi^2)/2 * ||y||^2/d * 4^(-b)

    Usage:
        >>> q = TurboQuantProd(d=128, b=3, seed=42)
        >>> idx, qjl, gamma = q.quantize(x)
        >>> x_tilde = q.dequantize(idx, qjl, gamma)

    Args:
        d: Vector dimension.
        b: Total bits per coordinate (MSE gets b-1, QJL gets 1).
        seed: Optional random seed.
        precompute_centroids: Use cache.
    """

    def __init__(self, d: int, b: int, seed: int | None = None,
                 precompute_centroids: bool = True):
        _validate_bits(b, "prod")
        self.d = d
        self.b = b

        if b < 2:
            raise ValueError(
                f"TurboQuantProd requires b>=2 (got b={b}). "
                f"Needs at least 1 bit for MSE + 1 bit for QJL.")

        b_mse = b - 1
        self.mse_quantizer = TurboQuantMSE(
            d=d, b=b_mse, seed=seed,
            precompute_centroids=precompute_centroids,
            internal=True)

        s_seed = (seed + 1000) if seed is not None else None
        self.S = np.random.default_rng(s_seed).standard_normal((d, d))

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantize x. Returns (idx, qjl, gamma)."""
        idx = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(idx)
        r = x - x_mse
        gamma = float(np.linalg.norm(r))
        qjl = qjl_quantize(self.S, r)
        return idx, qjl, gamma

    def dequantize(self, idx: np.ndarray, qjl: np.ndarray,
                   gamma: float) -> np.ndarray:
        """Reconstruct from (idx, qjl, gamma)."""
        return (self.mse_quantizer.dequantize(idx)
                + qjl_dequantize(self.S, qjl, gamma, self.d))


    def quantize_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize batch of vectors. X: (N, d) -> (indices: (N, d), qjl: (N, d), gamma: (N,))."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        N = X.shape[0]
        if X.shape[1] != self.d:
            raise ValueError(f"X.shape[1]={X.shape[1]} != d={self.d}")

        # Stage 1: MSE quantization (batch)
        indices = self.mse_quantizer.quantize_batch(X)
        X_mse = self.mse_quantizer.dequantize_batch(indices)

        # Residuals
        R = X - X_mse  # (N, d)

        # Compute norms
        gamma = np.linalg.norm(R, axis=1)  # (N,)

        # QJL on residuals: for each row, compute sign(S @ r_i)
        QJL = np.sign(R @ self.S.T).astype(np.float64)  # (N, d)

        return indices, QJL, gamma

    def dequantize_batch(self, indices: np.ndarray, qjl: np.ndarray,
                         gamma: np.ndarray) -> np.ndarray:
        """Reconstruct batch from (indices, qjl, gamma) -> (N, d)."""
        if indices.ndim != 2:
            raise ValueError(f"indices must be 2D, got {indices.ndim}D")

        # MSE reconstruction
        X_mse = self.mse_quantizer.dequantize_batch(indices)

        # QJL reconstruction: for each row, sqrt(pi/2)/d * gamma_i * S^T @ qjl_i
        # qjl @ S gives (N, d) since each row: qjl_i^T @ S = (S^T @ qjl_i)^T
        scale = math.sqrt(math.pi / 2.0) / self.d
        X_qjl = (scale * gamma[:, np.newaxis]) * (qjl @ self.S)

        return X_mse + X_qjl

    def __repr__(self) -> str:
        return (f"TurboQuantProd(d={self.d}, b={self.b}, "
                f"mse_b={self.b - 1}, qjl_b=1)")


# ============================================================================
# 6. Utility Functions
# ============================================================================

# ============================================================================
# Additional Utility Functions
# ============================================================================

def get_original_dtype(bits: int) -> np.dtype:
    """Return the original dtype before quantization. Always float32."""
    return np.dtype('float32')


def get_compressed_size_bits(d: int, num_vectors: int, b: int) -> int:
    """
    Compute total bits needed to store num_vectors vectors after quantization.

    Each vector: d coordinates * b bits/coordinate = d * b bits.
    Also tracks storage overhead: ceil(bits / 8) bytes per vector.

    Args:
        d: Vector dimension.
        num_vectors: Number of vectors to store.
        b: Bit-width per coordinate.

    Returns:
        Total bits for index storage.
    """
    return d * b * num_vectors


def get_bits_info(b: int) -> dict:
    """
    Get information about a bit-width configuration.

    Args:
        b: Bit-width (1, 2, 3, 4, or 8).

    Returns:
        dict with keys:
            'name': human-readable name ('2bit', '4bit', '8bit')
            'compression_ratio': 32 / b (relative to float32)
            'max_val': max value representable (2^b - 1)
    """
    name_map = {1: '1bit', 2: '2bit', 3: '3bit', 4: '4bit', 8: '8bit'}
    name = name_map.get(b, f'{b}bit')
    return {
        'name': name,
        'compression_ratio': 32.0 / b,
        'max_val': (1 << b) - 1,
    }


# ============================================================================
# Quantization Theory Bounds (from arXiv:2504.19874v1)
# ============================================================================

def mse_theoretical_bound(b: int) -> float:
    """Upper bound: sqrt(3*pi)/2 * 4^(-b). Theorem 1, arXiv:2504.19874v1."""
    return math.sqrt(3.0 * math.pi) / 2.0 * (4.0 ** (-b))


def inner_prod_theoretical_bound(b: int, d: int,
                                 y_norm_sq: float = 1.0) -> float:
    """Upper bound: sqrt(3*pi^2)/2 * ||y||^2/d * 4^(-b). Theorem 2."""
    return (math.sqrt(3.0 * math.pi ** 2) / 2.0
            * y_norm_sq / d * (4.0 ** (-b)))


def run_demo() -> None:
    """Run demo and validation of TurboQuant algorithms."""
    print("=" * 72)
    print("  TurboQuant Demo & Validation")
    print("  Reference: arXiv:2504.19874v1")
    print("=" * 72)

    d = 128
    num_tests = 1000
    rng = np.random.default_rng(42)

    # Precompute centroids
    print("\nPrecomputing centroids for d=128, b=1..8...")
    for b_val in range(1, 9):
        get_centroids(d, b_val)
    print(f"Cached {len(_centroid_cache)} configurations (d=128).")

    # Show centroids
    print("\n  Lloyd-Max Centroids (d=128, Gaussian N(0, 1/d) approx):")
    for b_val in [1, 2, 3, 4]:
        c = get_centroids(d, b_val)
        if len(c) <= 8:
            s = np.array2string(c, precision=5, suppress_small=True)
        else:
            s = f"[{c[0]:.5f}, ..., {c[-1]:.5f}]"
        print(f"    b={b_val} ({len(c)} levels): {s}")

    # ---- Test 1: MSE ----
    print("\n" + "-" * 72)
    print("  TEST 1: TurboQuantMSE -- MSE Distortion")
    print("-" * 72)

    print(f"\n  {'b':>3} {'Measured MSE':>14} {'Bound sqrt(3pi)/2*4^(-b)':>26}")
    print(f"  {'-'*50}")
    for b_val in [1, 2, 3, 4]:
        q = TurboQuantMSE(d=d, b=b_val, seed=42)
        mses = []
        for _ in range(num_tests):
            x = rng.standard_normal(d); x /= np.linalg.norm(x)
            idx = q.quantize(x)
            x_tilde = q.dequantize(idx)
            mses.append(float(np.sum((x - x_tilde) ** 2)))
        avg = np.mean(mses)
        bound = mse_theoretical_bound(b_val)
        print(f"  {b_val:>3} {avg:>14.6f} {bound:>26.6f}")

    # ---- Test 2: Inner Product (TurboQuantProd) ----
    print("\n" + "-" * 72)
    print("  TEST 2: TurboQuantProd -- Inner-Product Distortion")
    print("-" * 72)

    for b_val in [2, 3, 4]:
        q = TurboQuantProd(d=d, b=b_val, seed=42)
        ip_errors = []
        biases = []
        for _ in range(num_tests):
            x = rng.standard_normal(d); x /= np.linalg.norm(x)
            y = rng.standard_normal(d); y /= np.linalg.norm(y)
            true_ip = float(np.dot(y, x))
            idx, qjl, gamma = q.quantize(x)
            x_tilde = q.dequantize(idx, qjl, gamma)
            approx_ip = float(np.dot(y, x_tilde))
            ip_errors.append((true_ip - approx_ip) ** 2)
            biases.append(approx_ip - true_ip)
        print(f"\n  b={b_val}:")
        print(f"    Mean IP distortion = {np.mean(ip_errors):.8f}")
        print(f"    Mean estimation bias = {np.mean(biases):.8f} (should ~0)")
        bound = inner_prod_theoretical_bound(b_val, d)
        print(f"    Theoretical bound  = {bound:.8f}")

    # ---- Test 3: Speed ----
    print("\n" + "-" * 72)
    print("  TEST 3: Speed Benchmark")
    print("-" * 72)

    import time
    for b_val in [2, 4, 8]:
        q_mse = TurboQuantMSE(d=d, b=b_val, seed=42)
        q_prod = TurboQuantProd(d=d, b=max(b_val, 2), seed=42)
        x = rng.standard_normal(d); x /= np.linalg.norm(x)

        for _ in range(100):
            i = q_mse.quantize(x); q_mse.dequantize(i)
            i2, q2, g2 = q_prod.quantize(x); q_prod.dequantize(i2, q2, g2)

        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            i = q_mse.quantize(x); q_mse.dequantize(i)
        t_mse = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N):
            i2, q2, g2 = q_prod.quantize(x); q_prod.dequantize(i2, q2, g2)
        t_prod = (time.perf_counter() - t0) / N * 1e6

        print(f"\n  b={b_val}, d={d}: MSE={t_mse:.2f} us, Prod={t_prod:.2f} us")

    print("\n" + "=" * 72)
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
