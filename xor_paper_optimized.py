#!/usr/bin/env python3
"""
Optimized XOR implementation based on academic papers:
- Han & Ki (ASIACRYPT '22): Nibble extraction with conjugate reduction
- Cheon et al. (SPC '20): Even polynomial optimization
- Baby-step/giant-step for polynomial evaluation

## Optimization Comparison

| Optimization Item | Paper Proposal | Current Implementation | Evaluation |
|------------------|----------------|----------------------|------------|
| **Nibble Extraction** (Han & Ki) | 8-bit → nibbles: degree 255 → 15 via x^16 reuse | extract_nibbles() with x·P(x^16) evaluation | Same idea but needs level alignment |
| **Even-Polynomial** (Cheon et al.) | Even-only: p(x)=g(x^2) for depth/2 | _is_xor_even_poly() ready but returns False | Not utilized → XOR still depth 15 |
| **BSGS Evaluation** | Depth O(√d) | evaluate_polynomial_bsgs() exists | Giant powers recomputed → log depth lost |
| **Rotation Keys** | Han & Ki: Δ = 1,2,4,8,16 only | Same set (5 keys) | Matches |
| **Scale/Level Mgmt** | Papers assume CKKS level alignment | multiply() lacks level alignment | **Key difference** → errors/noise |
| **Coeff Cache** | Not in papers (implementation detail) | FFT + conjugate symmetry, npy cache | Good practical improvement |
"""
#!/usr/bin/env python3
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Any, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import scipy.fft
from desilofhe import Engine
from datetime import datetime
import logging.handlers
from collections import OrderedDict
import warnings

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"xor_paper_opt_{timestamp}.log"
    logger = logging.getLogger('xor')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging initialized. Log file: %s", log_file)
    logger.info("=" * 80)
    return log_file

log_file_path = setup_logging()
logger = logging.getLogger('xor')
Scalar = Union[int, float, complex]

@dataclass
class XORConfig:
    poly_modulus_degree: int = 16384
    precision_bits: int = 40
    use_nibbles: bool = True
    cache_coefficients: bool = True
    thread_count: int = 8
    mode: str = "parallel"
    device_id: int = 0
    use_conjugate_reduction: bool = True
    use_even_poly_opt: bool = True
    max_coeff_cache_size: int = 100

class XORPaperOptimized:
    def __init__(self, config: XORConfig):
        self.config = config
        logger.info("Initializing XOR with desilofhe API optimizations")
        self.engine = Engine(
            max_level=30,
            mode=config.mode,
            thread_count=config.thread_count,
            device_id=config.device_id if config.mode == "gpu" else 0
        )
        logger.info("Creating FHE keys...")
        start_time = time.time()
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relin_key = self.engine.create_relinearization_key(self.secret_key)
        self.conj_key = self.engine.create_conjugation_key(self.secret_key)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)
        self.fixed_rotation_keys = {}
        rotation_deltas = [1, 2, 4, 8, 16]
        for delta in rotation_deltas:
            if delta < self.engine.slot_count:
                self.fixed_rotation_keys[delta] = self.engine.create_fixed_rotation_key(
                    self.secret_key, delta
                )
        key_gen_time = time.time() - start_time
        logger.info(f"FHE key generation completed in {key_gen_time:.2f} seconds")
        self.scale = 2 ** config.precision_bits
        self.max_slots = self.engine.slot_count
        self.nibble_encoding = self._create_encoding(4)
        self.byte_encoding = self._create_encoding(8)
        self._coeff_cache = OrderedDict()
        self._load_cached_coefficients()
        self._precompute_coefficients()
        logger.info(f"XOR initialization complete. Slot count: {self.engine.slot_count}")
    
    def _create_encoding(self, bits: int) -> Dict[str, Any]:
        n = 1 << bits
        root = np.exp(2j * np.pi / n)
        return {
            'bits': bits,
            'n': n,
            'root': root,
            'all_roots': np.array([root**i for i in range(n)])
        }
    
    def encrypt(self, data: np.ndarray, level: Optional[int] = None) -> Any:
        if level is not None:
            return self.engine.encrypt(data, self.public_key, level=level)
        return self.engine.encrypt(data, self.public_key)
    
    def decrypt(self, ciphertext: Any) -> np.ndarray:
        return self.engine.decrypt(ciphertext, self.secret_key)
    
    def encode_array(self, values: Sequence[int], bits: int) -> np.ndarray:
        encoding = self.nibble_encoding if bits == 4 else self.byte_encoding
        root = encoding['root']
        n = encoding['n']
        return np.array([root ** (int(v) % n) for v in values], dtype=np.complex128)
    
    def decode_array(self, encoded: Sequence[complex], bits: int) -> np.ndarray:
        encoding = self.nibble_encoding if bits == 4 else self.byte_encoding
        n = encoding['n']
        result = []
        for c in encoded:
            c_clean = np.real_if_close(c, tol=1000)
            if np.iscomplex(c_clean):
                c_clean = c
            angle = float(np.angle(c_clean)) % (2 * np.pi)
            raw_idx = angle * n / (2 * np.pi)
            idx = int(np.rint(raw_idx)) % n
            result.append(idx)
        return np.array(result, dtype=np.int32)
    
    def evaluate_polynomial(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        non_zero_indices = [i for i, c in enumerate(coefficients) if abs(c) > 1e-14]
        if not non_zero_indices:
            return self.engine.multiply(x_enc, 0)
        if len(non_zero_indices) <= 5:
            return self._evaluate_sparse_poly(x_enc, coefficients, non_zero_indices)
        coeff_list = coefficients.tolist()
        return self.engine.evaluate_polynomial(x_enc, coeff_list, self.relin_key)
    
    def _evaluate_sparse_poly(self, x_enc: Any, coeffs: np.ndarray, indices: List[int]) -> Any:
        max_degree = max(indices)
        const_coeff = 0.0
        if 0 in indices:
            real_val = np.real_if_close(coeffs[0], tol=1000)
            if np.iscomplexobj(real_val):
                if abs(coeffs[0].imag) < 1e-10:
                    const_coeff = float(coeffs[0].real)
                else:
                    const_coeff = float(coeffs[0].real)  # Just use real part for polynomials
            else:
                const_coeff = float(real_val)
        if max_degree == 0:
            zero_data = np.zeros(self.max_slots, dtype=np.complex128)
            zero_ct = self.encrypt(zero_data, level=getattr(x_enc, 'level', 25))
            if abs(const_coeff) > 1e-14:
                return self.engine.add(zero_ct, const_coeff)
            else:
                return zero_ct
        powers = self.engine.make_power_basis(x_enc, max_degree, self.relin_key)
        ciphertexts = []
        weights = []
        for idx in indices:
            if idx == 0:
                continue
            ciphertexts.append(powers[idx - 1])
            real_val = np.real_if_close(coeffs[idx], tol=1000)
            if np.iscomplexobj(real_val):
                if abs(coeffs[idx].imag) < 1e-10:
                    weights.append(float(coeffs[idx].real))
                else:
                    weights.append(float(coeffs[idx].real))  # Just use real part
            else:
                weights.append(float(real_val))
        if not ciphertexts:
            zero_data = np.zeros(self.max_slots, dtype=np.complex128)
            zero_ct = self.encrypt(zero_data, level=getattr(x_enc, 'level', 25))
            if abs(const_coeff) > 1e-14:
                return self.engine.add(zero_ct, const_coeff)
            else:
                return zero_ct
        result = self.engine.weighted_sum(ciphertexts, weights)
        if abs(const_coeff) > 1e-14:
            result = self.engine.add(result, const_coeff)
        return result
    
    def rotate_batch(self, ciphertext: Any, deltas: List[int]) -> List[Any]:
        fixed_keys = []
        remaining_deltas = []
        for delta in deltas:
            if delta in self.fixed_rotation_keys:
                fixed_keys.append(self.fixed_rotation_keys[delta])
            else:
                remaining_deltas.append(delta)
        results = []
        if fixed_keys:
            results.extend(self.engine.rotate_batch(ciphertext, fixed_keys))
        if remaining_deltas:
            results.extend(self.engine.rotate_batch(ciphertext, self.rotation_key, remaining_deltas))
        return results
    
    def extract_nibbles(self, byte_enc: Any) -> Tuple[Any, Any]:
        upper_coeffs = self._get_nibble_coeffs_optimized(is_upper=True)
        lower_coeffs = self._get_nibble_coeffs_optimized(is_upper=False)
        x_16 = self.engine.square(self.engine.square(self.engine.square(self.engine.square(byte_enc, self.relin_key), self.relin_key), self.relin_key), self.relin_key)
        poly_coeffs = np.zeros(16, dtype=np.complex128)
        for i in range(16):
            idx = 1 + 16 * i
            if idx < len(upper_coeffs):
                poly_coeffs[i] = upper_coeffs[idx]
        poly_result = self.evaluate_polynomial(x_16, poly_coeffs)
        upper = self.engine.multiply(byte_enc, poly_result, self.relin_key)
        lower = x_16
        if 16 < len(lower_coeffs) and abs(lower_coeffs[16] - 1.0) > 1e-14:
            lower = self.engine.multiply(lower, lower_coeffs[16])
        return upper, lower
    
    def apply_nibble_xor(self, x_enc: Any, y_enc: Any) -> Any:
        coeffs_2d = self._get_xor_coeffs_2d(4)
        return self._evaluate_bivariate_poly(x_enc, y_enc, coeffs_2d)
    
    def _evaluate_bivariate_poly(self, x_enc: Any, y_enc: Any, coeffs_2d: np.ndarray) -> Any:
        non_zero_mask = np.abs(coeffs_2d) > 1e-14
        non_zero_indices = np.argwhere(non_zero_mask)
        if len(non_zero_indices) == 0:
            return self.engine.multiply(x_enc, 0)
        max_x_deg = max(idx[0] for idx in non_zero_indices)
        max_y_deg = max(idx[1] for idx in non_zero_indices)
        x_powers = [None] * (max_x_deg + 1)
        y_powers = [None] * (max_y_deg + 1)
        one_data = np.ones(self.max_slots, dtype=np.complex128)
        x_powers[0] = self.encrypt(one_data, level=getattr(x_enc, 'level', 25))
        y_powers[0] = x_powers[0]
        if max_x_deg > 0:
            x_basis = self.engine.make_power_basis(x_enc, max_x_deg, self.relin_key)
            for i in range(1, max_x_deg + 1):
                x_powers[i] = x_basis[i - 1]
        if max_y_deg > 0:
            y_basis = self.engine.make_power_basis(y_enc, max_y_deg, self.relin_key)
            for i in range(1, max_y_deg + 1):
                y_powers[i] = y_basis[i - 1]
        terms = []
        weights = []
        const_coeff = 0.0
        for idx_i, idx_j in non_zero_indices:
            coeff = coeffs_2d[idx_i, idx_j]
            real_val = np.real_if_close(coeff, tol=1000)
            if np.iscomplexobj(real_val):
                if abs(coeff.imag) < 1e-10:
                    coeff_real = float(coeff.real)
                else:
                    raise ValueError(f"Coefficient has non-negligible imaginary part: {coeff}")
            else:
                coeff_real = float(real_val)
            if idx_i == 0 and idx_j == 0:
                const_coeff = coeff_real
                continue
            elif idx_i == 0:
                terms.append(y_powers[idx_j])
                weights.append(coeff_real)
            elif idx_j == 0:
                terms.append(x_powers[idx_i])
                weights.append(coeff_real)
            else:
                xy_term = self.engine.multiply(x_powers[idx_i], y_powers[idx_j], self.relin_key)
                terms.append(xy_term)
                weights.append(coeff_real)
        if not terms:
            zero_ct = self.encrypt(np.zeros(self.max_slots, dtype=np.complex128), 
                                 level=getattr(x_enc, 'level', 25))
            if abs(const_coeff) > 1e-14:
                return self.engine.add(zero_ct, const_coeff)
            else:
                return zero_ct
        result = self.engine.weighted_sum(terms, weights)
        if abs(const_coeff) > 1e-14:
            result = self.engine.add(result, const_coeff)
        return result
    
    def apply_byte_xor_via_nibbles(self, x_enc: Any, y_enc: Any) -> Tuple[Any, Any]:
        x_hi, x_lo = self.extract_nibbles(x_enc)
        y_hi, y_lo = self.extract_nibbles(y_enc)
        hi_xor = self.apply_nibble_xor(x_hi, y_hi)
        lo_xor = self.apply_nibble_xor(x_lo, y_lo)
        return hi_xor, lo_xor
    
    def _compute_coefficients(self, lut_function, input_bits: int, output_encoding_bits: int) -> np.ndarray:
        n = 1 << input_bits
        encoding = self.nibble_encoding if output_encoding_bits == 4 else self.byte_encoding
        root = encoding['root']
        lut = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            output_val = lut_function(i)
            lut[i] = root ** output_val
        coeffs = np.fft.fft(lut) / n
        return self._apply_conjugate_symmetry(coeffs, n)
    
    def _apply_conjugate_symmetry(self, coeffs: np.ndarray, n: int) -> np.ndarray:
        result = coeffs.copy()
        for k in range(1, n//2):
            avg = (result[k] + np.conj(result[n-k])) / 2
            result[k] = avg
            result[n-k] = np.conj(avg)
        if n % 2 == 0:
            result[n//2] = np.real(result[n//2])
        return result
    
    def _get_nibble_coeffs_optimized(self, is_upper: bool) -> np.ndarray:
        cache_key = f"{'upper' if is_upper else 'lower'}_nibble_optimized"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        def nibble_func(byte_val: int) -> int:
            return (byte_val // 16) if is_upper else (byte_val % 16)
        coeffs = self._compute_coefficients(nibble_func, input_bits=8, output_encoding_bits=4)
        if self.config.use_conjugate_reduction:
            reduced = np.zeros_like(coeffs)
            if is_upper:
                for k in range(16):
                    idx = 1 + 16 * k
                    if idx < len(coeffs):
                        reduced[idx] = coeffs[idx]
            else:
                if abs(coeffs[0]) > 1e-14:
                    reduced[0] = coeffs[0]
                reduced[16] = coeffs[16]
            coeffs = reduced
        self._cache_coefficient(cache_key, coeffs)
        return coeffs
    
    def _get_xor_coeffs_2d(self, bits: int) -> np.ndarray:
        cache_key = f"xor_{bits}_2d"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        n = 1 << bits
        lut = np.empty((n, n), dtype=np.complex128)
        root = (self.nibble_encoding if bits == 4 else self.byte_encoding)['root']
        for i in range(n):
            for j in range(n):
                lut[i, j] = root ** (i ^ j)
        coeffs = scipy.fft.fftn(lut, norm='forward')
        self._cache_coefficient(cache_key, coeffs)
        return coeffs
    
    def _cache_coefficient(self, key: str, value: np.ndarray):
        if len(self._coeff_cache) >= self.config.max_coeff_cache_size:
            self._coeff_cache.popitem(last=False)
        self._coeff_cache[key] = value
    
    def _precompute_coefficients(self):
        logger.info("Pre-computing XOR coefficients...")
        self._get_nibble_coeffs_optimized(is_upper=True)
        self._get_nibble_coeffs_optimized(is_upper=False)
        self._get_xor_coeffs_2d(4)
        self.save_cached_coefficients()
        logger.info("Coefficient pre-computation complete")
    
    def _load_cached_coefficients(self):
        cache_dir = Path("xor_cache")
        if not cache_dir.exists():
            return
        for prefix in ["upper", "lower"]:
            file = cache_dir / f"{prefix}_nibble_optimized.npy"
            if file.exists():
                try:
                    coeffs = np.load(file)
                    self._coeff_cache[f"{prefix}_nibble_optimized"] = coeffs
                    logger.info(f"Loaded cached {prefix} nibble coefficients")
                except Exception as e:
                    logger.warning(f"Failed to load {prefix} nibble cache: {e}")
    
    def save_cached_coefficients(self):
        if not self.config.cache_coefficients:
            return
        cache_dir = Path("xor_cache")
        cache_dir.mkdir(exist_ok=True)
        for key, coeffs in self._coeff_cache.items():
            file = cache_dir / f"{key}.npy"
            np.save(file, coeffs)
            logger.info(f"Saved coefficients to {file}")

def decrypt_simd(xor_inst: XORPaperOptimized, hi_ct: Any, lo_ct: Any, length: int) -> np.ndarray:
    hi_dec = xor_inst.decrypt(hi_ct)[:length]
    lo_dec = xor_inst.decrypt(lo_ct)[:length]
    hi = xor_inst.decode_array(hi_dec, 4)
    lo = xor_inst.decode_array(lo_dec, 4)
    return (hi << 4) | lo
