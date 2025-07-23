#!/usr/bin/env python3
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Any, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import scipy.fft
from desilofhe import Engine
import desilofhe
from datetime import datetime
import logging.handlers
from collections import OrderedDict
import warnings

Scalar = Union[int, float, complex]



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
        self._power_basis_cache = {}
        self._precompute_coefficients()
        self._load_noise_reduction_coeffs()
        logger.info(f"XOR initialization complete. Slot count: {self.engine.slot_count}")
    def multiply_plain(self, ct: Any, val: Union[Scalar, np.ndarray, Any]) -> Any:
        """Multiply ciphertext by plaintext value, handling complex scalars."""
        if isinstance(val, (int, float)):
            # Real scalar - direct multiplication
            return self.engine.multiply(ct, val)
        elif isinstance(val, complex):
            # Complex scalar - need to encode
            scalar_array = np.full(self.max_slots, val, dtype=np.complex128)
            pt = self.engine.encode(scalar_array)
            return self.engine.multiply(ct, pt)
        elif isinstance(val, np.ndarray):
            if np.ptp(val) < 1e-12:
                return self.multiply_plain(ct, complex(val.flat[0]))
            pt = self.engine.encode(val.astype(np.complex128))
            return self.engine.multiply(ct, pt)
        elif hasattr(val, '__class__') and 'Plaintext' in str(val.__class__):
            return self.engine.multiply(ct, val)
        else:
            raise TypeError(f"multiply_plain: unsupported type {type(val)}")
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
        powers = self.engine.make_power_basis(x_enc, max_degree, self.relin_key) if max_degree > 0 else []
        
        # Separate real and complex coefficients
        real_terms = []
        real_weights = []
        complex_terms = []
        
        for idx in indices:
            coeff = coeffs[idx]
            if abs(coeff.imag) < 1e-10:
                # Real coefficient
                if idx == 0:
                    # Handle constant term separately
                    real_weights = [float(coeff.real)] + real_weights if real_weights else [float(coeff.real)]
                else:
                    real_terms.append(powers[idx - 1])
                    real_weights.append(float(coeff.real))
            else:
                # Complex coefficient - needs special handling
                if idx == 0:
                    # Complex constant - create one ciphertext
                    one_data = np.ones(self.max_slots, dtype=np.complex128)
                    one_ct = self.encrypt(one_data, level=getattr(x_enc, 'level', 25))
                    complex_terms.append((one_ct, coeff))
                else:
                    complex_terms.append((powers[idx - 1], coeff))
        
        # Process real coefficients with weighted_sum
        result = None
        if real_terms:
            # weighted_sum: first weight is constant term
            if len(real_weights) == len(real_terms) + 1:
                # Constant already included as first weight
                result = self.engine.weighted_sum(real_terms, real_weights)
            else:
                # Need to add constant term
                real_weights = [0.0] + real_weights
                result = self.engine.weighted_sum(real_terms, real_weights)
        elif real_weights:
            # Only constant term - use add_plain for efficiency
            result = self.add_plain(x_enc, real_weights[0])
        
        # Process complex coefficients individually
        for term, coeff in complex_terms:
            coeff_pt = self.engine.encode(
                np.full(self.max_slots, coeff, dtype=np.complex128)
            )
            term_result = self.multiply_plain(term, coeff_pt)
            result = term_result if result is None else self.engine.add(result, term_result)
        
        # Handle case where we only have real constant but no terms
        if result is None and real_weights:
            result = self.add_plain(x_enc, real_weights[0])
        
        # Return zero if no result
        if result is None:
            # Return zero ciphertext
            zero_pt = self.engine.encode(np.zeros(self.max_slots, dtype=np.complex128))
            result = self.engine.multiply(x_enc, zero_pt)
        
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
            lower = self.multiply_plain(lower, lower_coeffs[16])
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
        real_terms = []
        real_weights = []
        complex_terms = []
        
        for idx_i, idx_j in non_zero_indices:
            coeff = coeffs_2d[idx_i, idx_j]
            
            # Determine the term (monomial)
            if idx_i == 0 and idx_j == 0:
                # Constant term - always treat as complex to preserve accuracy
                complex_terms.append((x_powers[0], coeff))
                continue
            elif idx_i == 0:
                term = y_powers[idx_j]
            elif idx_j == 0:
                term = x_powers[idx_i]
            else:
                term = self.engine.multiply(x_powers[idx_i], y_powers[idx_j], self.relin_key)
            
            # Separate real and complex coefficients
            if abs(coeff.imag) < 1e-10:
                real_terms.append(term)
                real_weights.append(float(coeff.real))
            else:
                complex_terms.append((term, coeff))
        
        # Process real coefficients with weighted_sum
        result = None
        if real_terms:
            # weighted_sum expects weights = [constant] + [weights for terms]
            if len(real_weights) > len(real_terms):
                # Constant already at front
                result = self.engine.weighted_sum(real_terms, real_weights)
            else:
                # Need to add zero constant
                result = self.engine.weighted_sum(real_terms, [0.0] + real_weights)
        
        # Process complex coefficients individually
        for term, coeff in complex_terms:
            # Encode complex coefficient as plaintext to preserve imaginary part
            coeff_pt = self.engine.encode(
                np.full(self.max_slots, coeff, dtype=np.complex128)
            )
            term_result = self.multiply_plain(term, coeff_pt)
            result = term_result if result is None else self.engine.add(result, term_result)
        
        # If no result yet, return zero
        if result is None:
            # Return zero
            zero_pt = self.engine.encode(np.zeros(self.max_slots, dtype=np.complex128))
            result = self.engine.multiply(x_enc, zero_pt)
        
        return result
    
    def encrypt_nibbles(self, byte_array: np.ndarray, level: int = 20) -> Tuple[Any, Any]:
        """Encrypt bytes directly as separate hi/lo nibbles to avoid extraction."""
        hi = (byte_array >> 4) & 0xF
        lo = byte_array & 0xF
        ct_hi = self.encrypt(self.encode_array(hi, 4), level)
        ct_lo = self.encrypt(self.encode_array(lo, 4), level)
        return ct_hi, ct_lo
    
    def apply_nibble_xor_direct(self, x_hi: Any, x_lo: Any, y_hi: Any, y_lo: Any) -> Tuple[Any, Any]:
        """Apply XOR directly on pre-separated nibbles."""
        hi_xor = self.apply_nibble_xor(x_hi, y_hi)
        lo_xor = self.apply_nibble_xor(x_lo, y_lo)
        return hi_xor, lo_xor
    
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
        coeffs = scipy.fft.fftn(lut, norm='forward', workers=self.config.thread_count)
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

    def _load_noise_reduction_coeffs(self):
        """Load noise reduction polynomial coefficients from paper."""
        # Try to load from paper's CSV file
        noise_coeff_file = Path("xor_cache") / "noise_reduction_coeffs.csv"
        if noise_coeff_file.exists():
            try:
                coeffs = np.loadtxt(noise_coeff_file, dtype=np.complex128, delimiter=',')
                self._coeff_cache['noise_reduction_f'] = coeffs
                logger.info(f"Loaded noise reduction coefficients from {noise_coeff_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load noise coefficients: {e}")
        
        # Paper's noise reduction polynomial (degree 32-48)
        degree = 32
        coeffs = np.zeros(degree + 1, dtype=np.complex128)
        # Approximate coefficients from paper Section 3.3
        coeffs[0] = 1.0
        coeffs[1] = 0.0
        coeffs[2] = -0.5
        coeffs[4] = 0.25
        coeffs[8] = -0.125
        coeffs[16] = 0.0625
        coeffs[32] = -0.03125
        self._coeff_cache['noise_reduction_f'] = coeffs
    
    def apply_noise_reduction(self, ct: Any) -> Any:
        """Apply noise reduction polynomial f to reduce noise before next LUT."""
        coeffs = self._coeff_cache.get('noise_reduction_f')
        if coeffs is None:
            return ct
        result = self.evaluate_polynomial(ct, coeffs)
        # Rescale to manage noise
        result = self.engine.rescale(result)
        return result
    
    def _make_power_basis_cached(self, x_enc: Any, max_degree: int) -> List[Any]:
        """Create power basis with caching for shared use across LUTs."""
        cache_key = (id(x_enc), max_degree)
        if cache_key in self._power_basis_cache:
            return self._power_basis_cache[cache_key]
        powers = self.engine.make_power_basis(x_enc, max_degree, self.relin_key)
        self._power_basis_cache[cache_key] = powers
        return powers
    
    def evaluate_lut_bundle(self, lut_coeffs_list: List[np.ndarray], x_enc: Any) -> List[Any]:
        """Evaluate multiple LUTs simultaneously sharing power basis."""
        max_degree = max(len(coeffs) - 1 for coeffs in lut_coeffs_list)
        powers = self._make_power_basis_cached(x_enc, max_degree)
        
        results = []
        for coeffs in lut_coeffs_list:
            # Use shared power basis for each LUT
            non_zero_indices = [i for i, c in enumerate(coeffs) if abs(c) > 1e-14]
            if not non_zero_indices:
                results.append(self.engine.multiply(x_enc, 0))
                continue
            
            # Separate constant term and other terms
            constant_term = 0.0
            terms_list = []
            weights = []
            
            for idx in non_zero_indices:
                if idx == 0:
                    constant_term = float(coeffs[0].real)
                else:
                    terms_list.append(powers[idx - 1])
                    weights.append(float(coeffs[idx].real))
            
            # Build result
            if terms_list:
                # weighted_sum with constant as first weight
                all_weights = [constant_term] + weights
                result = self.engine.weighted_sum(terms_list, all_weights)
            else:
                # Only constant term
                result = self.add_plain(x_enc, constant_term)
            
            results.append(result)
        
        return results
    
    def add_plain(self, ct: Any, val: Union[Scalar, np.ndarray]) -> Any:
        """Add plaintext value to ciphertext efficiently."""
        if isinstance(val, (int, float, complex)):
            # desilofhe supports direct scalar addition
            return self.engine.add(ct, val)
        elif isinstance(val, np.ndarray):
            # For arrays, check if constant
            if np.ptp(val) < 1e-12:
                return self.engine.add(ct, complex(val.flat[0]))
            # Otherwise encode as plaintext
            pt = self.engine.encode(val.astype(np.complex128))
            return self.engine.add(ct, pt)
        else:
            raise TypeError(f"add_plain: unsupported type {type(val)}")

def decrypt_simd(xor_inst: XORPaperOptimized, hi_ct: Any, lo_ct: Any, length: int) -> np.ndarray:
    hi_dec = xor_inst.decrypt(hi_ct)[:length]
    lo_dec = xor_inst.decrypt(lo_ct)[:length]
    hi = xor_inst.decode_array(hi_dec, 4)
    lo = xor_inst.decode_array(lo_dec, 4)
    return (hi << 4) | lo
    
