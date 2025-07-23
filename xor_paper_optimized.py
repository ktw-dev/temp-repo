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

# Configure logging
def setup_logging():
    """Setup logging to both console and file with rotation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"xor_paper_opt_{timestamp}.log"
    
    logger = logging.getLogger('xor')
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.WARNING)  # Only warnings and errors to file
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Info and above to console
    
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

# Initialize logging
log_file_path = setup_logging()
logger = logging.getLogger('xor')

# Type aliases
Scalar = Union[int, float, complex]


@dataclass
class XORConfig:
    poly_modulus_degree: int = 16384
    precision_bits: int = 40
    use_nibbles: bool = True
    cache_coefficients: bool = True
    thread_count: int = 8
    mode: str = "parallel"  # "parallel", "serial", "gpu"
    device_id: int = 0
    # Paper-specific optimizations
    use_conjugate_reduction: bool = True  # Han & Ki '22
    use_even_poly_opt: bool = True       # Cheon et al. '20
    use_baby_giant_step: bool = True     # BSGS for polynomial eval
    bsgs_threshold: int = 16             # When to use BSGS
    # Cache management
    max_power_cache_size: int = 1000     # Max entries in power cache
    max_coeff_cache_size: int = 100      # Max entries in coefficient cache
    # Strict mode for testing
    strict_decrypt_check: bool = False   # Check decryption accuracy
    noise_budget_threshold: int = 10     # Min bits of noise budget


class XORPaperOptimized:
    """XOR implementation with paper-based optimizations."""
    
    def __init__(self, config: XORConfig):
        self.config = config
        logger.info("Initializing XOR with paper optimizations")
        logger.info(f"Config: {config}")
        
        # Initialize FHE engine using desilofhe
        self.engine = Engine(
            max_level=30,
            mode=config.mode,
            thread_count=config.thread_count,
            device_id=config.device_id if config.mode == "gpu" else 0
        )
        
        # Create FHE keys
        logger.info("Creating FHE keys...")
        start_time = time.time()
        
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relin_key = self.engine.create_relinearization_key(self.secret_key)
        self.conj_key = self.engine.create_conjugation_key(self.secret_key)
        
        # Create minimal rotation keys
        self.rotation_keys = {}
        rotation_deltas = [1, 2, 4, 8, 16]  # Minimal set for nibble ops
        for delta in rotation_deltas:
            if delta < self.engine.slot_count:
                self.rotation_keys[delta] = self.engine.create_fixed_rotation_key(
                    self.secret_key, delta
                )
        
        key_gen_time = time.time() - start_time
        logger.info(f"FHE key generation completed in {key_gen_time:.2f} seconds")
        
        # Initialize components
        self._init_components()
        
        # Initialize caches with LRU eviction
        self._coeff_cache = OrderedDict()
        self._power_cache = OrderedDict()  # Cache for computed powers to avoid redundant calculations
        self._constant_cache = {}  # Cache for constant ciphertexts
        self._load_cached_coefficients()
        
        # Check API compatibility
        self._check_api_compatibility()
        
        # Pre-compute common constants
        self._init_constant_cache()
        
        # Pre-compute XOR coefficients
        self._precompute_coefficients()
        
        logger.info("XOR initialization complete")
        logger.info(f"Slot count: {self.engine.slot_count}")
        logger.info(f"Max level: 30")
        logger.info("=" * 80)
    
    def _init_components(self):
        """Initialize all internal components."""
        self.scale = 2 ** self.config.precision_bits
        self.max_slots = self.engine.slot_count
        
        # Encoding parameters
        self.nibble_encoding = self._create_encoding(4)
        self.byte_encoding = self._create_encoding(8)
    
    def _init_constant_cache(self):
        """Pre-compute commonly used constant ciphertexts."""
        # Cache zero and one ciphertexts at different levels
        # Ensure minimum level 1 for CKKS compatibility
        for level in [30, 25, 20, 15, 10, 5, 1]:
            # Zero ciphertext
            zero_key = ('zero', level)
            zero_data = np.zeros(self.max_slots, dtype=np.complex128)
            self._constant_cache[zero_key] = self.encrypt(zero_data, level=max(level, 1))
            
            # One ciphertext
            one_key = ('one', level)
            one_data = np.ones(self.max_slots, dtype=np.complex128)
            self._constant_cache[one_key] = self.encrypt(one_data, level=max(level, 1))
    
    def _precompute_coefficients(self):
        """Pre-compute commonly used coefficients at module initialization."""
        logger.info("Pre-computing XOR coefficients...")
        
        # Pre-compute nibble extraction coefficients
        self._get_nibble_coeffs_optimized(is_upper=True)
        self._get_nibble_coeffs_optimized(is_upper=False)
        
        # Pre-compute 4-bit XOR coefficients
        self._get_xor_coeffs_2d(4)
        
        # Pre-compute 8→4 XOR coefficients (paper approach)
        self._get_xor_coeffs_1d_paper()
        
        # Save to disk for future runs
        self.save_cached_coefficients()
        
        logger.info("Coefficient pre-computation complete")
    
    def _create_encoding(self, bits: int) -> Dict[str, Any]:
        """Create encoding parameters for given bit size."""
        n = 1 << bits
        root = np.exp(2j * np.pi / n)
        return {
            'bits': bits,
            'n': n,
            'root': root,
            'all_roots': np.array([root**i for i in range(n)])
        }
    
    # === Core Operations ===
    
    def encrypt(self, data: np.ndarray, level: Optional[int] = None) -> Any:
        """Encrypt data directly without encoding to Plaintext first."""
        if level is not None:
            return self.engine.encrypt(data, self.public_key, level=level)
        else:
            return self.engine.encrypt(data, self.public_key)
    
    def decrypt(self, ciphertext: Any) -> np.ndarray:
        """Decrypt ciphertext."""
        return self.engine.decrypt(ciphertext, self.secret_key)
    
    def encode_array(self, values: Sequence[int], bits: int) -> np.ndarray:
        """Encode integer array as roots of unity."""
        encoding = self.nibble_encoding if bits == 4 else self.byte_encoding
        root = encoding['root']
        n = encoding['n']
        return np.array([root ** (int(v) % n) for v in values], dtype=np.complex128)
    
    def decode_array(self, encoded: Sequence[complex], bits: int) -> np.ndarray:
        """Decode roots of unity back to integers with noise tolerance."""
        encoding = self.nibble_encoding if bits == 4 else self.byte_encoding
        n = encoding['n']
        
        result = []
        for c in encoded:
            # Use np.real_if_close to handle small imaginary parts from noise
            c_clean = np.real_if_close(c, tol=1000)
            if np.iscomplex(c_clean):
                c_clean = c  # Keep original if still complex
            
            # Compute angle with proper handling
            angle = float(np.angle(c_clean)) % (2 * np.pi)
            
            # Decode with rounding
            raw_idx = angle * n / (2 * np.pi)
            idx = int(np.rint(raw_idx)) % n
            
            # Sanity check - if decoding seems far off, log warning
            rounding_error = abs(raw_idx - np.rint(raw_idx))
            if rounding_error > 0.4:  # Increased tolerance for CKKS
                logger.debug(f"Large rounding error in decode: {raw_idx:.3f} -> {idx}")
            
            # Apply modulo to ensure valid range
            idx = idx % n
            result.append(idx)
        
        return np.array(result, dtype=np.int32)
    
    def add(self, ct1: Any, ct2: Any) -> Any:
        """Add two ciphertexts."""
        return self.engine.add(ct1, ct2)
    
    def multiply(self, ct1: Any, ct2: Any) -> Any:
        """Multiply two ciphertexts with relinearization.
        
        Includes level alignment to prevent scale mismatches and reduce noise.
        Automatically rescales if scale grows too large.
        """
        # Align levels before multiplication
        ct1_aligned, ct2_aligned = self._align_levels(ct1, ct2)
        
        # Perform multiplication
        result = self.engine.multiply(ct1_aligned, ct2_aligned)
        result = self.engine.relinearize(result, self.relin_key)
        
        # Check if rescaling is needed to prevent scale explosion
        if hasattr(result, 'scale'):
            result_scale = getattr(result, 'scale', self.scale)
            if result_scale > self.scale * 2 and self.has_rescale:
                logger.debug(f"Rescaling after multiply: scale {result_scale:.2e} -> {self.scale:.2e}")
                if hasattr(self.engine, 'rescale_to_next'):
                    result = self.engine.rescale_to_next(result)
                elif hasattr(self.engine, 'rescale'):
                    result = self.engine.rescale(result)
        
        # Check level after multiplication
        result_level = getattr(result, 'level', 1)
        if result_level <= 2:
            logger.warning(f"Low level after multiply: {result_level}")
        
        return result
    
    def _align_levels(self, ct1: Any, ct2: Any) -> Tuple[Any, Any]:
        """Align ciphertext levels before operations.
        
        This prevents scale mismatches and reduces noise growth.
        """
        ct1_level = getattr(ct1, 'level', 30)
        ct2_level = getattr(ct2, 'level', 30)
        
        if ct1_level > ct2_level:
            ct1 = self._level_down(ct1, ct2_level)
        elif ct2_level > ct1_level:
            ct2 = self._level_down(ct2, ct1_level)
        
        # Also align scales if needed
        ct1, ct2 = self._align_scales(ct1, ct2)
        
        return ct1, ct2
    
    def _level_down(self, ct: Any, target_level: int) -> Any:
        """Level down with API compatibility check."""
        if hasattr(self.engine, 'level_down'):
            return self.engine.level_down(ct, target_level)
        elif hasattr(self.engine, 'mod_switch_to_level'):
            return self.engine.mod_switch_to_level(ct, target_level)
        else:
            warnings.warn("No level_down API found, returning original ciphertext")
            return ct
    
    def _align_scales(self, ct1: Any, ct2: Any) -> Tuple[Any, Any]:
        """Align ciphertext scales to prevent precision loss."""
        scale1 = getattr(ct1, 'scale', self.scale)
        scale2 = getattr(ct2, 'scale', self.scale)
        
        if abs(scale1 - scale2) > 1e-6:  # Scales differ significantly
            if hasattr(self.engine, 'rescale_to_next'):
                # Rescale the one with larger scale
                if scale1 > scale2:
                    ct1 = self.engine.rescale_to_next(ct1)
                else:
                    ct2 = self.engine.rescale_to_next(ct2)
            elif hasattr(self.engine, 'rescale'):
                if scale1 > scale2:
                    ct1 = self.engine.rescale(ct1)
                else:
                    ct2 = self.engine.rescale(ct2)
        
        return ct1, ct2
    
    def _check_api_compatibility(self):
        """Check which APIs are available in the engine."""
        self.has_level_down = hasattr(self.engine, 'level_down')
        self.has_mod_switch = hasattr(self.engine, 'mod_switch_to_level')
        self.has_rescale = hasattr(self.engine, 'rescale') or hasattr(self.engine, 'rescale_to_next')
        self.has_noise_budget = hasattr(self.engine, 'invariant_noise_budget')
        self.has_multiply_plain = hasattr(self.engine, 'multiply_plain')
        
        logger.info(f"API compatibility: level_down={self.has_level_down}, "
                   f"mod_switch={self.has_mod_switch}, rescale={self.has_rescale}, "
                   f"noise_budget={self.has_noise_budget}")
    
    def multiply_plain(self, ct: Any, scalar: Scalar) -> Any:
        """Multiply ciphertext by plaintext scalar.
        
        Uses the proper ciphertext-plaintext multiplication path to avoid
        scale explosion and noise growth.
        """
        if isinstance(scalar, complex):
            # For complex scalars, encode to plaintext
            scalar_data = np.full(self.max_slots, scalar, dtype=np.complex128)
            scalar_pt = self.engine.encode(scalar_data)
            # Use ciphertext-plaintext multiplication (no relinearization needed)
            if self.has_multiply_plain:
                return self.engine.multiply_plain(ct, scalar_pt)
            else:
                return self.engine.multiply(ct, scalar_pt)
        else:
            # For real scalars, use dedicated multiply_plain if available
            if self.has_multiply_plain:
                # Create plaintext from scalar
                scalar_data = np.full(self.max_slots, scalar, dtype=np.float64)
                scalar_pt = self.engine.encode(scalar_data)
                return self.engine.multiply_plain(ct, scalar_pt)
            else:
                # Fallback to regular multiply (engine handles conversion)
                return self.engine.multiply(ct, scalar)
    
    def add_plain(self, ct: Any, scalar: Scalar) -> Any:
        """Add plaintext scalar to ciphertext."""
        if isinstance(scalar, complex):
            # For complex scalars, create a ciphertext
            scalar_data = np.full(self.max_slots, scalar, dtype=np.complex128)
            scalar_ct = self.encrypt(scalar_data, level=getattr(ct, 'level', 25))
            return self.add(ct, scalar_ct)
        else:
            return self.engine.add(ct, scalar)
    
    # === Optimized Polynomial Evaluation ===
    
    def evaluate_polynomial_bsgs(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        """Evaluate polynomial using Baby-Step/Giant-Step algorithm.
        
        For degree d polynomial:
        - Baby steps: compute x, x^2, ..., x^k where k = sqrt(d)
        - Giant steps: compute x^k, x^(2k), ..., x^(mk) where m = ceil(d/k)
        - Total depth: O(log k + log m) instead of O(log d)
        """
        n = len(coefficients)
        non_zero_indices = [i for i in range(n) if abs(coefficients[i]) > 1e-14]
        
        if not non_zero_indices:
            return self.multiply_plain(x_enc, 0)
        
        max_degree = max(non_zero_indices)
        
        # Use BSGS for polynomials above threshold
        if max_degree > self.config.bsgs_threshold:
            logger.info(f"Using BSGS for degree {max_degree} polynomial")
            
            # Optimal baby/giant step size
            k = int(np.sqrt(max_degree)) + 1
            m = (max_degree // k) + 1
            
            # Baby steps: compute x^i for i = 0, 1, ..., k using binary exponentiation
            baby_powers = {}
            # x^0 = 1
            input_level = getattr(x_enc, 'level', 25)
            if input_level <= 0:
                input_level = 1  # Minimum positive level for CKKS
            baby_powers[0] = self._constant_cache.get(('one', input_level))
            if baby_powers[0] is None:
                baby_powers[0] = self.encrypt(np.ones(self.max_slots), level=input_level)
            baby_powers[1] = x_enc
            
            # Use binary exponentiation for other powers
            for i in range(2, k + 1):
                baby_powers[i] = self._compute_power_binary(x_enc, i)
            
            # Giant step base: x^k
            giant_base = baby_powers[k]
            
            # Precompute giant powers using binary exponentiation
            giant_powers = {}
            giant_powers[0] = baby_powers[0]  # x^0 = 1
            if m > 1:
                giant_powers[1] = giant_base
            
            # Evaluate polynomial by grouping terms
            result = None
            for j in range(m):
                # Terms with degrees j*k, j*k+1, ..., j*k+(k-1)
                group_result = None
                
                for i in range(k):
                    deg = j * k + i
                    if deg < n and abs(coefficients[deg]) > 1e-14:
                        term = self.multiply_plain(baby_powers[i], coefficients[deg])
                        group_result = term if group_result is None else self.add(group_result, term)
                
                if group_result is not None:
                    # Multiply by x^(j*k) using binary exponentiation
                    if j > 0:
                        if j not in giant_powers:
                            # Compute giant_base^j using binary exponentiation
                            giant_powers[j] = self._compute_power_binary(giant_base, j)
                        group_result = self.multiply(group_result, giant_powers[j])
                    
                    result = group_result if result is None else self.add(result, group_result)
            
            # Check level exhaustion
            if result is not None and getattr(result, 'level', 1) <= 1:
                logger.warning(f"Level nearly exhausted after BSGS: {getattr(result, 'level', 'unknown')}")
                raise RuntimeError("Level exhausted; increase starting level or reduce polynomial degree")
            
            return result
        else:
            # For small polynomials, use standard evaluation
            return self._evaluate_standard(x_enc, coefficients, non_zero_indices)
    
    def _evaluate_standard(self, x_enc: Any, coefficients: np.ndarray, non_zero_indices: List[int]) -> Any:
        """Standard polynomial evaluation for small degrees."""
        result = None
        x_powers = {}
        
        for idx in non_zero_indices:
            coeff = coefficients[idx]
            
            if idx == 0:
                # Use cached constant if coefficient is 0 or 1
                level = getattr(x_enc, 'level', 25)
                if abs(coeff) < 1e-14:
                    continue  # Skip zero coefficients
                elif abs(coeff - 1.0) < 1e-14:
                    term = self._constant_cache.get(('one', level))
                    if term is None:
                        term = self.encrypt(np.ones(self.max_slots), level=level)
                else:
                    term = self.encrypt(np.full(self.max_slots, coeff), level=level)
            else:
                # Compute power if not cached
                if idx not in x_powers:
                    x_powers[idx] = self._compute_power_binary(x_enc, idx)
                term = self.multiply_plain(x_powers[idx], coeff)
            
            result = term if result is None else self.add(result, term)
        
        # Ensure we never return None
        if result is None:
            logger.warning("All coefficients were zero in standard evaluation - returning zero ciphertext")
            zero_pt = self.encrypt(np.ones(self.max_slots) * 0, level=getattr(x_enc, 'level', 25))
            return zero_pt
        
        return result
    
    def _compute_power_binary(self, x_enc: Any, degree: int) -> Any:
        """Compute x^degree using binary exponentiation with caching.
        
        Caches results to avoid redundant computations, especially useful
        for nibble XOR operations where the same powers are reused.
        """
        # Check cache first
        cache_key = (id(x_enc), degree)
        if cache_key in self._power_cache:
            # Move to end for LRU
            self._power_cache.move_to_end(cache_key)
            return self._power_cache[cache_key]
        
        if degree == 0:
            # Return a constant 1 ciphertext at the same level
            level = getattr(x_enc, 'level', 25)
            result = self._constant_cache.get(('one', level))
            if result is None:
                ones_data = np.ones(self.max_slots, dtype=np.complex128)
                result = self.encrypt(ones_data, level=level)
        elif degree == 1:
            result = x_enc
        else:
            # Binary exponentiation with careful scale management
            result = None
            base = x_enc
            exp = degree
            
            while exp > 0:
                if exp & 1:
                    result = base if result is None else self.multiply(result, base)
                if exp > 1:
                    base = self.multiply(base, base)
                exp >>= 1
        
        # Cache the result with LRU eviction
        self._cache_power(cache_key, result)
        return result
    
    def _cache_power(self, key: Tuple, value: Any):
        """Cache power with size limit and LRU eviction."""
        if len(self._power_cache) >= self.config.max_power_cache_size:
            # Remove oldest entry
            self._power_cache.popitem(last=False)
        self._power_cache[key] = value
    
    def _cache_coefficient(self, key: str, value: np.ndarray):
        """Cache coefficient with size limit and LRU eviction."""
        if len(self._coeff_cache) >= self.config.max_coeff_cache_size:
            # Remove oldest entry
            self._coeff_cache.popitem(last=False)
        self._coeff_cache[key] = value
    
    # === Nibble Extraction with Conjugate Reduction ===
    
    def extract_nibbles(self, byte_enc: Any) -> Tuple[Any, Any]:
        """Extract upper and lower nibbles using Han & Ki optimization."""
        # Get optimized coefficients
        upper_coeffs = self._get_nibble_coeffs_optimized(is_upper=True)
        lower_coeffs = self._get_nibble_coeffs_optimized(is_upper=False)
        
        # Evaluate extractions
        upper = self._evaluate_upper_nibble_han_ki(byte_enc, upper_coeffs)
        lower = self._evaluate_lower_nibble_han_ki(byte_enc, lower_coeffs)
        
        return upper, lower
    
    def _evaluate_upper_nibble_han_ki(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        """Upper nibble extraction following Han & Ki paper.
        
        The polynomial has form: x * P(x^16) where P has degree 15.
        This reduces evaluation from degree 241 to degree 15.
        """
        logger.info("=== Upper Nibble Extraction (Han & Ki) ===")
        initial_level = getattr(x_enc, 'level', 25)
        initial_scale = getattr(x_enc, 'scale', self.scale)
        logger.debug(f"Initial scale: {initial_scale:.2e}")
        
        # Compute x^16 efficiently
        x_16 = self._compute_power_binary(x_enc, 16)
        
        # Extract coefficients for P(t) where t = x^16
        # Coefficients at indices 1, 17, 33, ..., 241 map to P
        poly_coeffs = np.zeros(16, dtype=np.complex128)
        for i in range(16):
            idx = 1 + 16 * i
            if idx < len(coefficients):
                poly_coeffs[i] = coefficients[idx]
        
        # Evaluate P(x^16) - only degree 15!
        if self.config.use_baby_giant_step:
            poly_result = self.evaluate_polynomial_bsgs(x_16, poly_coeffs)
        else:
            poly_result = self._evaluate_standard(x_16, poly_coeffs, 
                                                 [i for i in range(16) if abs(poly_coeffs[i]) > 1e-14])
        
        # Align levels before multiplication to prevent scale mismatch
        x_enc_aligned, poly_result_aligned = self._align_levels(x_enc, poly_result)
        
        # Multiply by x to get final result
        result = self.multiply(x_enc_aligned, poly_result_aligned)
        
        # Additional rescale if needed for nibble extraction
        if self.has_rescale and hasattr(result, 'scale'):
            if getattr(result, 'scale', self.scale) > self.scale * 2:
                if hasattr(self.engine, 'rescale_to_next'):
                    result = self.engine.rescale_to_next(result)
                elif hasattr(self.engine, 'rescale'):
                    result = self.engine.rescale(result)
        
        final_level = getattr(result, 'level', 0)
        logger.info(f"Level consumption: {initial_level - final_level} (initial: {initial_level}, final: {final_level})")
        
        return result
    
    def _evaluate_lower_nibble_han_ki(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        """Lower nibble extraction - typically just x^16."""
        logger.info("=== Lower Nibble Extraction ===")
        initial_level = getattr(x_enc, 'level', 25)
        
        # Lower nibble is x^16 with appropriate coefficient
        result = self._compute_power_binary(x_enc, 16)
        
        # Apply coefficient if not 1
        if 16 < len(coefficients) and abs(coefficients[16] - 1.0) > 1e-14:
            result = self.multiply_plain(result, coefficients[16])
        
        final_level = getattr(result, 'level', 0)
        logger.info(f"Level consumption: {initial_level - final_level} (initial: {initial_level}, final: {final_level})")
        
        return result
    
    # === XOR Operations ===
    
    def apply_xor(self, x_enc: Any, y_enc: Any, bits: int = 8) -> Any:
        """Apply XOR operation on encrypted values."""
        if bits == 4:
            return self.apply_nibble_xor(x_enc, y_enc)
        elif bits == 8:
            return self.apply_byte_xor_via_nibbles(x_enc, y_enc)
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
    
    def _pack_nibbles_for_xor(self, state_enc_8bit: Any, key_enc_8bit: Any) -> Any:
        """Pack state and key nibbles into single ciphertext for paper-style XOR.
        
        IMPORTANT: This expects inputs already encoded in 8-bit space:
        - state_enc_8bit: ζ₂₅₆^(16s) where s is the state nibble
        - key_enc_8bit: ζ₂₅₆^k where k is the key nibble
        
        Result: packed = ζ₂₅₆^(16s) * ζ₂₅₆^k = ζ₂₅₆^(16s + k)
        
        This multiplication in ciphertext space adds the exponents,
        giving us the correct input for the 8→4 LUT.
        """
        # Align levels before multiplication
        state_aligned, key_aligned = self._align_levels(state_enc_8bit, key_enc_8bit)
        
        # Multiply to add exponents: ζ₂₅₆^(16s) * ζ₂₅₆^k = ζ₂₅₆^(16s + k)
        packed = self.multiply(state_aligned, key_aligned)
        
        return packed
    
    def apply_nibble_xor_paper(self, state_nibbles: np.ndarray, key_nibbles: np.ndarray, 
                               pre_encoded: bool = False) -> Any:
        """Apply 4-bit XOR using paper's single-variable approach.
        
        Args:
            state_nibbles: Either raw nibble values (0-15) or pre-encoded ciphertext
            key_nibbles: Either raw nibble values (0-15) or pre-encoded ciphertext
            pre_encoded: If True, inputs are already 8-bit encoded ciphertexts
        
        This matches the paper by:
        1. Encoding nibbles in 8-bit space (state shifted by 16)
        2. Packing via multiplication (adds exponents)
        3. Using a single 8→4 LUT evaluation
        4. Achieving lower depth and noise (only 1 CT-CT multiplication)
        """
        logger.info("=== Applying nibble XOR (paper approach) ===")
        
        if not pre_encoded:
            # Encode state nibbles as ζ₂₅₆^(16s)
            state_shifted = state_nibbles * 16  # Shift to upper 4 bits
            state_enc_8bit = self.encrypt(self.encode_array(state_shifted, 8), level=20)
            
            # Encode key nibbles as ζ₂₅₆^k
            key_enc_8bit = self.encrypt(self.encode_array(key_nibbles, 8), level=20)
        else:
            state_enc_8bit = state_nibbles
            key_enc_8bit = key_nibbles
        
        # Pack nibbles into single ciphertext via multiplication
        packed = self._pack_nibbles_for_xor(state_enc_8bit, key_enc_8bit)
        
        # Get 8→4 XOR coefficients
        coeffs = self._get_xor_coeffs_1d_paper()
        
        # Evaluate single-variable polynomial
        result = self.evaluate_polynomial_bsgs(packed, coeffs)
        
        return result
    
    def apply_nibble_xor(self, x_enc: Any, y_enc: Any, use_paper_method: bool = True) -> Any:
        """Apply 4-bit XOR.
        
        By default uses the paper's single-variable approach for better performance.
        Falls back to 2D approach if paper method is disabled or unavailable.
        
        Note: For paper method, inputs should be raw nibble arrays or properly
        8-bit encoded ciphertexts.
        """
        logger.info("=== Applying nibble XOR ===")
        
        # Use paper approach if available and enabled
        if use_paper_method and hasattr(self, '_get_xor_coeffs_1d_paper'):
            # For backward compatibility, detect if inputs are ciphertexts
            # If so, we need to decrypt and re-encode in 8-bit space
            if hasattr(x_enc, 'level'):  # It's a ciphertext
                logger.warning("Paper XOR method requires 8-bit encoding; re-encoding inputs")
                # Decrypt to get nibble values
                x_dec = self.decrypt(x_enc)[:self.max_slots]
                y_dec = self.decrypt(y_enc)[:self.max_slots]
                x_nibbles = self.decode_array(x_dec, 4)
                y_nibbles = self.decode_array(y_dec, 4)
                # Use paper method with raw arrays
                return self.apply_nibble_xor_paper(x_nibbles, y_nibbles, pre_encoded=False)
            else:
                # Assume they're numpy arrays
                return self.apply_nibble_xor_paper(x_enc, y_enc, pre_encoded=False)
        
        # Fallback to 2D approach
        coeffs_2d = self._get_xor_coeffs_2d(4)
        return self._evaluate_bivariate_standard(x_enc, y_enc, coeffs_2d)
    
    def _is_xor_even_poly(self, coeffs_2d: np.ndarray) -> bool:
        """Check if XOR polynomial has even symmetry.
        
        XOR polynomials over GF(2^n) often exhibit specific symmetries:
        1. Even symmetry: f(x,y) = f(-x,-y) (all odd-degree terms are zero)
        2. For 4-bit XOR: Special structure due to the small field
        """
        n, m = coeffs_2d.shape
        
        # Analyze coefficient structure
        odd_terms = 0
        even_terms = 0
        total_nonzero = 0
        
        for i in range(n):
            for j in range(m):
                if abs(coeffs_2d[i, j]) > 1e-14:
                    total_nonzero += 1
                    total_degree = i + j
                    if total_degree % 2 == 1:
                        odd_terms += 1
                    else:
                        even_terms += 1
        
        # XOR typically has even symmetry only if there are even terms but no odd terms
        # For 4-bit XOR, the polynomial has non-zero coefficients only when (i+j) is odd,
        # so it actually doesn't have even symmetry in the traditional sense
        has_even_symmetry = (odd_terms == 0 and even_terms > 0)
        
        if has_even_symmetry:
            logger.info(f"XOR polynomial has even symmetry - {even_terms} even terms, "
                       f"0 odd terms out of {total_nonzero} total")
        else:
            logger.info(f"XOR polynomial lacks even symmetry - {even_terms} even terms, "
                       f"{odd_terms} odd terms out of {total_nonzero} total")
        
        return has_even_symmetry
    
    def _evaluate_bivariate_standard(self, x_enc: Any, y_enc: Any, coeffs_2d: np.ndarray) -> Any:
        """Standard bivariate polynomial evaluation with sparsity optimization.
        
        XOR polynomials have high sparsity, especially at conjugate positions.
        This implementation leverages that sparsity to reduce computations.
        """
        result = None
        
        # Create sparsity mask for non-zero coefficients
        non_zero_mask = np.abs(coeffs_2d) > 1e-14
        non_zero_indices = np.argwhere(non_zero_mask)
        
        logger.info(f"XOR polynomial sparsity: {len(non_zero_indices)}/{coeffs_2d.size} non-zero terms ({100*len(non_zero_indices)/coeffs_2d.size:.1f}%)")
        
        # Precompute small powers to reuse
        x_powers = {}
        y_powers = {}
        
        # Always need the constant term
        x_powers[0] = self.encrypt(np.ones(self.max_slots), level=getattr(x_enc, 'level', 25))
        y_powers[0] = x_powers[0]
        x_powers[1] = x_enc
        y_powers[1] = y_enc
        
        # Evaluate only non-zero monomials
        for idx_i, idx_j in non_zero_indices:
            coeff = coeffs_2d[idx_i, idx_j]
            
            # Compute powers as needed (with caching)
            if idx_i not in x_powers:
                x_powers[idx_i] = self._compute_power_binary(x_enc, idx_i)
            if idx_j not in y_powers:
                y_powers[idx_j] = self._compute_power_binary(y_enc, idx_j)
            
            # Compute monomial x^i * y^j * coeff
            if idx_i == 0 and idx_j == 0:
                # Constant term
                term = self.multiply_plain(x_powers[0], coeff)
            elif idx_i == 0:
                # Only y power
                term = self.multiply_plain(y_powers[idx_j], coeff)
            elif idx_j == 0:
                # Only x power
                term = self.multiply_plain(x_powers[idx_i], coeff)
            else:
                # Both x and y powers
                term = self.multiply(x_powers[idx_i], y_powers[idx_j])
                term = self.multiply_plain(term, coeff)
            
            result = term if result is None else self.add(result, term)
        
        # Ensure we never return None - return zero ciphertext if all coefficients were zero
        if result is None:
            logger.warning("All coefficients were zero in bivariate evaluation - returning zero ciphertext")
            zero_pt = self.encrypt(np.zeros(self.max_slots, dtype=np.complex128),
                                 level=getattr(x_enc, 'level', 25))
            return zero_pt
        
        return result
    
    def _evaluate_xor_even_poly(self, x_enc: Any, y_enc: Any, coeffs_2d: np.ndarray) -> Any:
        """Evaluate XOR polynomial using even polynomial optimization.
        
        For even polynomials f(x,y) = g(x²,y²), we can reduce depth by half.
        This is based on Cheon et al. (SPC '20) optimization.
        """
        logger.info("=== Even Polynomial Optimization Active ===")
        initial_x_level = getattr(x_enc, 'level', 25)
        initial_y_level = getattr(y_enc, 'level', 25)
        
        # Compute x² and y²
        x_squared = self.multiply(x_enc, x_enc)
        y_squared = self.multiply(y_enc, y_enc)
        
        # Extract even-indexed coefficients for g(u,v) where u=x², v=y²
        n, m = coeffs_2d.shape
        n_even = (n + 1) // 2
        m_even = (m + 1) // 2
        even_coeffs = np.zeros((n_even, m_even), dtype=np.complex128)
        
        for i in range(n_even):
            for j in range(m_even):
                if 2*i < n and 2*j < m:
                    even_coeffs[i, j] = coeffs_2d[2*i, 2*j]
        
        # Check if even_coeffs has any non-zero terms
        if np.all(np.abs(even_coeffs) < 1e-12):
            logger.warning("Even coefficients are all zero - falling back to standard evaluation")
            # This happens for 4-bit XOR where non-zero coeffs are at odd (i+j) positions
            return self._evaluate_bivariate_standard(x_enc, y_enc, coeffs_2d)
        
        # Evaluate g(x², y²) with reduced degree
        result = self._evaluate_bivariate_standard(x_squared, y_squared, even_coeffs)
        
        final_level = getattr(result, 'level', 0)
        logger.info(f"Even poly optimization - Level consumption: {initial_x_level - final_level} (depth reduced by ~50%)")
        
        return result
    
    def apply_byte_xor_via_nibbles(self, x_enc: Any, y_enc: Any) -> Tuple[Any, Any]:
        """Apply 8-bit XOR by extracting and XORing nibbles."""
        # Extract nibbles with optimized methods
        x_hi, x_lo = self.extract_nibbles(x_enc)
        y_hi, y_lo = self.extract_nibbles(y_enc)
        
        # XOR nibbles
        hi_xor = self.apply_nibble_xor(x_hi, y_hi)
        lo_xor = self.apply_nibble_xor(x_lo, y_lo)
        
        return hi_xor, lo_xor
    
    # === Coefficient Computation ===
    
    def _compute_coefficients(self, lut_function, input_bits: int, output_encoding_bits: int) -> np.ndarray:
        """Compute polynomial coefficients via FFT."""
        n = 1 << input_bits
        encoding = self.nibble_encoding if output_encoding_bits == 4 else self.byte_encoding
        root = encoding['root']
        
        # Create LUT
        lut = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            output_val = lut_function(i)
            lut[i] = root ** output_val
        
        # Compute FFT coefficients
        coeffs = np.fft.fft(lut) / n
        
        # Apply conjugate symmetry
        return self._apply_conjugate_symmetry(coeffs, n)
    
    def _apply_conjugate_symmetry(self, coeffs: np.ndarray, n: int) -> np.ndarray:
        """Apply conjugate symmetry constraint."""
        result = coeffs.copy()
        
        for k in range(1, n//2):
            avg = (result[k] + np.conj(result[n-k])) / 2
            result[k] = avg
            result[n-k] = np.conj(avg)
        
        if n % 2 == 0:
            result[n//2] = np.real(result[n//2])
        
        return result
    
    def _apply_conjugate_symmetry_2d(self, coeffs: np.ndarray) -> np.ndarray:
        """Apply conjugate symmetry constraint for 2D functions."""
        result = coeffs.copy()
        n, m = coeffs.shape
        
        for u in range(n):
            for v in range(m):
                u_neg = (-u) % n
                v_neg = (-v) % m
                
                if (u_neg < u) or (u_neg == u and v_neg < v):
                    continue
                
                if u == u_neg and v == v_neg:
                    result[u, v] = np.real(result[u, v])
                else:
                    avg = (result[u, v] + np.conj(result[u_neg, v_neg])) / 2
                    result[u, v] = avg
                    result[u_neg, v_neg] = np.conj(avg)
        
        return result
    
    def _get_nibble_coeffs_optimized(self, is_upper: bool) -> np.ndarray:
        """Return optimized coefficients for nibble extraction.

        * Han & Ki(’22) conjugate-reduction 적용
        * DC( k = 0 ) 항은 반드시 유지해 위상(평행이동) 오류를 방지
        """
        cache_key = f"{'upper' if is_upper else 'lower'}_nibble_optimized"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]

        # 1. LUT → FFT 로 원 계수 계산
        def nibble_func(byte_val: int) -> int:
            return (byte_val // 16) if is_upper else (byte_val % 16)

        coeffs = self._compute_coefficients(nibble_func, input_bits=8, output_encoding_bits=4)

        # 2. Han-&-Ki conjugate reduction
        if self.config.use_conjugate_reduction:
            reduced = np.zeros_like(coeffs)

            # ―― DC 항( k = 0 ) 보존 ――
           
            if is_upper:
                # 계수 k = 1 + 16·k’ 만 유지 → x · P(x¹⁶)
                for k in range(16):
                    idx = 1 + 16 * k
                    if idx < len(coeffs):
                        reduced[idx] = coeffs[idx]
            else:
                # 하위 니블: DC(k=0) 항과 k=16 항 모두 필요
                if abs(coeffs[0]) > 1e-14:
                    reduced[0] = coeffs[0]  # DC term for phase preservation
                reduced[16] = coeffs[16]    # Main x^16 term

            coeffs = reduced

        # 3. LRU 캐시에 저장 후 반환
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

        # ⚠️ **대칭 강제 삭제!**
        coeffs = scipy.fft.fftn(lut, norm='forward')

        self._cache_coefficient(cache_key, coeffs)
        return coeffs
    
    def _get_xor_coeffs_1d_paper(self) -> np.ndarray:
        """Get 8→4 single-variable XOR coefficients following the paper approach.
        
        Input: 8 bits (upper 4 bits = state nibble, lower 4 bits = key nibble)
        Output: 4 bits (state XOR key)
        
        This creates a 256-element LUT where LUT[i] = (i>>4) ^ (i&0xF)
        """
        cache_key = "xor_8to4_paper"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        
        logger.info("Computing 8→4 XOR LUT coefficients (paper approach)")
        
        # Create the 8→4 XOR LUT
        lut = np.zeros(256, dtype=np.complex128)
        root = self.nibble_encoding['root']  # 16th root of unity for 4-bit output
        
        for i in range(256):
            state_nibble = i >> 4  # Upper 4 bits
            key_nibble = i & 0x0F  # Lower 4 bits
            xor_result = state_nibble ^ key_nibble
            lut[i] = root ** xor_result
        
        # Compute FFT coefficients
        coeffs = np.fft.fft(lut)
        
        # Log sparsity
        non_zero = np.sum(np.abs(coeffs) > 1e-14)
        logger.info(f"8→4 XOR LUT has {non_zero}/256 non-zero coefficients ({non_zero/256*100:.1f}%)")
        
        self._cache_coefficient(cache_key, coeffs)
        return coeffs

    
    def _load_cached_coefficients(self):
        """Load cached coefficients from disk."""
        cache_dir = Path("xor_cache")
        if not cache_dir.exists():
            return
        
        # Load optimized coefficients
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
        """Save coefficients to disk."""
        if not self.config.cache_coefficients:
            return
        
        cache_dir = Path("xor_cache")
        cache_dir.mkdir(exist_ok=True)
        
        for key, coeffs in self._coeff_cache.items():
            file = cache_dir / f"{key}.npy"
            np.save(file, coeffs)
            logger.info(f"Saved coefficients to {file}")

# Utility functions
def decrypt_simd(xor_inst: XORPaperOptimized, hi_ct: Any, lo_ct: Any, length: int, 
                 strict_check: bool = False) -> np.ndarray:
    """Decrypt SIMD XOR result with optional strict checking."""
    # Check noise budget if available and in strict mode
    if strict_check and xor_inst.has_noise_budget:
        hi_budget = xor_inst.engine.invariant_noise_budget(hi_ct, xor_inst.secret_key)
        lo_budget = xor_inst.engine.invariant_noise_budget(lo_ct, xor_inst.secret_key)
        min_budget = min(hi_budget, lo_budget)
        
        if min_budget < xor_inst.config.noise_budget_threshold:
            warnings.warn(f"Low noise budget: {min_budget} bits remaining")
            logger.warning(f"Low noise budget: hi={hi_budget}, lo={lo_budget} bits")
    
    # Decrypt
    hi_dec = xor_inst.decrypt(hi_ct)[:length]
    lo_dec = xor_inst.decrypt(lo_ct)[:length]
    
    # Decode
    hi = xor_inst.decode_array(hi_dec, 4)
    lo = xor_inst.decode_array(lo_dec, 4)
    
    # Additional accuracy check in strict mode
    if strict_check:
        # Check if decoded values are valid nibbles
        if np.any(hi >= 16) or np.any(lo >= 16):
            warnings.warn("Decoded nibbles out of range")
            logger.warning(f"Invalid nibbles: hi_max={np.max(hi)}, lo_max={np.max(lo)}")
    
    return (hi << 4) | lo


