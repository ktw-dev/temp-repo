from copy import deepcopy
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Any, Sequence, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import scipy.fft
from enum import Enum
from abc import ABC, abstractmethod
import json
import os
from desilofhe import Engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Scalar = Union[int, float, complex]
@dataclass
class XORConfig:
    poly_modulus_degree: int = 16384
    precision_bits: int = 60
    chunk_size: int = 4
    use_nibbles: bool = True
    sparsify_threshold: float = 1e-10
    use_multivariate: bool = True
    cache_coefficients: bool = True
    thread_count: int = 256
    mode: str = "parallel"
    device_id: int = 0
class EngineWrapper:
    def __init__(self,
                 engine: Engine,
                 pk: Any = None,
                 sk: Any = None,
                 relin_key: Optional[Any] = None,
                 level_manager: Optional['LevelManager'] = None,
                 scale_manager: Optional['ScaleManager'] = None,
                 noise_manager: Optional['NoiseManager'] = None):
        self.engine = engine
        self.pk = pk
        self.sk = sk
        self.relin_key = relin_key
        self.level_manager = level_manager
        self.scale_manager = scale_manager
        self.noise_manager = noise_manager
        self._const_ct_cache: Dict[Tuple[complex, int], Any] = {}
        self._pt_cache: Dict[complex, Any] = {}
        self._hot_vals = {0.0, 1.0, -1.0}
    def encrypt(self, data: Union[np.ndarray, Any], level: Optional[int] = None) -> Any:
        if level is None:
            return self.engine.encrypt(data, self.pk)
        else:
            return self.engine.encrypt(data, self.pk, level=level)
    def decrypt(self, ciphertext: Any) -> np.ndarray:
        return self.engine.decrypt(ciphertext, self.sk)
    def _pt(self, value: Scalar) -> Any:
        val = complex(value)
        if val not in self._pt_cache:
            arr = np.full(self.engine.slot_count, val, dtype=np.complex128)
            self._pt_cache[val] = self.engine.encode(arr)
        return self._pt_cache[val]
    def _const_ct(self, value: Scalar, level: int) -> Any:
        key = (complex(value), level)
        if key in self._const_ct_cache:
            return self._const_ct_cache[key]
        for (v, lvl), ct in self._const_ct_cache.items():
            if v == complex(value) and lvl > level:
                ct2 = self.engine.mod_switch_to_level(ct, level)
                self._const_ct_cache[key] = ct2
                return ct2
        arr = np.full(self.engine.slot_count, value, dtype=np.complex128)
        ct = self.engine.encrypt(arr, self.pk, level=level)
        self._const_ct_cache[key] = ct
        return ct
    def multiply_plain(self,
                       ct: Any,
                       val: Union[Scalar, np.ndarray, Any]) -> Any:
        if isinstance(val, (int, float, complex)):
            pt = self._pt(val)
            if hasattr(self.engine, "multiply_plain"):
                return self.engine.multiply_plain(ct, pt)
            return self.engine.multiply(ct, pt)
        if isinstance(val, np.ndarray):
            if np.ptp(val) < 1e-12:
                return self.multiply_plain(ct, complex(val.flat[0]))
            pt = self.engine.encode(val.astype(np.complex128))
            if hasattr(self.engine, "multiply_plain"):
                return self.engine.multiply_plain(ct, pt)
            return self.engine.multiply(ct, pt)
        if hasattr(self.engine, "multiply_plain"):
            return self.engine.multiply_plain(ct, val)
        return self.engine.multiply(ct, val)
    def _get_scale(self, ct: Any):
        if hasattr(ct, "scale"):
            return ct.scale
        if hasattr(self.engine, "get_scale"):
            try:
                return self.engine.get_scale(ct)
            except Exception:
                pass
        return None
    def _match_scales(self, ct1: Any, ct2: Any, tol: float = 1e-3) -> Tuple[Any, Any]:
        s1, s2 = self._get_scale(ct1), self._get_scale(ct2)
        if s1 is None or s2 is None:
            return ct1, ct2
        if abs(s1 - s2) / max(s1, s2) < tol:
            return ct1, ct2
        big, small = (ct1, ct2) if s1 > s2 else (ct2, ct1)
        while True:
            prev_scale = self._get_scale(big)
            big = self.rescale(big)
            new_scale = self._get_scale(big)
            if new_scale is None:
                break
            if abs(new_scale - self._get_scale(small)) / new_scale < tol:
                break
            if new_scale >= prev_scale:
                break
        return (big, small) if s1 > s2 else (small, big)
    def clear_cache(self):
        self._const_ct_cache.clear()
        self._pt_cache.clear()
    def add(self, ct1: Any, ct2: Any) -> Any:
        ct1, ct2 = self._match_scales(ct1, ct2)
        result = self.engine.add(ct1, ct2)
        self._track_operation('add', result, ct1, ct2)
        return result
    def multiply(self, ct1: Any, ct2: Any, relin_key: Any = None) -> Any:
        ct1, ct2 = self._match_scales(ct1, ct2)
        min_level = min(getattr(ct1, 'level', 0), getattr(ct2, 'level', 0))
        if min_level < 1:
            raise ValueError(f"Insufficient levels for multiplication: {min_level}")
        rkey = relin_key or self.relin_key
        if rkey is None:
            result = self.engine.multiply(ct1, ct2)
        else:
            result = self.engine.multiply(ct1, ct2, rkey)
        self._track_operation('multiply', result, ct1, ct2)
        return result
    def _match_scales(self, ct1: Any, ct2: Any) -> Tuple[Any, Any]:
        if not hasattr(ct1, 'scale') or not hasattr(ct2, 'scale'):
            return ct1, ct2
        scale1, scale2 = ct1.scale, ct2.scale
        if abs(scale1 - scale2) / max(scale1, scale2) < 1e-10:
            return ct1, ct2
        if self.scale_manager:
            self.scale_manager.record_scale_change('scale_match', scale1, scale2)
        if scale1 > scale2:
            if hasattr(self.engine, 'rescale_to'):
                ct1 = self.engine.rescale_to(ct1, scale2)
            elif hasattr(self.engine, 'rescale'):
                ct1 = self.engine.rescale(ct1)
        else:
            if hasattr(self.engine, 'rescale_to'):
                ct2 = self.engine.rescale_to(ct2, scale1)
            elif hasattr(self.engine, 'rescale'):
                ct2 = self.engine.rescale(ct2)
        return ct1, ct2
    def _track_operation(self, op_name: str, result: Any, *operands):
        if self.level_manager and hasattr(result, 'level'):
            self.level_manager.reset(result.level)
        if self.scale_manager and hasattr(result, 'scale'):
            old_scale = operands[0].scale if hasattr(operands[0], 'scale') else 1.0
            new_scale = result.scale
            self.scale_manager.record_scale_change(op_name, old_scale, new_scale)
        if self.noise_manager:
            estimated_noise = self.noise_manager.estimate_noise(op_name, *operands)
            self.noise_manager.update_noise(result, estimated_noise)
    def __getattr__(self, name):
        return getattr(self.engine, name)
@dataclass
class FHEContext:
    public_key: Any
    secret_key: Any
    relinearization_key: Any
    conjugation_key: Optional[Any] = None
    rotation_keys: Optional[Dict[int, Any]] = None
    bootstrap_key: Optional[Any] = None
    slot_count: int = 16384
    max_level: int = 30
    scale: float = 2**40
    def has_rotation_key(self, steps: int) -> bool:
        return self.rotation_keys is not None and steps in self.rotation_keys
    def has_bootstrap_key(self) -> bool:
        return self.bootstrap_key is not None
    def get_rotation_key(self, steps: int) -> Any:
        if not self.has_rotation_key(steps):
            raise KeyError(f"Rotation key for {steps} steps not found")
        return self.rotation_keys[steps]
@dataclass
class RootOfUnityEncoding:
    n: int
    zeta: complex
    decode_threshold: float
    @classmethod
    def for_bits(cls, bits: int, *, threshold_scale: float = 1.0) -> "RootOfUnityEncoding":
        if bits <= 0:
            raise ValueError("bits must be positive")
        n = 1 << bits
        zeta = np.exp(2j * np.pi / n)
        threshold = (np.pi / n) * threshold_scale
        return cls(n=n, zeta=zeta, decode_threshold=threshold)
@dataclass
class CryptoParameters:
    block_size: int = 16
    key_size: int = 16
    num_rounds: int = 10
    nibble_bits: int = 4
    byte_bits: int = 8
    aes_modulus: int = 0x11B
    batch_size: int = 16
    slot_gap: int = 1024
    slot_count_hint: Optional[int] = None
    decode_guard_scale: float = 1.0
    _encodings: Dict[int, RootOfUnityEncoding] = field(default_factory=dict, init=False, repr=False)
    _gf_mul: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _gf_inv: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _sbox: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _inv_sbox: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _mix_mat: np.ndarray = field(default=None, init=False, repr=False)
    _inv_mix_mat: np.ndarray = field(default=None, init=False, repr=False)
    _shift_rows: Tuple[int, int, int, int] = (0, 1, 2, 3)
    _inv_shift_rows: Tuple[int, int, int, int] = (0, 3, 2, 1)
    
    def __post_init__(self) -> None:
        self._encodings[self.nibble_bits] = RootOfUnityEncoding.for_bits(
            self.nibble_bits, threshold_scale=self.decode_guard_scale)
        self._encodings[self.byte_bits] = RootOfUnityEncoding.for_bits(
            self.byte_bits, threshold_scale=self.decode_guard_scale)
        self._init_gf_tables()
        self._init_sboxes()
        self._init_mixcolumns_matrices()
    def get_encoding(self, bits: int) -> RootOfUnityEncoding:
        enc = self._encodings.get(bits)
        if enc is None:
            enc = RootOfUnityEncoding.for_bits(bits, threshold_scale=self.decode_guard_scale)
            self._encodings[bits] = enc
        return enc
    def encode_nibble(self, value: int) -> complex:
        if not 0 <= value < (1 << self.nibble_bits):
            raise ValueError(f"Nibble value must be 0..{(1<<self.nibble_bits)-1}, got {value}")
        enc = self.get_encoding(self.nibble_bits)
        return enc.zeta ** value
    def decode_nibble(self, encoded: complex, *, strict: bool = False) -> int:
        enc = self.get_encoding(self.nibble_bits)
        return self._decode_scalar(encoded, enc, strict=strict)
    def encode_byte(self, value: int) -> complex:
        if not 0 <= value < (1 << self.byte_bits):
            raise ValueError(f"Byte value must be 0..{(1<<self.byte_bits)-1}, got {value}")
        enc = self.get_encoding(self.byte_bits)
        return enc.zeta ** value
    def decode_byte(self, encoded: complex, *, strict: bool = False) -> int:
        enc = self.get_encoding(self.byte_bits)
        return self._decode_scalar(encoded, enc, strict=strict)
    def encode_scalar(self, value: int, bits: int) -> complex:
        enc = self.get_encoding(bits)
        return enc.zeta ** (value % enc.n)
    def decode_scalar(self, encoded: complex, bits: int, *, strict: bool = False) -> int:
        enc = self.get_encoding(bits)
        return self._decode_scalar(encoded, enc, strict=strict)
    def _decode_scalar(self, encoded: complex, enc: RootOfUnityEncoding, *, strict: bool) -> int:
        ang = float(np.angle(encoded)) % (2 * np.pi)
        idx = int(round(ang * enc.n / (2 * np.pi))) % enc.n
        if strict:
            target_ang = 2 * np.pi * idx / enc.n
            diff = abs(self._principal_angle(ang - target_ang))
            if diff > enc.decode_threshold:
                raise ValueError(f"Decode failure: angle diff {diff:.3e} exceeds threshold {enc.decode_threshold:.3e}")
        return idx
    @staticmethod
    def _principal_angle(theta: float) -> float:
        return (theta + np.pi) % (2 * np.pi) - np.pi
    def encode_array(self, values: Sequence[int], bits: int) -> np.ndarray:
        enc = self.get_encoding(bits)
        vals = np.asarray(values, dtype=np.int64) % enc.n
        ang = (2 * np.pi / enc.n) * vals
        return np.exp(1j * ang)
    def decode_array(self, encoded: Sequence[complex], bits: int, *, strict: bool = False) -> np.ndarray:
        enc = self.get_encoding(bits)
        arr = np.asarray(encoded, dtype=np.complex128)
        ang = np.angle(arr) % (2 * np.pi)
        idx = np.rint(ang * enc.n / (2 * np.pi)).astype(np.int64) % enc.n
        if strict:
            target_ang = (2 * np.pi / enc.n) * idx
            diff = np.abs(((ang - target_ang + np.pi) % (2 * np.pi)) - np.pi)
            bad = diff > enc.decode_threshold
            if np.any(bad):
                raise ValueError(f"Decode failure for {bad.sum()} element(s); max diff={diff.max():.3e} > {enc.decode_threshold:.3e}")
        return idx
    def encode_nibbles(self, values: Sequence[int]) -> np.ndarray:
        return self.encode_array(values, self.nibble_bits)
    def decode_nibbles(self, encoded: Sequence[complex], *, strict: bool = False) -> np.ndarray:
        return self.decode_array(encoded, self.nibble_bits, strict=strict)
    def encode_bytes(self, values: Sequence[int]) -> np.ndarray:
        return self.encode_array(values, self.byte_bits)
    def decode_bytes(self, encoded: Sequence[complex], *, strict: bool = False) -> np.ndarray:
        return self.decode_array(encoded, self.byte_bits, strict=strict)
    def _init_gf_tables(self) -> None:
        self._gf_mul = np.zeros((256, 256), dtype=np.uint8)
        for a in range(256):
            for b in range(256):
                self._gf_mul[a, b] = self._gf_multiply(a, b)
        self._gf_inv = np.zeros(256, dtype=np.uint8)
        self._gf_inv[0] = 0
        for i in range(1, 256):
            for j in range(1, 256):
                if self._gf_mul[i, j] == 1:
                    self._gf_inv[i] = j
                    break
    def _gf_multiply(self, a: int, b: int) -> int:
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            carry = a & 0x80
            a = (a << 1) & 0xFF
            if carry:
                a ^= self.aes_modulus & 0xFF
            b >>= 1
        return p
    def _init_sboxes(self) -> None:
        self._sbox = np.zeros(256, dtype=np.uint8)
        self._inv_sbox = np.zeros(256, dtype=np.uint8)
        affine_matrix = np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        affine_constant = 0x63
        for i in range(256):
            inv = self._gf_inv[i] if i != 0 else 0
            bits = np.array([(inv >> j) & 1 for j in range(8)], dtype=np.uint8)
            transformed = np.dot(affine_matrix, bits) % 2
            result = sum(bit << idx for idx, bit in enumerate(transformed))
            result ^= affine_constant
            self._sbox[i] = result
            self._inv_sbox[result] = i
    def _init_mixcolumns_matrices(self) -> None:
        self._mix_mat = np.array([
            [0x02, 0x03, 0x01, 0x01],
            [0x01, 0x02, 0x03, 0x01],
            [0x01, 0x01, 0x02, 0x03],
            [0x03, 0x01, 0x01, 0x02]
        ], dtype=np.uint8)
        self._inv_mix_mat = np.array([
            [0x0E, 0x0B, 0x0D, 0x09],
            [0x09, 0x0E, 0x0B, 0x0D],
            [0x0D, 0x09, 0x0E, 0x0B],
            [0x0B, 0x0D, 0x09, 0x0E]
        ], dtype=np.uint8)
    def gf_mul(self, a: int, b: int) -> int:
        return int(self._gf_mul[a, b])
    def gf_inv(self, a: int) -> int:
        return int(self._gf_inv[a])
    def sbox(self, val: int) -> int:
        return int(self._sbox[val])
    def inv_sbox(self, val: int) -> int:
        return int(self._inv_sbox[val])
class PackingStrategy(Enum):
    SEQUENTIAL = "sequential"
    INTERLEAVED = "interleaved"
    BLOCKED = "blocked"
    SPARSE = "sparse"
class SlotManager:
    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.allocated_slots = 0
        self.slot_map: Dict[str, Tuple[int, int]] = {}
    def allocate(self, name: str, count: int) -> Tuple[int, int]:
        if count > self.max_slots - self.allocated_slots:
            raise ValueError(f"Not enough slots: requested {count}, available {self.max_slots - self.allocated_slots}")
        start = self.allocated_slots
        self.allocated_slots += count
        self.slot_map[name] = (start, count)
        return start, count
    def pack_values(self, values: np.ndarray, 
                    strategy: PackingStrategy = PackingStrategy.SEQUENTIAL,
                    block_size: int = 16) -> np.ndarray:
        packed = np.zeros(self.max_slots, dtype=values.dtype)
        if strategy == PackingStrategy.SEQUENTIAL:
            n = min(len(values), self.max_slots)
            packed[:n] = values[:n]
        elif strategy == PackingStrategy.INTERLEAVED:
            n_values = len(values)
            n_groups = min(block_size, self.max_slots // n_values)
            for i in range(n_values):
                for j in range(n_groups):
                    idx = j * n_values + i
                    if idx < self.max_slots:
                        packed[idx] = values[i]
        elif strategy == PackingStrategy.BLOCKED:
            n_blocks = self.max_slots // block_size
            for i in range(min(len(values), n_blocks)):
                start = i * block_size
                packed[start:start+1] = values[i]
        elif strategy == PackingStrategy.SPARSE:
            gap = max(1, self.max_slots // len(values))
            for i, val in enumerate(values):
                if i * gap < self.max_slots:
                    packed[i * gap] = val
        return packed
    def unpack_values(self, packed: np.ndarray, count: int,
                      strategy: PackingStrategy = PackingStrategy.SEQUENTIAL,
                      block_size: int = 16) -> np.ndarray:
        if strategy == PackingStrategy.SEQUENTIAL:
            return packed[:count].copy()
        elif strategy == PackingStrategy.INTERLEAVED:
            values = []
            n_groups = self.max_slots // count
            for i in range(count):
                vals = [packed[j * count + i] for j in range(n_groups) 
                        if j * count + i < len(packed)]
                values.append(vals[0] if vals else 0)
            return np.array(values)
        elif strategy == PackingStrategy.BLOCKED:
            values = []
            for i in range(count):
                start = i * block_size
                if start < len(packed):
                    values.append(packed[start])
            return np.array(values)
        elif strategy == PackingStrategy.SPARSE:
            gap = max(1, self.max_slots // count)
            values = []
            for i in range(count):
                if i * gap < len(packed):
                    values.append(packed[i * gap])
            return np.array(values)
    def reset(self):
        self.allocated_slots = 0
        self.slot_map.clear()
class LevelManager:
    def __init__(self, max_level: int):
        self.max_level = max_level
        self.current_level = max_level
        self.level_history: List[Tuple[str, int]] = []
    def consume_level(self, operation: str, count: int = 1) -> int:
        if self.current_level < count:
            raise ValueError(f"Insufficient levels: need {count}, have {self.current_level}")
        self.current_level -= count
        self.level_history.append((operation, count))
        return self.current_level
    def check_level(self, required: int) -> bool:
        return self.current_level >= required
    def reset(self, level: Optional[int] = None):
        self.current_level = level if level is not None else self.max_level
        self.level_history.clear()
    def get_consumption_report(self) -> Dict[str, int]:
        report = {}
        for op, count in self.level_history:
            report[op] = report.get(op, 0) + count
        return report
class ScaleManager:
    def __init__(self, default_scale: float = 2**40):
        self.default_scale = default_scale
        self.scale_history: List[Tuple[str, float, float]] = []
    def match_scales(self, ct1: Any, ct2: Any) -> Tuple[Any, Any]:
        scale1 = getattr(ct1, 'scale', self.default_scale)
        scale2 = getattr(ct2, 'scale', self.default_scale)
        if abs(scale1 - scale2) < 1e-10:
            return ct1, ct2
        if scale1 > scale2:
            return ct1, ct2
        else:
            return ct1, ct2
    def record_scale_change(self, operation: str, before: float, after: float):
        self.scale_history.append((operation, before, after))
    def get_scale_report(self) -> List[Dict[str, Any]]:
        return [
            {"operation": op, "before": before, "after": after, "ratio": after/before}
            for op, before, after in self.scale_history
        ]
class NoiseManager:
    def __init__(self, noise_budget_bits: int = 60):
        self.noise_budget_bits = noise_budget_bits
        self.noise_estimates: Dict[int, float] = {}
        self.operation_noise: Dict[str, float] = {
            "add": 0.5,
            "multiply": 3.0,
            "rotate": 1.0,
            "conjugate": 1.0,
            "rescale": -20.0,
            "bootstrap": -50.0
        }
    def estimate_noise(self, operation: str, *operands) -> float:
        base_noise = max(self.noise_estimates.get(id(op), 0) for op in operands)
        return base_noise + self.operation_noise.get(operation, 0)
    def update_noise(self, ciphertext: Any, noise_bits: float):
        self.noise_estimates[id(ciphertext)] = noise_bits
    def needs_bootstrap(self, ciphertext: Any, threshold: float = 10.0) -> bool:
        noise = self.noise_estimates.get(id(ciphertext), 0)
        return noise > self.noise_budget_bits - threshold
    def clear(self):
        self.noise_estimates.clear()
class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        self.computation_times: Dict[str, float] = {}
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        return None
    def set(self, key: str, value: Any, computation_time: float = 0.0):
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_counts[lru_key]
            if lru_key in self.computation_times:
                del self.computation_times[lru_key]
        self.cache[key] = value
        self.access_counts[key] = 1
        if computation_time > 0:
            self.computation_times[key] = computation_time
    def clear(self):
        self.cache.clear()
        self.access_counts.clear()
        self.computation_times.clear()
    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "total_accesses": sum(self.access_counts.values()),
            "total_computation_time_saved": sum(
                self.computation_times.get(k, 0) * (self.access_counts.get(k, 1) - 1)
                for k in self.cache
            )
        }
class PerformanceMonitor:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.active_timers: Dict[str, float] = {}
    def start_timer(self, operation: str):
        self.active_timers[operation] = time.time()
    def end_timer(self, operation: str) -> float:
        if operation not in self.active_timers:
            return 0.0
        elapsed = time.time() - self.active_timers[operation]
        del self.active_timers[operation]
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(elapsed)
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        return elapsed
    def measure(self, operation: str):
        class TimerContext:
            def __init__(self, monitor, op):
                self.monitor = monitor
                self.op = op
            def __enter__(self):
                self.monitor.start_timer(self.op)
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.monitor.end_timer(self.op)
        return TimerContext(self, operation)
    def get_statistics(self) -> Dict[str, float]:
        stats = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    "count": len(times),
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
        return stats
    def reset(self):
        self.timings.clear()
        self.operation_counts.clear()
        self.active_timers.clear()
class EnhancedCoefficientComputerV2:
    def __init__(self, max_slots: int, scale_factor: float = 2**40, config: Optional[XORConfig] = None):
        self.max_slots = max_slots
        self.scale_factor = scale_factor
        self.config = config or XORConfig()
        self._coefficient_cache = {}
    def get_xor_coefficients(self, bits: int) -> np.ndarray:
        cache_key = f"xor_{bits}"
        if cache_key in self._coefficient_cache:
            return self._coefficient_cache[cache_key]
        coeffs = self._compute_xor_coefficients(bits)
        self._coefficient_cache[cache_key] = coeffs
        return coeffs
    def _compute_xor_coefficients(self, bits: int) -> np.ndarray:
        if bits == 4:
            n = 16
            lut = np.zeros(n * n, dtype=np.complex128)
            for x in range(n):
                for y in range(n):
                    idx = x * n + y
                    xor_val = x ^ y
                    lut[idx] = np.exp(2j * np.pi * xor_val / n)
            coeffs = np.fft.fft(lut) / (n * n)
            return coeffs
        else:
            n = 1 << bits
            lut = np.zeros(n * n, dtype=np.complex128)
            for x in range(n):
                for y in range(n):
                    idx = x * n + y
                    xor_val = x ^ y
                    lut[idx] = np.exp(2j * np.pi * xor_val / n)
            coeffs = np.fft.fft(lut) / (n * n)
            return coeffs
    def _compute_low_degree_xor_coeffs(self, bits: int, max_degree: int) -> np.ndarray:
        n = 2**bits
        num_samples = min(n * n, 1000)
        indices = np.random.choice(n * n, num_samples, replace=False)
        X = []
        y = []
        for idx in indices:
            x_val = idx // n
            y_val = idx % n
            xor_val = x_val ^ y_val
            features = []
            combined = x_val * n + y_val
            for deg in range(max_degree + 1):
                features.append(combined ** deg)
            X.append(features)
            y.append(np.exp(2j * np.pi * xor_val / n))
        X = np.array(X)
        y = np.array(y)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        full_coeffs = np.zeros(n * n, dtype=np.complex128)
        full_coeffs[:len(coeffs)] = coeffs
        return full_coeffs
    def _compute_standard_xor_coeffs(self, bits: int) -> np.ndarray:
        n = 2**bits
        lut_size = n * n
        lut = np.zeros(lut_size, dtype=np.complex128)
        for x in range(n):
            for y in range(n):
                idx = x * n + y
                xor_val = x ^ y
                lut[idx] = np.exp(2j * np.pi * xor_val / n)
        coeffs = scipy.fft.fft(lut) / lut_size
        if bits == 4:
            max_degree = 64
            coeffs = coeffs[:max_degree]
        return coeffs
    def _get_nibble_xor_coefficients_2d(self):
        cache_key = "nibble_xor_2d"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        n = 16
        lut = np.fromfunction(
            lambda i, j: np.exp(2j*np.pi*((i.astype(int) ^ j.astype(int)))/n),
            (n, n),
            dtype=int
        )
        coeffs = np.fft.fft2(lut)
        self._coeff_cache[cache_key] = coeffs
        self._save_cached_coefficients()
        return coeffs
    def sparsify_coefficients(self, coeffs: np.ndarray,
                            threshold: float = 1e-10) -> np.ndarray:
        base_mask = np.abs(coeffs) >= threshold
        if coeffs.ndim >= 2:
            rows, cols = np.indices(coeffs.shape)
            even_rule = (rows % 2 == 0) & np.isin(cols, [2, 6, 10, 14])
            odd_rule  = (rows % 2 == 1) & (cols % 2 == 1)
            keep = base_mask & (even_rule | odd_rule)
        else:
            keep = base_mask
        return np.where(keep, coeffs, 0)
class IMonomialBasisManager(ABC):
    @abstractmethod
    def compute_basis(self, x_enc: Any, max_degree: int) -> List[Any]:
        pass
class MonomialBasisManager(IMonomialBasisManager):
    def __init__(self, engine: EngineWrapper, context: FHEContext):
        self.engine_wrapper = engine
        self.engine = engine.engine if hasattr(engine, 'engine') else engine
        self.context = context
        self._basis_cache = {}
    def compute_basis(self, x_enc: Any, max_degree: int) -> List[Any]:
        cache_key = (id(x_enc), max_degree)
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]
        if hasattr(self.engine_wrapper, 'level_manager') and hasattr(x_enc, 'level'):
            self.engine_wrapper.level_manager.reset(x_enc.level)
        basis = [None] * (max_degree + 1)
        basis[0] = self.engine.encrypt(
            np.ones(self.engine.slot_count, dtype=np.complex128),
            self.context.public_key,
            level=x_enc.level if hasattr(x_enc, 'level') else 25
        )
        basis[1] = x_enc
        powers_computed = {0: basis[0], 1: basis[1]}
        for deg in range(2, max_degree + 1):
            if deg % 2 == 0:
                half = deg // 2
                if half in powers_computed:
                    basis[deg] = self.engine_wrapper.multiply(
                        powers_computed[half],
                        powers_computed[half],
                        self.context.relinearization_key
                    )
                else:
                    basis[half] = self._compute_power(x_enc, half, powers_computed)
                    powers_computed[half] = basis[half]
                    basis[deg] = self.engine_wrapper.multiply(
                        basis[half],
                        basis[half],
                        self.context.relinearization_key
                    )
            else:
                basis[deg] = self.engine_wrapper.multiply(
                    basis[deg-1] if basis[deg-1] is not None else self._compute_power(x_enc, deg-1, powers_computed),
                    x_enc,
                    self.context.relinearization_key
                )
            powers_computed[deg] = basis[deg]
        self._basis_cache[cache_key] = basis
        return basis
    def _compute_power(self, x_enc: Any, degree: int, computed: Dict[int, Any]) -> Any:
        if degree in computed:
            return computed[degree]
        result = None
        temp = x_enc
        deg = degree
        while deg > 0:
            if deg & 1:
                if result is None:
                    result = temp
                else:
                    result = self.engine_wrapper.multiply(
                        result, temp, self.context.relinearization_key
                    )
            temp = self.engine_wrapper.multiply(
                temp, temp, self.context.relinearization_key
            )
            deg >>= 1
        return result
    def compute_basis_with_conjugate(self, x_enc: Any, max_degree: int) -> List[Any]:
        if not self.context.conjugation_key:
            return self.compute_basis(x_enc, max_degree)
        basis = self.compute_basis(x_enc, max_degree // 2)
        extended_basis = basis[:]
        for i in range(len(basis), max_degree + 1):
            conj_idx = i - len(basis)
            if conj_idx < len(basis):
                extended_basis.append(
                    self.engine_wrapper.conjugate(basis[conj_idx])
                )
        return extended_basis
class ILUTEvaluator(ABC):
    @abstractmethod
    def evaluate_lut(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        pass
class EnhancedLUTEvaluator(ILUTEvaluator):
    def __init__(self, engine: EngineWrapper, monomial_manager: IMonomialBasisManager):
        self.engine_wrapper = engine
        self.engine = engine.engine if hasattr(engine, 'engine') else engine
        self.monomial_manager = monomial_manager
        self.perf_monitor = PerformanceMonitor()
        self.pk = engine.pk if hasattr(engine, 'pk') else None
        self.relin_key = engine.relin_key if hasattr(engine, 'relin_key') else None
    # =========================================
    # EnhancedLUTEvaluator  –  새 구현
    # =========================================
    def evaluate_lut(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        """
        • sparsify → 희소화
        • even-poly & conj-key 있으면 x² 폴리로 변환해 깊이½
        • 그외 Horner / baby-giant
        """
        with self.perf_monitor.measure("lut_evaluation"):

            # 0) 입력 확인
            if coefficients.size == 0:
                raise ValueError("No coefficients provided")

            # 1) 희소화
            
            coefficients = self.coeff_computer.sparsify_coefficients(
                    coefficients, self.config.sparsify_threshold)

            # 2) 짝수차수 + conj-key → 깊이 감소
            if (self.monomial_manager.context.conjugation_key
                and self._is_even_polynomial(coefficients)):
                logger.info("Even polynomial detected – applying conjugation optimizer")
                return self._evaluate_even_poly_conjugation(x_enc, coefficients)

            # 3) 깊이 체크 → baby-giant 여부
            max_degree = int(
                np.max(np.nonzero(np.abs(coefficients) > 1e-10))
                if np.any(np.abs(coefficients) > 1e-10) else 0)

            lvl = getattr(x_enc, "level", 25)
            if max_degree > lvl:
                logger.info(f"degree {max_degree} > level {lvl} → baby-giant")
                return self.lut_evaluator.evaluate_with_baby_giant_steps(
                    x_enc, coefficients,
                    baby_steps=min(16, int(np.sqrt(max_degree))+1))

            # 4) Horner
            coeff_hi = np.full(self.engine.slot_count,
                            coefficients[max_degree], dtype=np.complex128)
            result = self.engine.encrypt(coeff_hi, self.pk, level=lvl)

            for deg in range(max_degree-1, -1, -1):
                if getattr(result, "level", 0) < 1:
                    logger.warning(f"level exhausted at deg {deg}")
                    break
                result = self.engine_wrapper.multiply(result, x_enc, self.relin_key)
                if abs(coefficients[deg]) > 1e-10:
                    coeff_plain = self.engine.encode(
                        np.full(self.engine.slot_count, coefficients[deg],
                                dtype=np.complex128))
                    result = self.engine_wrapper.add(result, coeff_plain)
            return result


    def _evaluate_even_poly_conjugation(self, x_enc: Any,
                                        coeffs: np.ndarray) -> Any:
        """p(x)=Σa₂k x²k  ⇒  g(t)=Σa₂k tᵏ ,  t=x² 로 깊이½"""
        g_coeffs = coeffs[::2].copy()          # 짝수항만
        x_sq = self.engine_wrapper.multiply(x_enc, x_enc, self.relin_key)

        # conj-opt 재귀 무한루프 방지용 플래그
        saved_key = self.monomial_manager.context.conjugation_key
        self.monomial_manager.context.conjugation_key = None
        try:
            return self.evaluate_lut(x_sq, g_coeffs)
        finally:
            self.monomial_manager.context.conjugation_key = saved_key


    def evaluate_multivariate_lut(self, inputs: List[Any],
                                coefficients: np.ndarray,
                                var_degrees: List[int]) -> Any:
  
        with self.perf_monitor.measure("multivariate_lut"):

            if len(inputs) != len(var_degrees):
                raise ValueError("inputs vs var_degrees length mismatch")

            
            coefficients = self.coeff_computer.sparsify_coefficients(
                    coefficients, self.config.sparsify_threshold)

            bases = [ self.monomial_manager.compute_basis(x, d)
                    for x, d in zip(inputs, var_degrees) ]

            result = None
            coeff_idx = 0
            for degrees in self._iterate_degrees(var_degrees):
                if coeff_idx >= len(coefficients):
                    break
                coeff = coefficients[coeff_idx]
                if abs(coeff) > 1e-10:
                    term = None
                    for var_idx, deg in enumerate(degrees):
                        term = (bases[var_idx][deg] if term is None else
                                self.engine_wrapper.multiply(
                                    term, bases[var_idx][deg], self.relin_key))
                    term = self.engine_wrapper.multiply_plain(term, coeff)
                    result = term if result is None else self.engine_wrapper.add(result, term)
                coeff_idx += 1
            return result

    def _iterate_degrees(self, var_degrees: List[int]):
        import itertools
        ranges = [range(deg + 1) for deg in var_degrees]
        return itertools.product(*ranges)
    def evaluate_with_baby_giant_steps(self, x_enc: Any, coefficients: np.ndarray,
                                       baby_steps: int = 16) -> Any:
        with self.perf_monitor.measure("baby_giant_steps"):
            n = len(coefficients)
            max_degree = 0
            for i in range(n):
                if abs(coefficients[i]) > 1e-10:
                    max_degree = i
            if max_degree == 0:
                input_level = x_enc.level if hasattr(x_enc, 'level') else 25
                coeff_array = np.full(self.engine.slot_count, coefficients[0], dtype=np.complex128)
                return self.engine.encrypt(coeff_array, self.pk, level=input_level)
            baby_steps = min(baby_steps, int(np.sqrt(max_degree)) + 1)
            giant_steps = (max_degree + baby_steps) // baby_steps
            logger.info(f"Baby-Giant evaluation: degree={max_degree}, baby={baby_steps}, giant={giant_steps}")
            logger.info(f"Depth: baby={int(np.log2(baby_steps))}, giant={int(np.log2(giant_steps))}, total={int(np.log2(baby_steps)) + int(np.log2(giant_steps))}")
            baby_powers = self.monomial_manager.compute_basis(x_enc, baby_steps)
            x_baby_steps = baby_powers[baby_steps]
            giant_steps  = (max_degree + baby_steps) // baby_steps
            giant_powers = [None]*giant_steps
            giant_powers[0] = self.engine.encrypt(
                np.ones(self.engine.slot_count, dtype=np.complex128),
                self.pk, level=x_enc.level)
            if giant_steps > 1:
                giant_basis = self.monomial_manager.compute_basis(
                    x_baby_steps, giant_steps-1)
                for g in range(1, giant_steps):
                    giant_powers[g] = giant_basis[g]
            result = None
            for giant_idx in range(giant_steps):
                baby_result = None
                for baby_idx in range(baby_steps):
                    coeff_idx = giant_idx * baby_steps + baby_idx
                    if coeff_idx > max_degree:
                        break
                    if abs(coefficients[coeff_idx]) > 1e-10:
                        coeff_array = np.full(self.engine.slot_count, coefficients[coeff_idx], dtype=np.complex128)
                        if baby_idx == 0 and giant_idx == 0:
                            term = self.engine.encrypt(coeff_array, self.pk, level=x_enc.level if hasattr(x_enc, 'level') else 25)
                        else:
                            coeff_plain = self.engine.encode(coeff_array)
                            term = self.engine_wrapper.multiply_plain(baby_powers[baby_idx], coeff_plain)
                        if baby_result is None:
                            baby_result = term
                        else:
                            baby_result = self.engine_wrapper.add(baby_result, term)
                if baby_result is not None and giant_idx > 0:
                    baby_result = self.engine_wrapper.multiply(
                        baby_result, giant_powers[giant_idx], self.relin_key
                    )
                if baby_result is not None:
                    if result is None:
                        result = baby_result
                    else:
                        result = self.engine_wrapper.add(result, baby_result)
            return result
    def _compute_power(self, x_enc: Any, degree: int) -> Any:
        if degree == 0:
            return self.engine.encrypt(
                np.ones(self.engine.slot_count, dtype=np.complex128),
                self.pk,
                level=x_enc.level if hasattr(x_enc, 'level') else 25
            )
        result = x_enc
        temp = x_enc
        deg = degree - 1
        while deg > 0:
            if deg & 1:
                result = self.engine_wrapper.multiply(result, temp, self.relin_key)
            if deg > 1:
                temp = self.engine_wrapper.multiply(temp, temp, self.relin_key)
            deg >>= 1
        return result
class SlotwiseLUTEvaluator(ILUTEvaluator):
    def __init__(self, engine: EngineWrapper, monomial_manager: IMonomialBasisManager):
        self.engine_wrapper = engine
        self.engine = engine.engine if hasattr(engine, 'engine') else engine
        self.monomial_manager = monomial_manager
        self.pk = engine.pk if hasattr(engine, 'pk') else None
        self.relin_key = engine.relin_key if hasattr(engine, 'relin_key') else None
    def evaluate_lut(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        return self._horner_evaluation(x_enc, coefficients)
    def _horner_evaluation(self, x_enc: Any, coefficients: np.ndarray) -> Any:
        input_level = x_enc.level if hasattr(x_enc, 'level') else 25
        coeff_array = np.full(self.engine.slot_count, coefficients[-1], dtype=np.complex128)
        result = self.engine.encrypt(coeff_array, self.engine.pk, level=input_level)
        for i in range(len(coefficients) - 2, -1, -1):
            result = self.engine.multiply(result, x_enc, self.engine.relin_key)
            if coefficients[i] != 0:
                coeff_plain = self.engine.encode(
                    np.full(self.engine.slot_count, coefficients[i])
                )
                result = self.engine_wrapper.add(result, coeff_plain)
        return result
    def evaluate_multiple_luts(self, x_enc: Any, 
                             coefficient_sets: List[np.ndarray],
                             slot_allocation: Dict[int, Tuple[int, int]]) -> List[Any]:
        max_degree = max(len(coeffs) - 1 for coeffs in coefficient_sets)
        basis = self.monomial_manager.compute_basis(x_enc, max_degree)
        results = []
        for lut_idx, coeffs in enumerate(coefficient_sets):
            if lut_idx not in slot_allocation:
                continue
            start_slot, num_slots = slot_allocation[lut_idx]
            result = None
            for deg, coeff in enumerate(coeffs):
                if abs(coeff) < 1e-10:
                    continue
                coeff_array = np.zeros(self.engine.slot_count, dtype=np.complex128)
                coeff_array[start_slot:start_slot + num_slots] = coeff
                term = self.engine.multiply_plain(basis[deg], coeff_array)
                if result is None:
                    result = term
                else:
                    result = self.engine.add(result, term)
            results.append(result)
        return results
    def evaluate_packed_luts(self, x_enc: Any,
                            coefficient_matrix: np.ndarray,
                            packing_info: Dict[str, Any]) -> Any:
        num_luts, max_degree = coefficient_matrix.shape
        basis = self.monomial_manager.compute_basis(x_enc, max_degree - 1)
        input_level = x_enc.level if hasattr(x_enc, 'level') else 25
        zero_array = np.zeros(self.engine.slot_count, dtype=np.complex128)
        result = self.engine.encrypt(zero_array, self.engine.pk, level=input_level)
        for deg in range(max_degree):
            packed_coeffs = np.zeros(self.engine.slot_count, dtype=np.complex128)
            for lut_idx in range(num_luts):
                slots = packing_info.get(f"lut_{lut_idx}", {})
                start = slots.get("start", lut_idx * 1024)
                size = slots.get("size", 1)
                packed_coeffs[start:start + size] = coefficient_matrix[lut_idx, deg]
            if np.any(np.abs(packed_coeffs) > 1e-10):
                term = self.engine.multiply_plain(basis[deg], packed_coeffs)
                result = self.engine.add(result, term)
        return result
class XORLUTEvaluator:
    def __init__(self, config: XORConfig):
        self.config = config
        self.engine = Engine(
            mode = "cpu"
        )
        logger.info("Creating FHE keys...")
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relin_key = self.engine.create_relinearization_key(self.secret_key)
        self.conj_key = self.engine.create_conjugation_key(self.secret_key)
        self.rotation_keys = {}
        for delta in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            if delta < self.engine.slot_count:
                self.rotation_keys[delta] = self.engine.create_fixed_rotation_key(
                    self.secret_key, delta
                )
        self.slot_manager = SlotManager(max_slots=self.engine.slot_count)
        self.level_manager = LevelManager(max_level=30)
        self.scale_manager = ScaleManager(default_scale=2**config.precision_bits)
        self.noise_manager = NoiseManager()
        self.cache_manager = CacheManager()
        self.perf_monitor = PerformanceMonitor()
        self.engine_wrapper = EngineWrapper(
            self.engine, self.public_key, self.secret_key, self.relin_key,
            level_manager=self.level_manager,
            scale_manager=self.scale_manager,
            noise_manager=self.noise_manager
        )
        self.params = CryptoParameters()
        self.coeff_computer = EnhancedCoefficientComputerV2(
            max_slots=self.engine.slot_count,
            scale_factor=2**config.precision_bits,
            config=config
        )
        self.fhe_context = FHEContext(
            public_key=self.public_key,
            secret_key=self.secret_key,
            relinearization_key=self.relin_key,
            conjugation_key=self.conj_key,
            rotation_keys=self.rotation_keys,
            slot_count=self.engine.slot_count,
            scale=2**config.precision_bits,
            max_level=30
        )
        self.monomial_manager = MonomialBasisManager(
            self.engine_wrapper,
            self.fhe_context
        )
        self.lut_evaluator = EnhancedLUTEvaluator(
            self.engine_wrapper,
            self.monomial_manager
        )
        self.slotwise_evaluator = SlotwiseLUTEvaluator(
            self.engine_wrapper,
            self.monomial_manager
        )
        self._coeff_cache = {}
        self._load_cached_coefficients()
        self.lut_evaluator = EnhancedLUTEvaluator(
            self.engine_wrapper,
            self.monomial_manager
        )

        # ▼ 이 두 줄 추가
        self.lut_evaluator.coeff_computer = self.coeff_computer
        self.lut_evaluator.config         = self.config
    def _load_cached_coefficients(self):
        cache_dir = Path("xor_cache")
        if not cache_dir.exists():
            return
        npy_files = list(cache_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                key = npy_file.stem
                coeffs = np.load(npy_file, allow_pickle=False)
                self._coeff_cache[key] = coeffs
                logger.info(f"Loaded {key} coefficients from {npy_file}")
            except Exception as e:
                logger.warning(f"Failed to load {npy_file}: {e}")
        if not self._coeff_cache:
            json_file = cache_dir / "xor_coefficients.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        cache_data = json.load(f)
                    for key, coeffs in cache_data.items():
                        if isinstance(coeffs, dict) and 'real' in coeffs and 'imag' in coeffs:
                            self._coeff_cache[key] = np.array(coeffs['real']) + 1j * np.array(coeffs['imag'])
                        else:
                            self._coeff_cache[key] = np.array(coeffs)
                    logger.info(f"Loaded cached XOR coefficients from {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to load JSON cache: {e}")
    def _save_cached_coefficients(self):
        if not self.config.cache_coefficients:
            return
        cache_dir = Path("xor_cache")
        cache_dir.mkdir(exist_ok=True)
        for key, coeffs in self._coeff_cache.items():
            npy_file = cache_dir / f"{key}.npy"
            try:
                np.save(npy_file, coeffs, allow_pickle=False)
                logger.info(f"Saved {key} coefficients to {npy_file}")
            except Exception as e:
                logger.warning(f"Failed to save {key} coefficients: {e}")
        json_file = cache_dir / "xor_coefficients.json"
        try:
            cache_data = {}
            for key, coeffs in self._coeff_cache.items():
                if np.iscomplexobj(coeffs):
                    cache_data[key] = {
                        'real': np.real(coeffs).tolist(),
                        'imag': np.imag(coeffs).tolist()
                    }
                else:
                    cache_data[key] = coeffs.tolist()
            with open(json_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save JSON backup: {e}")
    def _get_xor_coefficients(self, bits: int) -> np.ndarray:
        cache_key = f"xor_{bits}"
        if cache_key in self._coeff_cache:
            coeffs = self._coeff_cache[cache_key]
            logger.info(f"Loaded XOR coefficients for {bits} bits from cache, shape: {coeffs.shape}")
            return coeffs
        coeffs = self.coeff_computer.get_xor_coefficients(bits)
        logger.info(f"Computed XOR coefficients for {bits} bits, shape: {coeffs.shape}")
        if self.config.cache_coefficients:
            self._coeff_cache[cache_key] = coeffs
            self._save_cached_coefficients()
        return coeffs
    def _get_nibble_xor_coefficients_2d(self) -> np.ndarray:
        cache_key = "nibble_xor_2d"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        coeffs_2d = np.zeros((16, 16), dtype=np.complex128)
        for x in range(16):
            for y in range(16):
                xor_val = x ^ y
                coeffs_2d[x, y] = np.exp(2j * np.pi * xor_val / 16)
        coeffs_2d = scipy.fft.fft2(coeffs_2d) / (16 * 16)
        if self.config.cache_coefficients:
            self._coeff_cache[cache_key] = coeffs_2d
            self._save_cached_coefficients()
        return coeffs_2d
    def encode_for_xor(self, values: np.ndarray, bits: int = 8) -> Any:
        encoded = self.params.encode_array(values, bits)
        plaintext = self.engine.encode(encoded)
        return plaintext
    def apply_xor(self, x_enc, y_enc, bits=8):
        if bits == 4:
            return self.apply_4bit_xor_multivariate(x_enc, y_enc)
        if bits == 8 and self.config.use_nibbles:
            return self.apply_byte_xor_via_nibbles(x_enc, y_enc)
        return self.apply_direct_xor(x_enc, y_enc, bits)
    def apply_4bit_xor_multivariate(self, x_encrypted, y_encrypted):
        with self.perf_monitor.measure("4bit_xor_multivariate"):
            coeffs_2d = self._get_4bit_xor_coefficients_2d()
            coeffs_flat = coeffs_2d.reshape(-1)
            return self.lut_evaluator.evaluate_multivariate_lut(
                [x_encrypted, y_encrypted],
                coeffs_flat,
                [15, 15]
            )
    def _get_4bit_xor_coefficients_2d(self):
        cache_key = "xor_4bit_2d"
        if cache_key in self._coeff_cache:
            return self._coeff_cache[cache_key]
        n = 16
        lut = np.fromfunction(
            lambda i, j: np.exp(2j*np.pi*((i.astype(int) ^ j.astype(int)))/n),
            (n, n),
            dtype=int
        )
        coeffs = np.fft.fft2(lut)
        self._coeff_cache[cache_key] = coeffs
        self._save_cached_coefficients()
        return coeffs
    def _compute_xor_coefficients(self, bits: int) -> np.ndarray:
        if bits == 4:
            coeffs_2d = self._get_4bit_xor_coefficients_2d()
            return coeffs_2d.reshape(-1)
        elif bits == 8:
            n = 256
            lut = np.zeros(n * n, dtype=np.complex128)
            for x in range(n):
                for y in range(n):
                    idx = x * n + y
                    xor_val = x ^ y
                    lut[idx] = np.exp(2j * np.pi * xor_val / n)
            coeffs = np.fft.fft(lut) / (n * n)
            return coeffs
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
    def apply_byte_xor_via_nibbles(self, x_enc, y_enc):
        x_hi, x_lo = self.extract_nibbles(x_enc)
        y_hi, y_lo = self.extract_nibbles(y_enc)
        hi_xor = self.apply_nibble_xor_lut(x_hi, y_hi)
        lo_xor = self.apply_nibble_xor_lut(x_lo, y_lo)
        return hi_xor, lo_xor
    def apply_nibble_xor_lut(self, lhs_enc, rhs_enc):
            coeffs_2d = self._get_4bit_xor_coefficients_2d()
            coeffs_flat = coeffs_2d.reshape(-1)
            return self.lut_evaluator.evaluate_multivariate_lut(
                [lhs_enc, rhs_enc],
                coeffs_flat,
                [15, 15]
            )

    def apply_direct_xor(self, x_encrypted, y_encrypted, bits: int = 8):
        with self.perf_monitor.measure(f"direct_xor_{bits}bit"):
            coeffs = self._get_xor_coefficients(bits)
            n = 1 << bits
            x_powers = [None] * n
            x_powers[0] = self.engine.encrypt(
                np.ones(self.engine.slot_count, dtype=np.complex128),
                self.public_key,
                level=x_encrypted.level if hasattr(x_encrypted, 'level') else 30
            )
            x_powers[1] = x_encrypted
            for i in range(2, n):
                x_powers[i] = self.engine_wrapper.multiply(
                    x_powers[i-1], x_encrypted, self.relin_key
                )
            y_powers = [None] * n
            y_powers[0] = self.engine.encrypt(
                np.ones(self.engine.slot_count, dtype=np.complex128),
                self.public_key,
                level=y_encrypted.level if hasattr(y_encrypted, 'level') else 30
            )
            y_powers[1] = y_encrypted
            for i in range(2, n):
                y_powers[i] = self.engine_wrapper.multiply(
                    y_powers[i-1], y_encrypted, self.relin_key
                )
            result = None
            idx = 0
            for i in range(n):
                for j in range(n):
                    if abs(coeffs[idx]) > 1e-10:
                        term = self.engine_wrapper.multiply(
                            x_powers[i], y_powers[j], self.relin_key
                        )
                        coeff_array = np.full(self.engine.slot_count, coeffs[idx], dtype=np.complex128)
                        coeff_plain = self.engine.encode(coeff_array)
                        term = self.engine_wrapper.multiply_plain(term, coeff_plain)
                        if result is None:
                            result = term
                        else:
                            result = self.engine_wrapper.add(result, term)
                    idx += 1
            return result
    def evaluate_lut(self, x_encrypted, coefficients: np.ndarray):
        if self.config.sparsify_threshold > 0:
            coefficients = self.coeff_computer.sparsify_coefficients(
                coefficients, self.config.sparsify_threshold
            )
        max_degree = 0
        for i in range(len(coefficients)):
            if abs(coefficients[i]) > 1e-10:
                max_degree = i
        input_level = x_encrypted.level if hasattr(x_encrypted, 'level') else 25
        if max_degree > 16 and max_degree > input_level // 2:
            logger.info(f"Using baby-giant steps for degree {max_degree} polynomial")
            return self.lut_evaluator.evaluate_with_baby_giant_steps(
                x_encrypted, coefficients, baby_steps=min(16, int(np.sqrt(max_degree)) + 1)
            )
        logger.info(f"Evaluating polynomial of degree {max_degree} using monomial basis")
        if max_degree > 0:
            if max_degree > input_level:
                logger.warning(f"Limiting polynomial degree from {max_degree} to {input_level}")
                max_degree = min(max_degree, input_level)
            basis = self.monomial_manager.compute_basis(x_encrypted, max_degree)
            result = None
            for degree in range(max_degree + 1):
                if abs(coefficients[degree]) < 1e-10:
                    continue
                coeff_array = np.full(self.engine.slot_count, coefficients[degree], dtype=np.complex128)
                if degree == 0:
                    input_level = x_encrypted.level if hasattr(x_encrypted, 'level') else 25
                    term = self.engine.encrypt(coeff_array, self.public_key, level=input_level)
                else:
                    coeff_plain = self.engine.encode(coeff_array)
                    term = self.engine_wrapper.multiply_plain(basis[degree], coeff_plain)
                if result is None:
                    result = term
                else:
                    result = self.engine_wrapper.add(result, term)
            return result
        else:
            input_level = x_encrypted.level if hasattr(x_encrypted, 'level') else 25
            coeff_array = np.full(self.engine.slot_count, coefficients[0], dtype=np.complex128)
            return self.engine.encrypt(coeff_array, self.public_key, level=input_level)


    def extract_nibbles(self, byte_encrypted):
        upper_coeffs_key = "upper_nibble_4"
        lower_coeffs_key = "lower_nibble_4"
        if upper_coeffs_key not in self._coeff_cache:
            upper_coeffs = self._compute_nibble_extraction_coeffs(is_upper=True)
            self._coeff_cache[upper_coeffs_key] = upper_coeffs
        else:
            upper_coeffs = self._coeff_cache[upper_coeffs_key]
        if lower_coeffs_key not in self._coeff_cache:
            lower_coeffs = self._compute_nibble_extraction_coeffs(is_upper=False)
            self._coeff_cache[lower_coeffs_key] = lower_coeffs
        else:
            lower_coeffs = self._coeff_cache[lower_coeffs_key]
        upper_ct = self.evaluate_lut(byte_encrypted, upper_coeffs)
        lower_ct = self.evaluate_lut(byte_encrypted, lower_coeffs)
        return upper_ct, lower_ct
    def xor_simd(self, x_bytes: np.ndarray, y_bytes: np.ndarray) -> Tuple[Any, Any]:
        assert x_bytes.shape == y_bytes.shape
        n = len(x_bytes)
        x_ct = self.engine.encrypt(
            self.params.encode_array(x_bytes, 8), self.public_key, level=25)
        y_ct = self.engine.encrypt(
            self.params.encode_array(y_bytes, 8), self.public_key, level=25)
        x_hi, x_lo = self.extract_nibbles(x_ct)
        y_hi, y_lo = self.extract_nibbles(y_ct)
        hi_xor = self.apply_nibble_xor_lut(x_hi, y_hi)
        lo_xor = self.apply_nibble_xor_lut(x_lo, y_lo)
        return hi_xor, lo_xor   
    def _compute_nibble_extraction_coeffs(self, is_upper: bool = True) -> np.ndarray:
        lut_size = 256
        lut = np.zeros(lut_size, dtype=np.complex128)
        zeta = np.exp(2j * np.pi / 16)
        for i in range(lut_size):
            nibble_val = (i // 16) if is_upper else (i % 16)
            lut[i] = zeta ** nibble_val
        coeffs = np.fft.fft(lut) / lut_size
        if self.config.sparsify_threshold > 0:
            coeffs = self.coeff_computer.sparsify_coefficients(
                coeffs, self.config.sparsify_threshold
            )
        return coeffs


    def _evaluate_lut_with_conjugation(self, x_encrypted, coefficients):
        n = len(coefficients)
        half_n = (n + 1) // 2
        half_coeffs = coefficients[:half_n:2]
        x_squared = self.engine_wrapper.multiply(x_encrypted, x_encrypted, self.relin_key)
        result = self.evaluate_lut(x_squared, half_coeffs)
        if n % 2 == 0 and abs(coefficients[-1]) > 1e-10:
            x_power = x_encrypted
            for _ in range((n-1)//2):
                x_power = self.engine_wrapper.multiply(x_power, x_squared, self.relin_key)
            last_term = self.engine_wrapper.multiply_plain(x_power, coefficients[-1])
            result = self.engine_wrapper.add(result, last_term)
        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.perf_monitor.get_statistics()
def decrypt_simd(xoreval: XORLUTEvaluator,
                    hi_ct: Any, lo_ct: Any, length: int) -> np.ndarray:
        hi = xoreval.params.decode_array(
                xoreval.engine.decrypt(hi_ct, xoreval.secret_key)[:length], 4)
        lo = xoreval.params.decode_array(
                xoreval.engine.decrypt(lo_ct, xoreval.secret_key)[:length], 4)
        return (hi << 4) | lo
def run_tests():
    from copy import deepcopy
    import numpy as np
    print("=== XOR Operation Tests (no-lift) ===\n")
    base_cfg = XORConfig(
        poly_modulus_degree = 16384,
        precision_bits      = 40,
        thread_count        = 256,
        mode                = "parallel",
        use_nibbles         = True,
    )
    print("Test 1: single 8-bit XOR (nibble domain)\n")
    xor1 = XORLUTEvaluator(deepcopy(base_cfg))
    x   = np.array([0x3A, 0x7F, 0xC4, 0xFF], dtype=np.uint16)
    y   = np.array([0x5C, 0xB2, 0x91, 0x28], dtype=np.uint16)
    exp = x ^ y
    x_ct = xor1.engine.encrypt(xor1.params.encode_array(x, 8), xor1.public_key, level=25)
    y_ct = xor1.engine.encrypt(xor1.params.encode_array(y, 8), xor1.public_key, level=25)
    hi_ct, lo_ct = xor1.apply_xor(x_ct, y_ct, bits=8)
    hi_dec = xor1.params.decode_array(
                 xor1.engine.decrypt(hi_ct, xor1.secret_key)[:len(x)], 4)
    lo_dec = xor1.params.decode_array(
                 xor1.engine.decrypt(lo_ct, xor1.secret_key)[:len(x)], 4)
    res = (hi_dec << 4) | lo_dec
    print("Input X :", [hex(v) for v in x])
    print("Input Y :", [hex(v) for v in y])
    print("Expect  :", [hex(v) for v in exp])
    print("Result  :", [hex(int(v)) for v in res])
    print("Match   :", np.allclose(res, exp), "\n")
    print("Test 2: batch 8-bit XOR")
    batch_sz = 3
    xs = [np.random.randint(0, 256, 12) for _ in range(batch_sz)]
    ys = [np.random.randint(0, 256, 12) for _ in range(batch_sz)]
    ok = True
    for i in range(batch_sz):
        x_ct = xor1.engine.encrypt(
            xor1.params.encode_array(xs[i], 8), xor1.public_key, level=25)
        y_ct = xor1.engine.encrypt(
            xor1.params.encode_array(ys[i], 8), xor1.public_key, level=25)
        hi_ct, lo_ct = xor1.apply_xor(x_ct, y_ct, bits=8)
        hi_dec = xor1.params.decode_array(
                     xor1.engine.decrypt(hi_ct, xor1.secret_key)[:len(xs[i])], 4)
        lo_dec = xor1.params.decode_array(
                     xor1.engine.decrypt(lo_ct, xor1.secret_key)[:len(xs[i])], 4)
        res = (hi_dec << 4) | lo_dec
        if not np.allclose(res, xs[i] ^ ys[i]):
            ok = False
            print(f"  Batch {i} mismatch\n   got {res}\n   exp {xs[i]^ys[i]}")
    print("All batches match:", ok, "\n")
    print("Test 3: SIMD full-slot XOR")
    slot_cnt = xor1.engine.slot_count
    length   = min( 4 * 1024, slot_cnt )
    x_simd = np.random.randint(0, 256, length, dtype=np.uint8)
    y_simd = np.random.randint(0, 256, length, dtype=np.uint8)
    exp_simd = x_simd ^ y_simd
    x_ct = xor1.engine.encrypt(xor1.params.encode_array(x_simd, 8),
                            xor1.public_key, level=25)
    y_ct = xor1.engine.encrypt(xor1.params.encode_array(y_simd, 8),
                            xor1.public_key, level=25)
    hi_ct, lo_ct = xor1.apply_xor(x_ct, y_ct, bits=8)
    hi_dec = xor1.params.decode_array(
                xor1.engine.decrypt(hi_ct, xor1.secret_key)[:length], 4).astype(np.uint8)
    lo_dec = xor1.params.decode_array(
                xor1.engine.decrypt(lo_ct, xor1.secret_key)[:length], 4).astype(np.uint8)
    res_simd = (hi_dec << 4) | lo_dec
    print("Length          :", length)
    print("First 16 expect :", exp_simd[:16])
    print("First 16 result :", res_simd[:16])
    print("SIMD match      :", np.allclose(res_simd, exp_simd), "\n")
    print("=== Performance Stats ===")
    for op, stat in xor1.get_performance_stats().items():
        if isinstance(stat, dict):
            print(f"\n{op}:")
            for k, v in stat.items():
                print(f"  {k}: {v:.3f}")
        else:
            print(f"{op}: {stat}")
if __name__ == "__main__":
    run_tests()
