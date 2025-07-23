# xor_wrapper.py  ────────────────────────────────────────────────────────────
import numpy as np
from xor_paper_optimized import XORConfig, XORPaperOptimized   # ← 300-line 버전

class XOR:
    """
    정적 레퍼런스:
      • XOR.xor(a, b)              : 단일 4-bit XOR → int
      • XOR.simdxor(a_vec, b_vec)  : 4-bit 벡터 XOR → np.ndarray
      • XOR.pack_nibbles(hi, lo)   : (hi<<4 | lo) 8-bit 결합
      • XOR.packedsimdxor(s, k)    : 4-bit 벡터 두 개를 암호화→XOR→복호
    """

    # ── ❶ 싱글턴 FHE 코어 ────────────────────────────────────────────────
    _cfg  = XORConfig(
        poly_modulus_degree = 16384,
        precision_bits      = 40,
        thread_count        = 8,
        mode                = "parallel",
        use_conjugate_reduction = True
    )
    _core = XORPaperOptimized(_cfg)

    # ── ❷ 단일 nibble XOR ────────────────────────────────────────────────
    @classmethod
    def xor(cls, a: int, b: int) -> int:
        core = cls._core
        ct_a = core.encrypt(core.encode_array([a], 4), level=20)
        ct_b = core.encrypt(core.encode_array([b], 4), level=20)
        ct_z = core.apply_nibble_xor(ct_a, ct_b)          # 2-변수 XOR
        dec  = core.decrypt(ct_z)[0]
        return int(core.decode_array([dec], 4)[0])

    # ── ❸ SIMD nibble XOR (평문 입력) ────────────────────────────────────
    @classmethod
    def simdxor(cls, a_vec, b_vec) -> np.ndarray:
        a_vec = np.asarray(a_vec, dtype=np.uint16)
        b_vec = np.asarray(b_vec, dtype=np.uint16)
        if a_vec.shape != b_vec.shape:
            raise ValueError("입력 벡터 길이가 일치해야 합니다")

        core = cls._core
        ct_a = core.encrypt(core.encode_array(a_vec, 4), level=20)
        ct_b = core.encrypt(core.encode_array(b_vec, 4), level=20)
        ct_z = core.apply_nibble_xor(ct_a, ct_b)
        dec  = core.decrypt(ct_z)[: len(a_vec)]
        return core.decode_array(dec, 4)

    # ── ❹ 4-bit 두 벡터를 8-bit 값으로 패킹 ──────────────────────────────
    @classmethod
    def pack_nibbles(
        cls,
        hi_vec,
        lo_vec,
        *,
        as_cipher: bool = False,
        level: int = 20,
    ):
        hi = np.asarray(hi_vec, dtype=np.uint16)
        lo = np.asarray(lo_vec, dtype=np.uint16)
        if hi.shape != lo.shape:
            raise ValueError("hi_vec 과 lo_vec 길이가 달라요")

        byte_vals = (hi << 4) | lo
        if not as_cipher:
            return byte_vals

        core = cls._core
        return core.encrypt(core.encode_array(byte_vals, 8), level=level)

    # ── ❺ Packed-SIMD XOR  (4-bit 벡터 두 개 → 결과 복호/암호문) ────────
    @classmethod
    def packedsimdxor(
        cls,
        state_nibbles,
        key_nibbles,
        *,
        level: int = 20,
        return_cipher: bool = False,
    ):
        s = np.asarray(state_nibbles, dtype=np.uint16)
        k = np.asarray(key_nibbles,  dtype=np.uint16)
        if s.shape != k.shape:
            raise ValueError("state_nibbles와 key_nibbles 길이가 달라요")

        core = cls._core
        ct_s = core.encrypt(core.encode_array(s, 4), level=level)
        ct_k = core.encrypt(core.encode_array(k, 4), level=level)
        ct_z = core.apply_nibble_xor(ct_s, ct_k)          # 2-변수 XOR

        if return_cipher:
            return ct_z

        dec  = core.decrypt(ct_z)[: len(s)]
        return core.decode_array(dec, 4)
from typing import Union, List, Optional, Dict, Tuple, Any, Sequence
def decrypt_simd(xor_inst: XORPaperOptimized, hi_ct: Any, lo_ct: Any, length: int) -> np.ndarray:
    hi_dec = xor_inst.decrypt(hi_ct)[:length]
    lo_dec = xor_inst.decrypt(lo_ct)[:length]
    hi = xor_inst.decode_array(hi_dec, 4)
    lo = xor_inst.decode_array(lo_dec, 4)
    return (hi << 4) | lo
# ─────── 간단 테스트 ────────────────────────────────────────────────────────
cfg  = XORConfig()
xor  = XORPaperOptimized(cfg)

state = np.random.randint(0, 256, 1024, dtype=np.uint8)
key   = np.random.randint(0, 256, 1024, dtype=np.uint8)

ct_s_hi, ct_s_lo = xor.encrypt_nibbles(state)
ct_k_hi, ct_k_lo = xor.encrypt_nibbles(key)

ct_hi, ct_lo = xor.apply_nibble_xor_direct(ct_s_hi, ct_s_lo,
                                           ct_k_hi, ct_k_lo)

res = decrypt_simd(xor, ct_hi, ct_lo, len(state))
assert np.all(res == (state ^ key))
print("✔ XOR bundle test passed. Level =", getattr(ct_hi, "level", "?"))
