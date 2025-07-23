"""
Homomorphic evaluation of the AES S‑Box via the 2‑variable
polynomial recovered from FFT/IFFT (복소수 계수 포함).

Functions
---------
sbox_poly(ct_hi, ct_lo, engine, rlk)
    : FHE 상에서 상위/하위 니블 암호문 → S‑Box 결과 암호문(byte)
"""

from __future__ import annotations
import json, os, numpy as np
from typing import Any, List, Tuple

# ---------- 1. 계수 로딩 --------------------------------------------------
_BASE = os.path.dirname(__file__)
_COEFF_PATH = os.path.join(_BASE, "coeffs", "sbox_coeffs.json")

with open(_COEFF_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)

# 16×16 복소수 행렬 복원
C_hi: np.ndarray = (np.array(_data["sbox_upper_mv_coeffs_real"])
                    + 1j*np.array(_data["sbox_upper_mv_coeffs_imag"]))
C_lo: np.ndarray = (np.array(_data["sbox_lower_mv_coeffs_real"])
                    + 1j*np.array(_data["sbox_lower_mv_coeffs_imag"]))

_HI_DEG, _LO_DEG = C_hi.shape[0]-1, C_hi.shape[1]-1
_EPS = 1e-12   # 0 계수 컷

# ---------- 2. 헬퍼 --------------------------------------------------------
def multiply_plain_complex(ct: Any, scalar: complex, engine) -> Any:
    """ciphertext × complex(scalar)"""
    # desilofhe.encode는 복소수 벡터를 지원 → 길이1 벡터로 인코딩
    pt = engine.encode(np.array([scalar], dtype=np.complex128))
    return engine.multiply(ct, pt)

def level_align(ct1: Any, ct2_or_level: Any, engine) -> Tuple[Any, Any]:
    """두 ciphertext를 같은 level로 맞춰 반환"""
    lvl1 = engine.get_level(ct1)
    lvl2 = (ct2_or_level if isinstance(ct2_or_level, int)
            else engine.get_level(ct2_or_level))
    if lvl1 > lvl2:
        ct1 = engine.level_down(ct1, lvl2)
    elif lvl2 > lvl1 and not isinstance(ct2_or_level, int):
        ct2_or_level = engine.level_down(ct2_or_level, lvl1)
    return ct1, ct2_or_level

# ---------- 3. 메인 함수 ---------------------------------------------------
def sbox_poly(ct_hi: Any, ct_lo: Any, engine, rlk) -> Any:
    """
    암호화된 상위·하위 니블(ct_hi, ct_lo)을 입력받아
    AES S‑Box(byte) 값을 암호화 상태로 돌려준다.
    """
    # 1) power‑basis 준비
    hi_basis: List[Any] = engine.make_power_basis(ct_hi, _HI_DEG, rlk)
    lo_basis: List[Any] = engine.make_power_basis(ct_lo, _LO_DEG, rlk)
    one_ct = engine.encrypt(np.array([1.0]))   # 상수항용

    terms_hi, terms_lo = [], []
    target_lvl = min(engine.get_level(x) for x in hi_basis+lo_basis) - 1

    # 2) 모든 모노미얼 평가
    for i in range(_HI_DEG+1):
        ct_hi_pow = one_ct if i == 0 else hi_basis[i-1]
        for j in range(_LO_DEG+1):
            coeff_hi, coeff_lo = C_hi[i, j], C_lo[i, j]
            if abs(coeff_hi) < _EPS and abs(coeff_lo) < _EPS:
                continue

            ct_lo_pow = one_ct if j == 0 else lo_basis[j-1]
            a, b = level_align(ct_hi_pow, ct_lo_pow, engine)
            monom = engine.relinearize(engine.multiply(a, b), rlk)

            if abs(coeff_hi) >= _EPS:
                term = multiply_plain_complex(monom, coeff_hi, engine)
                term, _ = level_align(term, target_lvl, engine)
                terms_hi.append(term)
            if abs(coeff_lo) >= _EPS:
                term = multiply_plain_complex(monom, coeff_lo, engine)
                term, _ = level_align(term, target_lvl, engine)
                terms_lo.append(term)

    # 3) 합산
    def sum_terms(lst: List[Any]) -> Any:
        if not lst:
            return engine.encrypt(np.array([0.0]))
        acc = lst[0]
        for t in lst[1:]:
            acc = engine.add(acc, t)
        return acc

    ct_hi_out = sum_terms(terms_hi)
    ct_lo_out = sum_terms(terms_lo)

    # 4) 허수부 제거(수치 오차) → (x + conj(x)) / 2
    conj_hi = engine.conjugate(ct_hi_out, engine.keys['conjugation'])
    conj_lo = engine.conjugate(ct_lo_out, engine.keys['conjugation'])
    ct_hi_real = engine.multiply_plain(
        engine.add(ct_hi_out, conj_hi), 0.5)
    ct_lo_real = engine.multiply_plain(
        engine.add(ct_lo_out, conj_lo), 0.5)

    # 5) byte 조립: (upper <<4) + lower
    shifted_hi = engine.multiply_plain(ct_hi_real, 16.0)
    shifted_hi, ct_lo_real = level_align(shifted_hi, ct_lo_real, engine)
    return engine.add(shifted_hi, ct_lo_real)

__all__ = ["sbox_poly"]
