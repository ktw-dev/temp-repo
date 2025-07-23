"""
FHE Engine wrapper for desilofhe (CKKS) – AES‑128 전용
"""

from __future__ import annotations
import numpy as np
from typing import Any, List
import desilofhe


class FHEEngine:
    _instance, _initialized = None, False

    # ------------------------ Singleton boilerplate ------------------------
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if FHEEngine._initialized:
            return
        self.engine: desilofhe.Engine | None = None
        self.keys: dict[str, Any] = {}
        self.scale = 2 ** 45
        self.poly_modulus_degree = 65536
        self.coeff_mod_bits = [60] + [55] * 28 + [60]  # 29 levels
        self.slot_count: int | None = None
        FHEEngine._initialized = True

    # ======================================================================
    # INITIALIZE
    # ======================================================================
    def initialize(
        self,
        poly_modulus_degree: int = 65536,
        coeff_mod_bits: List[int] | None = None,
        scale: float = 2 ** 45,
        security_level: int = 128,
    ) -> None:
        coeff_mod_bits = coeff_mod_bits or self.coeff_mod_bits
        max_lv = len(coeff_mod_bits) - 1
        mode_to_use = 'parallel'  # 사용할 모드를 변수에 저장

        # 1) 엔진 인스턴스
        self.engine = desilofhe.Engine(max_level=max_lv, mode = 'parallel')
        self.mode = mode_to_use  # 우리 클래스 속성에 mode 값을 직접 저장

        # 2) 키 생성
        sk = self.engine.create_secret_key()
        self.keys.update(
            secret=sk,
            public=self.engine.create_public_key(sk),
            relin=self.engine.create_relinearization_key(sk),
            rotation=self.engine.create_rotation_key(sk),
            conjugation=self.engine.create_conjugation_key(sk),
        )

        # 3) Info
        self.slot_count = getattr(self.engine, "slot_count", poly_modulus_degree // 2)
        self.scale = scale
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bits = coeff_mod_bits

        print("FHE Engine initialized with desilofhe:")
        print(f"  - Polynomial degree : {poly_modulus_degree}")
        print(f"  - Slot count        : {self.slot_count}")
        print(f"  - Scale             : 2^{int(np.log2(scale))}")
        print(f"  - Total levels      : {max_lv}")
        print(f"  - Execution Mode    : {self.mode}") # 클래스에 저장된 mode 값을 출력


    # ======================================================================
    # ENCODE / CONJUGATE
    # ======================================================================
    def encode(self, data, level: int | None = None):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif isinstance(data, (int, float, complex)):
            data = [data]
        else:
            data = list(data)
        return self.engine.encode(data, level) if level is not None else self.engine.encode(data)

    def conjugate(self, ct, conj_key: Any | None = None):
        conj_key = conj_key or self.keys["conjugation"]
        return self.engine.conjugate(ct, conj_key)

    # ======================================================================
    # I/O
    # ======================================================================
    def encrypt(
        self,
        plaintext: list | np.ndarray | float | complex,
        public_key: Any | None = None,
        level: int | None = None,
    ):
        public_key = public_key or self.keys["public"]
        if isinstance(plaintext, np.ndarray):
            pt_list = plaintext.tolist()
        elif isinstance(plaintext, (int, float, complex)):
            pt_list = [plaintext]
        else:
            pt_list = list(plaintext)
        ct = self.engine.encrypt(pt_list, public_key)
        if level is not None and ct.level > level:
            ct = self.level_down(ct, level)
        return ct

    def decrypt(self, ciphertext, size: int | None = None, secret_key: Any | None = None):
        secret_key = secret_key or self.keys["secret"]
        arr = np.array(self.engine.decrypt(ciphertext, secret_key))
        # ----- NEW: 허수부 제거 -----
        if np.iscomplexobj(arr):
            arr = np.real_if_close(arr, tol=1e6)  # |imag| < 10^-6 -> 실수화
            if np.iscomplexobj(arr):              # 여전히 복소?
                arr = arr.real
        return arr[:size] if size else arr

    # ======================================================================
    # BASIC ARITHMETIC
    # ======================================================================
    def add(self, x, y): return self.engine.add(x, y)
    def subtract(self, x, y): return self.engine.subtract(x, y)
    def multiply(self, x, y): return self.engine.multiply(x, y)

    def multiply_plain(self, ct, plain):
        """ciphertext × (scalar | vector)"""
        if isinstance(plain, complex):
            pt = self.encode([plain])
            return self.engine.multiply(ct, pt)
        if isinstance(plain, (int, float, np.integer, np.floating)):
            return self.engine.multiply(ct, float(plain))
        pt = self.encode(plain)
        return self.engine.multiply(ct, pt)

    def relinearize(self, ct, rlk=None):
        return self.engine.relinearize(ct, rlk or self.keys["relin"])

    def rotate(self, ct, k: int, rtk=None):
        return self.engine.rotate(ct, rtk or self.keys["rotation"], k)

    # ----------------------------------------------------------------------
    # LEVEL / POWER BASIS
    # ----------------------------------------------------------------------
    def get_level(self, ct): return ct.level
    def level_down(self, ct, lvl): return self.engine.level_down(ct, lvl)
    def make_power_basis(self, ct, d, rlk):
        return self.engine.make_power_basis(ct, d, rlk)
