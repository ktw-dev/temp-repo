# xor_wrapper.py  ────────────────────────────────────────────────────────────
import numpy as np
from xor_paper_optimized import XORConfig, XORPaperOptimized

class XOR:
    """편리한 정적 레퍼런스  –  XOR.xor(), XOR.simdxor(), XOR.packedsimdxor()"""

    # ── ❶ 싱글턴 FHE 엔진 ────────────────────────────────────────────────
    _cfg  = XORConfig(
        poly_modulus_degree = 16384,
        precision_bits      = 40,
        thread_count        = 8,
        mode                = "parallel",
        use_conjugate_reduction = True,
        use_baby_giant_step     = True,
    )
    _core = XORPaperOptimized(_cfg)

    # ── ❷ 단일 nibble XOR ────────────────────────────────────────────────
    @classmethod
    def xor(cls, a: int, b: int) -> int:
        enc_a = cls._core.encrypt(cls._core.encode_array([a], 4), level=20)
        enc_b = cls._core.encrypt(cls._core.encode_array([b], 4), level=20)
        enc_z = cls._core.apply_nibble_xor(enc_a, enc_b)
        dec_z = cls._core.decrypt(enc_z)[:1]
        return int(cls._core.decode_array(dec_z, 4)[0])

    # ── ❸ SIMD nibble XOR (평문 입력) ────────────────────────────────────
    @classmethod
    def simdxor(cls, a_vec, b_vec) -> np.ndarray:
        a_vec = np.asarray(a_vec, dtype=np.uint16)
        b_vec = np.asarray(b_vec, dtype=np.uint16)

        enc_z = cls._core.apply_nibble_xor_paper(a_vec, b_vec, pre_encoded=False)
        dec_z = cls._core.decrypt(enc_z)[:len(a_vec)]
        return cls._core.decode_array(dec_z, 4)
    def pack_nibbles(
        cls,
        hi_vec,                       # 상위 4-bit 값 시퀀스
        lo_vec,                       # 하위 4-bit 값 시퀀스
        *,
        as_cipher: bool = False,      # True → 8-bit 암호문 반환
        level: int = 20
    ):
        """
        4-bit SIMD 결과 둘을 (hi<<4 | lo) 8-bit 값으로 결합.

        Parameters
        ----------
        hi_vec, lo_vec : 1-D iterable of int (0‥15)  또는  numpy.ndarray
        as_cipher      : True 이면 8-bit 인코딩 CKKS **암호문** 반환
        level          : 암호화 레벨 (as_cipher=True 일 때만 사용)

        Returns
        -------
        numpy.ndarray | ciphertext
            as_cipher=False → 평문 numpy array (0‥255)
            as_cipher=True  → 8-bit 인코딩 ciphertext
        """
        hi = np.asarray(hi_vec, dtype=np.uint16)
        lo = np.asarray(lo_vec, dtype=np.uint16)
        if hi.shape != lo.shape:
            raise ValueError("hi_vec 과 lo_vec 길이가 달라요")

        byte_vals = (hi << 4) | lo        # 8-bit 평문 값

        if not as_cipher:
            return byte_vals              # ▶ 평문 numpy 배열

        # 8-bit 공간(ζ₂₅₆)으로 인코딩 후 암호화해서 반환
        core = cls._core
        enc = core.encrypt(core.encode_array(byte_vals, 8), level=level)
        return enc
    # ── ❹ ▶ NEW ◀ Packed-SIMD XOR  (8-bit 공간에 4+4 닙블 패킹) ───────────
    @classmethod
    def packedsimdxor(
        cls,
        state_nibbles,          # 상위 4-bit 로 갈 닙블 시퀀스
        key_nibbles,            # 하위 4-bit 로 갈 닙블 시퀀스
        *,
        level: int = 20,
        return_cipher: bool = False   # True → 암호문 그대로 반환
    ):
        """
        논문 방식(ζ₂₅₆^(16s) · ζ₂₅₆^k) 으로 이미 **패킹·암호화**한 뒤 XOR 수행.

        Parameters
        ----------
        state_nibbles, key_nibbles : 1-D iterable of int (0‥15), 같은 길이
        level                      : CKKS 레벨 (default 20)
        return_cipher              : 암호문 그대로 받을지 여부

        Returns
        -------
        ciphertext  |  numpy.ndarray
            - return_cipher=True  → 암호문 (FHE ciphertext)
            - False (default)     → 평문 numpy array (각 원소 0‥15)
        """
        s = np.asarray(state_nibbles, dtype=np.uint16)
        k = np.asarray(key_nibbles,  dtype=np.uint16)
        if s.shape != k.shape:
            raise ValueError("state_nibbles와 key_nibbles 길이가 달라요")

        core = cls._core
        # ① 8-bit 공간 인코딩 후 암호화
        enc_state = core.encrypt(core.encode_array(s * 16, 8), level=level)
        enc_key   = core.encrypt(core.encode_array(k,      8), level=level)

        # ② 논문 경로로 XOR (이미 패킹-암호화 됐으므로 pre_encoded=True)
        enc_out = core.apply_nibble_xor_paper(
            enc_state, enc_key, pre_encoded=True
        )

        if return_cipher:
            return enc_out                       # 암호문 그대로
        # ③ 결과 복호·디코드
        plain  = core.decrypt(enc_out)[:len(s)]
        return core.decode_array(plain, 4)

# ─────── 테스트용 실행부 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    state = np.random.randint(0, 16, 8)
    key   = np.random.randint(0, 16, 8)

    hi_xor = XOR.simdxor(state >> 2,  key >> 2)   # 임의 예시
    lo_xor = XOR.simdxor(state & 3,   key & 3)

    # ① 평문 8-bit 벡터로 패킹
    bytes_plain = XOR.pack_nibbles(hi_xor, lo_xor)
    print("Packed bytes (plain):", bytes_plain)

    # ② 8-bit 암호문으로 바로 패킹
    bytes_ct = XOR.pack_nibbles(hi_xor, lo_xor, as_cipher=True, level=20)
    print("Cipher level:", getattr(bytes_ct, 'level', 'N/A'))

