import pytest
import sys
import os
import numpy as np
import cmath  # 복소수 연산을 위해 cmath 모듈을 임포트합니다.

# 파이썬 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 필요한 모듈 임포트
from aes_128 import S_BOX
from src.fhe_aes.engine import FHEEngine
from src.fhe_aes.sbox_polynomial import sbox_poly

class TestFHESBoxPolynomial:
    @pytest.fixture(scope="class")
    def fhe_environment(self):
        """FHE 엔진과 키를 설정하는 픽스처"""
        engine = FHEEngine()
        # 병렬 처리를 위해 mode='parallel'을 적용합니다.
        # engine.initialize() 대신 아래 코드를 사용하거나 engine.py 자체를 수정하세요.
        if not engine.engine: # 이미 초기화되었는지 확인
            engine.initialize()
            if hasattr(engine.engine, 'mode') and engine.engine.mode != 'parallel':
                print("\nINFO: Re-initializing engine with mode='parallel' for testing.")
                engine.engine = None # Reset
                FHEEngine._initialized = False
                engine_temp = FHEEngine() # New instance to re-init
                engine_temp.initialize()
                # 수동으로 병렬 모드 설정 (라이브러리 버전에 따라 다를 수 있음)
                # 이 부분은 engine.py를 직접 수정하는 것이 더 안정적입니다.

        sk = engine.keys['secret']
        pk = engine.keys['public']
        rlk = engine.keys['relin']
        return engine, sk, pk, rlk

    # def test_all_256_values_with_zeta_encoding(self, fhe_environment):
    #     """
    #     [최종] '제타 필드' 인코딩을 적용하여 0x00~0xFF 모든 값에 대해 S-Box를 검증합니다.
    #     """
    #     engine, sk, pk, rlk = fhe_environment
    #     mismatches = []

    #     # 1. 0부터 255까지 모든 바이트에 대해 반복
    #     for byte in range(256):
    #         # 2. 니블 분리 및 '제타 필드' 값으로 인코딩
    #         high_nibble_int, low_nibble_int = byte >> 4, byte & 0x0F
    #         zeta = cmath.exp(-2j * cmath.pi / 16)
    #         encoded_hi = zeta ** high_nibble_int
    #         encoded_lo = zeta ** low_nibble_int
            
    #         # 3. 각 니블의 복소수 값을 별도의 암호문으로 암호화
    #         ct_hi = engine.encrypt(np.array([encoded_hi]), public_key=pk)
    #         ct_lo = engine.encrypt(np.array([encoded_lo]), public_key=pk)
            
    #         # 4. 동형 S-Box 연산 수행
    #         res_ct = sbox_poly(ct_hi, ct_lo, engine, rlk)
            
    #         # 5. 결과 검증
    #         dec = engine.decrypt(res_ct, secret_key=sk, size=1)
    #         out = int(round(dec[0])) & 0xFF
    #         expected = S_BOX[byte]
            
    #         # 진행 상황을 터미널에 실시간으로 표시
    #         print(f"Testing {hex(byte)} -> {hex(out)} (Expected: {hex(expected)})", end='\r', flush=True)
            
    #         if out != expected:
    #             mismatches.append((byte, out, expected))
        
    #     # 모든 테스트 후, 불일치 항목이 없으면 성공
    #     assert not mismatches, f"Mismatches found for {len(mismatches)} values: {mismatches[:5]}"
    #     print("\n\n✓ All 256 S-Box values correctly verified with Zeta Field encoding!")

    def test_all_256_values_simd(self, fhe_environment):
        """
        [최종/최적화] SIMD(배칭)를 사용하여 256개 모든 값을 단 한 번의 FHE 연산으로 검증합니다.
        """
        engine, sk, pk, rlk = fhe_environment

        # 1. 256개 모든 입력 바이트에 대한 평문 데이터 준비
        all_bytes = np.arange(256, dtype=np.uint8)
        
        # 상위/하위 니블 분리
        high_nibbles_int = all_bytes >> 4
        low_nibbles_int = all_bytes & 0x0F
        
        # '제타 필드' 인코딩 (NumPy 벡터 연산으로 한 번에 처리)
        zeta = cmath.exp(-2j * cmath.pi / 16)
        encoded_hi_all = zeta ** high_nibbles_int
        encoded_lo_all = zeta ** low_nibbles_int

        # 2. [핵심] 256개의 니블 데이터를 단 두 개의 암호문으로 암호화
        print("\nEncrypting all 256 nibbles in a batch...", flush=True)
        ct_hi_batch = engine.encrypt(encoded_hi_all, public_key=pk)
        ct_lo_batch = engine.encrypt(encoded_lo_all, public_key=pk)

        # 3. [핵심] sbox_poly 함수를 단 한 번만 호출
        print("Performing homomorphic SubBytes on the batch...", flush=True)
        res_ct_batch = sbox_poly(ct_hi_batch, ct_lo_batch, engine, rlk)
        
        # 4. 결과 검증
        print("Decrypting and verifying the batch result...", flush=True)
        dec_batch = engine.decrypt(res_ct_batch, secret_key=sk, size=256)
        out_batch = np.round(dec_batch).astype(np.uint8)
        
        expected_batch = S_BOX[all_bytes]

        # np.array_equal을 사용하여 전체 결과 배열을 한 번에 비교
        assert np.array_equal(out_batch, expected_batch), "SIMD S-Box verification failed!"
        
        print("\n\n✓ All 256 S-Box values correctly verified using SIMD batching!")
