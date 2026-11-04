# -*- coding: utf-8 -*-
"""
evaluate_shelf.py - 평가 전용 단일 스크립트 (저장소 루트, shelf_audit 패키지 밖. 나중에 이 파일만 지우면 된다)

test/normal 의 기준 사진으로 test/check 의 사진들을 판정하고 정답과 비교한다.

    python evaluate_shelf.py                       # 기본: --normal test/normal --check test/check --out eval_logs
    python evaluate_shelf.py --truth truth.json    # 정답표 파일 ({"사진.jpg": {"3": "upside", ...}})
    python evaluate_shelf.py --replay eval_logs/20260101_120000   # 모델 없이 그 로그로 재현

출력(사진마다):
    🔍 분석 중: 20260522_203708.jpg
    ├─ 총 13권 | 정상 10 | 이상 3
    ├─ ⚠️ 결과
    ├─ ⚠️ 3번 책: 위아래 뒤집힘 (abnormal_upside)
    ├─ 🧪 백엔드 품질: 신뢰 | 미확인 0/13
    └─ 📌 정답 : 3,5,9 뒤집혀짐  → ✅ 일치
끝에 전체 통계(책 단위 정확도 · 놓침 · 오탐 · 유형 오분류 · 혼동 행렬)를 찍는다.

로그 폴더(eval_logs/<시각>/)에는 파일 두 개만 남는다:
    summary.txt   사람이 보는 것. 콘솔 출력 + 책별 표(정답·예측·경로·플래그·핵심 증거값) + 기준 슬롯 정보
    replay.zip    분석자에게 보내는 것. 탐지 덤프 · 임베딩 기록 · 1280px 사진 · 사진별 전체 JSON · 크롭 몽타주 · 통계 · 설정이 들어 있다
                  (--replay eval_logs/<시각> 으로 모델도 원본 사진도 없이 같은 판정을 재현)
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트의 shelf_audit 패키지

from shelf_audit import package_hash                                   # noqa: E402
from shelf_audit.config import CFG, KR                                 # noqa: E402
from shelf_audit.detection.yolo_adapter import DumpDetector, get_detector, set_detector  # noqa: E402
from shelf_audit.errors import Diagnostics                             # noqa: E402
from shelf_audit.features.embedder import get_embedder, set_embedder   # noqa: E402
from shelf_audit.geometry import as_quad, clip_box, to_py, warp_spine  # noqa: E402
from shelf_audit.imaging import list_images, prepare_image             # noqa: E402
from shelf_audit.output.backend import build_backend_result            # noqa: E402
from shelf_audit.pipeline import ShelfAuditor                          # noqa: E402
from shelf_audit.reference.builder import build_shelf_reference, load_manifest  # noqa: E402
from shelf_audit.reference.catalog import ShelfCatalog                 # noqa: E402
from shelf_audit.reference.pool import RefPool, report_shelf_overlap   # noqa: E402


# ============================================================================ 재현(replay) 지원
# 모델(YOLO·ResNet) 없이도 같은 입력으로 판정 로직을 다시 돌릴 수 있게 한다.
#  * 탐지: 패키지의 observe() 가 CFG.DUMP_DIR/<사진 이름>.json 에 YOLO 원본 탐지를 남긴다 -> DumpDetector 가 읽는다.
#  * 임베딩: RecordingEmbedder 가 실제 임베더에 들어간 모든 배열과 출력 벡터를 기억했다가 npz 로 저장한다.
#           ReplayEmbedder 는 그 npz 를 읽어 같은 배열이 들어오면 저장된 벡터를 돌려준다. 기록에 없으면 근사 벡터.
def _key(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()


def _sig(a: np.ndarray) -> np.ndarray:
    """픽셀이 조금 달라도(JPEG 디코더·리사이즈 차이) 같은 크롭을 찾기 위한 지각 서명: 8x32 그레이 + 8x32 색상(ab)"""
    a = np.ascontiguousarray(a)
    small = cv2.resize(a, (8, 32), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    return np.concatenate([lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()]).astype(np.float32)


SIG_TOL = 0.05     # 서명 평균 절대차가 이보다 작으면 같은 크롭으로 본다 (디코더 차이는 보통 0.01 안팎, 다른 책은 0.1 이상)


class RecordingEmbedder:
    """실제 임베더를 감싸 입력/출력을 기록한다."""

    def __init__(self, inner):
        self.inner = inner
        self.store: Dict[str, np.ndarray] = {}

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        out = self.inner.embed(rgb_list, chunk)
        for a, v in zip(rgb_list, out):
            k = _key(a)
            if k not in self.store:
                self.store[k] = (np.asarray(v, np.float32), _sig(a), np.array(a.shape[:2], np.int32))
        return out

    def save(self, path: str) -> None:
        keys = list(self.store.keys())
        np.savez_compressed(path, keys=np.array(keys),
                            vecs=np.stack([self.store[k][0] for k in keys]) if keys else np.zeros((0, 1), np.float32),
                            sigs=np.stack([self.store[k][1] for k in keys]) if keys else np.zeros((0, 1), np.float32),
                            shapes=np.stack([self.store[k][2] for k in keys]) if keys else np.zeros((0, 2), np.int32))


class ReplayEmbedder:
    """기록된 npz 로 임베딩을 돌려준다. 기록에 없으면 근사 벡터(색 히스토그램 투영)."""

    def __init__(self, path: Optional[str] = None):
        self.store: Dict[str, np.ndarray] = {}
        self.sigs = None
        self.shapes = None
        self.vecs = None
        self.misses = 0
        self.hits = 0
        self.near = 0
        self._P = None
        if path:
            z = np.load(path, allow_pickle=False)
            self.store = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
            if "sigs" in z and len(z["sigs"]):
                self.sigs, self.vecs = z["sigs"].astype(np.float32), z["vecs"]
                self.shapes = z["shapes"] if "shapes" in z else None
        self.dim = next(iter(self.store.values())).shape[0] if self.store else 512

    def _approx(self, a: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(cv2.resize(a, (16, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hist = np.concatenate([cv2.calcHist([a], [c], None, [12], [0, 256]).ravel() for c in range(3)])
        v = np.concatenate([lab.ravel(), hist / max(hist.sum(), 1) * 4])
        if self._P is None:
            self._P = np.random.RandomState(1).randn(len(v), self.dim).astype(np.float32) / np.sqrt(len(v))
        f = v @ self._P
        return (f / max(np.linalg.norm(f), 1e-6)).astype(np.float32)

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        out = []
        for a in rgb_list:
            v = self.store.get(_key(a))
            if v is None and self.sigs is not None:
                d = np.abs(self.sigs - _sig(a)[None, :]).mean(axis=1)
                if self.shapes is not None:                    # 크기가 비슷한 기록을 우선 (가로세로 10% 이내)
                    h, w = a.shape[:2]
                    ok = (np.abs(self.shapes[:, 0] - h) <= 0.1 * h + 2) & (np.abs(self.shapes[:, 1] - w) <= 0.1 * w + 2)
                    d = np.where(ok, d, 9.0)
                k = int(np.argmin(d))
                if d[k] < SIG_TOL:
                    v = self.vecs[k]
                    self.near += 1
            if v is None:
                self.misses += 1
                v = self._approx(np.ascontiguousarray(a))
            else:
                self.hits += 1
            out.append(v)
        return np.stack(out).astype(np.float32)


# ============================================================================ 평가

# ---------------------------------------------------------------- 정답표
# 번호 = 기준 사진에서 왼쪽부터 0. 값은 "normal" 외 상태명, 또는 둘 이상을 인정하면 튜플.
# 항목마다 "ref" = 그 사진을 비교할 기준 사진(test/normal 의 파일명). 같은 책장을 다른 시각에 찍은 기준이 둘이면 자동 서가 식별이
# 흔들리므로 정답표가 기준을 못 박는다. "sig" 가 있으면 파일명이 달라도 사진 내용(24x18 밝기 서명)으로 찾는다.
# --truth 파일(JSON)로 덮어쓸 수 있다: [{"file": "...", "ref": "...", "slots": {"3": "upside"}, "sig": "..."}, ...]
SIG_W, SIG_H, SIG_MATCH = 24, 18, 0.02      # 서명 크기, 같은 사진으로 보는 평균 밝기차(0~1). 재인코딩 차이 ~0.0001, 다른 사진끼리 ≥0.05

REF_A = "20260522_203503.jpg"   # 1차 세트 기준 (13권)
REF_B = "20260522_204705.jpg"   # 2차 세트 기준 (14권: 0 캐릭터디자인 1 액션만화 2 잃어버린이름 3 픽셀아트 4 돈으로살수없는 5 에너지위기
                                 #                 6 안드로이드앱 7 자료구조 8 절대어휘 9 땅속세계 10 게임디자인플레이 11 C++ 12 국어사전 13 PowerVoca)

DEFAULT_TRUTH: List[Dict[str, Any]] = [
    # ---- 1차 세트 (기준 20260522_203503) ----
    {"file": "20260522_203503.jpg", "ref": REF_A, "slots": {}},
    {"file": "20260522_203708.jpg", "ref": REF_A, "slots": {3: "upside", 5: "upside", 9: "upside"}},
    {"file": "20260522_203732.jpg", "ref": REF_A, "slots": {1: "upside", 5: "upside", 10: "upside"}},
    {"file": "20260522_203815.jpg", "ref": REF_A, "slots": {0: "upside", 2: "upside", 8: "upside", 11: "upside"}},
    {"file": "20260522_205058.jpg", "ref": REF_A, "slots": {7: "stack"}},
    {"file": "20260522_205117.jpg", "ref": REF_A, "slots": {2: "stack"}},
    # 205137 의 11·12번: 글자는 바로 서 있고(SIFT 회전 ~0°) 약 30° 기울어 있음 → 뒤집힘·기울어짐 둘 다 인정
    {"file": "20260522_205137.jpg", "ref": REF_A, "slots": {6: "stack", 10: "stack", 11: ("upside", "tilted"), 12: ("upside", "tilted")}},
    {"file": "20260522_205210.jpg", "ref": REF_A, "slots": {0: "stack", 1: "stack"}},
    {"file": "20260522_210223.jpg", "ref": REF_A, "slots": {4: "paper", 11: "paper"}},
    {"file": "20260522_210248.jpg", "ref": REF_A, "slots": {10: "paper"}},
    {"file": "20260522_210313.jpg", "ref": REF_A, "slots": {1: "paper", 8: "paper", 12: "paper"}},
    {"file": "20260522_210803.jpg", "ref": REF_A, "slots": {2: "tilted"}},
    {"file": "20260522_210814.jpg", "ref": REF_A, "slots": {5: "tilted"}},

    # ---- 2차 세트 (기준 20260522_204705, 왼쪽에 책이 아닌 플라스틱 트레이 있음) ----
    # 파일명을 모르는 사진은 "file" 대신 "label" 로 부르고 서명(sig)으로 찾는다. 파일명을 알면 "file" 을 채워 넣으면 된다.
    {"file": "20260522_204705.jpg", "ref": REF_B, "slots": {},
     "sig": "Ii9PVVhaV09HSEhGQz86NjAjJik4VHltIiAjQExMTEQ6NTQyLysnIx0fJC04UXptIR4cHCUpLjAuLSwtSCsoQjAfIzA4UXtuIyAdHSlhPz07SU+Bs0JBqWMfIzI5T3ptJSMgHi9oSGlRX06TtYO3qGUdIjM4TnlqKCQhHyxxZ6qydlF+toy3rGQdUDw8THVoLCUgHyxqYZXBe02HuYy0nIVkeUdFT3RqLSogISpkWpy/sFSEj662oKSJgUtOUXJqMDAjJTBfZJWvr1x4k6bBoaiShU1PUHRqMDQoLDRYbKaktXZyo6PLlpaRik5NUnBoMzgvMzRMbqywuHZznafNkKGhilBSUXBlMj42N0JQlJ+1t4FjkKTPm4+cjlFUUm1kO0g8OE9rg5e/yINkkbXRk5CekFNWUmpkRE4/ZoOCcZXD1IlkkbzSkp2UlFRaVWljO0hiW3WCb3zH24toj7rYpq64mVphWWVhSGqLgXhyfXDFv5Nskb7MsrrHqnByaGFgj7e5xc3QtZvW36+YqM3awsnR0rp+c1xfs8LHyczO0NLV1dPQ09fV09HPzsuudVtf"},
    # 204131: 4(돈으로)도 뒤집혀 있고, 6(안드로이드)은 기준 사진과 같은 방향(둘 다 뒤집힘) → 기준 대비 정상. 9(땅속)는 확대해 보면 바로 서 있음
    {"file": "20260522_204131.jpg", "ref": REF_B, "slots": {4: "upside", 5: "upside", 6: "normal", 8: "upside", 10: "upside"},
     "sig": "Ih4eJystMC8sKyoqKSgnIyAaICInLTQ7Ih4dKCwxNzg3NDExOT5NKjI+LCIpMTc/JSAgKjFnWD1BPkVjZHudOHSkPSEtMDlCJyEjKjJOai6CP4RVdYepo5qmNB8wMTpGKCEkJjBKe2u2oKVVcIettaGpK0E3MzxILCIkJTE9fnGbwbFabIKqxZiVWHQ+Oj9IMSYkIy84eXKb0bpibIe0w5uadIFBQEJKNS0lJDM6bmyYzKppb4ezxpyZhIFERERLNTMpJTg8eWibwa5/Yoeux5eVioZHREVMNTguKjk9cVmkuK+MZICzyJONkYVLRUdQNj8zMUlQcnGvwq+GX3OxyZOaiIVOR0hTN0Q4NlVfaYObw7SIXnCWxpSbhYNQS0tVQE49S4ygoYOUx8CEZHXEtZKFjYBUTU1XRUBLjaCdjnKLzMBfZXLHvpSin35cUlFbOk9XZGBycnV6xclsa3bCxqC3s4tsYltcTIWcp6SyspR/y9l+eYrI0bPEz86Vbmhcj8PHycjLzMrR19TR0NDT1c7NzMrIjmlanrXCw8TGys3O0NDOz8/R083KysnGxIpb"},
    {"label": "2차-03 (1·7·11 뒤집힘)", "ref": REF_B, "slots": {1: "upside", 7: "upside", 11: "upside"},
     "sig": "HBgYHyUnKisqKCcmJSUkIR0YHiAmLDI8HhkaIig6VzM2NDRmVnxmLXKfJyAqLjdDIBscJC48aS1kQFBaYomIhqWoHx4tLjlIIhseIiw2cE6zdVVfYoyFrqabFyEuLjpKJBweIiwtgFijq1teY5CMtqWIH2c4NDtMKR4eICkpgmKisGtlaZCPtKOIWX05Nz5RLSMgJCksdmacrpBsaXiVp6KUa3k8PEFTLikkJzMxc2yQpItnZ4mdtKOMcns/PUNXLy4pKTQ0bkefppNmdZOZuaKIe3pCPURYMDMwLjc6blO4pZ5meYuguaWJf3VDP0RYMjozMklOcFWtqpdPdZCauqOKfnVGQkdaO0U5NmV/hV+os6VEdJilvZ6Jg29IQ0dZQENEfX10aWimt8RGcJenv56gjWtOR0tcNkhfWHGDfGmnub5UiKGlx6WikGdVTE1cQmuGgH2Gkmykt7dfep2xwKy7v5lrYFhZjbjAxsfIwpzN0MqYqr3Lz8bHx8acaGBSoLa+v8HDxsnLzc3LzM3OzMvLyMO/lV1SvL7AwMHDxMfIycvHysvKysjIx8XDv5Jp"},
    {"label": "2차-04 (0·2·3·10·12 뒤집힘)", "ref": REF_B, "slots": {0: "upside", 2: "upside", 3: "upside", 10: "upside", 12: "upside"},
     "sig": "GBkcIyUpLCwqKSgoKCcmIx8aGR0gIykuGRodJzI0OTk3NDQ/SExAKS9ALR4hJi0xGhsdKlByRTk7OkJYdYOBNmifUh0gKywzHBwcKD+GUEV9P05bj4+Jn52lShohLyw1HBsbKjKWVneqhlNeiJGQraihPh80My43HhobKCeLZG66nlRgfY2Qw6OUYVVcPjg7JRsfKypzd2bGrV9ohIyTxJ+Vn5FhRUE/KyAjKy5gc229sX5tl32SyKCUoYxeSEJALyUoKTFcc26/tHFrfoSfx5+di45cSUNCMyktKjRKfnO1snxsfY2fypiRno9YSkRENjAyNElMhm+3totrh4+oyZCPko1SS0VHOjg2NVpXin+YuI1SdoWlyZONnZBOTEhIRkk1X5eel4CLtZ1KeIi7vY6PrI1OT0lLQz1im46NgoGTuMFNcZLBuIubqIpUUklMOFJjUmFyfXGTs7ZXdZrBt6CimYVcVE5PX2iMgo2KjHeQxMRnd5rEtqy9x7Z3aV5VnbC0wcjMy6q21M2jjbPU1MTLzcy4cGZbvb3ExcfKzM3Q0dDNz9DT0szLy8rHq2lb"},
    # 210840: 7·8 확실히 기울어짐. 4·5·6 은 살짝 기울어 애매 → 정상/기울어짐 둘 다 인정
    {"file": "20260522_210840.jpg", "ref": REF_B, "slots": {4: ("normal", "tilted"), 5: ("normal", "tilted"), 6: ("normal", "tilted"), 7: "tilted", 8: "tilted"},
     "sig": "djgoJjtHTFFUVVRST05LRT85NikhJSk5eEAqIiAkMTM2OTo5NjQwKyYhGhshJCw9eUkuIiEbHSQnLS8wLy0rKSYgGR0iJzFBfFAxJyIdHiowOz9BPTg0MzAmJB0hKzRFfFk1LjU4Hi42QkVGQURGbkNYqDUeLTNLemU5MTFoHyZqRkBVNFqGqZGwrRcfMDZPcW85Mx58NFayYq1PMGaOpKqtmjdoOztTYXo9NSFpPFGsf6tJTnBar7qhnGt0QT9XVn5FNyVWVGOuqbU7ZZBQnbmfpYhyRkNZQ3xQOSteYoGZspYrcnVoocmPhZBzSUZcNnpYQDpnaIWnpX5Rdk2OnrmUjZBxTEleOnpeQkNki4K7xmRXeTmio6eepJNzUkxgL3RiREZZeILC1lpkaFe1tqKWk5xvV09hImZpR1JsYnrN225zYJC1uKCptqFtX1ZjI1dtUV52h2zNxmmAqqaxvq6+1seVdGljPWFsY4G8yb/d3b3K1c7N1dTU1NPQlnFgxappccLL1dfZ2tbW1dna2NXU1NLPyohfx8OxwM3Oz9DR0dHQ0dHRz87NzMvJx8Gp"},
    # 210824: 1~9 기울어짐. 0 은 트레이에 가려진 채 종이면을 보이며 기울어 있음(사용자 정답에 없음), 2(얇은 책)는 기운 책들 사이에 거의 안 보임
    #         10(게임디자인플레이)은 뒷표지가 드러나지만 제자리
    {"file": "20260522_210824.jpg", "ref": REF_B,
     "slots": {0: ("tilted", "normal", "unknown", "paper"), 2: ("tilted", "unknown", "absent"), **{k: "tilted" for k in (1, 3, 4, 5, 6, 7, 8, 9)}},
     "sig": "gWE1Q2ducHJycnBvbWtnYVhRTEQsKjZhgWQ4LSxDUFNWWVpaV1NPSEA6MCIlKDdifWo7LSgkKDQ3Oz07ODMuKiUfGh0kKTljc3A7MiglHicuNzw9PTk0MC0lHB4kLztkbHY5OSsoISw3REpNTEc/ODZZgSYiMDtmZH47PDQrIy04UWdMTEc+PXyZrxgfMj1nW4FBRm1CJCVidKSLWzw3TqikrRtJOEBoSYNMQUd6i7t7aYuzXDgwSrahnGNvQ0VnQIRRQDFwaqnDbHaaj2UvR7Skmn9wSEloNoBXPzgqdYC+t3iIjHw5TrSwmox1SktpMH9aQ0JTXnqhxY14kZprVq+xkJF2TU5oVH5dR0plX32OxcWDep+EYqm0noV5UE9oOXdgSU1aY2KDodStaIGnfKW6kop5U1FoJW9lTFVXd2VoerbZnHShop68mKB5W1ZoKGJrW2Bsi6m6iHXIwoCBxqrGuKp7ZV1nQWRsaH/G09XX0KvQ17+juNHMxdS+gXJjyJhof8TH1NXX2Nnc2tjZ2trY19XStnlhycGyyc/R09XW19jZ2trb2tfW1NPSz6Zk"},
    {"label": "2차-07 (10 얹어짐)", "ref": REF_B, "slots": {10: "stack"}, "sig": "QFtrbW9wcXBqYF1cWVZSTUY/Oi0qLz16Kyg1SVthZGZjVUZBPDgzLikjICQpMD9+KSUoJSc0Njo7OzczMC0sKiQfICQpNUGCKiQoJCMvNj9GR0M/PT9DRkE/Qz0qP0yBMCYpJiQxPE62u8LAyMK+ubWyonssSFd/Mi0qKCUuOkR0ck9MRWpzhXGqHxwxPFN8MzQsKCUrOUFOcVdQTHCXpoypGCQ1PVd7NDYtLCUnN0BFg5u9dXKLrpSlR3FDQl12NTcsNScoOkdIhpW9enmJs42kg3pLSmJyNjkyPC0qQEZLgZW+mod/oZqsmXVQTmRyOzg2PTYwQ0pQhZu0lYF3vJecm3hQUWdvPjo8QEA7V2Vmeae6kod0wZOUmXVUU2tvQ0BGRUxBaoCClanFkHZqvJimmnhbVmxvT0hXVlRYsbmhkKfSknFzxZiZo3VdWW5qWktZTVGAk5KJhq7YoHiAvqG1q3VmXm5uY05ZUoCSjaCdjanOq4CWu7vQzJh9bmp2a2JticvR19rcudTjzbbN2srW2dKMd2iHbWBysMPR0tTX2dzd3dra3NrY1tS7dmaU"},
    {"label": "2차-08 (8 얹어짐)", "ref": REF_B, "slots": {8: "stack"}, "sig": "KSkuRUxMS0U8NDEvLSwqJyQfICQsM0FIJiYjIy80O0A/PDs4NjY2MikjIyUsPkhSJicjIjJhsK62uauMr7q4uLWxraakVF9dJyokJDJBV2NGNDItPj0+H0hsMyUtOkVVKCwlJS02P089ICc0O1JWaYSdJhouO0ZaLC0mJSk3QFJVYnpcW3qFqrC3KU0+PkhfLTIqJSUyPk99c5V2aHV6samiRYBKRExiMTMyJyYzQ02GjKajeH2NuLGohIlLTVBmODQ9LCg5RUiIi62/fHuKurGqko9RUlJoPTJBNjE6R0yAm5+/f3iJurCdkZFUU1VsPzdDPzxEW2R7j7DDin2IwK6cmZFZVVhuRD5ESUNNa2l1ob3DeXZ+vq+mmpNcWVlvRkxOW0Z1tLiEmMnMe3eJxa6XnpFhXVtwS1tUTGOwtax5jcnVj3iOyq6srpBpZGFwSFVIW211d4aCc8XSmXyZybjG1bGBeGxtUV9elrS8wce7ltjmtprB2s7R2NiygnVqYWqGzdHT09bX2t7e3dva3dza19XTnXVoXXW2vM3Q0dTW19rb29nZ2dnY19XSzYVp"},
    {"label": "2차-09 (0·3 얹어짐)", "ref": REF_B, "slots": {0: "stack", 3: "stack"}, "sig": "KixBWmFlZFpUU1BMSEZDQDozKCgsLz5NKSgqKTxJSkk/OTc2MzEwLSkkIicsMUNUKSYpIyQxNj0/Pz48OTk8Pjs4PS4vPEtdLyQqJCQ2VFhxlJCan52jv726uq6kXVxnNCgrJyU1OkZhiXNwbmp2gkiAXiwuP0llNC8sJyUuOURHSB4oSGGCnJeuZhw0P0tpMjMwKCQsO0VHU194ZnSYtMCvY01KQ05sMjUxLiUpOkhKYnmZYHSPtsKfiHhTSlFtNDgxOCgqPklNYoHBh36Os72kpIRRUVRwNjsyPy8tREpQZI7Md3qIo8ejqItRU1ZxODw1QTczS1NYYpbIfHmNt8ybpo5UVFd0PD47Q0I9Y2psc43FfnuAt8uenJBZVlp3PkFHTE9FfpKThXXIdXB3usyenJJdWVx3Q0tZWE50uLedh4nHim52ws6XrpFjXmBzTUpWSWF4foF6fanClHR1wtO0uZFwaWhwWVNeX5Soo7atkMXmrnmAyc+/186SfXFrZmZvoc/S1NXVztrb2c7Q2d3b19XHhnNramGNssPNztHU1dbY2NfY29nX1tTRuXNq"},
    {"label": "2차-10 (7·12 얹어짐)", "ref": REF_B, "slots": {7: "stack", 12: "stack"}, "sig": "ND5icnR1dG9iT0c6Njk4NjU0MzU3Jy9nNS0pPlplZ2diSnKOhYyNj4qkoKywODVsOiomKSY3QUJDPnCIeIyXpJKqpqetJTtyOy0lKCMkLzlMTW+Af3x4eXpzbWBkJ0B2PDMmKSQjNUBfY2Ruc4F7dHt2W3JqPUl6OzQvKyUiOD9LYC8dIiqIID5oKCo0NUF9QjU2LSYiND9LZ1llLVayaox+EhYrNkGBTDQ3MSkjLz5KcXeunmW2oLmjJD9XP0aAVTY4MjEmLkJOc36ixm6zqrCYKFJnRUt8YDc9NjwsNUdNd3uq0IGXuq2RLlxnRVB6azo+NkI0OUpRdoykxXKqrrWNPmdpSFZ2dD5CPEo+SWJqfJC1v2qzsa+JTm9pSlxzfEJIRlNFV4F+fKTDtVKxr6+XWXllTV9zf0xTXVpNnry6gKTPuUm+tq6VbIRhUGJveVROVExukY+Se6HStlTFua2ynpddWWZud11VXHqTnZ6ehaXPunnMu7rJyZ9da2podWhsfMbY2tvdytjk3L/V3tPR1cyZeGhmbGhpsb7Q1NTX2t3f3tvf3dzb1tTPiWZq"},
    {"label": "2차-11 (4·12 종이면)", "ref": REF_B, "slots": {4: "paper", 12: "paper"}, "sig": "IiMgJDM1NjYzLywrKikoJSEcGyElJjE6ICMdHSYsMjg5ODYzMjExLiYgHiImLDQ7ICUgHygxOUFEREE6P0NQSy9SXSUmMDZAJCYjIig1O3VNQUJBYWyFgliVmR0nMThDKiclIyUvNl1cd1tXYYKUlbqrjBstMztHLSwnIyQvNGhoob2hZWmHnLWrh1RPOT1JLystIyQsMlprlL6laXiQn7eikmVXPkFMNCs1JigxNkx2k7u6g3iEoLGem3BVQkRPNiw6LzA5PkGClK+5e3WBocKgo3lSQ0dTODE6ODk8RUZ7mai/g3aMrMaYnoBRRklXOjc9QkNMWFV5mMPFhXiFqsibl4hQSE1aPkRGUU1QfpF+lL7KfG+Brsiak41PTFBeRVNOWkx4mIKAirzRj2+GwMmVnY9VU1NfR1VHTWGIkp2IdLzTnHSIvc6mp41gW1tkTFlMZYmSjZ6sc7nSonmNwMe1yruCb2ZjWGNrsLzI0NHNnNfbzK691NbM0M7AfW9hXGWevMfJy87Q09XV1dTT1dTS0M7KuHRhW5y4wcjKzM3O0dPT09LS09LQzszMy7KJ"},
    {"label": "2차-12 (5·10 종이면)", "ref": REF_B, "slots": {5: "paper", 10: "paper"}, "sig": "FhISGCAjKCwsKyonJiYnJiAaFBQXGh8lGBMUGSRjNzY1NDRQV2F7LS94eh0YHCUlGRUWGSRmRx81SlVTenKkUGKbkhQWHSklGhUWFiFaVFqvX2pgiImkn72XfRIVIisnHhYVFCBhWWi0pnhbaXGqpbeichdEPDArIxsVFR5OY2ueqIdZeH6jprSPkIltRDg0JSIXGB84ZW6atI9tfXyjuLKVnZNzQjw4JigeHyUrbmaWtJxrd36RvLGTnJR9PD87JisjKSoveWucsKtmdoSvvbCJk5qDPkA9JjAqMDI9c1GnqLVsdIWyw6iNi5t9REI+KTY1NjpOfGqvs7hsa3ynx6WZhJF5SUE/NUNGPkuNiHiYuMRtbH2uyaWZkpNzTEVBOUFFUod8dHiOvcd4a4PFzKCWhJlrUEhDNzpPc3qVemN/xcqAbY++z6Knq5plV01HQEd5g3V8jHNswbWEcJLD0LG2u6t2Y1pQUYistMXKzJOEztCehKfP07vHz8/DdmRdeLrBxMbGysbO09LQ0dTW1NLPz83Jtmlfq6m/w8XGycvMz8/Oz9HR0c/OzMvIxKdg"},
    # 0~3 은 종이면을 보이며 기울어 있음, 4~9 살짝 기울어 애매, 11(C++) 종이면, 12(국어사전) 없음
    {"label": "2차-13 (0~3 종이면+기울어짐, 11 종이면, 12 없음)", "ref": REF_B,
     # YOLO 가 0~3 을 한 박스로 묶어 버리는 사진 → 1~3 은 판정불가도 인정
     "slots": {0: ("paper", "tilted"), **{k: ("paper", "tilted", "unknown") for k in range(1, 4)}, **{k: ("normal", "tilted") for k in range(4, 10)},
               11: "paper", 12: ("absent", "unknown", "paper")},
     "sig": "HBwaGycqLjAuLCwqKikoJSAaGR8jIys1Hh4bGysuNjw6OTczMTEyLyYgHSEjKC43ICIdHS43QERBQD0/Tlh1MkCbvZMiKC04IiQeID2ZizoyOj9JZICoU3+bxXsjJy46JSUfISiSqKudSmxSaJ+ohLikxnMtKTA9IiogHyRvr8LDa7JUZXitgb6mxo8/MDdDJCwmICM6tsXPmLRYboyrlrutxpBANDpHKy8tIyQ1lrzQtbl5e5WKrqy1x4Y7ODtOLywzKCY7ebnFwbp3doaqp8G5xX87Nz5UMyw4LSxCXLXBy7V9eoiyrcC8t3s+OEBYOS05NTRRXKXB0LyJbn2ssbq+rHxAO0JdPTVAPUZtjKy6x8aQX3u2urjApXpDPUNhQ0NCW3CZn6HHscewXnfGwbfEpHZJQ0ZjREhJZG6Ei4TLtcmrY3zKw7rFpnVXS01iR1FThp6br6nCu9HMa3/Mx7bExbZyYlheWGWhx8rLzc/Q1dfWwsbY08K5wMqqa19dVI2jvsXHyc7Q0tXU0dLU1NTT0MrHmWFagb2/xsjJys3P0NLS0NHS0dDPzs3KxpaP"},
]
# 기준 사진 204705 에서 6번(안드로이드 앱 만들기)은 이미 뒤집혀 꽂혀 있다 (사진을 확대해 보면 글자가 거꾸로다).
# 판정은 '기준 대비 변화'이므로, 다른 사진에서 6번이 바로 서 있으면 '뒤집힘'이 맞다. 기준을 다시 찍으면 이 집합을 비우면 된다.
REF_B_FLIPPED_IN_BASELINE = {6}
for _e in DEFAULT_TRUTH:
    if _e["ref"] == REF_B and _e.get("file") != REF_B:
        for _s in REF_B_FLIPPED_IN_BASELINE:
            if _s not in _e["slots"]:
                _e["slots"][_s] = "upside"
            elif _e["slots"][_s] != "normal":
                _v = _e["slots"][_s]
                _e["slots"][_s] = tuple(_v) + ("upside",) if isinstance(_v, tuple) else (_v, "upside")

# ---------------------------------------------------------------- 3차 세트: 멀리서 찍은 사진 (32권) + 추가된 책
# 번호가 아니라 '책 이름'으로 정답을 적는다. YOLO 가 기준에서 몇 권을 놓쳐도 번호가 밀리지 않도록, 기준 사진(1280px 좌표)의
# 책 띠(x1, x2)로 슬롯 -> 이름을 잇는다 (슬롯 quad 의 가운데 x 가 들어가는 띠). 기준에서 검출되지 않은 책의 정답은 채점에서 빠지고 요약에 적힌다.
REF_C = "20260522_203633.jpg"
REF_C_BANDS = [   # (이름, x1, x2)  왼쪽부터. 기준 203633 의 YOLO 탐지(1280px)에서 딴 값
    ("멈추면", 248, 296), ("믿음A", 291, 341), ("믿음B", 336, 379), ("THINK", 374, 392), ("실크로드", 387, 415), ("귀머거리", 409, 449),
    ("외교관", 448, 487), ("토토", 484, 519), ("밤하늘", 515, 542), ("북극", 534, 562), ("색과빛", 557, 587), ("뉴문", 589, 661),
    ("비교정치", 655, 703), ("친구들", 699, 735), ("그건사랑", 731, 765), ("독수리", 759, 794), ("시골의사", 789, 826), ("나는나를", 826, 862),
    ("공자", 858, 894), ("바보빅터", 888, 925), ("STRUCT", 918, 960), ("주체이론", 956, 1003), ("나무", 1001, 1040), ("소나기", 1035, 1064),
    ("게임", 1059, 1103), ("변증법", 1096, 1131), ("글읽기", 1124, 1152), ("바보처럼", 1147, 1177), ("덕혜옹주", 1172, 1218), ("신약성경", 1211, 1239),
    ("Che", 1228, 1286), ("CARROTS", 1262, 1280),
]
# 기준 203633 에서는 32권 모두 바로 서 있다. (204315 는 같은 배열을 다른 각도에서 찍은 사진인데 믿음A·믿음B·비교정치·독수리 4권이
#  거꾸로 꽂혀 있다 -> 그 사진에서만 그 4권이 '뒤집힘')
_UP = "upside"
_NT = ("normal", "tilted")                       # 살짝 기울어 애매
_PT = ("paper", "tilted", "unknown")             # 앞마구리를 보이며 기울어짐
_COVER = ("paper", "unknown", "absent", "tilted", "stack")   # 표지를 앞으로 돌려 세움 (책등이 안 보임)
_STACK = ("stack", "absent", "unknown")          # 뽑아서 위에 눕힘 (자리는 빔)
_HID = ("normal", "unknown", "absent")           # 다른 책에 가려 거의 안 보임
_LEAN = {k: _NT for k in ("시골의사", "나는나를", "공자", "STRUCT", "주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주")}
DEFAULT_TRUTH += [
    {"file": REF_C, "ref": REF_C, "slots": {}, "sig": "1nFxZ3B0dHRzdHRzdXd2dnZ2dXRyb2xn2XR2al1eYGNjYF1bXV5fXlxbWllYV1VS23h4cV5PRk1OTElHRkRDQT8+Pj08Ozky2356eGZTO0VGQ2lCQUE/Pjs6PDk5NjEp3IV9fnBXPllJVX5ERURCQT5FPz47Ny8q3Y+FiXhuYKSDa38/cUdVR0t9RDZASist3ZqVoLPC0qZ3jYNOsLCpr8iVk2hHhnBr3aCXorfH0qGTiYZHnsu0mMyTroBWkos93KaZpLfF1qWQiYxFlMeogMGUsIpWjY1H3KubprfF16yQhJMrjau1g7SUuJZRf35I27CcqLfL2LCRiZEthK+kgsuiuqVEg4xG2rSeq6m62LeYkIk0dLGtWr2juLVFg5dK2LifrrXJ17+opJFdhrnDUbiwuMVGg5lF0r2isr3U2MGqpZR+k7zCVoylu7VBfqpYx8Gls7rO2cGfspmap8LNv725xsa2uL2yx8OlsMXO0MvAycfIzMnJxsjIx8a+v7+3zsOmw8rKysjHx8bHx8bFwsPExsbDxMO+0MrHzcvKzMrLzMvLy8rJysjGxMG/vLiv"},
    {"file": "20260522_204315.jpg", "label": "3차-01 (다른 각도, 믿음A·B·비교정치·독수리 거꾸로)", "ref": REF_C, "sig": "fHJqcnd3d3h5eXl6e3t7enl3dnVzcXBufHlrYWVoaWtrbGxucHFycnJycW9ubWxpfHxyZFpUWFpZV1VXWFlZWVlbW1xcXFlTf396bF9FQUVFUUNDQ0JAPjw8PDw7NzMvf4J/dWRISUtUeUhJSkdEPzw9Ozk3MCwyhIeHfmtXimZedkJKTEpGQlFAPjw5M0BUjJafq7fFoGeRdUeHZnFkcoFGOUVHL0Njj5qlu8W/o4OJeETBubexy5OHZE+LaWyfkZ2pu8nBpYOJgki0ypCTzJGqZlmebkWRk6CsvcnHqIaChzKmwYV+vJCwdlqVcUh2laOvu87Iqoh/jiyasJ+GuJW1hlOAc0VSlqSwqcfHrYuGjSyTq7J6waq1mU58jT9OmKizts7Is6CZkD6XpbVTta2ysFd1pEJWm6y4xNLOuqqhknWqqMVNqbmytWVtrC9ZnrC6wNbRsJ6nmYGrvcVxjqW0tWZotllaorC6ytHRxbXHrMLNysvDxMLDwry5vbKspLrKy8zLysnIycrKysvGyMbFwr29vrazwsrHx8jIycrIycnHx8bFxsbEw8C/vbi2",
     "slots": {"믿음A": _UP, "믿음B": _UP, "비교정치": _UP, "독수리": _UP, "Che": _NT, "CARROTS": _NT}},
    {"file": "20260522_204416.jpg", "label": "3차-02 (10권 뒤집힘)", "ref": REF_C, "sig": "eHVnYWlrbW5vbm1vcnR1dnd3dnV0cnFvenluX1ZTWFpaWFZWV1laWlpbXFxdXVpUe3t1Y1lBQERFVERFREREQkFBQkFAPDUufH56bl5FR0pOfUhKS0lIRUNEQ0E+NjA7foB/c2BKlHNOiURKTEtKR1lGRUI+OENOhpCVm7DAoH2RjUGXb3VlbYxYOklVNEFmiZaessTWppqqk0OhwK+WxZijbEiGc2iWi5mhtMTYppypoUaPz5x7uZmshV2DdC+PjZuktcLVsJyVlUOCxJ1mrJ+vkmV5kjlwj52otMrWsZqRiEp2ssBmtaWxmmlypi9SkKCrrsDYtYOidlhpr89cubqtp2h0sSpRkqKtq8HYwIelcWprtMVbsciouXxzuzlUlKayutHay42leIWLxMdoo9Kru4pmw0tLlam0uc7Zy4yikoKQuM+Ii5+uw4h4vIJ3maqwwM7UzZ/Lw7fRys/IycrKysTBxri0nqvEysvNzcvKys3Nzc3KzM3Ly8jGycG9ssnKycrLy83NzMzMy8zLzc/Qz8/OzcrIzc7MysrKysrKysnJycbAu7eyraiim5GG",
     "slots": {"THINK": _UP, "실크로드": _UP, "귀머거리": _UP, "밤하늘": _UP, "독수리": _UP, "그건사랑": _UP, "나는나를": _UP, "변증법": _UP,
               "글읽기": _UP, "덕혜옹주": _UP}},
    {"file": "20260522_204231.jpg", "label": "3차-03 (5권 뒤집힘)", "ref": REF_C, "sig": "2np1gYSFhYWFhIOCg4KAf317enl2c3Ny2350cXp7fHt8fHx8fX19fXx7eXd1cnJv2n97bWlsbm9vbm1ub3BxcXFxb21ramdg2oB+dGdWUVdXVlRSUE9PTUxKSUhGQDgx2oOAfGtTRUxMXlBJSEVEQD4/PDs3LC0824eEhXNWTltMdl5PTU1LR0pGQkE6Okhh3JCOk4SKmJR/cGJ0WWJVWH1BPERINl2G3JeZpLLRupeQgo+utrS1xp5/WEyZfYG125qbp77PvZifdpGgxLSVx5avX0Glak21252dqb7IxpqjcmRuv7F8s5y2ZUOWa06S2qCfrL/OyZmgcGpssLOFt5q5cUqGdkJ52aOhrq7QzYWma2hgsK90xJe6hVN/jj5v2KWjsLLQ1IisnG5jsrpVv52yoFVxojtm2KintcbS3JCxrHRqvMRVpaK9nkNjrzJk1qqqtsHS14uxv4aCv8uZtKzBt5meu6Ch1K2qtsjOzrLIxrrHyMfDxsTFw728wLa20K+zyMfIyMbGxcXFxcTBw8PExMLCw769z8bMysnKy8rLy8rKysrIyMfEwb+9t66k",
     "slots": {"귀머거리": _UP, "뉴문": _UP, "STRUCT": _UP, "주체이론": _UP, "게임": _UP}},
    # 210909: 29권 전부 왼쪽으로 8~9° 쏠림 (Che·CARROTS 만 수직). 사용자 기준(가까운 사진 세트)에서 10° 안팎의 도미노 쏠림은
    #         정상으로도 봤으므로 전부 둘 다 인정. (이 사진의 '정답'은 기울어짐 임계값을 몇 도로 두느냐의 문제)
    {"file": "20260522_210909.jpg", "label": "3차-05 (왼쪽부터 전체가 8~9° 쏠림)", "ref": REF_C, "sig": "fWx4fYCAgICAfn+AgH9+fXt5dnV3en6DfHBpcXd4eXl5eHl7fX19fXx6d3Rxb21senZrZGdrbm9ubW5wc3R2d3Z1dXRycXBteXhzZ19QUldXVlVXWVpZWVlaW1xdXFlRent4bWJISUtSSUhIRURBPj07OTg4MyojfYB+dmhITktwWk1QUVBMSEhHREE9Nj5QhIyLhpZyhoVnbFJWVk9HaUdGRUI9NEZhipabocrNmoJ9e1+8mZyfnnxITGpXXHihjpudkszLq4eKiDy+wquwrZ6gUHakQXS1kJ6fhcfHxnadgkeaxJWUrpemaV61RF+ulKCeiLXOzYmXc116r7CBrJiufUqcVUeOlaCfjpjL0KaJeG1Tr7ZuvKCukkiRZUuBlqKhlpjL0MGcnoRRuLpenK+ptEt9gkt1mKSmoKHW1M6aq5d7rMJ5eLuruGVinj9tmaelqrrJzdKVqqCYqcemi7OsvIBwr3uUm6W1v8LJy82+vr++xcfCwcTDw724ubCvpL3Av7/CxcHAwcDBwsLBwsHAwby7uLOzxsfGxsbGx8bGxsbGxsbGxsXEwsC/vLm4",
     "slots": {k: _NT for k in ("멈추면", "믿음A", "믿음B", "THINK", "실크로드", "귀머거리", "외교관", "토토", "밤하늘", "북극", "색과빛", "뉴문", "비교정치",
                                "친구들", "그건사랑", "독수리", "시골의사", "나는나를", "공자", "바보빅터", "STRUCT", "주체이론", "나무", "소나기", "게임",
                                "변증법", "글읽기", "바보처럼", "덕혜옹주")}},
    # 210901: 북극·색과빛·뉴문 12° 이상 쏠림. 밤하늘은 기준 대비 거의 안 기울었고, 비교정치·친구들·그건사랑은 7~9° (둘 다 인정)
    {"file": "20260522_210901.jpg", "label": "3차-06 (가운데 책들 쏠림)", "ref": REF_C, "sig": "wcLFyMzQ09bZ293c3NrY1tTSz8vCta6jkHZ8foGBgYB/fX5/gYKDg4KDgoKDhoiJk3Jyen+BgoF/fX+BgoKBgH99e3l3d3d1kntvaGttcXFwbW1wcnNzcnFxcXBvbWtikX13amNOUFNUUU9PTkxJRkVERENBOjAokoB/dGpNUFFeUFBRUE5KRkZDQj86MjtQk4OEfGtTZU95VFFTVFFLU0xIRkU9PlB0l4+Um7mshHlia291dXFuj248SFZKWIObmZScsc20kn1Rb2LBtLO2r56RTHuBZ6+ympegssvDjoY/gD22voubspiiWnuPRKqym5qjs8vJjoREcEeXsJuHqZysZmOHSny8nZ6nr87Ol45Ic2B5pqOBwZ6zbFaaQXHFnqGrp83UoJWBh3RhrLpXtJ63gU6vQGvFoKWxvtTcp5ullZF2usRhpau6lE+oRWfMoqqyu9PWppq3vqCWrcqPla66mmyqfJDGpaq8yczNwcXDv8HEysrExMTFwbq9tra7tcbIyMfIx8TCwcPExsfExsTEwby+uLi7y8vKysrKy8vMzMzLysrJyMfFw8K/vbu5",
     # 북극은 12° 이상. 색과빛·뉴문은 선반 앞턱 롤 기준으로 7~8° 라 둘 다 인정 (기준 203633 자체가 카메라 롤 -2.9° 에 책들이 +2.6° 쏠려 있음)
     "slots": {"북극": "tilted", **{k: _NT for k in ("색과빛", "뉴문", "밤하늘", "비교정치", "친구들", "그건사랑", "독수리")},
               "Che": _NT, "CARROTS": _NT, "실크로드": _HID, "나무": ("normal", "unknown")}},
    {"file": "20260522_205820.jpg", "label": "3차-07 (STRUCT 위에 눕힘 + 그림책 1권 추가)", "ref": REF_C, "sig": "d3mAgYKCgoB/f4CBgIB+fn17eXd3dnVwf29pb3JwbmdcVlhdYmhsb3FycnFwbmhdgXdnX19fYFpNQD5ARUpOUlZZWllSS0M8gn5uYVdCSlWMVzs6PEBCQ0NBPzkzMC5Eg4J4Z1dETX07W3t5Z1BKS0xIRDw5RW6PhoiEblZyaYM6IiI1U3N+d2FLQjxIX32QkJuhss+NeKFVY7Voc3FzS1dRaE9hiaexkqCsxdSYipBpZ8DItsOtqnZVmI1MlbG2laSwxNGjgYt+ULrFg7SprZBajKArfrC1mKezxNCtfIKEOq22k56lpp5na6IpTqu5m6u1xtO1gYSQO565iKa4oKV2VK85QZe8na2zqtK5lJCTOZe2kGy6nbiIUapXPHy9oLG7yNXFop2ddqyxpWCtn8SXUqBwNWPJo7W9yNfCmaKkbp6usGePoMKfU4+WNHu8p7O+zNLIq8S4us3NzMPEvcjDvsG+tLS8q8TMy8vJyMjJy8zMy8jMysnHw8fCvb2/zM7NzM3NzczMzMvLzc7Q0NDOzMzKx8bDra2sq6qpp6SioJ6bm6KppqOfmpeOiYN9", "extras": 1,
     "slots": {"STRUCT": _STACK, "Che": _NT, "밤하늘": ("normal", "unknown"),
               **{k: _NT for k in ("주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주")}}},
    {"file": "20260522_205729.jpg", "label": "3차-08 (친구들·바보빅터 위에 눕힘)", "ref": REF_C, "sig": "xHdvcXt9fn5/gH+BgoOEg4KAfXp4dnVyxXh4aWVqa2phWldZXmNpbnFzcnBubWpmx3d7bl9TUlVQRT89Pj5AQkVKT1NSUEtFyXl+dmNROkSKi3JdTj42ODk7PDo1MjEwzHqAfmtTP0lagTpTY3aPimZLRkA7NDhPz3+FhnJVU3Njg3OIiX+FmYd/VD83Q1Z00omXoK/LzXmchk6DV2lehVA9S1tHTXCX1IybpcHMxoOki2eAu8GzrKh/VHGlXHCu1o6dqMHMz4uTjXJssJ6KuJ6YbWGuVkyf2ZCfq8LM1JCKjm9Swo92qKGtdlyYaDds2pKir8DS05x9kXY+zoBuqqa9f1SAliBL3JOksKbS2KCFk3kvy4hoibO4qFVjtTRB3JantbzW3aGmsoBhv5t1YrmuuX1UrFo425qruMjY3KGrq5N9r8STWqClvoVLlYIp2p2ttsPY3KGmup2kt83Aq8O9xK+Vra2G16GsvcvQ0crFysjJysbFxMbGxsG9wLqz1Ka+ysjJy8jJx8fJyMXEw8XFwsC+v7m10cvPzczNzc3NzczMy8rKycnHxcPBvru5",
     "slots": {"친구들": _STACK, "바보빅터": _STACK, "독수리": _NT, "그건사랑": ("normal", "unknown"), "신약성경": _NT, **_LEAN}},
    {"file": "20260522_210700.jpg", "label": "3차-09 (비교정치·독수리 종이면, 나무 표지)", "ref": REF_C, "sig": "kHRueX9/fn58fHt6enp5eXh3dnVzcnBtk3pwaGxwcHFvbWxsbGtrampqZ2ZkY2Bblnp4a2FaWl5eW1hXVVNRT05NTEpKSEE3m3x9c2ZUPUdHUkRDQT8+PDk5OTY1MCsnoH6AemxXQU1OgUhLSkhIQ0BCQj89NjpHp4KFgHFXW39VkT1KSEZEPnE/Pz86MT9Xr42YmbnSwYaBljtai2ibnpx9PkxnUVyCs5Gco8rUt5CJk0hdzI25yKOXVEtstEZ7uJSfpsnSy42TiVhjxZiBv6egUVxkvEBmvZahqcnT14mXglZ9s8uiraWun51GqE1NwZilrcbV2I+VfmNnrMysrLqwrsBTolJNxZmnranW4JichmlQtci5gMWnsb1jnWJSyJurtMDY46qfsHtywMG2b9antbp7l3JOzJ6vuMnc4a+ls4R/u7ytd7GsubGXg4dIz6Gytsjd3rytvpq7wsK2rsu+zMC4qrWu0KSxxc/T0s7MysnLw8PHysvIyMfFv7mz0azIy8vLy8zKyMnJysrJycrHxsbEwbu109LS0M/Pz9DQzs7Nzc3Ly8zLycjFwsC9", "extras": 1,
     "slots": {"비교정치": _PT, "독수리": _PT, "나무": _COVER, "실크로드": ("normal", "unknown"), "CARROTS": _HID}},
    {"file": "20260522_210604.jpg", "label": "3차-10 (추가 1권 표지, 종이면 3권, 공자 표지)", "ref": REF_C, "sig": "hHlvdnt8fX5+fn+AgIGBgX9+fXt5eHh2g4BwZ2ltcHBub3Byc3Z2d3Z1dXRzc3Nwg4J3aWBUWVpaXF5fX2FfXl5gYWNjYlxShoZ+b2RHTExZTkxKSUlFQkJCQUE9NzIwh4iDd2VGVk52YFJTUVBMSElJRUQ7NT5VjJGOgo1Pl4B4YE5RT01JdE5MSEQ7TFx7kJujvtWGm4uLZ1uipJF4tXpAT15BVnygkp+pw9iIm5uIb1equL2Kw7mTU3SaaqDAmKWwxtiem5mPc2avuYWdw7WcZmq1OpXAnKq5y9qVnZKNdWK5vIm2wMGje1GvS2a2nq69zdmWrY+MfEi0uoK5usOteE2rTFKgoLC+sNuWtaGVgDi7wICJvb69gk+nW1GLorXE0d2XvLKvjH68u5BzvrvGm1KhZld2pbnI0eCUs7eson+ztKFsurjClkmQd1t7qLzH1+GtybrBtMLGutDIy8XOw6+4vrm5q77T19nR0dLNz9DKy8/O0dHNyMTIw7/BwtTT0tHP0tPQ0M/Pz87O0dHQzcvLyMbG09PR0dDQ0NHR0dHR0dDPzszKycjFwr63", "extras": 2,
     "slots": {"비교정치": "paper", "친구들": "paper", "STRUCT": "paper", "공자": _COVER,
               "믿음A": ("normal", "paper", "unknown"), "믿음B": _HID, "THINK": ("normal", "unknown"), "그건사랑": ("normal", "unknown"), "바보빅터": _HID, "CARROTS": _NT}},
    {"file": "20260522_210632.jpg", "label": "3차-11 (친구들·독수리 앞으로 빼 종이면, 추가 1권 표지)", "ref": REF_C, "sig": "lXVqb3h4eXp7e3p8fHt7e3p5eXh2dHRzlHdxZWJlaWxta2ptb29wcXFxcXBubm1plHp4aFxNT1RWVlVXWVlZWFlbXV5eXFZNln5/c2NHRkpJXkVFRENCPz4/P0E9NjEvmIGEemdFWE9ljElNTEpHQ0JEQ0I5MzpQnpCUjJ6nmIVwlUhbTUpFWlBHR0Q7Sll5oZehsdfXn4aMoFKibZKZo49FO2BBWn6jopqkttbXmpaBrECZnqO7uKSYMHWdZZu8pJ2mt9LSsZmDuz6msZiluJ2hOmiyO5S8pqCqt9fSt517vy21yrySq6aoY4y2RVy1qKOts9fVvaJ+tzmuwLKlx6etbXS+RU+mqaWvqNDUxbaGwEaqva2H0aG7gFnCVFCTq6m0vt7Xzb+Uzoa2uqKVzKrDkVbEXFKCra24vt7Nwbmaz42or6CopqvGfVDLfHGOsLC3yNfX1cLBysjUwMS/x8jJwZm4uLO2srfNz9LS0dLNz9DR0dDS0MzNycjQx8DBy9TS0dLS09PT0tLS0tLT1dXV09LRzszJsbGurKqopqOfnJqXlJKUoJqVj4mJeHJs", "extras": 1,
     "slots": {"친구들": ("paper", "tilted", "unknown", "upside"), "독수리": _PT, "그건사랑": _HID, "시골의사": ("normal", "paper", "unknown"),
               "게임": _HID, "변증법": _HID, "글읽기": _HID, "바보처럼": _HID, "CARROTS": _NT}},
]

TRUTH_KR = {"upside": "뒤집혀짐", "stack": "엎어짐", "paper": "종이가 보임", "tilted": "기울어짐", "absent": "없음"}
EV_KEYS = ("best_score", "evidence_quality", "tilt_signed", "tilt_src", "paper_score", "paper_ref", "paper_delta", "upside_total",
           "e_strip", "e_profile", "e_sift", "sift_theta", "sift_inliers", "aspect_rel", "above_row", "hyp90_margin", "best_hyp",
           "recovery", "paper_sig_distance", "slot_kind")
BACKEND_MIN_CONF, MIN_DET, MAX_DET, MAX_UNRES = 0.5, 0.8, 1.2, 0.3     # backend/analyzer.py 와 같은 값


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def short(state: str) -> str:
    return state.replace("abnormal_", "")


def image_sig(pil: Image.Image) -> np.ndarray:
    """사진 내용 서명(24x18 밝기). 파일명이 바뀌어도 같은 사진을 찾기 위한 것."""
    im = pil.convert("L")
    im.thumbnail((1280, 1280))
    return cv2.resize(np.array(im), (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)


def _decode_sig(b64: str) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(b64), np.uint8).astype(np.float32).reshape(SIG_H, SIG_W)


def _norm_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    slots = {}
    for k, v in (e.get("slots") or {}).items():
        key = int(k) if isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()) else str(k)
        slots[key] = tuple(v) if isinstance(v, (list, tuple)) else v
    out = {"file": e.get("file"), "label": e.get("label") or e.get("file"), "ref": e.get("ref"), "slots": slots,
           "sig": _decode_sig(e["sig"]) if e.get("sig") else None, "extras": int(e.get("extras", 0) or 0)}
    return out


REF_BANDS = {REF_C: REF_C_BANDS}


def slot_names(ref) -> Dict[int, str]:
    """기준 슬롯 번호 -> 책 이름 (REF_BANDS 에 띠가 있는 기준만). 슬롯 quad 의 가운데 x 가 들어가는 띠 (기운 책은 bbox 가 이웃과 겹치므로 bbox 를 쓰지 않는다)."""
    bands = REF_BANDS.get(ref.file)
    if not bands:
        return {}
    out = {}
    for s in ref.slots:
        cx = float(np.mean(np.asarray(s.quad, float)[:, 0])) if s.quad is not None else (s.box["x1"] + s.box["x2"]) / 2.0
        best, bd = None, 1e9
        for name, b1, b2 in bands:
            d = 0.0 if b1 <= cx <= b2 else min(abs(cx - b1), abs(cx - b2))
            if d < bd:
                best, bd = name, d
        if best is not None and bd <= 12:
            out[s.slot] = best
    return out


def resolve_truth_slots(t: Dict[Any, Any], names: Dict[int, str]) -> Tuple[Dict[int, Any], List[str]]:
    """이름 키 정답을 슬롯 번호 키로. 기준에서 검출되지 않은 이름은 건너뛰고 돌려준다."""
    by_name: Dict[str, List[int]] = {}
    for sn, nm in names.items():
        by_name.setdefault(nm, []).append(sn)
    out: Dict[int, Any] = {}
    missing: List[str] = []
    for k, v in t.items():
        if isinstance(k, int):
            out[k] = v
        elif k in by_name:
            for sn in by_name[k]:
                out[sn] = v
        else:
            missing.append(k)
    return out, missing


def load_truth(path: Optional[str]) -> List[Dict[str, Any]]:
    entries = [_norm_entry(e) for e in DEFAULT_TRUTH]
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):                                   # 옛 형식 {"사진.jpg": {"3": "upside"}}
            raw = [{"file": k, "slots": v} for k, v in raw.items()]
        for e in raw:
            ne = _norm_entry(e)
            entries = [x for x in entries if not (ne["file"] and x["file"] == ne["file"])] + [ne]
    return entries


def find_truth(entries: List[Dict[str, Any]], fn: str, pil: Image.Image) -> Optional[Dict[str, Any]]:
    """파일명으로 먼저, 없으면 사진 서명으로 정답 항목을 찾는다."""
    for e in entries:
        if e["file"] == fn:
            return e
    sg = image_sig(pil)
    best, bd = None, 9.0
    for e in entries:
        if e["sig"] is None:
            continue
        d = float(np.abs(e["sig"] - sg).mean()) / 255.0
        if d < bd:
            best, bd = e, d
    return best if bd < SIG_MATCH else None


def truth_text(t: Dict[int, Any]) -> str:
    if not t:
        return "전부 정상"
    groups: Dict[str, List[Any]] = {}
    for slot, v in sorted(t.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        key = v if isinstance(v, str) else "/".join(v)
        groups.setdefault(key, []).append(slot)
    return ", ".join(f"{','.join(map(str, s))}번 {'/'.join(TRUTH_KR.get(x, x) for x in k.split('/'))}" for k, s in groups.items())


def montage(crops: List[np.ndarray], labels: List[str], path: str, h: int = 240) -> None:
    tiles = []
    for c, lb in zip(crops, labels):
        if c is None or c.size == 0:
            c = np.full((h, 40, 3), 128, np.uint8)
        im = Image.fromarray(c)
        w = max(24, int(im.width * h / max(im.height, 1)))
        im = im.resize((min(w, 400), h))
        tile = Image.new("RGB", (im.width + 8, h + 34), (245, 245, 245))
        tile.paste(im, (4, 30))
        ImageDraw.Draw(tile).text((4, 4), lb, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    W = sum(t.width for t in tiles)
    out = Image.new("RGB", (W, h + 34), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    out.save(path, quality=85)


def backend_quality(items: List[dict], expected: int) -> Dict[str, Any]:
    """backend/analyzer.assess_session_quality 와 같은 규칙으로 '인식 신뢰도' 판정을 흉내 낸다."""
    detected = len(items)
    ident = [i["book_id"] for i in items if i["book_id"] != "UNKNOWN" and i["confidence_score"] >= BACKEND_MIN_CONF]
    unident = sum(1 for i in items if not (i["book_id"] != "UNKNOWN" and i["confidence_score"] >= BACKEND_MIN_CONF)
                  and i["visual_status"] != "abnormal_paper")
    dup = len(ident) - len(set(ident))
    reasons = []
    if detected == 0:
        reasons.append("인식 0권")
    else:
        r = detected / max(expected, 1)
        if r < MIN_DET:
            reasons.append(f"인식 부족 {detected}/{expected}")
        elif r > MAX_DET:
            reasons.append(f"인식 과다 {detected}/{expected}")
        if (unident + dup) / detected > MAX_UNRES:
            reasons.append(f"미확인 {unident} + 중복 {dup} / {detected}")
    return {"reliable": not reasons, "reasons": reasons, "unidentified": unident, "duplicated": dup, "detected": detected}


def crop_of(rgb: np.ndarray, r: dict) -> Optional[np.ndarray]:
    q = as_quad(r.get("quad")) if r.get("quad") else None
    c = warp_spine(rgb, q) if q is not None else None
    if c is None:
        H, W = rgb.shape[:2]
        b = clip_box(r["box"], W, H)
        c = rgb[b["y1"]:b["y2"], b["x1"]:b["x2"]]
    return c


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="shelf_audit 평가: 기준 사진으로 검사 사진을 판정하고 정답과 비교")
    p.add_argument("--normal", default=os.path.join("test", "normal"))
    p.add_argument("--check", default=os.path.join("test", "check"))
    p.add_argument("--truth", default=None, help="정답표 JSON (없으면 내장 정답표)")
    p.add_argument("--out", default="eval_logs")
    p.add_argument("--replay", default=None, help="이전 로그 폴더로 모델 없이 재현")
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--conf", type=float, default=None)
    a = p.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_replay" if a.replay else "")
    log_dir = os.path.join(a.out, stamp)
    work = os.path.join(log_dir, "_work")                 # 마지막에 replay.zip 으로 묶고 지운다
    os.makedirs(os.path.join(work, "images"), exist_ok=True)
    CFG.DUMP_DETECTIONS, CFG.DUMP_DIR = True, os.path.join(work, "detections")
    extracted = None
    if a.replay:
        rz = os.path.join(a.replay, "replay.zip")
        if os.path.isfile(rz) and not os.path.isdir(os.path.join(a.replay, "detections")):
            import zipfile
            extracted = os.path.join(a.replay, "_unzipped")
            with zipfile.ZipFile(rz) as z:
                z.extractall(extracted)
            a.replay = extracted
    if a.no_fallback:
        CFG.FALLBACK_DETECT = False
    if a.conf is not None:
        CFG.PIPELINE_CONF = a.conf
    tee = Tee(os.path.join(log_dir, "summary.txt"))
    sys.stdout = tee

    if a.replay:
        detector = DumpDetector(dump_dir=os.path.join(a.replay, "detections"))
        emb = ReplayEmbedder(os.path.join(a.replay, "embeddings.npz"))
        print(f"♻️ 재현 모드: {a.replay} (탐지 덤프 + 임베딩 기록 사용)")
        if os.path.isdir(os.path.join(a.replay, "photos", "normal")):     # 로그에 든 1280px 사진으로 재현 → 원본 사진 없이도 된다
            a.normal, a.check = os.path.join(a.replay, "photos", "normal"), os.path.join(a.replay, "photos", "check")
            print(f"   사진도 로그 안의 것을 씁니다: {os.path.join(a.replay, 'photos')}")
    else:
        detector = get_detector()
        emb = RecordingEmbedder(get_embedder())
    set_detector(detector)
    set_embedder(emb)
    truth = load_truth(a.truth)
    t_start = time.time()

    # ---------------------------------------------------------------- 1) 기준 사진
    normal_files = list_images(a.normal)
    if not normal_files:
        print(f"❌ 기준 사진이 없습니다: {a.normal}")
        return 1
    manifest = load_manifest()
    if not a.replay:                                                       # 재현용으로 1280px 사진을 로그에 넣는다
        for sub_dir, files in (("normal", normal_files), ("check", list_images(a.check))):
            os.makedirs(os.path.join(work, "photos", sub_dir), exist_ok=True)
            for f in files:
                prepare_image(Image.open(f).convert("RGB")).save(os.path.join(work, "photos", sub_dir, os.path.basename(f)), quality=92)
    for f in normal_files:
        manifest.setdefault(os.path.basename(f), {"shelf_key": os.path.splitext(os.path.basename(f))[0].split("_")[0]})
    refs = [build_shelf_reference(f, detector, emb, Diagnostics(), manifest) for f in normal_files]
    pool = RefPool(shelves=refs)
    if pool.multi:
        pool.shelf_min = max(CFG.SHELF_MIN_SCORE, report_shelf_overlap(pool, emb) + 0.05)
    pool.catalog = ShelfCatalog(refs)
    ref_by_file = {r.file: r for r in refs}
    # 기준마다 단독 풀/감사기: 정답표가 기준을 지정한 사진은 자동 서가 식별 없이 그 기준으로 판정한다.
    # 카탈로그(타 서가 도서 출처 찾기)는 전체 기준을 공유한다
    single = {}
    for r in refs:
        sp = RefPool(shelves=[r])
        sp.catalog = pool.catalog
        single[r.file] = (sp, ShelfAuditor(sp, detector=detector, emb=emb))
    with open(os.path.join(work, "reference.json"), "w", encoding="utf-8") as f:
        json.dump([{
            "file": r.file, "shelf_key": r.shelf_key, "n_slots": len(r.slots),
            "tau": {k: r.meta[k] for k in ("tau_floor", "tau_ok", "tau_neg")}, "med_h": r.meta["med_h"], "med_w": r.meta["med_w"],
            "audit": r.audit, "series": {g: s.members for g, s in r.series.items()},
            "slots": [{"slot": s.slot, "kind": s.kind, "source": s.source, "box": s.box, "w": s.w, "h": s.h, "paper_score": round(s.paper_score, 3),
                       "ambiguous_orient": s.ambiguous_orient, "M1": round(s.M1, 3), "M2": round(s.M2, 3), "edge": round(s.edge, 2),
                       "axis": s.axis, "lsd": s.lsd, "series_id": s.series_id} for s in r.slots]} for r in refs],
            f, ensure_ascii=False, indent=1, default=to_py)
    for r in refs:
        montage([s.crop for s in r.slots], [f"s{s.slot} {s.kind}{' fb' if s.source != 'yolo' else ''}" for s in r.slots],
                os.path.join(work, f"reference_slots{'_' + r.shelf_key if len(refs) > 1 else ''}.jpg"))

    # ---------------------------------------------------------------- 2) 검사 사진
    check_files = list_images(a.check)
    if not check_files:
        print(f"❌ 검사 사진이 없습니다: {a.check}")
        return 1
    auto_auditor = ShelfAuditor(pool, detector=detector, emb=emb)
    rows: List[dict] = []
    per_image: List[dict] = []
    confusion: Dict[str, Dict[str, int]] = {}
    print("\n========================================================")
    print(f"📁 검사 사진 {len(check_files)}장 | 기준 {', '.join(f'{r.file}({len(r.slots)}권)' for r in refs)}")
    print("========================================================\n")
    for path in check_files:
        fn = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        entry = find_truth(truth, fn, img)
        t = entry["slots"] if entry else None
        t_missing: List[str] = []
        if entry and entry["ref"] in single:
            ref = ref_by_file[entry["ref"]]
            pool_used, auditor = single[entry["ref"]]
            ref_how = "정답표 지정"
        elif len(refs) == 1:
            ref, pool_used, auditor, ref_how = refs[0], pool, auto_auditor, "유일한 기준"
        else:
            ref, pool_used, auditor, ref_how = None, pool, auto_auditor, "자동 서가 식별"
        an = auditor.analyze(img, fn)
        sm = an["summary"]
        if ref is None:
            ref = ref_by_file.get(sm.get("shelf", {}).get("file"), refs[0])
        names = slot_names(ref)
        if t is not None:
            t, t_missing = resolve_truth_slots(t, names)
        backend = build_backend_result(img, an, pool_used, fn, save_crops=False)
        items = backend.get("vision_items", [])
        q = backend_quality(items, len(ref.slots))
        results = an["results"]
        abn = [r for r in results if r["state"] != "normal"]
        n_norm = sum(1 for r in results if r["state"] == "normal")

        print(f"🔍 분석 중: {fn}" + (f"  [정답표: {entry['label']}]" if entry and entry["label"] != fn else ""))
        print(f"├─ 기준: {ref.file} ({ref_how}) | 총 {sm['total_books']}권 | 정상 {n_norm} | 이상 {len(abn)}")
        lines = [f"{r['slot']}번{(' ' + names[r['slot']]) if r['slot'] in names else ''} 책: {KR.get(r['state'], r['state'])} ({r['state']})" for r in abn]
        lines += [f"{i['slot']}번 책: 순서 이상 ({i['found_position']}→{i['expected_position']}번 자리)" for i in sm["order_issues"]]
        if not lines:
            print("├─ ✅ 결과 : 이상 없음")
        elif len(lines) == 1:
            print(f"├─ ⚠️ 결과 : {lines[0]}")
        else:
            print("├─ ⚠️ 결과")
            for ln in lines:
                print(f"├─ ⚠️ {ln}")
        warn_codes = [w["code"] for w in an["diagnostics"]["warnings"]]
        qtxt = "신뢰" if q["reliable"] else "낮음 (" + " / ".join(q["reasons"]) + ")"
        n_extra = sum(1 for e in sm.get("extra_boxes", []) if e.get("source", "yolo") == "yolo")
        exp_extra = entry["extras"] if entry else 0
        extra_txt = f" | 기준에 없는 책(미매칭 박스) {n_extra}" + (f" (예상 {exp_extra})" if exp_extra else "")
        print(f"├─ 🧪 백엔드 품질: {qtxt} | 미확인 {q['unidentified']}/{q['detected']}{extra_txt}" + (f" | 경고 {warn_codes}" if warn_codes else ""))
        if t_missing:
            print(f"├─ ℹ️ 기준에서 검출되지 않아 채점에서 뺀 책: {', '.join(t_missing)}")

        # 정답 비교 (책 단위)
        verdict = "정답 없음"
        img_ok = None
        if t is not None:
            miss = fp = typ = 0
            for r in results:
                exp = t.get(r["slot"], "normal")
                exp_set = set(exp) if isinstance(exp, (list, tuple, set)) else {exp}
                pred = short(r["state"])
                ok = pred in exp_set
                if not ok:
                    if "normal" in exp_set:
                        fp += 1
                    elif pred in ("normal", "unknown", "absent"):
                        miss += 1
                    else:
                        typ += 1
                tk = "|".join(sorted(exp_set))
                confusion.setdefault(tk, {})
                confusion[tk][pred] = confusion[tk].get(pred, 0) + 1
                rows.append({"file": fn, "slot": r["slot"], "name": names.get(r["slot"], ""), "truth": tk, "pred": pred, "ok": ok, "state_conf": round(r["state_conf"], 3),
                             "backend_conf": next((i["confidence_score"] for i in items if i["debug"].get("slot") == r["slot"]), None),
                             "source": r["source"], "flags": ",".join(r["flags"]), "match_similarity": round(float(r["match_similarity"] or 0), 4),
                             **{k: r["evidence"].get(k) for k in EV_KEYS}})
            extra_miss = max(0, exp_extra - n_extra)                    # 추가된 책을 기준 책으로 잘못 흡수했거나 못 찾음
            n_wrong = miss + fp + typ + extra_miss
            img_ok = n_wrong == 0 and not sm["order_issues"]
            verdict = "✅ 일치" if img_ok else f"❌ 불일치 {n_wrong}건 (놓침 {miss} · 오탐 {fp} · 유형 {typ}" + (f" · 추가책 놓침 {extra_miss}" if extra_miss else "") + (f" · 순서오탐 {len(sm['order_issues'])}" if sm["order_issues"] else "") + ")"
        print(f"└─ 📌 정답 : {truth_text(t) if t is not None else '-'}  → {verdict}\n")

        per_image.append({"file": fn, "ok": img_ok, "quality": q, "n_abnormal": len(abn), "ref": ref.file, "ref_how": ref_how,
                          "extras_found": n_extra, "extras_expected": exp_extra,
                          "truth_label": entry["label"] if entry else None, "truth": {str(k): v for k, v in (t or {}).items()}})
        with open(os.path.join(work, "images", os.path.splitext(fn)[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump({"analysis": an, "backend": backend, "quality": q, "truth": {str(k): v for k, v in (t or {}).items()}},
                      f, ensure_ascii=False, indent=1, default=to_py)
        rgb = np.array(prepare_image(img))
        montage([crop_of(rgb, r) for r in results],
                [f"s{r['slot']} {short(r['state'])} {r['state_conf']:.2f} {r['source'][:6]}" + (f" T:{short(str(t.get(r['slot'], 'normal')))}" if t is not None else "") for r in results],
                os.path.join(work, "images", os.path.splitext(fn)[0] + ".jpg"))

    # ---------------------------------------------------------------- 3) 통계
    scored = [r for r in rows]
    n = len(scored)
    n_ok = sum(1 for r in scored if r["ok"])
    abn_rows = [r for r in scored if r["truth"] != "normal"]
    norm_rows = [r for r in scored if r["truth"] == "normal"]
    miss = sum(1 for r in abn_rows if not r["ok"] and r["pred"] in ("normal", "unknown", "absent"))
    typ = sum(1 for r in abn_rows if not r["ok"] and r["pred"] not in ("normal", "unknown", "absent"))
    fp = sum(1 for r in norm_rows if not r["ok"] and r["pred"] not in ("unknown",))
    unk_norm = sum(1 for r in norm_rows if r["pred"] == "unknown")
    imgs = [i for i in per_image if i["ok"] is not None]
    stats = {
        "images": len(check_files), "images_scored": len(imgs), "images_exact": sum(1 for i in imgs if i["ok"]),
        "books": n, "books_ok": n_ok, "book_accuracy": round(n_ok / max(n, 1), 4),
        "abnormal_total": len(abn_rows), "abnormal_caught": sum(1 for r in abn_rows if r["ok"]),
        "missed_abnormal": miss, "type_confused": typ, "false_alarm_on_normal": fp, "unknown_on_normal": unk_norm,
        "normal_total": len(norm_rows), "normal_ok": sum(1 for r in norm_rows if r["ok"]),
        "backend_low_quality_images": [i["file"] for i in per_image if not i["quality"]["reliable"]],
        "confusion": confusion, "elapsed_sec": round(time.time() - t_start, 1), "replay": a.replay, "package_hash": package_hash(),
    }
    if not a.replay:
        emb.save(os.path.join(work, "embeddings.npz"))
    else:
        stats["replay_embed_hits"], stats["replay_embed_misses"] = emb.hits, emb.misses
    print("========================================================")
    print(f"📊 전체 통계 (정답표가 있는 {len(imgs)}장 / 책 {n}권)")
    print(f"   이미지 단위 완전 일치 : {stats['images_exact']}/{len(imgs)}")
    print(f"   책 단위 정확도        : {n_ok}/{n} ({stats['book_accuracy'] * 100:.1f}%)")
    print(f"   이상 책 맞힘          : {stats['abnormal_caught']}/{len(abn_rows)}  (이상→정상 놓침 {miss} · 유형 오분류 {typ})")
    print(f"   정상 책을 이상으로 오탐 : {fp}/{len(norm_rows)}  (정상→미확인 {unk_norm})")
    print(f"   백엔드 '인식 신뢰도 낮음' 이 뜨는 사진: {len(stats['backend_low_quality_images'])}장 {stats['backend_low_quality_images']}")
    unk_rows = [r for r in scored if r["pred"] == "unknown"]
    unk_ok = sum(1 for r in unk_rows if r["ok"])
    stats["unknown_total"], stats["unknown_allowed"] = len(unk_rows), unk_ok
    print(f"   판정불가(unknown) 책      : {len(unk_rows)}권 / {n} ({len(unk_rows) / max(n, 1) * 100:.1f}%)  "
          f"(가려짐·표지 등 정답표가 허용한 자리 {unk_ok} · 그 외 {len(unk_rows) - unk_ok})")
    by_ref: Dict[str, List[dict]] = {}
    for i in imgs:
        by_ref.setdefault(i["ref"], []).append(i)
    if len(by_ref) > 1:
        print("   기준별 사진 일치 : " + " | ".join(f"{k} {sum(1 for i in v if i['ok'])}/{len(v)}" for k, v in by_ref.items()))
    ex_imgs = [i for i in imgs if i["extras_expected"]]
    if ex_imgs:
        print(f"   추가된 책(기준에 없는 책) 찾기 : {sum(min(i['extras_found'], i['extras_expected']) for i in ex_imgs)}/{sum(i['extras_expected'] for i in ex_imgs)} "
              f"(사진 {len(ex_imgs)}장)")
    n_auto = sum(1 for i in per_image if i["ref_how"] == "자동 서가 식별")
    if n_auto:
        print(f"   정답표에 기준이 없어 자동 서가 식별로 판정한 사진: {n_auto}장")
    print("   혼동 행렬 (정답 → 예측):")
    for tk, preds in confusion.items():
        print(f"      {tk:8s} → " + ", ".join(f"{k} {v}" for k, v in sorted(preds.items(), key=lambda kv: -kv[1])))
    if a.replay:
        print(f"   재현 임베딩: 기록 적중 {emb.hits} (그중 근사 일치 {emb.near}) / 근사 대체 {emb.misses}")
    print(f"   소요 {stats['elapsed_sec']}s | 로그: {os.path.abspath(log_dir)}")
    print("========================================================")

    with open(os.path.join(work, "books.csv"), "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(work, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({**stats, "per_image": per_image}, f, ensure_ascii=False, indent=1, default=to_py)
    with open(os.path.join(work, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: getattr(CFG, k) for k in dir(CFG) if k.isupper()}, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(work, "env.txt"), "w", encoding="utf-8") as f:
        f.write(f"python {platform.python_version()} {platform.platform()}\nnumpy {np.__version__}\n")
        try:
            import cv2
            f.write(f"opencv {cv2.__version__}\n")
        except Exception:
            pass
        try:
            import torch
            f.write(f"torch {torch.__version__} cuda={torch.cuda.is_available()}\n")
        except Exception:
            f.write("torch (없음)\n")
        f.write(f"shelf_audit {package_hash()}\n")

    # 아래 두 표는 summary.txt 에만 쓴다 (콘솔에는 안 찍는다)
    print(f"📋 기준 슬롯 · 책별 상세(증거값) 표는 {os.path.join(log_dir, 'summary.txt')} 에 있습니다")
    sys.stdout = tee.f
    print("\n📋 기준 슬롯 (번호 = 왼쪽부터 0)")
    for r in refs:
        print(f"   {r.file}: 매칭 하한 {r.meta['tau_floor']:.3f} / 확신 {r.meta['tau_ok']:.3f}")
        nm = slot_names(r)
        print("   " + " | ".join(f"{s.slot}:{s.kind[0]}{'*' if s.source != 'yolo' else ''} w{int(s.w)}" + (f" {nm[s.slot]}" if s.slot in nm else "") for s in r.slots))
        if REF_BANDS.get(r.file):
            missing = [n for n, _, _ in REF_BANDS[r.file] if n not in nm.values()]
            if missing:
                print(f"   기준에서 검출되지 않은 책: {', '.join(missing)}")
        for au in r.audit:
            print(f"   - [{au.get('level', 'warn')}] {au['message']}")
    print("\n📋 책별 상세 (file | slot | 정답→예측 | OK | conf | 경로 | 플래그 | 핵심 증거)")
    for r in rows:
        ev_txt = " ".join(f"{k}={r[k]}" for k in ("best_score", "upside_total", "e_sift", "sift_inliers", "tilt_signed", "paper_score", "paper_delta", "aspect_rel") if r.get(k) is not None)
        print(f"   {r['file']} | {r['slot']:2d}{(' ' + r['name']) if r.get('name') else ''} | {r['truth']}→{r['pred']} | {'O' if r['ok'] else 'X'} | {r['state_conf']} | {r['source']} | {r['flags'] or '-'} | {ev_txt}")
    sys.stdout = tee.out
    tee.f.close()

    import shutil
    import zipfile
    zpath = os.path.join(log_dir, "replay.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(work):
            for fn in fs:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, work))
        z.write(os.path.join(log_dir, "summary.txt"), "summary.txt")
    shutil.rmtree(work, ignore_errors=True)
    if extracted:
        shutil.rmtree(extracted, ignore_errors=True)
    print(f"📦 남긴 파일: {os.path.join(log_dir, 'summary.txt')} (읽는 용) · {zpath} (보내는 용)")
    return 0


if __name__ == "__main__":
    sys.exit(main())