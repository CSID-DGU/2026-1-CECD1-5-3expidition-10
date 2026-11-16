# -*- coding: utf-8 -*-
"""명령행 진입점. `python -m shelf_audit [--image x.jpg]` (v4 의 `python MakeResult.py` 대체)."""
import argparse
import json
import sys

from . import analyze_image_to_dict, build_normal_reference_pool, run_test_folder_to_json
from .config import CFG


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="서가 이상 탐지 v5 (모듈 구조)")
    p.add_argument("--image", help="사진 한 장만 분석해 백엔드 JSON 을 표준출력으로 낸다")
    p.add_argument("--test-dir", help="테스트 폴더(기본: dataset/test, test)")
    p.add_argument("--out", default="test_results.json")
    p.add_argument("--conf", type=float, default=None, help="YOLO conf 임계값 덮어쓰기")
    p.add_argument("--no-fallback", action="store_true", help="비YOLO 보정 탐지 끄기")
    a = p.parse_args(argv)
    if a.conf is not None:
        CFG.PIPELINE_CONF = a.conf
    if a.no_fallback:
        CFG.FALLBACK_DETECT = False

    print("💡 서가 이상 탐지 시스템 v5 (모듈 구조 + 보정 탐지 + 시리즈 구분 + 타 서가 식별) 가동을 시작합니다.")
    pool = build_normal_reference_pool()
    if not pool:
        print("❌ 정상(Normal) 참조 데이터를 생성하지 못해 분석을 중단합니다.")
        return 1
    if a.image:
        print(json.dumps(analyze_image_to_dict(a.image, pool), ensure_ascii=False, indent=2))
        return 0
    run_test_folder_to_json(pool, test_dir=a.test_dir, output_json=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
