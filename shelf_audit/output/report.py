# -*- coding: utf-8 -*-
"""[평가 전용] 정답표 비교 + test 폴더 대량 분석 -> test_results.json (형태는 v4 와 동일)."""
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

from ..config import IMG_EXTS, KR, TEST_DIRS
from ..geometry import to_py
from ..imaging import list_images
from ..pipeline import makeResult
from ..reference.pool import RefPool
from ..serialize import pool_from_any
from .backend import build_backend_result

# 번호는 정상 사진의 왼쪽부터 0번. 값은 upside / stack / paper / tilted. 괄호로 묶으면 둘 다 정답.
SHELF_TRUTH: Dict[str, str] = {}
GROUND_TRUTH: Dict[str, Dict[int, Any]] = {
    "20260522_203503.jpg": {},
    "20260522_203708.jpg": {3: "upside", 5: "upside", 9: "upside"},
    "20260522_203732.jpg": {1: "upside", 5: "upside", 10: "upside"},
    "20260522_203815.jpg": {0: "upside", 2: "upside", 8: "upside", 11: "upside"},
    "20260522_205058.jpg": {7: "stack"},
    "20260522_205117.jpg": {2: "stack"},
    "20260522_205137.jpg": {6: "stack", 10: "stack", 11: ("upside", "tilted"), 12: ("upside", "tilted")},
    "20260522_205210.jpg": {0: "stack", 1: "stack"},
    "20260522_210223.jpg": {4: "paper", 11: "paper"},
    "20260522_210248.jpg": {10: "paper"},
    "20260522_210313.jpg": {1: "paper", 8: "paper", 12: "paper"},
    "20260522_210803.jpg": {2: "tilted"},
    "20260522_210814.jpg": {5: "tilted"},
}
EV_KEYS = ("best_score", "evidence_quality", "tilt_signed", "paper_score", "paper_delta", "upside_total", "e_strip", "e_profile",
           "e_sift", "sift_inliers", "aspect_rel", "above_row")


def _short(state: str) -> str:
    return state.replace("abnormal_", "")


def score_against_truth(filename: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    truth = GROUND_TRUTH.get(filename)
    if truth is None:
        return None
    rows, wrong = [], []
    for r in analysis["results"]:
        exp = truth.get(r["slot"], "normal")
        exp_set = set(exp) if isinstance(exp, (tuple, list, set)) else {exp}
        pred = _short(r["state"])
        ok = pred in exp_set
        rows.append({"file": filename, "slot": r["slot"], "truth": "|".join(sorted(exp_set)), "pred": pred, "ok": ok,
                     "source": r["source"], "flags": ",".join(r["flags"]), **{k: r["evidence"].get(k) for k in EV_KEYS}})
        if not ok:
            wrong.append((r, "|".join(sorted(exp_set)), pred))
    return {"ok": not wrong, "wrong": wrong, "rows": rows}


def run_test_folder_to_json(reference_pool, test_dir: Any = None, output_json: str = "test_results.json") -> None:
    reference_pool: RefPool = pool_from_any(reference_pool)
    print("\n========================================================")
    print("📁 [테스트 폴더 분석] (번호는 정상 사진 왼쪽부터 0번)")
    print("========================================================")
    dirs = [test_dir] if isinstance(test_dir, str) else list(test_dir or TEST_DIRS)
    files: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"   - {d}: 폴더 없음")
            continue
        imgs = list_images(d)
        if not files and imgs:
            files = imgs
            print(f"   - {d}: 이미지 {len(imgs)}장  ← 이 폴더를 사용")
        else:
            print(f"   - {d}: 이미지 {len(imgs)}장" + ("  (사용 안 함)" if imgs else ""))
    if not files:
        print(f"⚠️ 분석할 이미지({', '.join(IMG_EXTS)})가 없습니다. 읽은 폴더: {dirs}")
        return
    print(f"총 {len(files)}개의 테스트 이미지를 분석합니다...\n")
    final = {"reference_file": reference_pool.meta["file"], "test_results": []}
    report_rows, overview = [], []
    n_files = n_files_ok = n_books = n_books_ok = n_abn = n_abn_ok = n_fp = n_miss = n_type = n_ord = 0
    n_shelf_truth = n_shelf_ok = 0
    for path in files:
        fn = os.path.basename(path)
        print(f"🔍 분석 중: {fn}")
        try:
            img = Image.open(path).convert("RGB")
            an = makeResult(img, reference_pool, fn)
            sm = an["summary"]
            sh = sm.get("shelf") or {}
            backend = build_backend_result(img, an, reference_pool, fn)
            for w in an.get("diagnostics", {}).get("warnings", []):
                print(f"   ├─ ℹ️ [{w['code']}] {w['message']}")
            if reference_pool.multi:
                scs = ", ".join(f"{k} {v:.2f}" for k, v in sh.get("scores", {}).items())
                if sm.get("shelf_unknown"):
                    print(f"   ├─ 🗄️ 서가 식별 실패 ({sh.get('reason')}) | {scs}\n   └─ ⏭️ 판정을 건너뜁니다.\n")
                    final["test_results"].append({**backend, "truth_check": None})
                    overview.append({"file": fn, "shelf": "(식별 실패)", "books": sm["total_books"], "abn": "판정 건너뜀", "truth": "-"})
                    continue
                ok_txt = ""
                if fn in SHELF_TRUTH:
                    n_shelf_truth += 1
                    good = SHELF_TRUTH[fn] == sh.get("file")
                    n_shelf_ok += int(good)
                    ok_txt = " ✅" if good else f" ❌ (정답 {SHELF_TRUTH[fn]})"
                print(f"   ├─ 🗄️ 서가: {sh.get('file')}{'' if sh.get('confident') else ' ⚠️ 불확실'}{ok_txt} | {scs}")
            print(f"   ├─ 슬롯 {sm['total_slots']} | 검출 {sm['total_books']} | 정상 {sm['normal_count']} | 이상 {sm['abnormal_count']} | 누락 {sm['absent_count']}")
            shown = False
            for r in an["results"]:
                if r["state"] != "normal":
                    shown = True
                    print(f"   ├─ ⚠️ {r['slot']}번 책: {KR.get(r['state'], r['state'])} ({r['state']}) [{','.join(r['flags'])}]")
            for iss in sm["order_issues"]:
                shown = True
                print(f"   ├─ 🔀 {iss['slot']}번 책: 순서 이상 ({iss['found_position']}번 자리 -> {iss['expected_position']}번 자리)")
            for e in sm["extra_boxes"]:
                if e.get("foreign"):
                    shown = True
                    print(f"   ├─ 📚 타 서가 도서: {e['foreign']['book_id']} (출처 {e['foreign']['shelf_key']}, 인라이어 {e['foreign']['inliers']})")
            if not shown:
                print("   ├─ ✅ 이상 없음")

            sc = score_against_truth(fn, an)
            if sc is not None:
                n_files += 1
                n_files_ok += int(sc["ok"] and not sm["order_issues"])
                for row in sc["rows"]:
                    n_books += 1
                    n_books_ok += int(row["ok"])
                    if row["truth"] != "normal":
                        n_abn += 1
                        n_abn_ok += int(row["ok"])
                        if not row["ok"]:
                            if row["pred"] in ("normal", "unknown", "absent"):
                                n_miss += 1
                            else:
                                n_type += 1
                    elif not row["ok"]:
                        n_fp += 1
                n_ord += len(sm["order_issues"])
                report_rows.extend(sc["rows"])
                if sc["ok"] and not sm["order_issues"]:
                    print("   └─ 📌 정답 비교: ✅ 전부 일치\n")
                else:
                    print(f"   └─ 📌 정답 비교: ❌ 불일치 {len(sc['wrong'])}건" + (f", 순서 오탐 {len(sm['order_issues'])}건" if sm["order_issues"] else ""))
                    for r, exp, pred in sc["wrong"]:
                        ev = {k: r["evidence"].get(k) for k in EV_KEYS if r["evidence"].get(k) is not None}
                        print(f"        ❌ {r['slot']}번: 정답 {exp} / 예측 {pred} | 경로 {r['source']} | flags {r['flags']}\n           근거 {ev}")
                    print()
            else:
                print("   └─ 🔎 분류만 표시 (정답표 없음)\n")
            abn = [f"{r['slot']}번 {KR.get(r['state'], r['state'])}" for r in an["results"] if r["state"] != "normal"]
            abn += [f"{i['slot']}번 순서이상" for i in sm["order_issues"]]
            overview.append({"file": fn, "shelf": sh.get("file") or "-", "books": sm["total_books"], "abn": ", ".join(abn) or "없음",
                             "truth": "정답 없음" if sc is None else ("✅ 일치" if sc["ok"] and not sm["order_issues"] else "❌ 불일치")})
            final["test_results"].append({**backend, "truth_check": None if sc is None else {"ok": sc["ok"], "wrong_slots": [w[0]["slot"] for w in sc["wrong"]]}})
        except Exception as e:                      # noqa: BLE001
            import traceback
            print(f"❌ {fn} 분석 중 오류 발생: {e}")
            traceback.print_exc()
            overview.append({"file": fn, "shelf": "-", "books": "-", "abn": f"오류: {e}", "truth": "-"})
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=4, default=to_py)
    scored = [o for o in overview if o["truth"] in ("✅ 일치", "❌ 불일치")]
    plain = [o for o in overview if o not in scored]
    print("========================================================")
    print(f"📋 분석한 사진 {len(overview)}장 = 정답 채점 {len(scored)}장 + 분류만 {len(plain)}장")
    for o in scored + plain:
        print(f"   {o['file']} | 서가 {o['shelf']} | 검출 {o['books']} | 이상: {o['abn']} | {o['truth']}")
    if n_shelf_truth:
        print(f"🗄️ 서가 식별 정확도: {n_shelf_ok}/{n_shelf_truth}")
    if report_rows:
        pd.DataFrame(report_rows).to_csv("book_level_report.csv", index=False, encoding="utf-8-sig")
    if n_files:
        print("========================================================")
        print(f"📊 정답 비교 요약 (정답표가 있는 {n_files}장)")
        print(f"   이미지 단위 완전 일치: {n_files_ok}/{n_files}")
        print(f"   책 단위 정확도      : {n_books_ok}/{n_books} ({n_books_ok / max(n_books, 1) * 100:.1f}%)")
        print(f"   이상 책 맞힘        : {n_abn_ok}/{n_abn}  (놓침 {n_miss} + 유형 오분류 {n_type})")
        print(f"   정상 책을 이상으로 오판: {n_fp}건 | 순서 오탐: {n_ord}건")
        print("   책별 상세는 'book_level_report.csv'에 저장되었습니다.")
        print("========================================================")
    print(f"\n✅ 테스트 폴더 분석 완료! 결과가 '{output_json}' 파일로 저장되었습니다.")
