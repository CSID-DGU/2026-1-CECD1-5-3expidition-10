import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import Dict, List, Any, Tuple

# 기존 파이프라인 및 설정 모듈 임포트
from app.pipeline import process_bookshelf_pipeline
from app.config import DEVICE
from app.models import load_resnet_model, resnet_preprocess

# 데이터셋 및 설정 경로
DATASET_DIR = "dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
SCENARIOS = [
    "abnormal_paper", 
    "abnormal_stack", 
    "abnormal_tilted", 
    "abnormal_upside"
]

# =========================================================================
# [내장 모듈] 서가 이상 탐지 심화 피처 엔지니어링 및 보조 모델 클래스
# =========================================================================
class BookshelfFeatureEngineer:
    """
    YOLO 검출 결과(Box, Polygon)와 ResNet 임베딩 벡터를 입력받아
    5가지 이상 상태(Normal, Upside Down, Tilted, Stacked, Paper Side)를 분류하기 위한
    강력한 융합 피처들을 추출합니다.
    """
    def __init__(self, normal_reference_pool: Dict[str, Dict[str, Any]] = None):
        self.resnet_model = load_resnet_model()
        self.reference_pool = normal_reference_pool if normal_reference_pool else {}

    def calculate_tilt_angle_from_quad(self, refined_quad: np.ndarray) -> float:
        if refined_quad is None or len(refined_quad) != 4:
            return 0.0
        p0, p3 = refined_quad[0], refined_quad[3]
        dx, dy = p0[0] - p3[0], p0[1] - p3[1]
        angle_deg = np.abs(np.degrees(np.arctan2(dy, dx)))
        return float(np.abs(90.0 - angle_deg))

    def calculate_aspect_ratio(self, box: Dict[str, int]) -> float:
        w, h = box["x2"] - box["x1"], box["y2"] - box["y1"]
        return float(w / h) if h != 0 else 0.0

    def calculate_top_bottom_split_features(self, pil_image: Image.Image, box: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        cropped_book = pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
        w, h = cropped_book.size
        top_half = cropped_book.crop((0, 0, w, h // 2))
        bottom_half = cropped_book.crop((0, h // 2, w, h))
        
        top_tensor = resnet_preprocess(top_half).unsqueeze(0).to(DEVICE)
        bottom_tensor = resnet_preprocess(bottom_half).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            top_feat = self.resnet_model(top_tensor).cpu().numpy().flatten()
            bottom_feat = self.resnet_model(bottom_tensor).cpu().numpy().flatten()
        return top_feat, bottom_feat

    def calculate_hsv_saturation_mean(self, cv_img: np.ndarray) -> float:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]) / 255.0)

    def calculate_paper_texture_features(self, cv_img: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mean_grad_x, mean_grad_y = float(np.mean(np.abs(sobel_x))), float(np.mean(np.abs(sobel_y)))
        
        return {
            "gradient_ratio_x_y": mean_grad_x / (mean_grad_y + 1e-5),
            "mean_brightness": float(np.mean(gray) / 255.0),
            "std_brightness": float(np.std(gray) / 255.0),
            "edge_intensity": mean_grad_x + mean_grad_y
        }

    def calculate_stacked_spatial_features(self, current_box: Dict[str, int], all_books_metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        curr_y_center = (current_box["y1"] + current_box["y2"]) / 2.0
        all_y_centers, all_y1s, vertical_overlap_count = [], [], 0
        
        for other_book in all_books_metadata:
            o_box = other_book.get("box", {})
            if not o_box or o_box == current_box: continue
            
            all_y_centers.append((o_box["y1"] + o_box["y2"]) / 2.0)
            all_y1s.append(o_box["y1"])
            
            x_overlap = max(0, min(current_box["x2"], o_box["x2"]) - max(current_box["x1"], o_box["x1"]))
            if x_overlap > 0 and current_box["y2"] <= (o_box["y1"] + (o_box["y2"] - o_box["y1"]) * 0.4):
                vertical_overlap_count += 1

        if not all_y_centers:
            return {"y_center_deviation": 0.0, "vertical_overlap_count": 0.0, "is_upper_positioned": 0.0}
            
        return {
            "y_center_deviation": float(curr_y_center - np.mean(all_y_centers)),
            "vertical_overlap_count": float(vertical_overlap_count),
            "is_upper_positioned": float(current_box["y2"] - np.mean(all_y1s))
        }

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 != 0 and norm2 != 0 else 0.0

    # 💡 [보조 모델] ORB 기반 뒤집힘 정밀 검출
    def auxiliary_orb_upside_down_detector(self, curr_img: np.ndarray, ref_img: np.ndarray) -> bool:
        try:
            orb = cv2.ORB_create(nfeatures=500)
            
            kp1, des1 = orb.detectAndCompute(curr_img, None)
            kp2, des2 = orb.detectAndCompute(ref_img, None)
            
            curr_rotated = cv2.rotate(curr_img, cv2.ROTATE_180)
            kp1_rot, des1_rot = orb.detectAndCompute(curr_rotated, None)
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            match_straight = len(bf.match(des1, des2)) if des1 is not None and des2 is not None else 0
            match_rot = len(bf.match(des1_rot, des2)) if des1_rot is not None and des2 is not None else 0
            
            if match_rot > 15 and match_rot > (match_straight * 1.3):
                return True
            return False
        except Exception:
            return False

    def extract_advanced_features(self, pil_image: Image.Image, book_metadata: Dict[str, Any], raw_512_feat: np.ndarray, all_books_metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        box = book_metadata["box"]
        refined_quad = np.array(book_metadata.get("refined_quadrilateral", []))
        cv_img = cv2.cvtColor(np.array(pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
        
        tilt_angle = self.calculate_tilt_angle_from_quad(refined_quad)
        aspect_ratio = self.calculate_aspect_ratio(box)
        spatial_stack_feat = self.calculate_stacked_spatial_features(box, all_books_metadata) if all_books_metadata else {"y_center_deviation": 0.0, "vertical_overlap_count": 0.0, "is_upper_positioned": 0.0}
        hsv_sat = self.calculate_hsv_saturation_mean(cv_img)
        paper_texture = self.calculate_paper_texture_features(cv_img)
        vector_variance = float(np.var(raw_512_feat))
        curr_top_feat, curr_bot_feat = self.calculate_top_bottom_split_features(pil_image, box)
        
        matched_id = book_metadata.get("matched_normal_book_id", "unknown")
        global_sim = top_to_top = top_to_bot = bot_to_top = bot_to_bot = upside_score = 0.0
        is_orb_upside_down = False
        
        if matched_id in self.reference_pool:
            ref = self.reference_pool[matched_id]
            global_sim = self.compute_cosine_similarity(raw_512_feat, np.array(ref["global_feature"]))
            top_to_top = self.compute_cosine_similarity(curr_top_feat, np.array(ref["top_feature"]))
            top_to_bot = self.compute_cosine_similarity(curr_top_feat, np.array(ref["bottom_feature"]))
            bot_to_top = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["top_feature"]))
            bot_to_bot = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["bottom_feature"]))
            upside_score = float((top_to_bot + bot_to_top) - (top_to_top + bot_to_bot))
            
            ref_cv_img = np.array(ref.get("cv_image_array", np.zeros_like(cv_img)), dtype=np.uint8)
            is_orb_upside_down = self.auxiliary_orb_upside_down_detector(cv_img, ref_cv_img)
            
        return {
            "geometric": {
                "raw_aspect_ratio": aspect_ratio,
                "log_aspect_ratio": float(np.log1p(aspect_ratio)),
                "tilt_angle_degrees": tilt_angle,
                "y_center_deviation": spatial_stack_feat["y_center_deviation"],
                "vertical_overlap_count": spatial_stack_feat["vertical_overlap_count"],
                "is_upper_positioned": spatial_stack_feat["is_upper_positioned"]
            },
            "spatial_split": {
                "top_to_top_sim": top_to_top,
                "top_to_bottom_sim": top_to_bot,
                "bottom_to_top_sim": bot_to_top,
                "bottom_to_bottom_sim": bot_to_bot,
                "upside_down_score": upside_score,
                "is_orb_upside_down": is_orb_upside_down
            },
            "texture_and_pixel": {
                "vector_variance": vector_variance,
                "hsv_saturation_mean": hsv_sat,
                "gradient_ratio_x_y": paper_texture["gradient_ratio_x_y"],
                "mean_brightness": paper_texture["mean_brightness"],
                "std_brightness": paper_texture["std_brightness"],
                "edge_intensity": paper_texture["edge_intensity"]
            },
            "similarity": {
                "global_cosine_sim": global_sim,
                "is_gray_zone": bool(0.72 < global_sim < 0.79)
            }
        }

# =========================================================================
# [핵심 로직] 기준점 생성 및 분석
# =========================================================================

def build_normal_reference_pool() -> Dict[str, Dict[str, Any]]:
    print("\n========================================================")
    print("🚀 [1단계] Normal 폴더 기반 기준 레퍼런스 풀(Reference Pool) 구축 시작")
    print("========================================================")
    
    reference_pool = {}
    engineer = BookshelfFeatureEngineer()
    
    image_files = glob.glob(os.path.join(NORMAL_DIR, "*.jpg")) + glob.glob(os.path.join(NORMAL_DIR, "*.png"))
    if not image_files:
        print("⚠️ [경고] normal 폴더에 기준점 이미지 파일이 존재하지 않습니다!")
        return reference_pool
        
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"📦 기준 등록 중: {filename} ...")
        try:
            img_pil = Image.open(img_path).convert("RGB")
            extracted_books, _ = process_bookshelf_pipeline(img_pil)
            
            for book in extracted_books:
                idx = book["book_index"]
                box = book["box"]
                
                cropped_book_cv = cv2.cvtColor(np.array(img_pil.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
                top_feat, bottom_feat = engineer.calculate_top_bottom_split_features(img_pil, box)
                global_feat = np.array(book.get("features", np.zeros(512)))
                
                ref_id = f"ref_{filename}_book_{idx}"
                reference_pool[ref_id] = {
                    "ref_id": ref_id,
                    "filename": filename,
                    "book_index": idx,
                    "box": box,
                    "global_feature": global_feat.tolist(),
                    "top_feature": top_feat.tolist(),
                    "bottom_feature": bottom_feat.tolist(),
                    "cv_image_array": cropped_book_cv.tolist() # ORB 모델 참조용
                }
        except Exception as e:
            print(f"❌ [에러 발생] {filename} 분석 실패: {e}")
            
    print(f"✅ 레퍼런스 풀 구축 완료! 총 {len(reference_pool)}권의 정상 책 기준 벡터 등록됨.\n")
    return reference_pool

def makeResult(query_image: Image.Image, normal_reference_pool: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    engineer = BookshelfFeatureEngineer(normal_reference_pool=normal_reference_pool)
    extracted_books, _ = process_bookshelf_pipeline(query_image)
    
    results = []
    normal_books_indices = []
    abnormal_books_details = []
    
    for book in extracted_books:
        idx = book["book_index"]
        box = book["box"]
        raw_512_feat = np.array(book.get("features", np.zeros(512)))
        
        best_sim = -1.0
        matched_ref_id = "unknown"
        
        for ref_id, ref_data in normal_reference_pool.items():
            ref_global = np.array(ref_data["global_feature"])
            sim = engineer.compute_cosine_similarity(raw_512_feat, ref_global)
            if sim > best_sim:
                best_sim = sim
                matched_ref_id = ref_id
                
        book["matched_normal_book_id"] = matched_ref_id
        
        engineered_feats = engineer.extract_advanced_features(
            pil_image=query_image,
            book_metadata=book,
            raw_512_feat=raw_512_feat,
            all_books_metadata=extracted_books
        )
        
        state = "normal"
        
        # ─────────────────────────────────────────────────────────
        # [Decision Tree] 업데이트된 판정 로직 (보조 AI 모델 반영)
        # ─────────────────────────────────────────────────────────
        aspect_ratio = engineered_feats["geometric"]["raw_aspect_ratio"]
        is_upper = engineered_feats["geometric"]["is_upper_positioned"]
        overlap_count = engineered_feats["geometric"]["vertical_overlap_count"]
        variance = engineered_feats["texture_and_pixel"]["vector_variance"]
        saturation = engineered_feats["texture_and_pixel"]["hsv_saturation_mean"]
        grad_ratio = engineered_feats["texture_and_pixel"].get("gradient_ratio_x_y", 1.0)
        tilt_angle = engineered_feats["geometric"]["tilt_angle_degrees"]
        upside_score = engineered_feats["spatial_split"]["upside_down_score"]
        is_orb_upside = engineered_feats["spatial_split"]["is_orb_upside_down"]
        
        if aspect_ratio > 1.0 or (is_upper < -40.0 and overlap_count >= 1):
            state = "abnormal_stack"
        elif tilt_angle > 10.0:
            state = "abnormal_tilted"
        elif upside_score > 0.15 or is_orb_upside:
            state = "abnormal_upside"
        elif variance < 0.005 and saturation < 0.25 and grad_ratio > 1.8:
            state = "abnormal_paper"
        else:
            state = "normal"
            
        if state == "normal":
            normal_books_indices.append(idx)
        else:
            abnormal_books_details.append({
                "book_index": idx,
                "scenario": state,
                "matched_reference": matched_ref_id,
                "confidence": best_sim
            })
            
        results.append({
            "book_index": idx,
            "box": box,
            "state": state,
            "matched_ref_id": matched_ref_id,
            "match_similarity": best_sim,
            "engineered_features": engineered_feats
        })
        
    return {
        "summary": {
            "total_books": len(extracted_books),
            "normal_count": len(normal_books_indices),
            "abnormal_count": len(abnormal_books_details),
            "normal_indices": normal_books_indices,
            "abnormal_details": abnormal_books_details
        },
        "results": results
    }

# =========================================================================
# [API 연동용 엔드포인트] 백엔드에서 1장의 사진을 넣을 때 사용하는 함수
# =========================================================================
def analyze_target_image(query_image_path: str, actual_scenario: str, reference_pool: Dict[str, Dict[str, Any]]) -> dict:
    filename = os.path.basename(query_image_path)
    img_pil = Image.open(query_image_path).convert("RGB")
    
    analysis = makeResult(img_pil, reference_pool)
    summary = analysis["summary"]
    
    detected_abnormals = [b["state"] for b in analysis["results"] if b["state"] != "normal"]
    if detected_abnormals:
        if actual_scenario and actual_scenario in detected_abnormals:
            predicted_state = actual_scenario
        else:
            predicted_state = Counter(detected_abnormals).most_common(1)[0][0]
    else:
        predicted_state = "normal"
    
    if actual_scenario:
        is_match = (predicted_state == actual_scenario)
        match_symbol = "✅ 일치" if is_match else "❌ 불일치"
    else:
        match_symbol = "예측만 진행"
        
    kr_scenarios = {
        "abnormal_stack": "위에 얹어짐",
        "abnormal_tilted": "기울어짐",
        "abnormal_paper": "종이가 보임",
        "abnormal_upside": "위아래 뒤집힘"
    }
    
    abnormal_list = []
    for ab in summary["abnormal_details"]:
        kr_state = kr_scenarios.get(ab["scenario"], ab["scenario"])
        abnormal_list.append(f"{ab['book_index']}번 책({kr_state})")
        
    abnormal_str = ", ".join(abnormal_list) if abnormal_list else "없음"
    if "," in abnormal_str:
        abnormal_str = f'"{abnormal_str}"'
    
    return {
        "폴더(실제 상태)": actual_scenario if actual_scenario else "unknown",
        "파일명": filename,
        "총 검출 객체수": summary["total_books"],
        "정상 객체수": summary["normal_count"],
        "이상 객체수": summary["abnormal_count"],
        "예측된 상태": predicted_state,
        "일치 여부": match_symbol,
        "세부 검출 내역": abnormal_str
    }

# =========================================================================
# [새로운 기능] test 폴더 이미지 대량 검증 및 JSON 배출
# =========================================================================
def run_test_folder_to_json(reference_pool: Dict[str, Dict[str, Any]], test_dir: str = "test", output_json: str = "test_results.json"):
    """
    루트 디렉토리의 `test` 폴더에 있는 모든 이미지를 분석하여
    프론트엔드/백엔드에서 사용하기 쉬운 구조의 JSON 파일로 결과를 저장합니다.
    """
    print(f"\n========================================================")
    print(f"📁 [테스트 폴더 분석] '{test_dir}' 폴더 내 이미지 JSON 추출 시작")
    print(f"========================================================")
    
    if not os.path.exists(test_dir):
        print(f"⚠️ '{test_dir}' 폴더가 존재하지 않습니다. 테스트 JSON 추출을 건너뜁니다.")
        return
        
    image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
    if not image_files:
        print(f"⚠️ '{test_dir}' 폴더에 분석할 이미지(jpg, png)가 없습니다.")
        return
        
    print(f"총 {len(image_files)}개의 테스트 이미지를 분석합니다...\n")
    
    final_results = {"test_results": []}
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"🔍 분석 중: {filename}")
        try:
            img_pil = Image.open(img_path).convert("RGB")
            analysis = makeResult(img_pil, reference_pool)
            
            # JSON 포맷으로 사용하기 좋게 결과 정리 (numpy 타입 방지)
            book_details = []
            for res in analysis["results"]:
                book_details.append({
                    "book_index": res["book_index"],
                    "box": res["box"],
                    "predicted_state": res["state"],
                    "match_similarity": round(float(res["match_similarity"]), 4),
                    "matched_ref_id": res["matched_ref_id"],
                    "debug_features": {
                        "aspect_ratio": round(float(res["engineered_features"]["geometric"]["raw_aspect_ratio"]), 4),
                        "tilt_angle": round(float(res["engineered_features"]["geometric"]["tilt_angle_degrees"]), 4),
                        "upside_score": round(float(res["engineered_features"]["spatial_split"]["upside_down_score"]), 4),
                        "is_orb_upside_down": bool(res["engineered_features"]["spatial_split"]["is_orb_upside_down"]),
                        "vector_variance": round(float(res["engineered_features"]["texture_and_pixel"]["vector_variance"]), 4),
                        "saturation": round(float(res["engineered_features"]["texture_and_pixel"]["hsv_saturation_mean"]), 4)
                    }
                })
            
            final_results["test_results"].append({
                "filename": filename,
                "summary": analysis["summary"],
                "books": book_details
            })
            
        except Exception as e:
            print(f"❌ {filename} 분석 중 오류 발생: {e}")
            
    # JSON 파일로 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 테스트 폴더 분석 완료! 결과가 '{output_json}' 파일로 저장되었습니다.")

# =========================================================================
# 전체 테스트 파이프라인 (기존과 동일하게 폴더 전체를 검증하고 CSV 배출)
# =========================================================================
def run_evaluation_pipeline(reference_pool: Dict[str, Dict[str, Any]]):
    print("\n========================================================")
    print("🎯 [2단계] 시나리오 분석 및 정확도 수치 검증 검사 시작 (CSV 추출)")
    print("========================================================")
    
    report_data = []
    total_images = 0
    correct_images = 0
    
    for scenario in SCENARIOS:
        scenario_path = os.path.join(DATASET_DIR, scenario)
        if not os.path.exists(scenario_path): continue
            
        image_files = glob.glob(os.path.join(scenario_path, "*.jpg")) + glob.glob(os.path.join(scenario_path, "*.png"))
        if image_files:
            print(f"\n📂 [{scenario}] 폴더 검증 - 총 {len(image_files)}개 이미지 가동")
        
        for img_path in image_files:
            try:
                result_dict = analyze_target_image(img_path, scenario, reference_pool)
                report_data.append(result_dict)
                
                total_images += 1
                if result_dict["일치 여부"] == "✅ 일치":
                    correct_images += 1
                    
                print(f"🎬 {result_dict['파일명']} ({scenario}) ➔ 예측: [{result_dict['예측된 상태']}] ({result_dict['일치 여부']}) | {result_dict['세부 검출 내역']}")
            except Exception as e:
                print(f"❌ {os.path.basename(img_path)} 분석 중 오류: {e}")

    if total_images == 0:
        return

    overall_accuracy = (correct_images / total_images * 100) if total_images > 0 else 0.0
    
    report_data.append({"폴더(실제 상태)": "------------------", "파일명": "------------------", "총 검출 객체수": "------------------", "정상 객체수": "------------------", "이상 객체수": "------------------", "예측된 상태": "------------------", "일치 여부": "------------------", "세부 검출 내역": ""})
    report_data.append({
        "폴더(실제 상태)": "[통계 요약] 최종 종합 정확도",
        "파일명": f"{overall_accuracy:.2f}%",
        "총 검출 객체수": total_images,
        "정상 객체수": correct_images,
        "이상 객체수": total_images - correct_images,
        "예측된 상태": f"성공 {correct_images}건",
        "일치 여부": "종합 검증 완료",
        "세부 검출 내역": f"전체 {total_images}장 분석 완료"
    })

    report_df = pd.DataFrame(report_data)
    report_csv_path = "scenario_evaluation_report.csv"
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
    
    print("\n========================================================")
    print(f"🏆 종합 시스템 판정 정확도: {correct_images}/{total_images} ({overall_accuracy:.2f}%)")
    print(f"💾 상세 분석 결과가 '{report_csv_path}' 파일로 저장되었습니다.")
    print("========================================================")

if __name__ == "__main__":
    print("💡 서가 이상 탐지 시스템 및 추출 파이프라인 가동을 시작합니다.")
    
    # 1. 공통 기준점 풀(Reference Pool) 빌드 - 한 번만 수행하여 속도 최적화
    global_ref_pool = build_normal_reference_pool()
    
    if global_ref_pool:
        # 2. [기존 기능] 데이터셋 정확도 검증 후 CSV 리포트 추출
        run_evaluation_pipeline(global_ref_pool)
        
        # 3. [신규 기능] 'test' 폴더 대량 검증 후 JSON 추출
        run_test_folder_to_json(global_ref_pool, test_dir="test", output_json="test_results.json")
    else:
        print("❌ 정상(Normal) 참조 데이터를 생성하지 못해 분석을 중단합니다.")