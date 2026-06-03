import os
import glob
import json
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Any, Tuple

# 기존 파이프라인 모듈 임포트 (원본 훼손 없음)
from app.pipeline import process_bookshelf_pipeline
from app.config import DEVICE
from app.models import load_resnet_model, resnet_preprocess

# 임시 DB(파일 참조)용 경로
NORMAL_DIR = "dataset/normal"

# =========================================================================
# [엔진] 피처 추출 및 상태 판정 클래스
# =========================================================================
class BookshelfFeatureEngineer:
    def __init__(self, normal_reference_pool: Dict[str, Dict[str, Any]]):
        self.resnet_model = load_resnet_model()
        self.reference_pool = normal_reference_pool

    def get_global_feature(self, pil_image: Image.Image, box: Dict[str, int]) -> np.ndarray:
        cropped_book = pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
        tensor = resnet_preprocess(cropped_book).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = self.resnet_model(tensor).cpu().numpy().flatten()
        return feat

    def calculate_tilt_angle_from_quad(self, refined_quad: np.ndarray) -> float:
        if refined_quad is None or len(refined_quad) != 4:
            return 0.0
        p0, p3 = refined_quad[0], refined_quad[3]
        dx, dy = p0[0] - p3[0], p0[1] - p3[1]
        angle_deg = np.abs(np.degrees(np.arctan2(dy, dx)))
        return float(np.abs(90.0 - angle_deg))

    def calculate_quad_aspect_ratio(self, refined_quad: np.ndarray, box: Dict[str, int]) -> float:
        if refined_quad is None or len(refined_quad) != 4:
            w = box["x2"] - box["x1"]
            h = box["y2"] - box["y1"]
            return float(w / h) if h != 0 else 0.0
            
        p0, p1, p2, p3 = refined_quad
        width1 = np.linalg.norm(p0 - p1)
        width2 = np.linalg.norm(p3 - p2)
        true_width = (width1 + width2) / 2.0
        
        height1 = np.linalg.norm(p0 - p3)
        height2 = np.linalg.norm(p1 - p2)
        true_height = (height1 + height2) / 2.0
        
        return float(true_width / true_height) if true_height != 0 else 0.0

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
            "gradient_ratio_y_x": mean_grad_y / (mean_grad_x + 1e-5),
            "mean_brightness": float(np.mean(gray) / 255.0),
            "std_brightness": float(np.std(gray) / 255.0),
            "edge_intensity": mean_grad_x + mean_grad_y
        }

    def calculate_stacked_spatial_features(self, current_quad: np.ndarray, current_box: Dict[str, int], all_books_metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        def get_y_bounds(quad, box):
            if quad is not None and len(quad) == 4:
                ys = np.array(quad)[:, 1]
                return np.min(ys), np.max(ys)
            return box["y1"], box["y2"]
            
        def get_x_bounds(quad, box):
            if quad is not None and len(quad) == 4:
                xs = np.array(quad)[:, 0]
                return np.min(xs), np.max(xs)
            return box["x1"], box["x2"]

        curr_y1, curr_y2 = get_y_bounds(current_quad, current_box)
        curr_x1, curr_x2 = get_x_bounds(current_quad, current_box)
        curr_y_center = (curr_y1 + curr_y2) / 2.0
        
        all_y_centers, all_y1s, vertical_overlap_count = [], [], 0
        
        for other_book in all_books_metadata:
            o_quad = np.array(other_book.get("refined_quadrilateral", []))
            o_box = other_book.get("box", {})
            if not o_box or o_box == current_box: continue
            
            o_y1, o_y2 = get_y_bounds(o_quad, o_box)
            o_x1, o_x2 = get_x_bounds(o_quad, o_box)
            
            all_y_centers.append((o_y1 + o_y2) / 2.0)
            all_y1s.append(o_y1)
            
            x_overlap = max(0, min(curr_x2, o_x2) - max(curr_x1, o_x1))
            if x_overlap > 0 and curr_y2 <= (o_y1 + (o_y2 - o_y1) * 0.4):
                vertical_overlap_count += 1

        if not all_y_centers:
            return {"y_center_deviation": 0.0, "vertical_overlap_count": 0.0, "is_upper_positioned": 0.0}
            
        return {
            "y_center_deviation": float(curr_y_center - np.mean(all_y_centers)),
            "vertical_overlap_count": float(vertical_overlap_count),
            "is_upper_positioned": float(curr_y2 - np.mean(all_y1s))
        }

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 != 0 and norm2 != 0 else 0.0

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

    def get_book_state(self, pil_image: Image.Image, book_metadata: Dict[str, Any], raw_512_feat: np.ndarray, all_books_metadata: List[Dict[str, Any]], matched_id: str) -> Tuple[str, dict]:
        box = book_metadata["box"]
        refined_quad = np.array(book_metadata.get("refined_quadrilateral", []))
        cv_img = cv2.cvtColor(np.array(pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
        
        tilt_angle = self.calculate_tilt_angle_from_quad(refined_quad)
        true_aspect_ratio = self.calculate_quad_aspect_ratio(refined_quad, box)
        spatial_stack_feat = self.calculate_stacked_spatial_features(refined_quad, box, all_books_metadata)
        hsv_sat = self.calculate_hsv_saturation_mean(cv_img)
        paper_texture = self.calculate_paper_texture_features(cv_img)
        vector_variance = float(np.var(raw_512_feat))
        curr_top_feat, curr_bot_feat = self.calculate_top_bottom_split_features(pil_image, box)
        
        global_sim = top_to_top = top_to_bot = bot_to_top = bot_to_bot = upside_score = 0.0
        is_orb_upside = False
        
        if matched_id in self.reference_pool:
            ref = self.reference_pool[matched_id]
            global_sim = self.compute_cosine_similarity(raw_512_feat, np.array(ref["global_feature"]))
            
            top_to_bot = self.compute_cosine_similarity(curr_top_feat, np.array(ref["bottom_feature"]))
            bot_to_top = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["top_feature"]))
            top_to_top = self.compute_cosine_similarity(curr_top_feat, np.array(ref["top_feature"]))
            bot_to_bot = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["bottom_feature"]))
            
            upside_score = float((top_to_bot + bot_to_top) - (top_to_top + bot_to_bot))
            ref_cv_img = np.array(ref.get("cv_image_array", np.zeros_like(cv_img)), dtype=np.uint8)
            is_orb_upside = self.auxiliary_orb_upside_down_detector(cv_img, ref_cv_img)

        is_upper = spatial_stack_feat["is_upper_positioned"]
        overlap_count = spatial_stack_feat["vertical_overlap_count"]
        grad_ratio_x_y = paper_texture["gradient_ratio_x_y"]
        is_laid_down = bool(true_aspect_ratio > 1.2)
        
        state = "normal"
        
        # 💡 [핵심 임계값 튜닝 1] 얹어짐 및 기울어짐 현실화
        if is_laid_down or (is_upper < -40.0 and overlap_count >= 1):
            state = "abnormal_stack"
        elif tilt_angle > 7.5: # 4.5에서 7.5로 복구하여 자연스러운 기대짐(정상)과 실제 도미노(비정상)를 정확히 구분
            state = "abnormal_tilted"
        elif upside_score > 0.1 or is_orb_upside: 
            state = "abnormal_upside"
        # 💡 [핵심 임계값 튜닝 2] 조명에 속지 않는 Paper 탐지 (Variance 복구)
        # 종이면은 특징점이 없어 분산(Variance)이 0.55 이하로 극단적으로 떨어지는 AI 특성을 활용합니다.
        elif vector_variance < 0.55 or (hsv_sat < 0.4 and grad_ratio_x_y > 1.5):
            state = "abnormal_paper"
            
        debug_info = {
            "true_aspect_ratio": true_aspect_ratio,
            "is_laid_down": is_laid_down,
            "tilt_angle": tilt_angle,
            "upside_score": upside_score,
            "is_orb_upside": is_orb_upside,
            "variance": vector_variance,
            "hsv_sat": hsv_sat,
            "grad_ratio_x_y": grad_ratio_x_y
        }
        return state, debug_info


# =========================================================================
# [API 연동 인터페이스] 백엔드에서 호출하는 메인 기능들
# =========================================================================
class BookshelfAnalyzerAPI:
    def __init__(self):
        print("[System] API 모듈 초기화 및 정상 데이터(DB) 로딩 중...")
        self.reference_pool, self.reference_shelves = self._build_temp_database()
        self.engineer = BookshelfFeatureEngineer(self.reference_pool)
        print(f"[System] 정상 데이터 {len(self.reference_pool)}건 로드 완료. API 준비됨.")

    def _get_min_x_for_sorting(self, book_item: Dict[str, Any]) -> float:
        quad = book_item.get("refined_quadrilateral")
        if quad is not None and len(quad) == 4:
            return np.min(np.array(quad)[:, 0])
        return book_item.get("box", {}).get("x1", 0)

    def _build_temp_database(self) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        pool = {}
        shelves = {}
        temp_engineer = BookshelfFeatureEngineer({})
        
        if not os.path.exists(NORMAL_DIR):
            return pool, shelves
            
        for img_path in glob.glob(os.path.join(NORMAL_DIR, "*.jpg")) + glob.glob(os.path.join(NORMAL_DIR, "*.png")):
            filename = os.path.basename(img_path)
            shelves[filename] = []
            try:
                img_pil = Image.open(img_path).convert("RGB")
                extracted_books, _ = process_bookshelf_pipeline(img_pil)
                extracted_books.sort(key=self._get_min_x_for_sorting)
                
                for normal_index, book in enumerate(extracted_books):
                    box = book["box"]
                    cropped_cv = cv2.cvtColor(np.array(img_pil.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
                    top_feat, bot_feat = temp_engineer.calculate_top_bottom_split_features(img_pil, box)
                    
                    global_feat = np.array(book.get("features", []))
                    if global_feat.size != 512 or np.all(global_feat == 0):
                        global_feat = temp_engineer.get_global_feature(img_pil, box)
                    
                    ref_id = f"ref_{filename}_book_{normal_index}"
                    ref_data = {
                        "ref_id": ref_id, 
                        "normal_index": normal_index + 1, 
                        "global_feature": global_feat.tolist(),
                        "top_feature": top_feat.tolist(),
                        "bottom_feature": bot_feat.tolist(),
                        "cv_image_array": cropped_cv.tolist()
                    }
                    pool[ref_id] = ref_data
                    shelves[filename].append(ref_data)
            except Exception:
                pass
        return pool, shelves

    def _global_unique_matching(self, detected_features: List[np.ndarray], target_refs: List[Dict]) -> Dict[int, Tuple[str, float]]:
        """
        💡 [핵심 수정] 2단계 분리 매칭 (Two-Phase Separated Matching)

        기존 문제:
        - is_laid_down=True(얹어짐)인 책들이 리스트 뒤에 붙으면서 total_det가 커짐
        - 결과적으로 세워진 책들의 rel_d = d_idx/total_det 비율이 실제보다 작게 계산됨
        - 위치 기반 매칭이 어긋나며 정상 세워진 책도 sequence_order != matched_idx 발생

        수정 방법:
        - 1단계: is_laid_down=False인 세워진 책들(upright)만 따로 분리하여 
                 upright 전용 d_idx/upright_count 비율로 위치 기반 매칭 → 순서 보장
        - 2단계: is_laid_down=True인 얹어진 책들은 전체 ref 풀에서 similarity만으로 자유 매칭
                 (이미 1단계에서 배정된 ref는 제외)

        NOTE: abnormal_paper / abnormal_tilted는 is_laid_down=False이므로 1단계에 포함됨.
              이 책들은 표지가 안 보여 similarity가 낮으므로 weight=0.9(위치 기반)가 적용됨.
              즉, 위치 기반으로 자기 자리 ref에 배정됨 → sequence_order와 matched_idx가 
              여전히 다를 수 있지만, 이는 "책이 실제로 그 자리에 있으나 식별 불가" 상황이라
              정상적인 결과임. 순서 구분이 가능한 세워진 책들은 정확히 매칭됨.
        """
        ref_list = sorted(target_refs, key=lambda x: x["normal_index"])
        total_refs = max(1, len(ref_list))

        # --- 1단계: 세워진 책(upright) 인덱스 분리 ---
        upright_indices = [i for i, feat in enumerate(detected_features)
                           if not getattr(self, "_laid_down_flags", [False] * len(detected_features))[i]]
        laid_down_indices = [i for i in range(len(detected_features)) if i not in upright_indices]

        total_upright = max(1, len(upright_indices))

        matches = []

        # 1단계: upright 책들 - upright 내부 순서 기준으로 위치 패널티 계산
        for rank, d_idx in enumerate(upright_indices):
            d_feat = detected_features[d_idx]
            for r_idx, ref in enumerate(ref_list):
                sim = self.engineer.compute_cosine_similarity(d_feat, np.array(ref["global_feature"]))
                rel_d = rank / total_upright       # upright 내 상대 위치
                rel_r = r_idx / total_refs
                spatial_penalty = abs(rel_d - rel_r)

                # 💡 [핵심 개조 3] 다이내믹 밸런스 매칭 (Dynamic Balance Matching)
                # 시각 정보의 신뢰도(유사도 점수)에 따라 위치 페널티 비중을 지능적으로 조절합니다.
                if sim > 0.85:
                    weight = 0.1  # 표지가 뚜렷하면 위치 무시 (순서 뒤바뀜 캐치)
                elif sim < 0.70:
                    weight = 0.9  # 백지/노이즈라 헷갈리면 철저하게 위치 기반으로 자기 자리 할당 (B011 스틸 차단)
                else:
                    weight = 0.35  # 일반적인 상황의 균형점

                score = sim - (weight * spatial_penalty)
                matches.append((d_idx, ref["ref_id"], score, sim))

        # 2단계: laid_down 책들 - 위치 패널티 없이 similarity만으로 매칭
        for d_idx in laid_down_indices:
            d_feat = detected_features[d_idx]
            for r_idx, ref in enumerate(ref_list):
                sim = self.engineer.compute_cosine_similarity(d_feat, np.array(ref["global_feature"]))
                # 얹어진 책은 표지가 옆으로 누워 있어 위치 비교가 의미 없음 → 패널티 없음
                matches.append((d_idx, ref["ref_id"], sim, sim))

        matches.sort(key=lambda x: x[2], reverse=True)
        assigned_d = {}
        assigned_r = set()

        for d_idx, ref_id, score, sim in matches:
            if d_idx not in assigned_d and ref_id not in assigned_r:
                assigned_d[d_idx] = (ref_id, sim)
                assigned_r.add(ref_id)

        return assigned_d

    def analyze_image_to_dict(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"파일을 찾을 수 없습니다: {image_path}"}

        try:
            img_pil = Image.open(image_path).convert("RGB")
            extracted_books, _ = process_bookshelf_pipeline(img_pil)
            
            for book in extracted_books:
                refined_quad = np.array(book.get("refined_quadrilateral", []))
                true_ar = self.engineer.calculate_quad_aspect_ratio(refined_quad, book.get("box", {}))
                book["_is_laid_down_temp"] = bool(true_ar > 1.2)
                book["_min_x_temp"] = self._get_min_x_for_sorting(book)
                
            # 세워진 책 먼저(x좌표 순), 누운 책 나중
            extracted_books.sort(key=lambda b: (b["_is_laid_down_temp"], b["_min_x_temp"]))
            
            detected_features = []
            for book in extracted_books:
                raw_512_feat = np.array(book.get("features", []))
                if raw_512_feat.size != 512 or np.all(raw_512_feat == 0):
                    raw_512_feat = self.engineer.get_global_feature(img_pil, book["box"])
                detected_features.append(raw_512_feat)

            # 💡 [수정] _global_unique_matching에서 is_laid_down 플래그를 참조할 수 있도록
            #    임시 속성으로 전달 (클래스 인스턴스 상태를 통해 공유)
            self._laid_down_flags = [book["_is_laid_down_temp"] for book in extracted_books]
                
            best_shelf_id = None
            best_shelf_score = -1
            
            if self.reference_shelves:
                for shelf_id, refs in self.reference_shelves.items():
                    shelf_score = 0
                    for d_feat in detected_features:
                        max_sim = max([self.engineer.compute_cosine_similarity(d_feat, np.array(r["global_feature"])) for r in refs])
                        shelf_score += max_sim
                    
                    if shelf_score > best_shelf_score:
                        best_shelf_score = shelf_score
                        best_shelf_id = shelf_id
            
            target_refs = self.reference_shelves.get(best_shelf_id, []) if best_shelf_id else []
            assigned_matches = self._global_unique_matching(detected_features, target_refs)
            
            # 매칭 후 플래그 정리
            self._laid_down_flags = []
            
            # ── 1차 패스: 상태 판정 및 raw 결과 수집 ──────────────────────
            raw_results = []
            normal_count = 0
            abnormal_count = 0

            for detected_order, book in enumerate(extracted_books):
                box = book["box"]
                raw_512_feat = detected_features[detected_order]

                matched_id, confidence = assigned_matches.get(detected_order, ("unknown", 0.0))

                state, debug_info = self.engineer.get_book_state(
                    img_pil, book, raw_512_feat, extracted_books, matched_id
                )

                if state == "normal":
                    normal_count += 1
                else:
                    abnormal_count += 1

                mapped_normal_index = -1
                if matched_id in self.reference_pool:
                    mapped_normal_index = self.reference_pool[matched_id]["normal_index"]

                raw_results.append({
                    "_x_pos": book["_min_x_temp"],          # 물리적 x 위치 (정렬 보조용)
                    "book_id": f"B{mapped_normal_index:03d}" if mapped_normal_index != -1 else "UNKNOWN",
                    "matched_normal_index": mapped_normal_index,
                    "box": box,
                    "is_laid_down": debug_info["is_laid_down"],
                    "visual_status": state,
                    "confidence_score": round(confidence, 4),
                    "matched_db_id": matched_id,
                    "debug_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in debug_info.items()}
                })

            # ── 2차 패스: 출력 순서 재정렬 ────────────────────────────────
            # 목표: 순서 확정 가능한 책 → 순서 불확실한 책 → 얹어진 책
            #
            # [그룹 A] 순서 확정: normal / abnormal_upside / abnormal_tilted (세워져 있고 표지 식별 가능)
            #          matched_normal_index 오름차순으로 정렬 → DB 번호 = 실제 순서
            # [그룹 B] 순서 불확실: abnormal_paper (표지가 안 보여 어떤 책인지 불확실)
            #          물리적 x 위치(왼→오) 순으로 정렬
            # [그룹 C] 얹어진 책: abnormal_stack (is_laid_down=True)
            #          물리적 x 위치 순으로 정렬

            def _order_group(item):
                s = item["visual_status"]
                laid = item["is_laid_down"]
                if laid or s == "abnormal_stack":
                    return 2                      # 그룹 C
                elif s == "abnormal_paper":
                    return 1                      # 그룹 B
                else:
                    return 0                      # 그룹 A

            group_a = sorted(
                [r for r in raw_results if _order_group(r) == 0],
                key=lambda r: r["matched_normal_index"]   # DB 번호 오름차순
            )
            group_b = sorted(
                [r for r in raw_results if _order_group(r) == 1],
                key=lambda r: r["_x_pos"]                 # 물리적 위치 순
            )
            group_c = sorted(
                [r for r in raw_results if _order_group(r) == 2],
                key=lambda r: r["_x_pos"]                 # 물리적 위치 순
            )

            ordered = group_a + group_b + group_c

            # sequence_order 재부여 + 내부 보조 키 제거
            results_list = []
            for new_seq, item in enumerate(ordered, start=1):
                item.pop("_x_pos", None)
                item["sequence_order"] = new_seq
                # JSON 필드 순서를 sequence_order → book_id → ... 로 맞춤
                results_list.append({
                    "sequence_order": item["sequence_order"],
                    "book_id": item["book_id"],
                    "matched_normal_index": item["matched_normal_index"],
                    "box": item["box"],
                    "is_laid_down": item["is_laid_down"],
                    "visual_status": item["visual_status"],
                    "confidence_score": item["confidence_score"],
                    "matched_db_id": item["matched_db_id"],
                    "debug_metrics": item["debug_metrics"],
                })

            total_books = len(extracted_books)
            is_warning_all_same_match = False
            
            return {
                "status": "success",
                "filename": os.path.basename(image_path),
                "matched_shelf_image": best_shelf_id,
                "summary": {
                    "total_detected": total_books,
                    "normal_count": normal_count,
                    "abnormal_count": abnormal_count,
                    "unique_matched_db_ids": len(assigned_matches),
                    "warning_all_same_match": is_warning_all_same_match
                },
                "vision_items": results_list
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_test_folder_to_json(self, test_dir: str = "test", output_json: str = "test_results.json"):
        print(f"\n========================================================")
        print(f"📁 [테스트 폴더 분석] '{test_dir}' 내 이미지 JSON 대량 추출 시작")
        print(f"========================================================")
        
        if not os.path.exists(test_dir):
            print(f"⚠️ '{test_dir}' 폴더가 존재하지 않습니다.")
            return
            
        image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
        if not image_files:
            print(f"⚠️ '{test_dir}' 폴더에 이미지가 없습니다.")
            return
            
        print(f"총 {len(image_files)}개의 이미지를 분석합니다...\n")
        
        final_results = {"test_results": []}
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            print(f"🔍 분석 중: {filename}")
            
            result_dict = self.analyze_image_to_dict(img_path)
            
            if result_dict.get("status") == "success":
                final_results["test_results"].append(result_dict)
            else:
                print(f"❌ {filename} 분석 실패: {result_dict.get('message')}")
                
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
            
        print(f"\n✅ 테스트 완료! 모든 결과가 '{output_json}'에 저장되었습니다.")

if __name__ == "__main__":
    print("💡 서가 이상 탐지 시스템 API 가동을 시작합니다.")
    api = BookshelfAnalyzerAPI()
    api.run_test_folder_to_json(test_dir="test", output_json="test_results.json")