import os
import glob
import json
import numpy as np
import cv2
import torch
from PIL import Image
from collections import Counter
from typing import Dict, List, Any, Tuple

# 기존 파이프라인 모듈 임포트
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

    def get_book_state(self, pil_image: Image.Image, book_metadata: Dict[str, Any], raw_512_feat: np.ndarray, all_books_metadata: List[Dict[str, Any]]) -> Tuple[str, str, float, dict]:
        """개별 책의 특징을 추출하고 상태를 판정하여 리턴합니다."""
        box = book_metadata["box"]
        refined_quad = np.array(book_metadata.get("refined_quadrilateral", []))
        cv_img = cv2.cvtColor(np.array(pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
        
        tilt_angle = self.calculate_tilt_angle_from_quad(refined_quad)
        aspect_ratio = self.calculate_aspect_ratio(box)
        spatial_stack_feat = self.calculate_stacked_spatial_features(box, all_books_metadata)
        hsv_sat = self.calculate_hsv_saturation_mean(cv_img)
        paper_texture = self.calculate_paper_texture_features(cv_img)
        vector_variance = float(np.var(raw_512_feat))
        curr_top_feat, curr_bot_feat = self.calculate_top_bottom_split_features(pil_image, box)
        
        best_sim = -1.0
        matched_id = "unknown"
        
        # 1. DB(Reference)에서 가장 유사한 정상 책 찾기
        for ref_id, ref_data in self.reference_pool.items():
            sim = self.compute_cosine_similarity(raw_512_feat, np.array(ref_data["global_feature"]))
            if sim > best_sim:
                best_sim = sim
                matched_id = ref_id
                
        global_sim = top_to_top = top_to_bot = bot_to_top = bot_to_bot = upside_score = 0.0
        is_orb_upside = False
        
        # 2. 매칭된 기준 데이터가 있다면 정밀 대조
        if matched_id in self.reference_pool:
            ref = self.reference_pool[matched_id]
            top_to_bot = self.compute_cosine_similarity(curr_top_feat, np.array(ref["bottom_feature"]))
            bot_to_top = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["top_feature"]))
            top_to_top = self.compute_cosine_similarity(curr_top_feat, np.array(ref["top_feature"]))
            bot_to_bot = self.compute_cosine_similarity(curr_bot_feat, np.array(ref["bottom_feature"]))
            
            upside_score = float((top_to_bot + bot_to_top) - (top_to_top + bot_to_bot))
            ref_cv_img = np.array(ref.get("cv_image_array", np.zeros_like(cv_img)), dtype=np.uint8)
            is_orb_upside = self.auxiliary_orb_upside_down_detector(cv_img, ref_cv_img)

        # 3. 상태 분기(Decision Tree)
        is_upper = spatial_stack_feat["is_upper_positioned"]
        overlap_count = spatial_stack_feat["vertical_overlap_count"]
        grad_ratio = paper_texture["gradient_ratio_x_y"]
        
        state = "normal"
        if aspect_ratio > 1.0 or (is_upper < -40.0 and overlap_count >= 1):
            state = "abnormal_stack"
        elif tilt_angle > 10.0:
            state = "abnormal_tilted"
        elif upside_score > 0.15 or is_orb_upside:
            state = "abnormal_upside"
        elif vector_variance < 0.005 and hsv_sat < 0.25 and grad_ratio > 1.8:
            state = "abnormal_paper"
            
        debug_info = {
            "aspect_ratio": aspect_ratio,
            "tilt_angle": tilt_angle,
            "upside_score": upside_score,
            "variance": vector_variance
        }
        return state, matched_id, best_sim, debug_info


# =========================================================================
# [API 연동 인터페이스] 백엔드에서 호출하는 메인 기능들
# =========================================================================

class BookshelfAnalyzerAPI:
    def __init__(self):
        """
        서버 기동 시 1회 호출되어 DB(Reference)를 메모리에 로드합니다.
        (추후 실제 DB 연동 시 이 부분을 SQL/NoSQL Fetch로 변경하면 됩니다.)
        """
        print("[System] API 모듈 초기화 및 정상 데이터(DB) 로딩 중...")
        self.reference_pool = self._build_temp_database()
        self.engineer = BookshelfFeatureEngineer(self.reference_pool)
        print(f"[System] 정상 데이터 {len(self.reference_pool)}건 로드 완료. API 준비됨.")

    def _build_temp_database(self) -> Dict[str, Any]:
        """임시로 normal 폴더의 데이터를 읽어 기준 DB 객체를 만듭니다."""
        pool = {}
        # 임베딩 추출을 돕기 위해 빈 엔지니어 객체 임시 사용
        temp_engineer = BookshelfFeatureEngineer({})
        
        if not os.path.exists(NORMAL_DIR):
            return pool
            
        for img_path in glob.glob(os.path.join(NORMAL_DIR, "*.jpg")) + glob.glob(os.path.join(NORMAL_DIR, "*.png")):
            filename = os.path.basename(img_path)
            try:
                img_pil = Image.open(img_path).convert("RGB")
                extracted_books, _ = process_bookshelf_pipeline(img_pil)
                
                for book in extracted_books:
                    idx = book["book_index"]
                    box = book["box"]
                    cropped_cv = cv2.cvtColor(np.array(img_pil.crop((box["x1"], box["y1"], box["x2"], box["y2"]))), cv2.COLOR_RGB2BGR)
                    top_feat, bot_feat = temp_engineer.calculate_top_bottom_split_features(img_pil, box)
                    global_feat = np.array(book.get("features", np.zeros(512)))
                    
                    ref_id = f"ref_{filename}_book_{idx}"
                    pool[ref_id] = {
                        "global_feature": global_feat.tolist(),
                        "top_feature": top_feat.tolist(),
                        "bottom_feature": bot_feat.tolist(),
                        "cv_image_array": cropped_cv.tolist()
                    }
            except Exception:
                pass
        return pool

    def analyze_image_to_json(self, image_path: str) -> str:
        """
        백엔드 라우터(Controller)에서 호출하는 최종 함수입니다.
        타겟 이미지를 분석하고 결과를 JSON 문자열로 직렬화하여 반환합니다.
        """
        if not os.path.exists(image_path):
            return json.dumps({"status": "error", "message": "파일을 찾을 수 없습니다."})

        try:
            img_pil = Image.open(image_path).convert("RGB")
            extracted_books, _ = process_bookshelf_pipeline(img_pil)
            
            results_list = []
            normal_count = 0
            abnormal_count = 0
            
            for book in extracted_books:
                idx = book["book_index"]
                box = book["box"]
                raw_512_feat = np.array(book.get("features", np.zeros(512)))
                
                # 핵심 상태 판정
                state, matched_id, confidence, debug_info = self.engineer.get_book_state(
                    img_pil, book, raw_512_feat, extracted_books
                )
                
                if state == "normal":
                    normal_count += 1
                else:
                    abnormal_count += 1
                
                # API 응답 규격 작성 (백엔드가 다루기 편한 dict 형태)
                results_list.append({
                    "book_index": idx,
                    "box": box,
                    "predicted_state": state,
                    "confidence_score": round(confidence, 4),
                    "matched_db_id": matched_id,
                    "debug_metrics": {k: round(v, 4) for k, v in debug_info.items()}
                })
            
            # 최종 JSON 딕셔너리 구성
            response_dict = {
                "status": "success",
                "filename": os.path.basename(image_path),
                "summary": {
                    "total_detected": len(extracted_books),
                    "normal_count": normal_count,
                    "abnormal_count": abnormal_count
                },
                "book_details": results_list
            }
            
            # JSON 텍스트로 직렬화하여 리턴 (한글 및 특수문자 깨짐 방지)
            return json.dumps(response_dict, ensure_ascii=False, indent=4)
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# =========================================================================
# [사용 예시] 백엔드 프레임워크(FastAPI 등)에서의 호출 방법
# =========================================================================
if __name__ == "__main__":
    # 1. 서버 시작 시 전역(Global) 인스턴스로 API 모듈을 한 번만 로드합니다.
    #    (이때 Normal 폴더의 데이터를 읽어 캐싱해 둡니다)
    analyzer = BookshelfAnalyzerAPI()
    
    # 2. 클라이언트로부터 요청(Request)이 들어오면 함수를 호출합니다.
    # 2. 테스트 폴더 경로 지정 (ach 폴더 안에서 실행된다고 가정)
    test_dir = os.path.join("dataset", "test")
    
    # 해당 폴더 안의 이미지 파일(jpg, jpeg, png 등) 목록을 모두 가져옵니다.
    image_files = glob.glob(os.path.join(test_dir, "*.[jp][pn]*g"))
    
    # 이미지가 없는 경우 방어 로직
    if not image_files:
        print(f"❌ [에러] '{test_dir}' 폴더 안에 테스트할 이미지가 없습니다.")
        print("-> dataset/test/ 폴더를 만들고 사진을 한 장 넣어주세요!")
        exit(1)
        
    # 3. 폴더 안의 첫 번째 이미지를 타겟으로 자동 선택합니다.
    test_image_path = image_files[0]

    print(f"\n[Request] 분석 시작: {test_image_path}")
    
    # 3. JSON 결과를 반환받습니다. (백엔드는 이 문자열을 바로 클라이언트에게 Return)
    json_result = analyzer.analyze_image_to_json(test_image_path)
    
    print("\n[Response] 클라이언트에게 전달할 JSON 데이터:")
    print(json_result)

    # 6. 🚀 결과를 'output.json' 파일로 물리적으로 저장합니다. (백엔드 연동용)
    output_dir = "vision_output"
    os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 자동으로 만들어줍니다!
    
    output_filename = os.path.join(output_dir, "output.json")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(json_result)
        
    print(f"\n✅ [성공] 분석 결과가 '{output_filename}' 파일로 완벽하게 저장되었습니다!")