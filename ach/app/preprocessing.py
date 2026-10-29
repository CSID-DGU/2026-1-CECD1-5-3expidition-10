import cv2
import numpy as np
from PIL import Image
import logging

# 서브 AI 모델 (U-2-Net 기반 배경 제거) 도입
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except Exception as e:
    REMBG_AVAILABLE = False
    logging.warning(f"rembg 모듈을 로드할 수 없습니다 ({e}). 'pip install \"rembg[cpu]\"' 설치 시 AI 기반 배경 제거가 활성화됩니다. 현재는 원본 보존 모드로 작동합니다.")

class BookshelfPreprocessor:
    def __init__(self):
        pass

    def apply_sub_ai_background_removal(self, pil_image: Image.Image) -> Image.Image:
        """
        서브 AI 모델을 사용하여 책이 아닌 배경(선반, 빈 공간)을 어둡게 지워버립니다.
        상단 영역에 있는 누워있는 책들이 AI 모델에 의해 배경으로 오인되어 삭제되는 현상을 방지합니다.
        """
        if REMBG_AVAILABLE:
            try:
                no_bg_image = remove(pil_image)
                w, h = pil_image.size
                
                # 상단 32%는 100% 원본 유지, 32%~50% 구간은 부드럽게 AI 마스크로 전이 (가로 책 영역 확장 보완)
                grad = np.ones((h, w), dtype=np.float32)
                t1 = int(h * 0.32)
                t2 = int(h * 0.50)
                grad[t1:t2, :] = np.linspace(1.0, 0.0, t2 - t1)[:, None]
                grad[t2:, :] = 0.0
                
                grad_im = Image.fromarray((grad * 255).astype(np.uint8), mode='L')
                
                background = Image.new("RGBA", no_bg_image.size, (20, 20, 20, 255))
                ai_result = Image.composite(no_bg_image, background, no_bg_image)
                
                final_image = Image.composite(pil_image.convert("RGBA"), ai_result, grad_im)
                return final_image.convert("RGB")
            except Exception as e:
                logging.warning(f"rembg remove 실패: {e}. 원본 이미지를 보존합니다.")
                return pil_image
        else:
            open_cv_image = np.array(pil_image)
            img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            mask = cv2.dilate(edges, kernel, iterations=3)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            
            h, w = mask.shape
            mask[:int(h * 0.35), :] = 255
            
            norm_mask = mask.astype(np.float32) / 255.0
            norm_mask = np.clip(norm_mask + 0.3, 0.0, 1.0)
            mask_3d = cv2.merge([norm_mask, norm_mask, norm_mask])
            suppressed_bgr = (img_bgr.astype(np.float32) * mask_3d).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(suppressed_bgr, cv2.COLOR_BGR2RGB))

    def suppress_bookshelf_boundaries(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        좌우 양끝의 책장 판넬 벽면 및 상단 천장의 선반 노출부 억제.
        [수정] 천장에 완전히 맞닿은 책(보라색 책)이 훼손되지 않도록 상단 마진을 8% -> 2%로 대폭 축소.
        """
        h, w, _ = img_bgr.shape
        edge_margin = int(w * 0.075) 
        top_margin = int(h * 0.02)   # 8% -> 2% (최상단 날카로운 모서리만 제거)
        
        grad_left = np.linspace(0.15, 1.0, edge_margin, dtype=np.float32)
        grad_right = np.linspace(1.0, 0.15, edge_margin, dtype=np.float32)
        
        grad_left_3d = np.tile(grad_left[None, :, None], (h, 1, 3))
        grad_right_3d = np.tile(grad_right[None, :, None], (h, 1, 3))
        
        suppressed_bgr = img_bgr.copy()
        suppressed_bgr[:, :edge_margin, :] = (suppressed_bgr[:, :edge_margin, :] * grad_left_3d).astype(np.uint8)
        suppressed_bgr[:, -edge_margin:, :] = (suppressed_bgr[:, -edge_margin:, :] * grad_right_3d).astype(np.uint8)
        
        grad_top = np.linspace(0.15, 1.0, top_margin, dtype=np.float32)
        grad_top_3d = np.tile(grad_top[:, None, None], (1, w, 3))
        suppressed_bgr[:top_margin, :, :] = (suppressed_bgr[:top_margin, :, :] * grad_top_3d).astype(np.uint8)
        
        return suppressed_bgr

    def suppress_page_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        책의 옆면(종이면/Fore-edge)을 탐지하여 어둡게 누릅니다.
        [수정] 가로로 누운 책이 아래쪽으로 더 내려온 상황을 반영하여 보호 구역을 상단 45%로 연장합니다.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_intensity = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = cv2.boxFilter(edge_intensity, -1, (15, 15))
        
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        page_candidate = (s < 40) & (v > 100) & (edge_density > 15)
        
        mask = np.zeros_like(gray)
        mask[page_candidate] = 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0) / 255.0
        
        # [수정] 보호 구역 확장 (35% -> 45%) - 두 번째 가로 책 "언리얼 엔진 4" 보호
        h, w = mask.shape
        grad_suppress = np.ones((h, w), dtype=np.float32)
        grad_suppress[:int(h * 0.45), :] = 0.2
        mask_blur = mask_blur * grad_suppress
        
        suppressed_bgr = img_bgr.copy()
        for c in range(3):
            suppressed_bgr[:, :, c] = (img_bgr[:, :, c] * (1.0 - mask_blur * 0.65)).astype(np.uint8)
            
        return suppressed_bgr

    def enhance_for_detection(self, pil_image: Image.Image) -> Image.Image:
        """
        자연스러움을 유지하면서, 노이즈는 억제하고 틈새 윤곽선을 강하게 단절시킵니다.
        """
        clean_image = self.apply_sub_ai_background_removal(pil_image)
        open_cv_image = np.array(clean_image)
        img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        img_bgr = self.suppress_bookshelf_boundaries(img_bgr)
        img_bgr = self.suppress_page_edges(img_bgr)

        smoothed = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)

        # ---------------------------------------------------------
        # 형태학적 블랙햇(Morphological Black-Hat)을 이용한 물리적 절단 및 검은색 책 보호막
        # ---------------------------------------------------------
        gray_smoothed = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        knife_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        blackhat = cv2.morphologyEx(gray_smoothed, cv2.MORPH_BLACKHAT, knife_kernel)
        
        # [수정] 조명을 감안하여 어두운 책등 보호 기준 대폭 상향 (75 -> 95)
        # 이제 조명을 살짝 더 받은 "3D 게임 프로그래밍 입문"이나 "언리얼 엔진 4"도 완전히 안전하게 보호됩니다.
        dark_mask = gray_smoothed < 95
        blackhat[dark_mask] = 0
        
        blackhat_bgr = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)
        smoothed = cv2.subtract(smoothed, (blackhat_bgr * 0.5).astype(np.uint8))
        # ---------------------------------------------------------

        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
        final_bgr = cv2.addWeighted(img_clahe, 1.3, blurred, -0.3, 0)
        
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)

    def get_rotated_variants(self, pil_image: Image.Image):
        rotated_90 = pil_image.rotate(90, expand=True)
        return rotated_90
        
    def transform_boxes_back(self, boxes, orig_width):
        restored_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            new_x1 = orig_width - y2
            new_y1 = x1
            new_x2 = orig_width - y1
            new_y2 = x2
            restored_boxes.append([new_x1, new_y1, new_x2, new_y2])
        return restored_boxes

    def transform_polygons_back(self, polygons, orig_width):
        restored_polygons = []
        for poly in polygons:
            restored_poly = []
            for pt in poly:
                x_rot, y_rot = pt
                x_orig = orig_width - y_rot
                y_orig = x_rot
                restored_poly.append([x_orig, y_orig])
            restored_polygons.append(np.array(restored_poly, dtype=np.float32))
        return restored_polygons

preprocessor = BookshelfPreprocessor()