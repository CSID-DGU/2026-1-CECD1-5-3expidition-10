"""
순찰 사진 속 책 위치 (층 화면에서 문제 책에 번호 테두리를 그리는 용도)

AI(MakeResult.py)는 책마다 debug.box = {x1, y1, x2, y2}를 돌려줍니다. 이 좌표는 AI가 분석한 이미지,
즉 EXIF 회전을 보정하고 긴 변을 AI_LONG_SIDE(1280)로 맞춘 이미지 기준입니다. (MakeResult.prepare_image)
화면의 사진 크기와 상관없이 그릴 수 있도록 사진 가로 · 세로에 대한 0~1 비율로 바꿔 "x1,y1,x2,y2" 문자열로 저장합니다.
브라우저도 사진을 EXIF 회전대로 보여 주므로 비율 좌표가 그대로 맞습니다.
"""
from typing import Optional

from PIL import Image, ImageOps

AI_LONG_SIDE = 1280   # app/pipeline.py의 TARGET_SIZE (AI가 사진을 줄이는 긴 변 길이)


def photo_size(path: str) -> Optional[tuple]:
    """EXIF 회전을 보정한 사진 크기 (가로, 세로). 읽지 못하면 None"""
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).size
    except Exception:
        return None


def normalized_spine_box(box, size: Optional[tuple]) -> Optional[str]:
    """AI 좌표 box → "x1,y1,x2,y2" (0~1 비율, 소수 넷째 자리). 좌표나 사진 크기가 없으면 None"""
    if not box or not size:
        return None
    try:
        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    width, height = size
    scale = AI_LONG_SIDE / max(width, height)
    ai_w, ai_h = round(width * scale), round(height * scale)
    clamp = lambda v: min(1.0, max(0.0, v))
    x1, x2 = sorted((clamp(x1 / ai_w), clamp(x2 / ai_w)))
    y1, y2 = sorted((clamp(y1 / ai_h), clamp(y2 / ai_h)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return ",".join(f"{v:.4f}" for v in (x1, y1, x2, y2))


SPINE_BOX_COLUMN = ("VARCHAR(60) NULL COMMENT '순찰 사진 속 책 위치 x1,y1,x2,y2 "
                    "(사진 가로 · 세로에 대한 0~1 비율, 층 화면 번호 테두리용)'")


def ensure_spine_box_column(conn):
    """VISION_DATA.spine_box가 없으면 추가 (setup_db.py를 다시 실행하지 않은 DB도 서버를 켜면 바로 쓸 수 있게)"""
    cursor = conn.cursor()
    try:
        cursor.execute("""SELECT COUNT(*) FROM information_schema.COLUMNS
                          WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'VISION_DATA' AND COLUMN_NAME = 'spine_box'""")
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"ALTER TABLE VISION_DATA ADD COLUMN spine_box {SPINE_BOX_COLUMN}")
            conn.commit()
            print("  ✔️ VISION_DATA.spine_box 컬럼 추가 (층 화면 번호 테두리용 책 위치)")
    finally:
        cursor.close()


def parse_spine_box(text: Optional[str]) -> Optional[list]:
    """저장된 "x1,y1,x2,y2" → [x1, y1, x2, y2] (없거나 잘못되면 None)"""
    if not text:
        return None
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        return None
    return values if len(values) == 4 else None
