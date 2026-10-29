import io
from typing import Tuple

from PIL import Image, UnidentifiedImageError

# 사서가 대시보드에서 올리는 이미지(도서관 지도, 정상 상태 기준 사진) 검사
# 확장자가 아니라 실제 파일 내용으로 형식을 판단해, 이미지가 아닌 파일이 저장되지 않게 합니다.
FORMAT_EXTS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def detect_image_ext(data: bytes, allowed=("JPEG", "PNG", "WEBP")) -> Tuple[str, Tuple[int, int]]:
    """이미지 바이트의 (저장 확장자, (가로, 세로))를 반환합니다. 허용하지 않는 형식이면 ValueError."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt, size = img.format, img.size
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("이미지 파일이 아니거나 손상된 파일입니다.")
    if fmt not in allowed:
        names = ", ".join(FORMAT_EXTS[f].lstrip(".") for f in allowed)
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {fmt} ({names}만 가능)")
    return FORMAT_EXTS[fmt], size
