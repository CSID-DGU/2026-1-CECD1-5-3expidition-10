# -*- coding: utf-8 -*-
from typing import List, Protocol

from PIL import Image

from ..types import Detection


class DetectorSource(Protocol):
    """책 탐지기 공통 인터페이스. 입력은 파이프라인 좌표계(긴 변 1280)의 PIL 이미지."""
    name: str

    def detect(self, image: Image.Image) -> List[Detection]: ...
