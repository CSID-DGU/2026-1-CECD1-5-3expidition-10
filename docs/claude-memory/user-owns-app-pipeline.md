---
name: user-owns-app-pipeline
description: "사용자는 이 캡스톤 저장소에서 app/ (YOLO seg → 책등 크롭 → ResNet 특징) AI 파이프라인 담당이며, 검사·수정 요청은 app/ 범위로 한정한다"
metadata:
  node_type: memory
  type: user
  originSessionId: 6b6be9be-145c-425e-bd52-3fd495736acf
  modified: 2026-11-03T17:05:03.425Z
---

사용자(권준일)는 2026-1 CECD1 캡스톤 팀에서 **app/ 폴더의 AI 파이프라인**(YOLO 세그멘테이션 → 책등 사각형 추정·투시변환 크롭 → ResNet18 특징 추출, `app/pipeline.py`)을 맡고 있다. backend/, 프론트, run_master.py 등 다른 파트는 팀원 소관.

**Why:** 2026-11-04 "기울어진 도서 오동작 전수조사" 요청 때 "일단 app/ 폴더만 검사해주면 될 거야. 내가 맡은 파트에 대해서만 모듈별로 테스트하고 있어"라고 범위를 한정했다.

**How to apply:** 원인 조사·리팩토링·테스트는 app/ 안에서 하고, 다른 폴더를 건드려야 하면 먼저 물어본다. 테스트 입력은 `dataset/`(normal, abnormal_tilted/stack/upside/paper)와 `test/` 의 사진을 쓰면 된다. 모델은 `model/yolo26l_1280_v3.pt`, 실행 환경은 `.venv`(CPU torch).
