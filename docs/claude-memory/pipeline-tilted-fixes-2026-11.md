---
name: pipeline-tilted-fixes-2026-11
description: 2026-11-04 app/pipeline.py 기울어진 책 오동작 원인 조사 결론과 적용한 수정(미커밋), 남은 모델 한계
metadata:
  type: project
---

기울어진 책등이 깨지던 원인은 데이터셋보다 app/pipeline.py 쪽이 컸다. 적용한 수정(2026-11-04 기준 **미커밋**, 사용자가 "먼저 직접 확인" 선택):
1. YOLO 박스 NMS iou 0.7→0.95(`BOX_NMS_IOU`), max_det 600 — 축정렬 박스라 인접한 기울어진 얇은 책등이 IoU>0.7 로 지워졌음. 중복 제거는 mask_nms 가 담당.
2. `extract_mask_polygons()`: masks.xy 병합 폴리곤 대신 비트맵 최대 연결성분 + 같은 축선상 조각(`_collinear_contours`) 사용 — 이웃 책등 위성 조각이 opening 커널을 폭주시켜 책 5~6권짜리 사각형이 나오던 연쇄를 끊음. retina_masks=True.
3. `_quad_from_vertical_mask`: 끝변 기울기 기본값을 좌우변에 수직(m_perp)으로, clip 도 그 기준 편차로 — 32° 이상에서 끝변이 항상 수평이 되던 버그.
4. opening 커널 MORPH_ELLIPSE, opening 후 축선상 조각 유지, 면적비 검사는 볼록껍질 기준.
5. `order_quad(pts, lying=...)` 로 45° 부근 standing/lying 판정 일치.
결과: 원본 사진 기울어진 책 27→55권, 폭 100px 넘는 사각형 7→0, 45° 스택 길이 198→273px; 데이터셋 17장 세운 책 결과 동일.

남은 모델 한계: 하단 청구기호 라벨이 마스크에 없는 책(6/37), 45° 마스크 톱니 경계, 기울어진 책등 조각화 — 자체 데이터 재학습으로 해결 예정([[own-photos-testcase]]). 마스크 임계값 0.5→0.38 실험은 폭 +1px/조각화 감소지만 이웃 겹침 증가라 미적용(custom predictor 필요).

**How to apply:** pipeline 을 다시 손댈 때 위 설계 의도를 유지할 것. 커밋 여부는 사용자가 결정.
