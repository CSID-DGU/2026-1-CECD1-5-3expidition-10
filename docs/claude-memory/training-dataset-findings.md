---
name: training-dataset-findings
description: 현재 YOLO 모델 학습에 쓴 Roboflow 데이터셋(P2)의 문제점과 대안 데이터셋 조사 결과, Roboflow 계정 사정(무료=Public 만)
metadata:
  type: reference
---

- 학습에 쓴 데이터셋: Roboflow Universe "Book spine detection (P2)" https://universe.roboflow.com/book-detection/book-spine-detection-p2 (소스 7,063 / v2 18,052, CC BY 4.0). 문제: 같은 사진의 회전·반전·크롭 복사본이 소스에 그대로 들어 있고(첫 50장 중 고유 16개) 그 복사본이 train/valid/test 에 걸쳐 있어(IMG_5159_JPEG: Train4+Valid3 등) 검증 점수가 부풀려짐; 서양 서점·스톡 사진 위주라 한국 도서관 얇은 책등·청구기호 라벨·한 권만 기운 케이스가 거의 없음; 증강은 전체 회전 ±15° 뿐.
- Universe 의 다른 대형 책등 데이터셋(Yis Workspace 11.7k, Harald Varner 1.46k, dobs spine-seg 1.66k, ShelfLove 6.6k, emptybookslotdetect)은 모두 같은 원본 풀(IMG_48xx_JPEG, crop-NNN_png, IMG_20191014_*)의 재탕.
- 가장 닮은 공개 데이터: **CNU HMLAB "Book spine dataset-library"** https://universe.roboflow.com/cnu-hmlab-syko-and-ash/book-spine-dataset-library (312장, CC BY 4.0, 한국 대학도서관, 청구기호 라벨 포함, 기운 책 있음, 4점 폴리곤). 증강 없는 원본으로 받을 것(v1/v2 는 640 stretch, v3 는 512×384+90° 회전).
- 기타: Open Shelves https://github.com/capjamesg/open-shelves (CC BY-NC-SA), IP&M 2022 논문 데이터는 비공개. AI Hub 에는 해당 데이터 없음.
- 사용자의 Roboflow 계정: 기존 프로젝트 "Book spine instance segmentation"(1463장, Harald 포크). **무료 플랜이라 Private 프로젝트 불가(Public 만)** — 사용자가 공공도서관 촬영본 저작권을 걱정해 공개 업로드는 안 하기로 함. 브라우저 패널에서는 OS 파일 선택창을 못 열어 업로드는 사용자가 드래그해야 함.

**How to apply:** 재학습 데이터는 자체 사진([[own-photos-testcase]]) 중심 + CNU 312장 보조. P2 를 쓰려면 파일명 기준으로 복사본을 묶어 원본당 한 split 으로 재분할.

**2026-11-09 CNU 데이터 확보:** 공개 v1~v3 은 전부 리사이즈·증강본이라 사용자 Roboflow 계정으로 포크(공개 프로젝트) → 전처리·증강 없는 v1 생성 → COCO Segmentation zip 을 `dataset_cnu/` 에 받음. `dataset_cnu/to_labelme.py` 로 `images/cnu_000~311.jpg` + labelme JSON 변환(312장, 2048×1536, 라벨 5,934개 전부 book-spine 초안). 사용자가 `dataset_cnu/검수_labelme_실행.bat` 로 book-paper 분리 검수 예정. CNU 원본도 가운데 서가만 라벨돼 있고 위·아래 층·뒤편 서가 종이면이 라벨 없이 보임 → 학습 전 dataset_own 과 같은 크롭 필요(`from_labelme.py` 를 dataset_cnu 에 쓰려면 train/val 목록과 정리 패턴 일반화 필요, 미구현). Roboflow 다운로드·열람은 로그인 필수, 버전 생성은 2만 장당 1크레딧(무료 크레딧 차감).

**2026-11-10 추가 데이터셋 조사 (v4·v5 가 기운 책에서 개선 없어서):**
- **Zhong (Shenzhen Univ.) 책등 데이터셋 — 1순위.** (a) 4TU.ResearchData https://data.4tu.nl/datasets/2c3f156f-8bfc-4ca4-8c8a-f3d4d02d6674 : 661장, **CC0**, labelme JSON(이미지 내장, 폴리곤), data.zip 766MB, 로그인 없이 다운로드. Sensors 2024 논문(PMC11679322) 설명: 선전대 도서관 스마트폰 촬영, 661장 중 283장에 기운 책(촬영 각도 + 배치), 378장은 거의 수직. (b) IEEE DataPort https://ieee-dataport.org/documents/shelf-book-spine-dataset-instance-segmentation : 1,849장(train 1,387 / test 462), 약 3000×4000, "극도로 촘촘하고 기운 더미", COCO 형식, CC BY, 4.31GB, **IEEE 로그인 필요**(무료 계정으로 되는지는 미확인). (a) 가 (b) 의 부분집합일 가능성 높음 → 둘 다 쓰면 중복 제거 필요.
- Roboflow `minhaz/book_spine_instance_segmentation-uqybb`: 898장, CC BY 4.0, 클래스 book. 썸네일상 도서관 서가의 촘촘한 얇은 책을 여러 각도로 찍은 것(item-NNN.png) — Zhong 데이터의 재업로드일 가능성 있음(미확인).
- Roboflow `book-iypdg/booksegment`: 380장이지만 원본 약 20장을 인위 회전한 것(`15_rotated_12.jpg`) → 가치 낮음. `leo-ueno/book-spine-segmentation` 193장: 서양 고서 서가, 반듯함 → 도메인 불일치. 별마당도서관(123장)은 분류용, Autonomous Book-Sorting Robot 은 0장. `DL-On-shelf-Books-Recognition`(Boatsure) BookID 는 Baidu Pan + 접근코드라 사실상 받기 어려움.
- **book-paper(종이면) 클래스가 있는 공개 데이터셋은 못 찾음** → 직접 촬영뿐. AI Hub 에도 책등 세그멘테이션 없음(재확인).
- 주의: 위 데이터는 전부 1클래스. 위·아래 층 라벨 여부, 청구기호 라벨 포함 여부는 받아봐야 앎. 기울기가 "사진 전체 회전"에서 온 것이 많으면 회전 증강과 효과가 비슷할 수 있음.

**2026-11-10 Zhong 4TU 661장 실물 확인 — 이미 P2 에 들어 있는 사진임:** 사용자가 받은 위치 내려받아 푼 폴더의 `data/dataset_661/`. labelme JSON 661개(이미지 base64 내장, **인코딩 GBK** → gb18030 으로 읽어야 함), 라벨 이름은 전부 `"1"`, 폴리곤 15,454개, 4032×3024 / 3968×2976, 파일명 전부 `IMG_20191014_HHMMSS`. **P2(v3 학습 데이터)를 Roboflow 에서 `filename:IMG_20191014` 로 검색하면 252건, 첫 50개 이름이 4TU 파일과 50/50 일치** → P2 의 `IMG_20191014_*` 풀이 바로 이 데이터. 즉 v3 는 이 사진들의 상당수를 이미 학습함(새 데이터 아님).
내용: 서가 한 층을 가까이서 찍은 사진(책등 높이가 사진 높이의 중앙값 80%, 26%는 위아래가 프레임에 잘림), 청구기호 라벨 포함해 꼼꼼히 라벨, 다른 층은 거의 안 보임. 기운 책등(15~60°) 1,327개(8.6%)지만 사진 단위로 기운 것은 63장뿐이고 대부분 촬영 각도·원근에서 옴. → v3 출발 학습에서 기운 책 개선용으로는 기대 낮음. 쓸모: COCO 출발 학습의 다양성 보강(v5 의 바닥·천장 오검출 완화), 또는 망각 방지용 재학습 데이터. IEEE 1,849장도 같은 촬영분의 확장일 가능성이 높아 같은 한계. 결론: 얇은 책이 촘촘히 기운 사진은 직접 촬영이 답.

**2026-11-11 P2 v1 원본 확보·선별 (`dataset_p2/`):** P2 의 v1 은 "no augmentations"(Auto-Orient 만) 이라 포크 없이 받을 수 있다(7,063장 628MB, 받는 주소는 `https://app.roboflow.com/ds/<link>?key=<key>` — universe.roboflow.com 주소는 curl 이 Cloudflare 에 막힘). 내용: 이름 기준 원본 1,836개(묶음 470개가 split 에 걸침), CNU 312·Zhong 251 제외 후 dHash 로 합치면 **고유 1,011장**, 그중 1280 이상은 291장뿐이고 720장은 640×640 축소본. 다른 책과 15~60° 어긋난 책등 3권 이상인 사진은 31장뿐(기운 책 드묾 재확인).
수확: (1) **한국 도서관 4K 63장**(원래 이름 `1-A`~`17-G`, 서가 9곳 × 7상태 — 정상/한 권만 기울임/여러 권 기울임, 우리 과제와 같은 구성), (2) `Real_train/val/test_*` 117장(1920×1080, 촘촘하고 기운 도서관 서가, 두 층 라벨된 사진 있음). 종이면 후보는 v5 추론으로 44장 나왔지만 눈으로 보면 대부분 오검출(선반 판자·뒤편 서가·무늬 없는 책등) → P2 는 종이면 공급원으로 약함.
`dataset_p2/` 에 342장을 labelme 편집용으로 내보냄(p2_001~186 고해상도가 핵심, 187~ 은 640 저해상도). 라벨은 P2 원본 그대로 book-spine, 사용자가 편집 예정. `to_labelme.py` 는 기존 JSON 을 덮어쓰지 않음. 분석 스크립트는 세션 scratchpad 에만 있음(재현하려면 README 의 기준대로 다시 작성).
**2026-11-11 P2 검수 완료·병합:** 사용자가 22장 삭제(흑백 13장 포함, `removed.csv`; selection.json 에서도 뺌 — 안 빼면 to_labelme.py 가 되살림) → 320장, 책등 6,990 / 종이면 45. 남은 사진을 SIFT+RANSAC 으로 서로 대조: 책상에서 같은 책 15권을 반복 촬영한 640 근접 34장과 동영상 연속 프레임 1장은 `dataset_p2/exclude.txt` 로 목록에서만 뺌(파일은 유지). 한국 도서관 63장(서가 9곳×7번)은 의도된 같은 서가라 유지, train/val 걸침 없음. 학습에 쓰는 P2 = train 263 / val 20 (라벨 안 된 책이 띠 안에 남은 p2_125 도 exclude).
`dataset_merged/build.py` 가 own+cnu+p2 를 합침(`python build.py own cnu` 로 일부만 가능): train 583장(책등 13,238 / 종이면 104), **val.txt 는 own+cnu 45장 그대로**(P2 는 v3 가 본 사진이라 val_p2.txt 로만; v4~v6 과 점수 비교 유지). `train_tilt.txt` 847줄 = own 4회/드문 장면 8회(비중 31%), cnu 1/3, p2 1/2; 기운 책 4.6%→12.3%, 종이면 104→512. 권장 학습 명령은 그대로(`--weights v3 --data data_tilt.yaml --finetune`, 이름 v7_v3_p2). from_labelme.py 는 넓이 20px² 미만 라벨을 버림.

**2026-11-11 v7 학습 중 OOM:** 서버에서 55 에폭에 메모리 부족으로 중단(P2 의 촘촘한 사진으로 인스턴스 수 증가). `train.py` 에 `--resume`(last.pt 에서 한 번 이어감)과 `--keep-going [N]`(죽을 때마다 자식 프로세스로 자동 재개, 세 번 연속 진행 없으면 중단) 추가. 재개 시 바꿀 수 있는 것은 batch·workers·device 뿐(Ultralytics check_resume 제한). 폴더는 사용자가 `dataset_merged_v2` 로 이름 변경.

**2026-11-13 회전 증강 각도 (다음 학습 제안, 미실행):** v9·v10 학습의 회전 증강은 ±20°(`degrees=20`)였다. 반듯한 줄(책등 약 13,000개)은 최대 20° 까지만 돌려 봤고, 30~60° 기운 책등은 실제 기운 사진의 419개(3%), 그중 폭 12px 미만은 25개뿐. own_023(학습 사진, 얇은 책이 40~45° 로 촘촘)에서 51권 중 42권만 찾는데, 줄을 잘라 반듯하게 돌리면 같은 모델이 51권을 다 찾음 → 모델이 '촘촘하게 40° 넘게 기운 얇은 책'을 못 배운 것. 사용자는 파이프라인 2단계 검출 대신 모델 쪽 개선을 원함. `dataset_merged/train.py` 에 `--degrees`(기본 20) 추가, 다음 시도 = v9 와 같은 명령 + `--degrees 45`(이름 v11_v3_deg45). 효과는 미확인 — 학습 후 own_023·val2·own val 로 v9 와 비교할 것(반듯한 책·종이면이 떨어지지 않는지도).

**2026-11-13 해상도 질문(1280 vs 1440/1600/1920) — 추론 해상도만 바꿔 본 결과 (v9 last, 학습에 안 쓴 18장·책등 473):** 찾은 수 1280: 456 / 1440: 461 / 1600: 460 / 1920: 462. 늘어난 것은 거의 전부 폭 12px 미만 반듯한 책(12권 중 6 → 10 → 9 → 11), 1440 에서 대부분 얻고 그 뒤는 평평. 얇고 기운 것은 해상도와 무관하게 3~4/8. 사진당 시간(CPU, 검출+정제): 6.3 / 7.6 / 9.8 / 14.1초. 학습 해상도를 올린 실험은 없음(추정: GPU 메모리 1280 배치2 에서 11~15GB → 1440 약 1.27배, 1600 약 1.56배, 1920 약 2.25배라 24GB 공유 카드에서 1920 은 배치 1 필요). 권고: 회전 45° 학습은 1280 그대로(한 번에 하나만 변경), 해상도는 1440 이 가성비 구간.

## v11 = 회전 증강 ±50° (2026-11-13 학습, 11-15 평가) — v9 와 차이 없음, 앱은 v9 last 유지
- 서버 run `v11_deg50`: v3 시작, data_tilt.yaml, `--finetune --degrees 50`, 나머지는 v9 와 같음. 최고 46에폭, 76에폭에서 조기 종료. 검증 mask mAP50-95 0.815 (v9 0.838).
  가중치: `model/yolo26l_1280_v11_deg50_best.pt`, `_last.pt` (git 미추적).
- 학습에 안 쓴 사진 (앱과 같은 검출·사각형, IoU>=0.5):
  - val2 8장: 기운 책 100 → v9 96 / v11 best 97 / last 97. 반듯 43 → 37/37/37. 눕힌 8 → 8/8/8. 종이면 111(conf>=0.5) → 104/103/105.
  - own val 10장: 반듯 235 → 230/230/232. 기운 85 → 82/81/80 (own_051 에서 20 → 19 → 18). 종이면 11 → 10/10/11, 종이면 예측 수 20 → 23 → 24 (오검출 약간 증가).
  - CNU 35장: 반듯 665 → 665/664/665. 종이면 3 → 3/1/3 (best 는 2개를 책등으로 읽음).
  - 집 15장: 권수 거의 같음 (기운 책 폴더 40 → 39 → 39).
  - 18장 책등 473권 두께·기울기 표: 합계 v9 456 / v11 best 455 / last 456. 30~60° 기운 101권: 97/98/98. 얇은(<12px) 기운 8권: 3/3/3 — 그대로.
- 학습 사진이지만 문제의 사진: own_023(얇은 책이 40~45° 로 촘촘, 51권) v9 42 → v11 best 43 / last 45 (IoU>=0.7: 38 → 40 → 43). 같은 사진을 '대상 줄만 잘라 반듯하게 돌려' v9 에 넣으면 51 이었다.
  own_025(반듯한 얇은 그림책, 78권) 71 → 67 / 70.
- 결론: "회전 증강이 ±20° 뿐이라서" 라는 추정은 틀렸거나 일부만 맞다. 증강을 50° 로 올려도 촘촘히 기운 얇은 책은 풀리지 않는다.
  남는 설명(추정, 미검증): 기운 얇은 책이 촘촘하면 축 정렬 박스가 서로 거의 같아져서(박스로 후보를 가르고 박스 안에서 마스크를 자르는 구조) 학습 방식과 상관없이 가르기 어렵다. 사진을 돌려 책을 세우면 박스가 분리된다 → 2단계(줄 자르기 + 돌리기) 검출만 지금까지 효과가 확인된 방법.
