"""
직접 촬영한 서가 사진 -> '가운데 서가' 책등만 의사라벨(pseudo-label) 생성.

    python dataset_own/make_labels.py <사진 폴더> [출력 폴더=dataset_own]

출력
  images/own_NNN.jpg      EXIF 회전 보정 + 긴 변 2560 으로 저장 (ASCII 파일명; mapping.csv 에 원본 이름)
  labels/own_NNN.txt      YOLO-seg: "0 x1 y1 x2 y2 x3 y3 x4 y4" (0~1 정규화, 좌상→우상→우하→좌하)
  review/own_NNN.jpg      검수용: 선택된 서가의 라벨(노랑+번호), 버려진 다른 층 검출(회색), 선반선(시안), 선반까지 늘린 부분(마젠타)
  annotations_coco.json   같은 라벨의 COCO 형식 (Roboflow/CVAT 업로드용)
  data.yaml, train.txt, val.txt, mapping.csv

라벨은 app/pipeline.py 의 검출·사각형 추정을 그대로 쓴 것이라 "초안"이다. 빠진 책, 두 권이 합쳐진 것, 라벨 스티커가
잘린 것은 Roboflow/CVAT 에서 손으로 고쳐야 한다. 서가 선택이 틀린 사진은 review/ 에서 바로 보인다.
"""
import sys, os, csv, json, glob
import numpy as np, cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None            # 12000x9000 원본은 PIL 기본 한도를 넘는다
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from app import pipeline as P

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "dataset_own")
SAVE_LONG_SIDE = 2560
CONF = 0.2
for d in ("images", "labels", "review"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)


def detect_spines(work):
    """pipeline 의 per-spine 처리를 그대로 재현해 (quad, lying, tilt) 목록을 돌려준다 (1280 좌표계)."""
    r = P.yolo_model(work, conf=CONF, iou=P.BOX_NMS_IOU, agnostic_nms=True, imgsz=P.TARGET_SIZE,
                     retina_masks=True, max_det=P.MAX_DET, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return []
    polys, extras = P.extract_mask_polygons(r)
    confs = r.boxes.conf.cpu().numpy(); cls_ids = r.boxes.cls.cpu().numpy()
    keep = P.mask_nms(polys, confs, (work.height, work.width))
    H, W = work.height, work.width
    out = []
    for i in keep:
        if int(cls_ids[i]) != 0:
            continue
        pts = polys[i].astype(np.int32)
        if len(pts) < 3:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(pts)
        k = max(5, int(min(rw, rh) * 0.38)); k += (k % 2 == 0)
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [pts] + extras[i], 255)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = [pts]
        if cs:
            big = max(cs, key=cv2.contourArea); pts = big.reshape(-1, 2)
            parts = [pts] + [c.reshape(-1, 2) for c in P._collinear_contours(big, [c for c in cs if c is not big])]
        if len(pts) < 3:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(pts)
        if max(rw, rh) / max(min(rw, rh), 1) < 1.5 or max(rw, rh) < 25:
            continue
        m2 = np.zeros((H, W), np.uint8); cv2.fillPoly(m2, [np.asarray(p, np.int32) for p in parts], 255)
        q = P.extract_spine_quad_perspective(pts, m2, fallback_fn=P.robust_extract_spine_quad_via_pca)
        out.append(dict(quad=np.asarray(q, np.float64), lying=bool(P.is_lying(q)), tilt=P.spine_tilt_deg(q), conf=float(confs[i])))
    return out


def pick_middle_shelf(spines, H):
    """세운 책의 '아래끝 y' 로 층을 묶고(선반마다 바닥이 맞춰져 있음), 화면 중앙에 가장 가까운 층을 고른다."""
    standing = [s for s in spines if not s["lying"] and abs(s["tilt"]) < 60]
    if not standing:
        return [], None
    L_med = np.median([max(P.quad_size(s["quad"])) for s in standing])
    order = sorted(standing, key=lambda s: s["quad"][:, 1].max())
    rows, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if b["quad"][:, 1].max() - a["quad"][:, 1].max() > 0.3 * L_med:
            rows.append(cur); cur = []
        cur.append(b)
    rows.append(cur)
    rows = [r for r in rows if len(r) >= 3] or rows
    row = min(rows, key=lambda r: abs(np.mean([s["quad"][:, 1].mean() for s in r]) - H / 2))
    y_top = min(s["quad"][:, 1].min() for s in row); y_bot = max(s["quad"][:, 1].max() for s in row)
    # 같은 층 높이 안에 있는 눕힌 책·급경사 책도 포함
    in_row = {id(s) for s in row}
    others = [s for s in spines if id(s) not in in_row and y_top - 0.15 * L_med <= s["quad"][:, 1].mean() <= y_bot]
    return row + others, (y_top, y_bot)


def extend_to_shelf(row, L_med):
    """세운 책의 아래끝이 선반선보다 위에서 끝나면(라벨 스티커가 마스크에서 빠진 경우) 옆변 방향으로 선반선까지 늘린다."""
    standing = [s for s in row if not s["lying"] and abs(s["tilt"]) < 45]
    if len(standing) < 4:
        return None
    pts = np.array([[(s["quad"][2][0] + s["quad"][3][0]) / 2, (s["quad"][2][1] + s["quad"][3][1]) / 2] for s in standing], np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    line_y = lambda x: y0 + (x - x0) * vy / vx
    for s in standing:
        q = s["quad"]
        for bi, ti in ((2, 1), (3, 0)):            # 우하←우상, 좌하←좌상
            d = q[bi] - q[ti]
            if abs(d[1]) < 1e-6:
                continue
            target = line_y(q[bi][0])
            t = (target - q[bi][1]) / d[1]           # 옆변 방향으로 얼마나 더 가야 선반선에 닿는지 (길이 비율)
            if 4 / max(abs(d[1]), 1) < t <= 0.15:
                q[bi] = q[bi] + d * t
                s["extended"] = True
    return line_y


def main():
    paths = sorted(glob.glob(os.path.join(SRC, "*.jpg")) + glob.glob(os.path.join(SRC, "*.jpeg")) + glob.glob(os.path.join(SRC, "*.png")))
    coco = dict(images=[], annotations=[], categories=[dict(id=0, name="book-spine")])
    mapping, ann_id, names = [], 0, []
    font = ImageFont.load_default()
    for idx, p in enumerate(paths):
        name = f"own_{idx:03d}"
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        s = SAVE_LONG_SIDE / max(img.size)
        saved = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img
        saved.save(os.path.join(OUT, "images", name + ".jpg"), quality=92)
        work, _ = P.resize_to_target(saved)
        W, H = work.size
        spines = detect_spines(work)
        row, yrange = pick_middle_shelf(spines, H)
        L_med = np.median([max(P.quad_size(x["quad"])) for x in row]) if row else 0
        line_y = extend_to_shelf(row, L_med) if row else None
        row.sort(key=lambda x: (x["lying"], (x["quad"][0][0] + x["quad"][3][0]) / 2))

        # ---- 라벨 ----
        with open(os.path.join(OUT, "labels", name + ".txt"), "w") as f:
            for x in row:
                q = np.clip(x["quad"], [0, 0], [W - 1, H - 1])
                f.write("0 " + " ".join(f"{v[0]/W:.6f} {v[1]/H:.6f}" for v in q) + "\n")
        sx, sy = saved.width / W, saved.height / H
        coco["images"].append(dict(id=idx, file_name=name + ".jpg", width=saved.width, height=saved.height))
        for x in row:
            q = np.clip(x["quad"], [0, 0], [W - 1, H - 1]) * [sx, sy]
            xs, ys = q[:, 0], q[:, 1]
            coco["annotations"].append(dict(id=ann_id, image_id=idx, category_id=0, iscrowd=0,
                                            segmentation=[[float(v) for pt in q for v in pt]],
                                            bbox=[float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                                            area=float(cv2.contourArea(q.astype(np.float32)))))
            ann_id += 1

        # ---- 검수 이미지 ----
        vis = work.copy(); d = ImageDraw.Draw(vis, "RGBA")
        in_row = {id(x) for x in row}
        for x in spines:
            if id(x) not in in_row:
                d.polygon([tuple(map(float, v)) for v in x["quad"]], outline=(160, 160, 160, 255), width=1)
        if line_y is not None:
            d.line([(0, float(line_y(0))), (W - 1, float(line_y(W - 1)))], fill=(0, 255, 255, 255), width=1)
        for i, x in enumerate(row):
            col = (255, 0, 255, 255) if x.get("extended") else (255, 255, 0, 255)
            d.polygon([tuple(map(float, v)) for v in x["quad"]], outline=col, width=2)
            d.text((float(x["quad"][:, 0].min()) + 2, float(x["quad"][:, 1].min()) + 2), str(i), fill=(255, 255, 255, 255), font=font)
        d.rectangle((0, 0, 520, 14), fill=(0, 0, 0, 200))
        d.text((3, 1), f"{name}  <- {os.path.basename(p)}   labels={len(row)}  (all detections={len(spines)})", fill=(255, 255, 0, 255), font=font)
        vis.save(os.path.join(OUT, "review", name + ".jpg"), quality=88)

        mapping.append((name + ".jpg", os.path.basename(p), len(row), len(spines)))
        names.append(name)
        print(f"{name} <- {os.path.basename(p)}: labels={len(row)} / detections={len(spines)}  extended={sum(1 for x in row if x.get('extended'))}", flush=True)

    with open(os.path.join(OUT, "annotations_coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    with open(os.path.join(OUT, "mapping.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["file", "original", "labels", "all_detections"]); w.writerows(mapping)
    rng = np.random.RandomState(0); perm = rng.permutation(len(names)); n_val = max(1, len(names) // 5)
    with open(os.path.join(OUT, "val.txt"), "w") as f:
        f.writelines(f"images/{names[i]}.jpg\n" for i in sorted(perm[:n_val]))
    with open(os.path.join(OUT, "train.txt"), "w") as f:
        f.writelines(f"images/{names[i]}.jpg\n" for i in sorted(perm[n_val:]))
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(OUT)}\ntrain: train.txt\nval: val.txt\nnames:\n  0: book-spine\n")
    print("done:", len(names), "images,", ann_id, "labels")


if __name__ == "__main__":
    main()
