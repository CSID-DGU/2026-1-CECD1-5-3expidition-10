"""
labelme 로 검수한 images/*.json -> labels/*.txt (YOLO-seg) 와 annotations_coco.json 을 다시 만든다.
검수가 끝난 뒤 한 번 실행하면 data.yaml 로 바로 학습할 수 있다.

    python dataset_own/from_labelme.py [dataset_own 폴더=이 파일의 폴더]
"""
import sys, os, json, glob
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "labels"), exist_ok=True)
names = [l.strip() for l in open(os.path.join(ROOT, "labels.txt"), encoding="utf-8") if l.strip() and l.strip() != "__ignore__"] \
    if os.path.exists(os.path.join(ROOT, "labels.txt")) else ["book-spine"]
cls = {n: i for i, n in enumerate(names)}
coco = dict(images=[], annotations=[], categories=[dict(id=i, name=n) for n, i in cls.items()])
ann_id = n_lab = 0
for jp in sorted(glob.glob(os.path.join(ROOT, "images", "*.json"))):
    d = json.load(open(jp, encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
    stem = os.path.splitext(os.path.basename(jp))[0]
    img_id = len(coco["images"])
    coco["images"].append(dict(id=img_id, file_name=d["imagePath"], width=W, height=H))
    lines = []
    for s in d["shapes"]:
        if s.get("shape_type") not in (None, "polygon", "rectangle") or s["label"] not in cls:
            continue
        pts = np.array(s["points"], dtype=float)
        if s.get("shape_type") == "rectangle":              # labelme 사각형은 두 점 -> 네 꼭짓점
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if len(pts) < 3:
            continue
        pts = np.clip(pts, [0, 0], [W - 1, H - 1])
        lines.append(f"{cls[s['label']]} " + " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in pts))
        xs, ys = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        coco["annotations"].append(dict(id=ann_id, image_id=img_id, category_id=cls[s["label"]], iscrowd=0,
                                        segmentation=[[float(v) for p in pts for v in p]],
                                        bbox=[float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                                        area=float(area)))
        ann_id += 1
    with open(os.path.join(ROOT, "labels", stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    n_lab += len(lines)
with open(os.path.join(ROOT, "annotations_coco.json"), "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False)
print(f"images {len(coco['images'])}장, 라벨 {n_lab}개 -> labels/ 와 annotations_coco.json 갱신")
