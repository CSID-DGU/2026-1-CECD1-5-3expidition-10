"""
검수 우선순위: 라벨 통계만으로 의심 구간을 찾아 review_flags.csv 와 review/flag_*.jpg 로 남긴다.
  - wide  : 폭이 같은 사진 중앙값의 2.5배 이상 (두 권이 하나로 합쳐졌을 가능성)
  - gap   : 이웃 라벨 사이 빈 폭이 책등 폭의 80% 이상 (사이에 빠진 책이 있을 가능성) -> 빨간 세로선
    python dataset_own/review_flags.py
"""
import os, csv, glob
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
rows = []
for lp in sorted(glob.glob(os.path.join(ROOT, "labels", "*.txt"))):
    name = os.path.splitext(os.path.basename(lp))[0]
    img = Image.open(os.path.join(ROOT, "review", name + ".jpg")).convert("RGB")
    W, H = img.size
    quads = []
    for line in open(lp):
        v = list(map(float, line.split()[1:]))
        quads.append(np.array(v).reshape(4, 2) * [W, H])
    if not quads:
        rows.append((name, 0, 0, 0)); continue
    widths = np.array([max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])) for q in quads])
    heights = np.array([max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])) for q in quads])
    standing = [i for i in range(len(quads)) if heights[i] > widths[i]]
    w_med = np.median(widths[standing]) if standing else np.median(widths)
    wide = [i for i in standing if widths[i] > 2.5 * w_med]
    # 세운 책을 왼→오 정렬해 이웃 간격(중심 x 차이) 검사
    # 이웃 라벨 사이의 '빈 폭'(중심 간격 - 두 폭의 평균)이 얇은 쪽 책등 폭의 80% 를 넘으면 사이에 빠진 책이 있을 가능성
    order = sorted(standing, key=lambda i: quads[i][:, 0].mean())
    gaps = []
    for a, b in zip(order, order[1:]):
        empty = (quads[b][:, 0].mean() - quads[a][:, 0].mean()) - (widths[a] + widths[b]) / 2
        if empty > max(0.8 * min(widths[a], widths[b]), 6):
            gaps.append((a, b))
    d = ImageDraw.Draw(img)
    for i in wide:
        d.polygon([tuple(map(float, p)) for p in quads[i]], outline=(255, 0, 0), width=4)
    for a, b in gaps:
        x = (quads[a][:, 0].max() + quads[b][:, 0].min()) / 2
        y0 = min(quads[a][:, 1].min(), quads[b][:, 1].min()); y1 = max(quads[a][:, 1].max(), quads[b][:, 1].max())
        d.line([(x, y0), (x, y1)], fill=(255, 0, 0), width=4)
    if wide or gaps:
        img.save(os.path.join(ROOT, "review", "flag_" + name + ".jpg"), quality=88)
    rows.append((name, len(quads), len(wide), len(gaps)))

with open(os.path.join(ROOT, "review_flags.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["file", "labels", "wide_labels", "gaps_maybe_missed"]); w.writerows(rows)
rows.sort(key=lambda r: -(r[2] + r[3]))
print(f"{'file':>8} {'labels':>6} {'wide':>5} {'gaps':>5}")
for r in rows:
    print(f"{r[0]:>8} {r[1]:>6} {r[2]:>5} {r[3]:>5}")
print("flagged images:", sum(1 for r in rows if r[2] + r[3] > 0), "/", len(rows))
