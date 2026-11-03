"""
annotations_coco.json -> labelme JSON (이미지당 1개, images/ 옆에 같은 이름으로) 변환.
labelme 는 이미지 폴더를 열면 같은 이름의 .json 을 자동으로 읽어 폴리곤을 보여준다.

    python dataset_own/to_labelme.py [dataset_own 폴더=이 파일의 폴더]
"""
import sys, os, json

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
coco = json.load(open(os.path.join(ROOT, "annotations_coco.json"), encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
by_img = {}
for a in coco["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

n = 0
for im in coco["images"]:
    shapes = []
    for a in by_img.get(im["id"], []):
        seg = a["segmentation"][0]
        pts = [[float(seg[i]), float(seg[i + 1])] for i in range(0, len(seg), 2)]
        shapes.append(dict(label=cats[a["category_id"]], points=pts, group_id=None, description="",
                           shape_type="polygon", flags={}, mask=None))
    out = dict(version="5.5.0", flags={}, shapes=shapes, imagePath=im["file_name"], imageData=None,
               imageHeight=im["height"], imageWidth=im["width"])
    with open(os.path.join(ROOT, "images", os.path.splitext(im["file_name"])[0] + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    n += 1
with open(os.path.join(ROOT, "labels.txt"), "w", encoding="utf-8") as f:
    f.write("__ignore__\n" + "\n".join(cats[k] for k in sorted(cats)) + "\n")
print(f"labelme json {n}개 생성 -> {os.path.join(ROOT, 'images')}")
