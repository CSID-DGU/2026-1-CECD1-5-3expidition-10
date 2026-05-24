import os
import torch
import torch.nn.functional as F
from PIL import Image
from app.pipeline import process_bookshelf_pipeline

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder_path, filename)).convert("RGB")
            images.append((filename, img))
    return images

def extract_features_from_folder(folder_path):
    images = load_images_from_folder(folder_path)
    all_features = []
    for filename, img in images:
        books, _ = process_bookshelf_pipeline(img)
        for book in books:
            all_features.append({
                "folder": folder_path,
                "filename": filename,
                "book_index": book["book_index"],
                "feature_vector": book["feature_vector"]
            })
    return all_features

def compute_centroid(features):
    tensors = [torch.tensor(f["feature_vector"]) for f in features]
    stacked = torch.stack(tensors)
    return stacked.mean(dim=0)

def detect_abnormal_books(features, centroid, threshold=0.8):
    results = []
    for f in features:
        sim = F.cosine_similarity(
            torch.tensor(f["feature_vector"]), centroid, dim=0
        ).item()
        is_abnormal = sim < threshold   # 🔑 단순화: 0.8 밑이면 abnormal
        results.append({
            "folder": f["folder"],
            "filename": f["filename"],
            "book_index": f["book_index"],
            "similarity": sim,
            "abnormal": is_abnormal
        })
    return results

def summarize_abnormal(results):
    summary = {}
    for r in results:
        key = f"{r['folder']}/{r['filename']}"
        if r["abnormal"]:
            summary.setdefault(key, []).append(r["book_index"])
        else:
            summary.setdefault(key, [])
    return summary

if __name__ == "__main__":
    normal_path = "dataset/normal"
    abnormal_paths = [
        "dataset/abnormal_tilted",
        "dataset/abnormal_paper",
        "dataset/abnormal_upside",
        "dataset/abnormal_stack"
    ]

    normal_features = extract_features_from_folder(normal_path)
    if not normal_features:
        raise RuntimeError("⚠️ 정상 폴더에 이미지가 없습니다.")
    centroid = compute_centroid(normal_features)
    print("✅ 정상 centroid 계산 완료")

    for path in abnormal_paths:
        features = extract_features_from_folder(path)
        if not features:
            print(f"\n📂 {path} → 이미지 없음")
            continue
        results = detect_abnormal_books(features, centroid, threshold=0.8)
        summary = summarize_abnormal(results)
        print(f"\n📂 {path} 결과:")
        for img, abnormal_books in summary.items():
            if abnormal_books:
                print(f"{img} → abnormal books: {abnormal_books}")
            else:
                print(f"{img} → 모든 책 정상")
