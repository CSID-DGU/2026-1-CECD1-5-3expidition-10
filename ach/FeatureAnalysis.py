import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 기존 파이프라인 모듈 임포트
from app.pipeline import process_bookshelf_pipeline

# 데이터셋 경로 설정
DATASET_DIR = "dataset"
SCENARIOS = [
    "abnormal_paper", 
    "abnormal_stack", 
    "abnormal_tilted", 
    "abnormal_upside", 
    "normal"
]

OUTPUT_JSON = "extracted_features.json"

def extract_and_save_features():
    """
    1. 모든 폴더를 순회하며 이미지를 파이프라인에 통과시킵니다.
    2. 추출된 특징 벡터들을 모아 JSON 파일로 저장합니다.
    """
    all_data = []
    
    for scenario in SCENARIOS:
        scenario_path = os.path.join(DATASET_DIR, scenario)
        if not os.path.exists(scenario_path):
            print(f"경로를 찾을 수 없습니다: {scenario_path}")
            continue
            
        # jpg, png 이미지 모두 검색
        image_files = glob.glob(os.path.join(scenario_path, "*.jpg")) + glob.glob(os.path.join(scenario_path, "*.png"))
        print(f"[{scenario}] 폴더에서 {len(image_files)}개의 이미지를 처리합니다...")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                filename = os.path.basename(img_path)
                
                # 파이프라인 실행 (시각화 바이트는 무시하고 메타데이터만 받음)
                metadata, _ = process_bookshelf_pipeline(img)
                
                for book in metadata:
                    all_data.append({
                        "scenario": scenario,
                        "filename": filename,
                        "book_index": book["book_index"],
                        "box": book["box"],
                        "features": book["feature_vector"] # 512차원 벡터
                    })
            except Exception as e:
                print(f"이미지 처리 중 에러 발생 ({img_path}): {e}")

    # JSON으로 저장 (재사용 및 분석 용이)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 총 {len(all_data)}개의 책 데이터가 '{OUTPUT_JSON}'에 성공적으로 저장되었습니다!")
    return all_data


def analyze_cosine_similarity(data):
    """
    추출된 데이터를 바탕으로 시나리오 간의 코사인 유사도(Cosine Similarity)를 분석합니다.
    """
    print("\n📊 --- 시나리오 간 코사인 유사도 분석 --- 📊")
    df = pd.DataFrame(data)
    
    if df.empty:
        print("데이터가 없습니다.")
        return

    # 각 시나리오별 특징 벡터의 중심점(Centroid, 평균 벡터) 계산
    centroids = {}
    for scenario in df['scenario'].unique():
        # 해당 시나리오의 모든 512차원 벡터들을 numpy 배열로 변환
        vectors = np.array(df[df['scenario'] == scenario]['features'].tolist())
        # 평균을 내어 이 시나리오를 대표하는 하나의 512차원 벡터를 만듦
        centroid = np.mean(vectors, axis=0)
        centroids[scenario] = centroid

    # 시나리오 리스트 정렬
    scenario_names = list(centroids.keys())
    
    # 유사도 매트릭스 계산용 배열
    centroid_matrix = np.array([centroids[name] for name in scenario_names])
    
    # Scikit-learn을 이용한 코사인 유사도 계산
    similarity_matrix = cosine_similarity(centroid_matrix)
    
    # 결과를 보기 좋은 DataFrame으로 변환
    sim_df = pd.DataFrame(similarity_matrix, index=scenario_names, columns=scenario_names)
    
    print("\n[시나리오 간 특징 벡터(Centroid) 코사인 유사도 행렬]")
    print(sim_df.round(4)) # 소수점 4자리까지 출력
    
    # CSV로도 저장
    sim_df.to_csv("scenario_similarity_matrix.csv")
    print("\n✅ 유사도 분석 결과가 'scenario_similarity_matrix.csv'로 저장되었습니다.")

if __name__ == "__main__":
    # 1. 특정 벡터가 아직 안 뽑혔다면 아래 주석을 풀고 실행하세요.
    print("특징 벡터 추출을 시작합니다...")
    dataset_features = extract_and_save_features()
    
    # 2. 이미 추출해둔 JSON이 있다면 파일에서 불러오기만 해도 됩니다.
    # with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
    #     dataset_features = json.load(f)
        
    # 3. 코사인 유사도 분석 실행
    analyze_cosine_similarity(dataset_features)