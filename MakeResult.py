import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
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

def calculate_tilt_angle(polygon):
    """
    다각형(Polygon) 좌표를 기반으로 책이 수직 상태에서 얼마나 기울어졌는지(Tilt Angle) 계산합니다.
    """
    if not polygon or len(polygon) < 3:
        return 0.0
    poly_arr = np.array(polygon, dtype=np.float32)
    
    # cv2.fitLine으로 다각형을 관통하는 중심축 벡터 추출
    [vx, vy, x, y] = cv2.fitLine(poly_arr, cv2.DIST_L2, 0, 0.01, 0.01)
    
    angle = np.degrees(np.arctan2(vy[0], vx[0]))
    tilt = abs(90.0 - abs(angle))
    
    if tilt > 45.0:
        tilt = abs(tilt - 90.0)
        
    return float(tilt)

def engineer_features(raw_data):
    """
    추출된 JSON 원본 데이터를 입력받아 분류에 유용한 파생 변수들을 생성합니다.
    """
    print("\n⚙️ --- 피쳐 엔지니어링 수행 중 --- ⚙️")
    engineered_data = []
    
    for row in raw_data:
        box = row.get("box", {})
        poly = row.get("polygon", [])
        
        x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        area = width * height
        y_center = (y1 + y2) / 2.0
        
        aspect_ratio = width / float(height)
        tilt_angle = calculate_tilt_angle(poly)
        
        engineered_data.append({
            "scenario": row.get("scenario", "unknown"),
            "filename": row.get("filename", "unknown"),
            "book_index": row.get("book_index", -1),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "tilt_angle": tilt_angle,
            "y_center": y_center,
            "area": area,
            "features": row.get("features", []) 
        })
        
    df = pd.DataFrame(engineered_data)
    return df

def evaluate_and_report(df):
    """
    'Normal' 상태를 건강한 기준점(Baseline)으로 삼아 이상(Anomaly)을 탐지하고 문서화합니다.
    """
    print("\n🔎 --- 사진 촬영 기반 상태 구분(Anomaly Detection) 검증 중 --- 🔎")
    
    # [최적화 핵심] 'normal' 데이터를 다른 분류와 경쟁시키는 대신, 이상 탐지의 유일한 기준점(Baseline)으로 삼습니다.
    normal_df = df[df['scenario'] == 'normal']
    if not normal_df.empty:
        normal_vectors = np.array(normal_df['features'].tolist())
        normal_centroid = np.mean(normal_vectors, axis=0).reshape(1, -1)
        
        # 정상 책들이 정상 중심점(평균)과 얼마나 유사한지 분포(정규분포)를 구합니다.
        normal_sims = cosine_similarity(normal_vectors, normal_centroid).flatten()
        mean_sim = np.mean(normal_sims)
        std_sim = np.std(normal_sims)
        
        # 💡 [피쳐 튜닝] 더 예민한 이상 탐지를 위해 하위 약 6.7% (평균 - 1.5 Standard Deviations)로 임계값을 상향 조정
        visual_anomaly_threshold = mean_sim - (1.5 * std_sim)
        
        print(f"   ✓ [Baseline] 정상 상태 책들의 평균 유사도: {mean_sim:.4f}")
        print(f"   ✓ [Threshold] 시각적 이상(Upside 등) 판별 임계값: {visual_anomaly_threshold:.4f} 미만")
    else:
        print("⚠️ 'normal' 데이터가 부족하여 시각적 이상 탐지 기준을 설정할 수 없습니다.")
        normal_centroid = np.zeros((1, 512))
        visual_anomaly_threshold = 0.0
    
    report_data = []
    grouped = df.groupby(['scenario', 'filename'])
    
    scenario_stats = {scen: {'total': 0, 'correct': 0} for scen in SCENARIOS}
    correct_count = 0
    total_count = len(grouped)
    
    for (true_scenario, filename), group in grouped:
        # 사진 내 주변 책들의 평균 두께 (종이 끼임 판별 기준치)
        img_median_width = group['width'].median()
        img_predictions = set()
        book_details = []
        
        for idx, row in group.iterrows():
            book_idx = row['book_index']
            pred = "normal" # 기본값은 무죄 추정(정상)
            
            # [규칙 1] 가로세로 비율 역전 (누운 책)
            if row['aspect_ratio'] > 0.8:
                pred = "abnormal_stack"
            # [규칙 2] 기울기 7도 이상 (기울어진 책)
            elif row['tilt_angle'] > 7.0:
                pred = "abnormal_tilted"
            # 💡 [피쳐 튜닝] 얇은 종이를 더 잘 잡아내도록 평균 너비 대비 45% 미만으로 조건 완화
            elif row['width'] < (img_median_width * 0.45):
                pred = "abnormal_paper"
            # [규칙 4] 시각적 이상 탐지 (뒤집힌 책 등)
            else:
                vec = np.array(row['features']).reshape(1, -1)
                sim_normal = cosine_similarity(vec, normal_centroid)[0][0]
                
                # 해당 책이 '정상 평균'의 허용 범위(Threshold)를 벗어날 정도로 이질적이라면 이상으로 간주
                if sim_normal < visual_anomaly_threshold:
                    pred = "abnormal_upside"
                    
            img_predictions.add(pred)
            book_details.append(f"Book {book_idx}({pred})")
        
        # [사진의 최종 상태 판별] 
        # 사진에서 발견된 가장 심각한 이상(Anomaly) 하나를 대표 상태로 출력합니다.
        if "abnormal_stack" in img_predictions:
            final_pred = "abnormal_stack"
        elif "abnormal_paper" in img_predictions:
            final_pred = "abnormal_paper"
        elif "abnormal_tilted" in img_predictions:
            final_pred = "abnormal_tilted"
        elif "abnormal_upside" in img_predictions:
            final_pred = "abnormal_upside"
        else:
            final_pred = "normal"
            
        is_match = (final_pred == true_scenario)
        
        # 정확도 통계 누적
        if true_scenario in scenario_stats:
            scenario_stats[true_scenario]['total'] += 1
            if is_match:
                scenario_stats[true_scenario]['correct'] += 1
                correct_count += 1
            
        report_data.append({
            "폴더(실제 상태)": true_scenario,
            "파일명": filename,
            "예측된 상태": final_pred,
            "일치 여부": "✅ 일치" if is_match else "❌ 불일치",
            "책별 분석 상세": " | ".join(book_details)
        })
        
    # 💡 [신규 추가] 결과 데이터프레임 맨 아래에 정확도 통계 행 추가
    report_data.append({
        "폴더(실제 상태)": "---",
        "파일명": "---",
        "예측된 상태": "---",
        "일치 여부": "---",
        "책별 분석 상세": "---"
    })
    
    report_data.append({
        "폴더(실제 상태)": "[통계 요약]",
        "파일명": "",
        "예측된 상태": "",
        "일치 여부": "",
        "책별 분석 상세": ""
    })

    for scen, stats in scenario_stats.items():
        tot = stats['total']
        acc = (stats['correct'] / tot * 100) if tot > 0 else 0
        report_data.append({
            "폴더(실제 상태)": f"{scen} 정확도:",
            "파일명": f"{acc:.1f}%",
            "예측된 상태": f"({stats['correct']}/{tot})",
            "일치 여부": "",
            "책별 분석 상세": ""
        })

    overall_acc = (correct_count / total_count * 100) if total_count > 0 else 0
    report_data.append({
        "폴더(실제 상태)": "✨ 총 전체 정확도:",
        "파일명": f"{overall_acc:.1f}%",
        "예측된 상태": f"({correct_count}/{total_count})",
        "일치 여부": "",
        "책별 분석 상세": ""
    })

    report_df = pd.DataFrame(report_data)
    
    # 결과를 CSV 문서로 저장 (한글 깨짐 방지 utf-8-sig)
    report_df.to_csv("scenario_evaluation_report.csv", index=False, encoding='utf-8-sig')
    
    # 콘솔 요약 출력
    print(f"\n📊 검증 완료! 총 {total_count}개의 이미지 중 {correct_count}개 일치 (정확도: {overall_acc:.2f}%)")
    print("✅ 상세 결과 및 각 책의 판정 내역, 그리고 정확도 통계가 'scenario_evaluation_report.csv'에 저장되었습니다.")
    
    # 불일치 항목 콘솔 출력 (통계 부분 제외하고 필터링)
    actual_results_df = pd.DataFrame([r for r in report_data if not r['폴더(실제 상태)'].startswith('---') and not r['폴더(실제 상태)'].startswith('[통계 요약]') and '정확도' not in r['폴더(실제 상태)']])

    mismatches = actual_results_df[actual_results_df['일치 여부'] == "❌ 불일치"]
    if not mismatches.empty:
        print(f"\n⚠️ 분류 불일치 의심 이미지 목록 ({len(mismatches)}개):")
        print(mismatches[['폴더(실제 상태)', '파일명', '예측된 상태']].head(10).to_string(index=False))
        if len(mismatches) > 10:
            print("... (나머지는 CSV 파일 참고)")
            
    return report_df

def extract_and_save_features():
    """데이터셋 폴더 내 모든 이미지를 처리하고 특징을 추출하여 JSON으로 저장"""
    pass

if __name__ == "__main__":
    if os.path.exists(OUTPUT_JSON):
        print(f"기존 추출된 '{OUTPUT_JSON}' 데이터를 로드합니다...")
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 1. 피쳐 엔지니어링 수행 (기하학 변수 계산)
        engineered_df = engineer_features(raw_data)
            
        # 2. 모든 사진 평가 및 검증 리포트 생성 (최적화된 이상 탐지 로직 적용)
        evaluate_and_report(engineered_df)
    else:
        print(f"'{OUTPUT_JSON}' 파일이 없습니다. 이미지 파이프라인 처리를 먼저 수행해야 합니다.")