import webbrowser
import subprocess
import os
import sys
import time

# 📂 폴더 경로 설정 (팀장님의 실제 폴더명에 맞게 수정하세요)
ACH_DIR = "ach"
BACKEND_DIR = "backend"

def run_script(script_name, cwd_path):
    """지정된 폴더에서 파이썬 스크립트를 실행하고 끝날 때까지 대기합니다."""
    print(f"\n{'='*60}")
    print(f"🚀 [STEP] 실행 중: {script_name} (경로: {cwd_path})")
    print(f"{'='*60}")
    
    # 해당 폴더가 존재하는지 확인
    if not os.path.exists(cwd_path):
        print(f"❌ [경로 에러] '{cwd_path}' 폴더를 찾을 수 없습니다.")
        sys.exit(1)
        
    script_path = os.path.join(cwd_path, script_name)
    if not os.path.exists(script_path):
        print(f"❌ [파일 에러] '{script_path}' 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 서브프로세스로 파이썬 스크립트 실행
    result = subprocess.run([sys.executable, script_name], cwd=cwd_path)
    
    if result.returncode != 0:
        print(f"\n❌ [실행 에러] {script_name} 실행 중 치명적인 문제가 발생하여 파이프라인을 중단합니다.")
        sys.exit(1)
    else:
        print(f"\n✅ [STEP 완료] {script_name} 성공적으로 종료됨.\n")
        time.sleep(1) # 다음 스텝 전 약간의 딜레이

def main():
    print("🌟 VLM 도서관 서고 관리 플랫폼 - End-to-End 통합 파이프라인 시작 🌟\n")

    # ---------------------------------------------------------
    # 1단계: KJI/ach (Vision 파이프라인) - 이미지 분석 및 특징 벡터 추출
    # ---------------------------------------------------------
    # 결과물: 2026-1-CECD1-5-3expidition-10-ach/extracted_features.json 생성
    print("▶️ [1/3] Edge AI 가동: 서가 이미지에서 책의 특징 벡터를 추출합니다...")
    run_script("FeatureAnalysis.py", cwd_path=ACH_DIR)

    # ---------------------------------------------------------
    # 2단계: ach (물리적 상태 분류) - 이상 탐지 리포트 생성
    # ---------------------------------------------------------
    # 결과물: 2026-1-CECD1-5-3expidition-10-ach/scenario_evaluation_report.csv 생성
    print("▶️ [2/3] Edge AI 가동: 추출된 벡터를 바탕으로 누운 책, 뒤집힌 책을 판별합니다...")
    run_script("MakeResult.py", cwd_path=ACH_DIR)

    # ---------------------------------------------------------
    # 3단계: Backend (팀장님 API 서버 연동) - CSV 읽기 및 융합 검증
    # ---------------------------------------------------------
    # 주의: 이 단계를 실행하기 전 반드시 백엔드 서버(FastAPI)가 켜져 있어야 합니다.
    print("▶️ [3/3] Cloud 서버 연동: 판별 결과를 서버로 전송하고 최종 리포트를 생성합니다...")
    
    # 파일 복사 트릭: test_full_pipeline.py가 읽을 수 있도록 ach 폴더의 CSV를 backend 폴더로 복사
    source_csv = os.path.join(ACH_DIR, "scenario_evaluation_report.csv")
    target_csv = os.path.join(BACKEND_DIR, "scenario_evaluation_report.csv")
    
    if os.path.exists(source_csv):
        import shutil
        shutil.copy(source_csv, target_csv)
        print(f"📦 [데이터 전달] CSV 리포트를 Backend 폴더로 안전하게 복사했습니다.")
    else:
        print("⚠️ [경고] 2단계에서 생성된 CSV 파일을 찾을 수 없습니다.")

    # ... (기존 3단계 코드 유지) ...
    # 백엔드 API 테스트 코드 실행
    run_script("test_full_pipeline.py", cwd_path=BACKEND_DIR)

    print("🎉 [축하합니다!] 엣지(AI)부터 클라우드(DB)까지 모든 파이프라인이 성공적으로 완료되었습니다!")
    
    # 파이프라인이 다 돌면 1.5초 뒤에 자동으로 대시보드를 띄웁니다.
    print("🌐 분석 결과를 확인하기 위해 대시보드 웹페이지를 엽니다...")
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000/dashboard")

if __name__ == "__main__":
    main()