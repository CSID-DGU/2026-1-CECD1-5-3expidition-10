import os
import shutil
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG, PATROL_INBOX_DIR, SPINE_STORE_DIR

# 분석 세션 데이터와 순찰 사진 수신함을 초기화합니다.
# 가상 서가/도서 정보(SHELF_INFO, BOOK_MASTER)는 지우지 않습니다. (재설정: python setup_db.py)

def reset_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("🧹 데이터베이스 초기화를 시작합니다...")

        # 외래키 제약조건 잠시 해제 (안전하고 빠른 삭제를 위해)
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

        # 비워야 할 핵심 테이블 목록
        tables_to_clear = ['ANALYSIS_RESULT', 'VISION_DATA', 'RFID_DATA', 'SHELF_SESSION', 'PATROL_PHOTO']
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (DB_CONFIG['database'],))
        existing = {row[0].upper() for row in cursor.fetchall()}

        for table in tables_to_clear:
            if table not in existing:   # 예전 DB에는 PATROL_PHOTO가 없을 수 있음
                continue
            cursor.execute(f"TRUNCATE TABLE {table};")
            print(f"  ✔️ {table} 테이블 비우기 완료!")

        # 외래키 제약조건 원상복구
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

        conn.commit()
        print("\n✨ [성공] 모든 테스트 데이터가 깔끔하게 삭제되었습니다!")
        print("🌐 이제 대시보드를 새로고침 해보세요. 데이터가 0건으로 표시될 것입니다.")
        return True

    except Error as e:
        print(f"❌ [에러 발생] DB 초기화 중 문제가 생겼습니다: {e}")
        return False
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def clear_spine_store():
    """세션별 책등 크롭 보관소(spine_store)의 세션 폴더를 모두 삭제합니다."""
    if not os.path.isdir(SPINE_STORE_DIR):
        print("  ✔️ spine_store 폴더가 없어 건너뜁니다.")
        return
    session_dirs = [d for d in os.listdir(SPINE_STORE_DIR) if os.path.isdir(os.path.join(SPINE_STORE_DIR, d))]
    for d in session_dirs:
        shutil.rmtree(os.path.join(SPINE_STORE_DIR, d))
    print(f"  ✔️ spine_store 책등 이미지 세션 폴더 {len(session_dirs)}개 삭제 완료!")

def clear_patrol_inbox():
    """로봇 순찰 사진 수신함(patrol_inbox/<구역>/<책꽂이>/)의 사진을 모두 삭제합니다. (폴더 구조는 유지)"""
    if not os.path.isdir(PATROL_INBOX_DIR):
        return
    removed = 0
    for dirpath, _, filenames in os.walk(PATROL_INBOX_DIR):
        for name in filenames:
            os.remove(os.path.join(dirpath, name))
            removed += 1
    print(f"  ✔️ patrol_inbox 순찰 사진 {removed}장 삭제 완료!")

if __name__ == "__main__":
    print("=========================================")
    print(" 🚨 VLM 도서관 DB 완전 초기화 도구 🚨")
    print("=========================================")
    confirm = input("⚠️ 정말로 DB의 모든 데이터를 삭제하시겠습니까? 복구할 수 없습니다. (y/n): ")

    if confirm.lower() == 'y':
        # DB 초기화에 성공했을 때만 이미지도 지워, DB 기록과 이미지가 어긋나지 않게 합니다.
        if reset_database():
            clear_spine_store()
            clear_patrol_inbox()
    else:
        print("🛑 초기화 작업이 취소되었습니다.")
