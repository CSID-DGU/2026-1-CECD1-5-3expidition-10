import mysql.connector
from mysql.connector import Error

# 팀장님의 DB 환경설정
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1234',
    'database': 'library_ai_db',
    'port': 3306
}

def reset_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("🧹 데이터베이스 초기화를 시작합니다...")

        # 외래키 제약조건 잠시 해제 (안전하고 빠른 삭제를 위해)
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

        # 비워야 할 핵심 테이블 목록
        tables_to_clear = ['ANALYSIS_RESULT', 'VISION_DATA', 'RFID_DATA', 'SHELF_SESSION']
        
        for table in tables_to_clear:
            cursor.execute(f"TRUNCATE TABLE {table};")
            print(f"  ✔️ {table} 테이블 비우기 완료!")

        # 외래키 제약조건 원상복구
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

        conn.commit()
        print("\n✨ [성공] 모든 테스트 데이터가 깔끔하게 삭제되었습니다!")
        print("🌐 이제 대시보드를 새로고침 해보세요. 데이터가 0건으로 표시될 것입니다.")

    except Error as e:
        print(f"❌ [에러 발생] DB 초기화 중 문제가 생겼습니다: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    print("=========================================")
    print(" 🚨 VLM 도서관 DB 완전 초기화 도구 🚨")
    print("=========================================")
    confirm = input("⚠️ 정말로 DB의 모든 데이터를 삭제하시겠습니까? 복구할 수 없습니다. (y/n): ")
    
    if confirm.lower() == 'y':
        reset_database()
    else:
        print("🛑 초기화 작업이 취소되었습니다.")