import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG


def get_db_connection():
    """DB 연결을 반환합니다. 실패하면 None."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"🚨 DB 연결 에러: {e}")
        return None
