# api/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# جلب URL قاعدة البيانات من المتغيرات البيئية
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("SQLALCHEMY_DATABASE_URL is not set in environment variables.")

# إنشاء المحرك
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# إعداد الجلسة
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# قاعدة أساسية للنماذج
Base = declarative_base()

# دالة للحصول على الجلسة
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
