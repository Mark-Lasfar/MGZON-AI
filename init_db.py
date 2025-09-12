import os
import logging
from api.database import async_engine, Base, User, OAuthAccount, Conversation, Message, AsyncSessionLocal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    logger.info("Starting database initialization...")

    # إنشاء الجداول (sync version for init_db.py)
    try:
        from sqlalchemy import create_engine
        sync_engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///./data/mgzon_users.db"))
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

    # تنظيف البيانات غير المتسقة (sync for simplicity in init_db)
    try:
        from sqlalchemy import select, delete
        from sqlalchemy.orm import sessionmaker
        sync_engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///./data/mgzon_users.db"))
        SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
        with SyncSessionLocal() as session:
            # حذف سجلات oauth_accounts اللي مش مرتبطة بمستخدم موجود
            stmt = delete(OAuthAccount).where(
                OAuthAccount.user_id.notin_(select(User.id))
            )
            result = session.execute(stmt)
            deleted_count = result.rowcount
            session.commit()
            logger.info(f"Deleted {deleted_count} orphaned OAuth accounts.")

            # التأكد من إن كل المستخدمين ليهم is_active=True
            users = session.execute(select(User)).scalars().all()
            for user in users:
                if not user.is_active:
                    user.is_active = True
                    logger.info(f"Updated user {user.email} to is_active=True")
            session.commit()

            # اختبار إنشاء مستخدم ومحادثة (اختياري)
            test_user = session.query(User).filter_by(email="test@example.com").first()
            if not test_user:
                test_user = User(
                    email="test@example.com",
                    hashed_password="$2b$12$examplehashedpassword",  # استبدل بكلمة مرور مشفرة حقيقية
                    is_active=True,
                    display_name="Test User"
                )
                session.add(test_user)
                session.commit()
                logger.info("Test user created successfully.")

            test_conversation = session.query(Conversation).filter_by(user_id=test_user.id).first()
            if not test_conversation:
                test_conversation = Conversation(
                    conversation_id="test-conversation-1",
                    user_id=test_user.id,
                    title="Test Conversation"
                )
                session.add(test_conversation)
                session.commit()
                logger.info("Test conversation created successfully.")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

    logger.info("Database initialization completed.")

if __name__ == "__main__":
    init_db()
