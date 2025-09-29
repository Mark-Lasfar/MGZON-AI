# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from api.database import async_engine, Base, User, OAuthAccount, Conversation, Message, AsyncSessionLocal
from passlib.context import CryptContext

# إعداد اللوج
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد تشفير كلمة المرور
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def init_db():
    logger.info("Starting database initialization...")

    # إنشاء الجداول
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

    # تنظيف البيانات غير المتسقة
    async with AsyncSessionLocal() as session:
        try:
            # حذف سجلات oauth_accounts اللي مش مرتبطة بمستخدم موجود
            stmt = delete(OAuthAccount).where(
                OAuthAccount.user_id.notin_(select(User.id))
            )
            result = await session.execute(stmt)
            deleted_count = result.rowcount
            await session.commit()
            logger.info(f"Deleted {deleted_count} orphaned OAuth accounts.")

            # التأكد من إن كل المستخدمين ليهم is_active=True
            users = (await session.execute(select(User))).scalars().all()
            for user in users:
                if not user.is_active:
                    user.is_active = True
                    logger.info(f"Updated user {user.email} to is_active=True")
            await session.commit()

            # اختبار إنشاء مستخدم ومحادثة (اختياري)
            test_user = (await session.execute(
                select(User).filter_by(email="test@example.com")
            )).scalar_one_or_none()
            if not test_user:
                # استخدام كلمة مرور أقصر لتجنب مشكلة bcrypt
                test_password = "testpass"
                if len(test_password.encode('utf-8')) > 72:
                    logger.error("Test password is too long for bcrypt (>72 bytes)")
                    raise ValueError("Test password is too long for bcrypt (>72 bytes)")
                test_user = User(
                    email="test@example.com",
                    hashed_password=pwd_context.hash(test_password),
                    is_active=True,
                    display_name="Test User"
                )
                session.add(test_user)
                await session.commit()
                logger.info("Test user created successfully.")

            test_conversation = (await session.execute(
                select(Conversation).filter_by(user_id=test_user.id)
            )).scalar_one_or_none()
            if not test_conversation:
                test_conversation = Conversation(
                    conversation_id="test-conversation-1",
                    user_id=test_user.id,
                    title="Test Conversation"
                )
                session.add(test_conversation)
                await session.commit()
                logger.info("Test conversation created successfully.")

        except Exception as e:
            await session.rollback()
            logger.error(f"Error during initialization: {e}")
            raise
        finally:
            await session.close()

    logger.info("Database initialization completed.")

if __name__ == "__main__":
    try:
        asyncio.run(init_db())
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
