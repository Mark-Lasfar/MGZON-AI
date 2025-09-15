# create_dummy_user.py

import asyncio
from api.database import get_db, User
from passlib.hash import bcrypt

async def create_dummy_user():
    async for session in get_db():
        existing = await session.execute(
            User.__table__.select().where(User.email == "admin@example.com")
        )
        if existing.scalar():
            print("⚠️ User already exists.")
            return

        user = User(
            email="admin@example.com",
            hashed_password=bcrypt.hash("00000000"),
            is_active=True,
            is_superuser=True,
            is_verified=True,
            display_name="Admin"
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        print(f"✅ User created: {user.email}")

if __name__ == "__main__":
    asyncio.run(create_dummy_user())
