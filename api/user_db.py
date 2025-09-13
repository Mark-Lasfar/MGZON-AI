# api/user_db.py
# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.database import User, OAuthAccount  # استيراد جداولك

logger = logging.getLogger(__name__)

class CustomSQLAlchemyUserDatabase(SQLAlchemyUserDatabase[User, int]):
    """
    قاعدة بيانات مخصَّصة لمكتبة fastapi‑users.
    تضيف طريقة parse_id التي تُحوِّل الـ ID من str → int.
    """

    def parse_id(self, value: Any) -> int:
        logger.debug(f"Parsing user id: {value} (type={type(value)})")
        # إذا كان الـ ID نصًا (من JWT) → حوّله إلى int
        return int(value) if isinstance(value, str) else value

    # ---------- وظائف مساعدة ----------
    async def get_by_email(self, email: str) -> Optional[User]:
        logger.info(f"Looking for user with email: {email}")
        stmt = select(self.user_table).where(self.user_table.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, create_dict: Dict[str, Any]) -> User:
        logger.info(f"Creating new user: {create_dict.get('email')}")
        user = self.user_table(**create_dict)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

# ---------- Dependency يُستَخدم في باقي المشروع ----------
async def get_user_db(
    session: AsyncSession = Depends(lambda: None)   # سيتم استبداله في database.py
) -> AsyncGenerator[CustomSQLAlchemyUserDatabase, None]:
    """
    يُستَخدم كـ Depends في جميع المسارات التي تحتاج إلى قاعدة بيانات المستخدم.
    سيتم تمرير الـ AsyncSession الفعلي من `api/database.py`.
    """
    yield CustomSQLAlchemyUserDatabase(session, User, OAuthAccount)
