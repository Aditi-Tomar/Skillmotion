import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import aiomysql
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MySQLService:
    def __init__(self):
        self.host = os.getenv("MYSQL_HOST", "localhost")
        self.user = os.getenv("MYSQL_USER", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "")
        self.database = os.getenv("MYSQL_DATABASE", "skillmotion_ai")
        self.port = int(os.getenv("MYSQL_PORT", 3306))

        self.pool = None
        logger.info("✅ MySQL client initialized with configuration")

    async def connect(self):
        """Initialize connection pool"""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=5,
                maxsize=10,
                autocommit=True
            )
            logger.info("✅ MySQL connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise

    async def close(self):
        """Close connection pool"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("✅ MySQL connection pool closed")

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM user_profiles WHERE user_id = %s",
                        (user_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None

    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    columns = ["user_id"] + list(profile_data.keys())
                    values = [user_id] + list(profile_data.values())
                    placeholders = ", ".join(["%s"] * len(columns))
                    column_names = ", ".join(columns)

                    query = f"INSERT INTO user_profiles ({column_names}) VALUES ({placeholders})"
                    await cursor.execute(query, values)
                    await conn.commit()

                    # Fetch and return the created profile
                    await cursor.execute(
                        "SELECT * FROM user_profiles WHERE user_id = %s",
                        (user_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return None

    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    set_clause = ", ".join([f"{k} = %s" for k in profile_data.keys()])
                    values = list(profile_data.values()) + [user_id]

                    query = f"UPDATE user_profiles SET {set_clause} WHERE user_id = %s"
                    await cursor.execute(query, values)
                    await conn.commit()

                    # Fetch and return the updated profile
                    await cursor.execute(
                        "SELECT * FROM user_profiles WHERE user_id = %s",
                        (user_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return None

    async def create_session(self, user_id: str, mode: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO conversation_sessions (user_id, mode, is_active, started_at) VALUES (%s, %s, %s, %s)",
                        (user_id, mode, True, datetime.utcnow())
                    )
                    await conn.commit()

                    session_id = cursor.lastrowid
                    await cursor.execute(
                        "SELECT * FROM conversation_sessions WHERE id = %s",
                        (session_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None

    async def get_active_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM conversation_sessions WHERE user_id = %s AND is_active = TRUE ORDER BY started_at DESC LIMIT 1",
                        (user_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching active session: {e}")
            return None

    async def end_session(self, session_id: str) -> bool:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "UPDATE conversation_sessions SET is_active = FALSE, ended_at = %s WHERE id = %s",
                        (datetime.utcnow(), session_id)
                    )
                    await conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False

    async def save_message(self, session_id: str, user_id: str, message_type: str, content: str,
                           metadata: Dict[str, Any] = None, was_interrupted: bool = False) -> Optional[Dict[str, Any]]:
        try:
            import json
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO conversation_messages (session_id, user_id, message_type, content, metadata, was_interrupted, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (session_id, user_id, message_type, content, json.dumps(metadata or {}), was_interrupted,
                         datetime.utcnow())
                    )
                    await conn.commit()

                    message_id = cursor.lastrowid
                    await cursor.execute(
                        "SELECT * FROM conversation_messages WHERE id = %s",
                        (message_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return None

    async def get_session_messages(self, session_id: str, limit: int = 50) -> list:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM conversation_messages WHERE session_id = %s ORDER BY created_at ASC LIMIT %s",
                        (session_id, limit)
                    )
                    return await cursor.fetchall() or []
        except Exception as e:
            logger.error(f"Error fetching session messages: {e}")
            return []

    async def add_pending_query(self, session_id: str, user_id: str, query: str, priority: int = 0) -> Optional[
        Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO pending_queries (session_id, user_id, query, priority, status, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                        (session_id, user_id, query, priority, "pending", datetime.utcnow())
                    )
                    await conn.commit()

                    query_id = cursor.lastrowid
                    await cursor.execute(
                        "SELECT * FROM pending_queries WHERE id = %s",
                        (query_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error adding pending query: {e}")
            return None

    async def get_pending_queries(self, session_id: str) -> list:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM pending_queries WHERE session_id = %s AND status = 'pending' ORDER BY priority, created_at",
                        (session_id,)
                    )
                    return await cursor.fetchall() or []
        except Exception as e:
            logger.error(f"Error fetching pending queries: {e}")
            return []

    async def update_query_status(self, query_id: str, status: str) -> bool:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    processed_at = datetime.utcnow() if status == "completed" else None
                    await cursor.execute(
                        "UPDATE pending_queries SET status = %s, processed_at = %s WHERE id = %s",
                        (status, processed_at, query_id)
                    )
                    await conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error updating query status: {e}")
            return False

    async def create_support_request(self, user_id: str, subject: str, description: str,
                                     context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        try:
            import json
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO support_requests (user_id, subject, description, context, status, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                        (user_id, subject, description, json.dumps(context or {}), "pending", datetime.utcnow())
                    )
                    await conn.commit()

                    request_id = cursor.lastrowid
                    await cursor.execute(
                        "SELECT * FROM support_requests WHERE id = %s",
                        (request_id,)
                    )
                    return await cursor.fetchone()
        except Exception as e:
            logger.error(f"Error creating support request: {e}")
            return None
