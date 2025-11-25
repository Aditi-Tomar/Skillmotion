import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
from mysql_client import MySQLService

logger = logging.getLogger(__name__)

@dataclass
class PendingQuery:
    query: str
    priority: int
    timestamp: datetime
    query_id: Optional[str] = None

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    mode: str
    is_active: bool = True
    current_speaking: bool = False
    pending_queries: List[PendingQuery] = field(default_factory=list)
    resume_data: Optional[Dict[str, Any]] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)

class SessionManager:
    def __init__(self, mysql_service: MySQLService):
        self.mysql = mysql_service
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_timeout = 1800

    async def create_session(self, user_id: str, mode: str) -> SessionContext:
        existing_session = await self.get_active_session(user_id)
        if existing_session:
            await self.end_session(existing_session.session_id)

        session_data = await self.mysql.create_session(user_id, mode)

        if session_data:
            context = SessionContext(
                session_id=session_data['id'],
                user_id=user_id,
                mode=mode,
                is_active=True
            )
            self.active_sessions[user_id] = context
            logger.info(f"Created new session {context.session_id} for user {user_id}")
            return context

        raise Exception("Failed to create session")

    async def get_active_session(self, user_id: str) -> Optional[SessionContext]:
        if user_id in self.active_sessions:
            context = self.active_sessions[user_id]
            if context.is_active:
                return context

        session_data = await self.mysql.get_active_session(user_id)
        if session_data:
            context = SessionContext(
                session_id=session_data['id'],
                user_id=user_id,
                mode=session_data['mode'],
                is_active=session_data['is_active']
            )
            self.active_sessions[user_id] = context
            return context

        return None

    async def end_session(self, session_id: str):
        for user_id, context in list(self.active_sessions.items()):
            if context.session_id == session_id:
                context.is_active = False
                await self.mysql.end_session(session_id)
                del self.active_sessions[user_id]
                logger.info(f"Ended session {session_id}")
                break

    async def save_message(self, user_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None, was_interrupted: bool = False):
        session = await self.get_active_session(user_id)
        if session:
            session.last_activity = datetime.utcnow()
            await self.mysql.save_message(
                session.session_id,
                user_id,
                message_type,
                content,
                metadata,
                was_interrupted
            )

    async def handle_interruption(self, user_id: str, current_response: str, new_query: str):
        session = await self.get_active_session(user_id)
        if not session:
            logger.warning(f"No active session for user {user_id}")
            return

        if session.current_speaking and current_response:
            await self.mysql.save_message(
                session.session_id,
                user_id,
                'ai',
                current_response,
                {'interrupted': True},
                was_interrupted=True
            )

            pending = PendingQuery(
                query=f"Continue from: {current_response[:100]}...",
                priority=1,
                timestamp=datetime.utcnow()
            )

            pending_data = await self.mysql.add_pending_query(
                session.session_id,
                user_id,
                pending.query,
                pending.priority
            )

            if pending_data:
                pending.query_id = pending_data['id']
                session.pending_queries.append(pending)
                logger.info(f"Added interrupted query to pending queue for user {user_id}")

        session.current_speaking = False

    async def process_pending_queries(self, user_id: str) -> List[PendingQuery]:
        session = await self.get_active_session(user_id)
        if not session:
            return []

        pending_queries = await self.mysql.get_pending_queries(session.session_id)

        queries = []
        for query_data in pending_queries:
            query = PendingQuery(
                query=query_data['query'],
                priority=query_data['priority'],
                timestamp=query_data['created_at'] if isinstance(query_data['created_at'], datetime) else datetime.fromisoformat(str(query_data['created_at'])),
                query_id=query_data['id']
            )
            queries.append(query)

        return queries

    async def mark_query_completed(self, query_id: str):
        await self.mysql.update_query_status(query_id, 'completed')

    async def update_resume_data(self, user_id: str, resume_data: Dict[str, Any]):
        session = await self.get_active_session(user_id)
        if session:
            session.resume_data = resume_data

        profile = await self.mysql.get_user_profile(user_id)
        if profile:
            await self.mysql.update_user_profile(user_id, {
                'resume_data': str(resume_data),
                'skills': str(resume_data.get('skills', []))
            })
        else:
            await self.mysql.create_user_profile(user_id, {
                'resume_data': str(resume_data),
                'skills': str(resume_data.get('skills', []))
            })

    async def get_session_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        session = await self.get_active_session(user_id)
        if not session:
            return None

        messages = await self.mysql.get_session_messages(session.session_id, limit=20)

        return {
            'session_id': session.session_id,
            'mode': session.mode,
            'resume_data': session.resume_data,
            'recent_messages': messages,
            'has_pending': len(session.pending_queries) > 0
        }

    async def cleanup_inactive_sessions(self):
        current_time = datetime.utcnow()
        for user_id, session in list(self.active_sessions.items()):
            elapsed = (current_time - session.last_activity).total_seconds()
            if elapsed > self.session_timeout:
                await self.end_session(session.session_id)
                logger.info(f"Cleaned up inactive session for user {user_id}")
