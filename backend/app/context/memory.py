from sqlalchemy.orm import Session
from backend.app.db.models import ChatHistory, ChatRole
from typing import List, Dict

class MemoryManager:
    def __init__(self, db: Session):
        self.db = db
        
    def get_recent_history(self, session_id: int, limit: int = 5) -> List[Dict[str, str]]:
        """
        Fetches the last `limit` messages from the session's chat history.
        Returns them in chronological order.
        """
        history = self.db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.created_at.desc()).limit(limit).all()
        
        # Reverse to get chronological order
        history.reverse()
        
        return [{"role": msg.role.value, "content": msg.content} for msg in history]
        
    def add_message(self, session_id: int, role: ChatRole, content: str):
        """
        Adds a single message to the session's chat history.
        """
        msg = ChatHistory(session_id=session_id, role=role, content=content)
        self.db.add(msg)
        self.db.commit()
