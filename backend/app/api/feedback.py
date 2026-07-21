from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from backend.app.db.database import get_db
from backend.app.db.models import UserFeedback, ChatHistory

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackCreate(BaseModel):
    message_id: int
    rating: int # 1 for upvote, -1 for downvote
    comment: Optional[str] = None

@router.post("/")
def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    # Verify message exists
    message = db.query(ChatHistory).filter(ChatHistory.id == feedback.message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
        
    # Check if feedback already exists for this message
    existing_feedback = db.query(UserFeedback).filter(UserFeedback.message_id == feedback.message_id).first()
    
    if existing_feedback:
        existing_feedback.rating = feedback.rating
        existing_feedback.comment = feedback.comment
    else:
        new_feedback = UserFeedback(
            message_id=feedback.message_id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        db.add(new_feedback)
        
    db.commit()
    return {"status": "success", "message": "Feedback recorded"}
