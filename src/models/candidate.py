"""Data models for the job assessment system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


@dataclass
class AssessmentResult:
    """Structured assessment result."""
    skill: str
    score: float
    grade: str
    timestamp: str
    details: Dict[str, Any]


@dataclass
class CandidateProfile:
    """Enhanced candidate profile with comprehensive tracking."""
    candidate_id: str
    name: str
    applied_role: str
    assessment_history: List[AssessmentResult] = field(default_factory=list)
    skill_levels: Dict[str, float] = field(default_factory=dict)
    current_assessment: Optional[str] = None
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    assessment_status: str = "started"  # "started", "in_progress", "completed"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionMetadata:
    """Session metadata for tracking assessment progress."""
    total_assessments: int = 0
    completed_skills: List[str] = field(default_factory=list)
    pending_skills: List[str] = field(default_factory=list)
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QuizState:
    """State for label reading quiz."""
    labels: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    current_question: int = 0
    correct_answers: int = 0
    quiz_active: bool = False


def create_initial_candidate_state(
    candidate_name: str, 
    candidate_id: str = None, 
    role: str = None
) -> Dict[str, Any]:
    """Create comprehensive initial state for a candidate."""
    if not candidate_id:
        candidate_id = str(uuid.uuid4())[:8]
    
    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "applied_role": role or "unknown",
        "role_identified": bool(role),
        "assessment_history": [],
        "skill_levels": {},
        "current_assessment": None,
        "interaction_history": [],
        "assessment_status": "started",
        "current_label_image": "",
        "created_at": datetime.now().isoformat(),
        "session_metadata": {
            "total_assessments": 0,
            "completed_skills": [],
            "pending_skills": [],
            "last_activity": datetime.now().isoformat()
        }
    }
