"""Stateful job assessment system."""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from . import Config
from src.models import create_initial_candidate_state
from src.utils import add_user_query_to_history, add_agent_response_to_history
from src.agents import create_master_agent

logger = logging.getLogger(__name__)


class StatefulJobAssessmentSystem:
    """Enhanced multi-agent job assessment system with comprehensive state management."""
    
    def __init__(self, prompts_dir: str = None):
        self.prompts_dir = prompts_dir or Config.ASSETS_DIR
        self.session_service = InMemorySessionService()
        self._load_knowledge()
        self.master_agent = self._create_master_agent()
        self.runner = Runner(
            agent=self.master_agent,
            app_name=Config.APP_NAME,
            session_service=self.session_service
        )
        
    def _load_knowledge(self):
        """Load knowledge base files."""
        try:
            competency_map_path = os.path.join(self.prompts_dir, Config.COMPETENCY_MAP_FILE)
            sub_agent_library_path = os.path.join(self.prompts_dir, Config.SUB_AGENT_LIBRARY_FILE)
            
            with open(competency_map_path, 'r') as f:
                self.competency_map = json.load(f)
            with open(sub_agent_library_path, 'r') as f:
                self.sub_agent_library = json.load(f)
                
        except FileNotFoundError as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def _create_master_agent(self):
        """Create enhanced master agent with state awareness."""
        return create_master_agent(self.competency_map, self.sub_agent_library)
    
    async def create_candidate_session(
        self, 
        candidate_name: str, 
        candidate_id: str = None, 
        role: str = None
    ) -> str:
        """Create new candidate session with comprehensive state."""
        if not candidate_id:
            candidate_id = str(uuid.uuid4())[:8]
        
        session_id = f"candidate_{candidate_id}"
        initial_state = create_initial_candidate_state(candidate_name, candidate_id, role)
        
        try:
            await self.session_service.create_session(
                app_name=Config.APP_NAME,
                user_id=candidate_id,
                session_id=session_id,
                state=initial_state
            )
            
            role_info = f" for {role}" if role else " (role to be identified)"
            logger.info(f"Created session for {candidate_name} (ID: {candidate_id}){role_info}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating candidate session: {e}")
            raise
    
    async def get_candidate_session(self, session_id: str, user_id: str):
        """Retrieve candidate session."""
        try:
            session = await self.session_service.get_session(
                app_name=Config.APP_NAME,
                user_id=user_id,
                session_id=session_id
            )
            return session
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            return None
    
    async def process_candidate_interaction(
        self, 
        session_id: str, 
        user_id: str, 
        user_message: str, 
        image_path: str = None
    ) -> str:
        """Process candidate interaction with proper ADK state management."""
        try:
            # Add user query to interaction history
            await add_user_query_to_history(
                self.session_service, Config.APP_NAME, user_id, session_id,
                user_message, bool(image_path)
            )
            
            # Prepare message with image context if provided
            if image_path:
                message_text = f"{user_message} [Image: {image_path}]"
            else:
                message_text = user_message
            
            # Create content for agent
            content = types.Content(role='user', parts=[types.Part(text=message_text)])
            
            # Run agent with proper session context
            events = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )
            
            # Collect response - capture all text parts
            response_texts = []
            async for event in events:
                if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text and part.text.strip():
                            response_texts.append(part.text.strip())
            
            # Join all text parts to get complete response
            response_text = "\n\n".join(response_texts) if response_texts else ""
            
            # Add agent response to interaction history
            await add_agent_response_to_history(
                self.session_service, Config.APP_NAME, user_id, session_id,
                response_text or "Assessment completed"
            )
            
            return response_text or "Assessment completed successfully"
            
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return f"Error processing request: {str(e)}"
    
    async def get_assessment_summary(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive assessment summary."""
        try:
            session = await self.get_candidate_session(session_id, user_id)
            if not session:
                return {"error": "Session not found"}
            
            state = session.state
            
            # Calculate final verdict
            final_verdict = self._calculate_final_verdict(state)
            
            summary = {
                "candidate_info": {
                    "id": state.get("candidate_id"),
                    "name": state.get("candidate_name"),
                    "role": state.get("applied_role"),
                    "status": state.get("assessment_status")
                },
                "assessment_results": state.get("assessment_history", []),
                "skill_levels": state.get("skill_levels", {}),
                "session_stats": state.get("session_metadata", {}),
                "final_verdict": final_verdict,
                "total_interactions": len(state.get("interaction_history", []))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating assessment summary: {e}")
            return {"error": str(e)}
    
    def _calculate_final_verdict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final verdict based on competency map and assessment results."""
        try:
            role = state.get("applied_role", "")
            assessment_history = state.get("assessment_history", [])
            
            if not role or role not in self.competency_map.get("roles", {}):
                return {
                    "decision": "INCOMPLETE",
                    "reason": "Unknown role or missing assessments"
                }
            
            role_requirements = self.competency_map["roles"][role]
            required_skills = role_requirements.get("required_skills", [])
            thresholds = role_requirements.get("passing_thresholds", {})
            
            results = {}
            overall_pass = True
            
            for skill in required_skills:
                skill_assessment = None
                for assessment in assessment_history:
                    assessment_skill = assessment.get("skill", "").lower()
                    if skill.lower() in assessment_skill or assessment_skill in skill.lower():
                        skill_assessment = assessment
                        break
                
                if not skill_assessment:
                    results[skill] = {"status": "MISSING", "pass": False}
                    overall_pass = False
                    continue
                
                # Check against thresholds
                skill_thresholds = thresholds.get(skill, {})
                skill_pass = True
                
                if "min_quality_rating" in skill_thresholds:
                    min_rating = skill_thresholds["min_quality_rating"]
                    if skill_assessment["score"] < min_rating:
                        skill_pass = False
                
                if "required_professional_grade" in skill_thresholds:
                    required_grades = skill_thresholds["required_professional_grade"]
                    if skill_assessment["grade"] not in required_grades:
                        skill_pass = False
                
                if "min_accuracy" in skill_thresholds:
                    min_accuracy = skill_thresholds["min_accuracy"]
                    if skill_assessment["score"] < min_accuracy:
                        skill_pass = False
                
                results[skill] = {
                    "status": "COMPLETED",
                    "pass": skill_pass,
                    "score": skill_assessment["score"],
                    "grade": skill_assessment["grade"]
                }
                
                if not skill_pass:
                    overall_pass = False
            
            return {
                "decision": "PASS" if overall_pass else "FAIL",
                "skill_results": results,
                "overall_pass": overall_pass,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating final verdict: {e}")
            return {
                "decision": "ERROR",
                "reason": str(e)
            }
