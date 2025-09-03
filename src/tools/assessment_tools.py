"""Assessment tools for the job assessment system."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from google.adk.tools import ToolContext
from ..utils.image_utils import load_image_from_path, validate_image

logger = logging.getLogger(__name__)


def complete_skill_assessment(
    skill_name: str, 
    score: float, 
    grade: str, 
    details: Dict[str, Any], 
    tool_context: ToolContext
) -> str:
    """Tool to complete a skill assessment and update candidate profile."""
    try:
        # Create assessment result
        assessment_result = {
            "skill": skill_name,
            "score": score,
            "grade": grade,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        # Get current assessment history and update it
        current_history = tool_context.state.get("assessment_history", [])
        current_history.append(assessment_result)
        tool_context.state["assessment_history"] = current_history
        
        # Get current skill levels and update
        current_skill_levels = tool_context.state.get("skill_levels", {})
        current_skill_levels[skill_name] = score
        tool_context.state["skill_levels"] = current_skill_levels
        
        # Get current session metadata and update
        current_metadata = tool_context.state.get("session_metadata", {
            "completed_skills": [], 
            "pending_skills": []
        })
        
        if skill_name not in current_metadata["completed_skills"]:
            current_metadata["completed_skills"].append(skill_name)
        
        # Remove from pending if exists
        if skill_name in current_metadata.get("pending_skills", []):
            current_metadata["pending_skills"].remove(skill_name)
        
        # Update metadata
        current_metadata["total_assessments"] = len(current_history)
        current_metadata["last_activity"] = datetime.now().isoformat()
        
        tool_context.state["session_metadata"] = current_metadata
        tool_context.state["current_assessment"] = None
        tool_context.state["assessment_status"] = "completed"
        
        logger.info(f"Completed {skill_name} assessment: score={score}, grade={grade}")
        logger.info(f"Updated session state - total assessments: {len(current_history)}")
        return f"Successfully completed {skill_name} assessment with score {score}/10 and grade {grade}"
        
    except Exception as e:
        logger.error(f"Error completing skill assessment: {e}")
        return f"Error completing assessment: {str(e)}"


def start_skill_assessment(skill_name: str, tool_context: ToolContext) -> str:
    """Tool to start a new skill assessment."""
    try:
        # Update current assessment
        tool_context.state["current_assessment"] = skill_name
        tool_context.state["assessment_status"] = "in_progress"
        
        # Get current session metadata and update
        current_metadata = tool_context.state.get("session_metadata", {"pending_skills": []})
        
        if skill_name not in current_metadata.get("pending_skills", []):
            current_metadata["pending_skills"].append(skill_name)
        
        current_metadata["last_activity"] = datetime.now().isoformat()
        tool_context.state["session_metadata"] = current_metadata
        
        logger.info(f"Started {skill_name} assessment")
        candidate_name = tool_context.state.get('candidate_name', 'candidate')
        return f"Started {skill_name} assessment for {candidate_name}"
        
    except Exception as e:
        logger.error(f"Error starting skill assessment: {e}")
        return f"Error starting assessment: {str(e)}"


def get_candidate_profile(tool_context: ToolContext) -> str:
    """Tool to retrieve candidate profile information."""
    try:
        state = tool_context.state
        
        profile_summary = {
            "candidate_id": state.get("candidate_id", "unknown"),
            "name": state.get("candidate_name", "unknown"),
            "role": state.get("applied_role", "unknown"),
            "role_identified": state.get("role_identified", False),
            "status": state.get("assessment_status", "unknown"),
            "completed_skills": state.get("session_metadata", {}).get("completed_skills", []),
            "skill_levels": state.get("skill_levels", {}),
            "total_assessments": len(state.get("assessment_history", []))
        }
        
        return json.dumps(profile_summary, indent=2)
        
    except Exception as e:
        logger.error(f"Error retrieving candidate profile: {e}")
        return f"Error retrieving profile: {str(e)}"


def update_candidate_role(role: str, tool_context: ToolContext) -> str:
    """Tool to update candidate's role once identified through conversation."""
    try:
        state = tool_context.state
        
        # Update role information
        state["applied_role"] = role
        state["role_identified"] = True
        state["session_metadata"]["last_activity"] = datetime.now().isoformat()
        
        # Add to interaction history
        if "interaction_history" not in state:
            state["interaction_history"] = []
        
        state["interaction_history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "role_identification",
            "content": f"Role identified as: {role}",
            "metadata": {"previous_role": state.get("applied_role", "unknown")}
        })
        
        logger.info(f"Updated candidate role to: {role}")
        return f"Successfully updated candidate role to {role}"
        
    except Exception as e:
        logger.error(f"Error updating candidate role: {e}")
        return f"Error updating role: {str(e)}"


def retrieve_image_from_path(image_path: str, tool_context: ToolContext) -> str:
    """Enhanced image retrieval tool with state updates."""
    try:
        image_data = load_image_from_path(image_path)
        
        # Store image data in tool context
        tool_context.state['image_data'] = image_data
        tool_context.state['image_path'] = image_path
        tool_context.state['image_size'] = len(image_data)
        
        # Update interaction history
        if "interaction_history" not in tool_context.state:
            tool_context.state["interaction_history"] = []
        
        tool_context.state["interaction_history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "image_upload",
            "content": f"Image uploaded: {image_path}",
            "metadata": {"file_size": len(image_data)}
        })
        
        return f"Successfully retrieved image from {image_path} ({len(image_data)} bytes)"
        
    except Exception as e:
        logger.error(f"Error retrieving image from path {image_path}: {str(e)}")
        raise


def validate_image_data(tool_context: ToolContext) -> str:
    """Enhanced image validation with comprehensive checks."""
    try:
        image_data = tool_context.state.get('image_data')
        image_path = tool_context.state.get('image_path', 'unknown')
        
        if not image_data:
            raise ValueError("No image data found in context. Call retrieve_image_from_path first.")
        
        validation_result = validate_image(image_data, image_path)
        
        # Store validation results
        tool_context.state['validation_result'] = validation_result
        
        if validation_result.get("valid"):
            return (f"Image validation successful: {validation_result['format']} format, "
                   f"{validation_result['size']} pixels")
        else:
            error_msg = f"Image validation failed: {validation_result.get('error', 'Unknown error')}"
            return error_msg
        
    except Exception as e:
        error_msg = f"Image validation failed: {str(e)}"
        logger.error(error_msg)
        tool_context.state['validation_result'] = {"valid": False, "error": str(e)}
        return error_msg
