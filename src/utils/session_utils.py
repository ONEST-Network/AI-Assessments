"""Session management utilities."""

import logging
from datetime import datetime
from typing import Dict, Any
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)


async def update_interaction_history(
    session_service: InMemorySessionService, 
    app_name: str, 
    user_id: str, 
    session_id: str, 
    interaction_data: Dict[str, Any]
):
    """Update interaction history following proper ADK pattern."""
    try:
        current_session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id, 
            session_id=session_id
        )
        
        if current_session:
            # Copy existing state
            updated_state = current_session.state.copy()
            
            # Ensure interaction_history exists
            if "interaction_history" not in updated_state:
                updated_state["interaction_history"] = []
            
            # Add timestamp to interaction
            interaction_entry = {
                "timestamp": datetime.now().isoformat(),
                **interaction_data
            }
            
            # Append new interaction
            updated_state["interaction_history"].append(interaction_entry)
            updated_state["session_metadata"]["last_activity"] = datetime.now().isoformat()
            
            # Create new session with updated state using keyword arguments
            await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id, 
                state=updated_state
            )
            
    except Exception as e:
        logger.error(f"Failed to update interaction history: {e}")


async def add_user_query_to_history(
    session_service: InMemorySessionService, 
    app_name: str, 
    user_id: str, 
    session_id: str, 
    query: str, 
    image_provided: bool = False
):
    """Add user query to interaction history."""
    await update_interaction_history(session_service, app_name, user_id, session_id, {
        "action": "user_query",
        "query": query,
        "image_provided": image_provided
    })


async def add_agent_response_to_history(
    session_service: InMemorySessionService, 
    app_name: str, 
    user_id: str, 
    session_id: str, 
    response: str
):
    """Add agent response to interaction history."""
    await update_interaction_history(session_service, app_name, user_id, session_id, {
        "action": "agent_response", 
        "response": response
    })
