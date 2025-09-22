"""
A2A SDK-based Job Assessment Agent Executor

This module integrates the existing StatefulJobAssessmentSystem with the A2A SDK
to provide standard A2A protocol compliance while preserving all existing functionality.
"""

import os
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    Task,
    TaskState,
    TextPart,
    FilePart,
    FileWithUri,
    FileWithBytes,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_agent_parts_message,
    new_task,
)
from a2a.utils.errors import ServerError
from app import _set_a2a_context_id, _set_adk_ids, _set_shared_runner

# Assessment system will be imported lazily to avoid blocking startup

logger = logging.getLogger(__name__)


class JobAssessmentAgentExecutor(AgentExecutor):
    """A2A SDK-compliant executor for the job assessment system."""
    
    def __init__(self):
        """Initialize the job assessment executor."""
        # Lazy initialization - don't create assessment system until needed
        self.assessment_system = None
        
        # Map A2A context IDs to your session information
        self.a2a_sessions: Dict[str, Dict[str, str]] = {}
        
        logger.info("JobAssessmentAgentExecutor initialized")
        
        # Eagerly initialize the assessment system at server startup to avoid first-request delay
        try:
            self._get_assessment_system()
        except Exception as e:
            logger.error(f"Failed to initialize StatefulJobAssessmentSystem on startup: {e}")

    def _get_assessment_system(self):
        """Get or initialize the assessment system lazily."""
        if self.assessment_system is None:
            logger.info("Initializing StatefulJobAssessmentSystem...")
            from app import StatefulJobAssessmentSystem
            self.assessment_system = StatefulJobAssessmentSystem()
            logger.info("StatefulJobAssessmentSystem initialized successfully")
        return self.assessment_system

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute job assessment interaction."""
        try:
            # Extract user input and context
            user_input = context.get_user_input()
            context_id = context.context_id
            
            # DEBUG: Log context information
            logger.info(f"DEBUG: Received context_id from RequestContext: {context_id}")
            logger.info(f"DEBUG: context object type: {type(context)}")
            logger.info(f"DEBUG: context.message: {context.message}")
            if hasattr(context.message, 'context_id'):
                logger.info(f"DEBUG: Message context_id: {context.message.context_id}")
            if hasattr(context, 'current_task') and context.current_task:
                logger.info(f"DEBUG: Current task context_id: {context.current_task.context_id}")
                logger.info(f"DEBUG: Current task id: {context.current_task.id}")
            
            # Extract image path if present in message parts
            image_path = self._extract_image_path(context.message)
            
            logger.info(f"Processing A2A request - Context: {context_id}, Input: '{user_input[:50]}...', Has Image: {bool(image_path)}")

            # Get or create task for this interaction
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            updater = TaskUpdater(event_queue, task.id, task.context_id)

            # Get or create assessment session
            session_info = await self._get_or_create_session(context_id)
            # Provide shared runner and IDs to sub-tools
            try:
                _set_shared_runner(self._get_assessment_system().runner, "job_assessment_app")
                _set_a2a_context_id(context_id)
                _set_adk_ids(session_info["user_id"], session_info["session_id"])
            except Exception:
                pass
            
            # Skip emitting a noisy "Processing..." message; go straight to the next response

            # A2A context already set via shared context helpers
            
            # Process the interaction using your existing assessment system
            assessment_system = self._get_assessment_system()
            response_text = await assessment_system.process_candidate_interaction(
                session_info["session_id"],
                session_info["user_id"],
                user_input or "",
                image_path
            )

            # Determine if this requires user input or is complete
            requires_input = self._requires_user_input(response_text)
            
            if requires_input:
                # Remove status indicators from response
                clean_response = self._clean_response_text(response_text)
                
                # Create response parts (text + images if needed)
                response_parts = await self._create_response_parts(clean_response, context_id)
                
                # Update task status as input required
                await updater.update_status(
                    TaskState.input_required,
                    new_agent_parts_message(
                        response_parts,
                        task.context_id,
                        task.id
                    ),
                    final=True,
                )
            else:
                # Assessment is complete
                clean_response = self._clean_response_text(response_text)
                
                # Add final response as artifact
                await updater.add_artifact(
                    [Part(root=TextPart(text=clean_response))],
                    name='assessment_result',
                )
                
                await updater.complete()

        except Exception as e:
            logger.error(f"Error in job assessment execution: {e}")
            
            # Handle error case
            if 'task' in locals() and 'updater' in locals():
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        f"Assessment error: {str(e)}", 
                        task.context_id, 
                        task.id
                    ),
                    final=True,
                )
            else:
                raise ServerError(f"Job assessment failed: {str(e)}")

    async def cancel(
        self, 
        context: RequestContext, 
        event_queue: EventQueue
    ) -> Optional[Task]:
        """Cancel is not supported for job assessments."""
        raise ServerError(error=UnsupportedOperationError())

    def _extract_image_path(self, message) -> Optional[str]:
        """Extract image path from A2A message parts."""
        if not hasattr(message, 'parts') or not message.parts:
            return None
            
        for part in message.parts:
            if hasattr(part, 'root') and hasattr(part.root, '__class__'):
                part_type = part.root.__class__.__name__
                
                if part_type == 'FilePart':
                    # Handle FilePart - check file.uri structure
                    if hasattr(part.root, 'file') and hasattr(part.root.file, 'uri'):
                        uri = part.root.file.uri
                        if uri.startswith('file://'):
                            return uri[7:]  # Remove file:// prefix
                        elif uri.startswith('http'):
                            # Could download and cache, for now just return URI
                            return uri
                    elif hasattr(part.root, 'uri'):
                        uri = part.root.uri
                        if uri.startswith('file://'):
                            return uri[7:]  # Remove file:// prefix
                        elif uri.startswith('http'):
                            return uri
                    elif hasattr(part.root, 'path'):
                        return part.root.path
                        
        return None

    async def _get_or_create_session(self, context_id: str) -> Dict[str, str]:
        """Get or create assessment session for A2A context."""
        logger.info(f"DEBUG: _get_or_create_session called with context_id: {context_id}")
        logger.info(f"DEBUG: Current a2a_sessions keys: {list(self.a2a_sessions.keys())}")
        logger.info(f"DEBUG: Context exists in sessions: {context_id in self.a2a_sessions}")
        
        if context_id not in self.a2a_sessions:
            # Create new candidate session
            candidate_name = f"A2A_User_{str(uuid.uuid4())[:8]}"
            assessment_system = self._get_assessment_system()
            session_id = await assessment_system.create_candidate_session(candidate_name)
            user_id = session_id.split("_")[1]
            
            self.a2a_sessions[context_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "candidate_name": candidate_name,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"DEBUG: Created new assessment session {session_id} for A2A context {context_id}")
            logger.info(f"DEBUG: Updated a2a_sessions keys: {list(self.a2a_sessions.keys())}")
        else:
            logger.info(f"DEBUG: Using existing session for context {context_id}: {self.a2a_sessions[context_id]}")
        
        return self.a2a_sessions[context_id]

    def _requires_user_input(self, response_text: str) -> bool:
        """Determine if response requires user input."""
        return "[STATUS:input_required]" in response_text

    def _clean_response_text(self, response_text: str) -> str:
        """Remove status indicators from response text."""
        return (response_text
                .replace("[STATUS:completed]", "")
                .replace("[STATUS:input_required]", "")
                .strip())

    async def _create_response_parts(self, response_text: str, context_id: str) -> list[Part]:
        """Create A2A response parts with text and images."""
        parts = []
        
        try:
            # Check if response contains questions with images
            if "Looking at [Image:" in response_text or "Image: " in response_text:
                import re
                image_paths = []
                # Prefer canonical prompt lines to avoid capturing [Image:] inside JSON blobs
                for line in response_text.splitlines():
                    line = line.strip()
                    if line.startswith("Looking at [Image:"):
                        m = re.search(r'\[Image:\s*([^\]]+)\]', line)
                        if m:
                            image_paths.append(m.group(1))
                # Fallback to older format: Image: path
                if not image_paths:
                    for line in response_text.splitlines():
                        line = line.strip()
                        if line.startswith("Image:"):
                            m = re.search(r'Image:\s*([^\n\s]+)', line)
                            if m:
                                image_paths.append(m.group(1))
                for image_path in image_paths:
                    try:
                        if not isinstance(image_path, str) or not image_path.strip():
                            continue
                        # Normalize
                        image_path = image_path.strip()
                        # Determine mime type by extension
                        mime_type = "image/jpeg"
                        lp = image_path.lower()
                        if lp.endswith('.png'):
                            mime_type = "image/png"
                        elif lp.endswith('.webp'):
                            mime_type = "image/webp"
                        elif lp.endswith('.gif'):
                            mime_type = "image/gif"
                        elif lp.endswith('.bmp'):
                            mime_type = "image/bmp"
                        elif lp.endswith('.tiff') or lp.endswith('.tif'):
                            mime_type = "image/tiff"

                        # Compute base URL once
                        base_url = os.getenv('BASE_URL')
                        if not base_url:
                            host = os.getenv('HOST', 'localhost')
                            port = os.getenv('PORT', '5000')
                            base_url = f"http://{host}:{port}"

                        # 1) Absolute URLs: use as-is, but repair placeholder host:port if present
                        if lp.startswith('http://') or lp.startswith('https://'):
                            # Replace placeholder 'http://host:port' with real base_url if present
                            fixed_uri = image_path.replace('http://host:port', base_url)
                            parts.append(Part(root=FilePart(file=FileWithUri(uri=fixed_uri, mime_type=mime_type))))

                        # 2) Label dataset relative → host via /label-media
                        elif lp.startswith('label_dataset/'):
                            image_uri = f"{base_url}/{image_path.replace('label_dataset/', 'label-media/')}"
                            parts.append(Part(root=FilePart(file=FileWithUri(uri=image_uri, mime_type=mime_type))))

                        # 3) Electrician relative → host via /ElectricianAssessment
                        elif lp.startswith('electricianassessment/') or image_path.lstrip('/').lower().startswith('electricianassessment/'):
                            image_uri = f"{base_url}/{image_path.lstrip('/')}"
                            parts.append(Part(root=FilePart(file=FileWithUri(uri=image_uri, mime_type=mime_type))))

                        # 4) Absolute path containing ElectricianAssessment → extract relative part
                        elif '/electricianassessment/' in lp:
                            # Extract the filename from absolute path
                            filename = os.path.basename(image_path)
                            image_uri = f"{base_url}/ElectricianAssessment/{filename}"
                            parts.append(Part(root=FilePart(file=FileWithUri(uri=image_uri, mime_type=mime_type))))

                        # 5) Fallback: treat as absolute/remote
                        else:
                            parts.append(Part(root=FilePart(file=FileWithUri(uri=image_path, mime_type=mime_type))))
                    except Exception as e:
                        logger.error(f"Error creating image part for {image_path}: {e}")
        except Exception as e:
            logger.error(f"Error creating image parts: {e}")
        
        # Always add text part
        parts.append(Part(root=TextPart(text=response_text)))
        
        return parts
