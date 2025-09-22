import logging
from typing import Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_agent_parts_message,
    new_task,
)
from a2a.utils.errors import ServerError

from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


logger = logging.getLogger(__name__)


class InterviewPrepAgentExecutor(AgentExecutor):
    """A2A SDK-compliant executor for the interview prep agent."""

    def __init__(self):
        """Initialize the interview prep executor and ADK runner."""
        # Import the ADK agent locally to avoid side effects at import time
        try:
            from agent import root_agent
        except Exception as e:
            logger.error(f"Failed to import root_agent: {e}")
            raise

        self._agent = root_agent
        self._user_id = 'interview_prep_user'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        logger.info("InterviewPrepAgentExecutor initialized")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a single turn of the interview prep agent."""
        try:
            # Extract input/context
            user_input = context.get_user_input() or ""
            context_id = context.context_id

            # Get or create task
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            updater = TaskUpdater(event_queue, task.id, task.context_id)

            # Processing indicator
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Processing your request...",
                    task.context_id,
                    task.id,
                ),
            )

            # Prepare or reuse session (keyed by A2A context_id)
            session = await self._runner.session_service.get_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=context_id,
            )
            if session is None:
                session = await self._runner.session_service.create_session(
                    app_name=self._agent.name,
                    user_id=self._user_id,
                    state={},
                    session_id=context_id,
                )

            # Build content and run the agent turn
            content = genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=user_input)],
            )

            final_text = ""
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session.id,
                new_message=content,
            ):
                if event.is_final_response():
                    # Aggregate text parts
                    if event.content and event.content.parts:
                        texts = [p.text for p in event.content.parts if getattr(p, 'text', None)]
                        final_text = "\n".join(t for t in texts if t)
                    final_text = final_text.strip()

            # Build response (text-only)
            response_parts = [Part(root=TextPart(text=final_text or ""))]

            # For a conversational agent, keep the task open (input_required)
            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message(
                    response_parts,
                    task.context_id,
                    task.id,
                ),
                final=True,
            )

        except Exception as e:
            logger.error(f"Error in interview prep execution: {e}")
            if 'task' in locals() and 'updater' in locals():
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        f"Interview prep error: {str(e)}",
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
            else:
                raise ServerError(f"Interview prep failed: {str(e)}")

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> Optional[Task]:
        """Cancel is not supported."""
        raise ServerError(error=UnsupportedOperationError())


