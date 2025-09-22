import os
import logging
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TransportProtocol,
)

from interview_prep_executor import InterviewPrepAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_agent_card(host: str = 'localhost', port: int = 5002) -> AgentCard:
    """Create the AgentCard for the interview prep agent."""
    skills = [
        AgentSkill(
            id='interview_prep_tutor',
            name='Interview Preparation Tutor',
            description=(
                'Conversational tutor for blue/grey-collar candidates. '
                'Teaches modules in small chunks, asks MCQs, and tracks progress.'
            ),
            tags=['interview', 'training', 'education', 'blue-collar', 'grey-collar'],
            examples=[
                'Help me prepare for electrician interview',
                'List electrician modules',
                'Teach me wiring fundamentals',
                'Show my learning progress',
            ],
        ),
    ]

    base_url_env = os.getenv('BASE_URL')
    agent_url = (base_url_env.rstrip('/') + '/') if base_url_env else f'http://{host}:{port}/'

    return AgentCard(
        name='Interview Prep Agent',
        description='An ADK-powered interview preparation agent with progress tracking.',
        url=agent_url,
        version='1.0.0',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        skills=skills,
        preferred_transport=TransportProtocol.http_json,
        security=[{'apiKey': []}] if os.getenv('A2A_API_KEY') else None,
    )


class InterviewPrepA2AServer:
    """A2A SDK-based server for the interview prep agent."""

    def __init__(self, host: str = 'localhost', port: int = 5002):
        self.host = host
        self.port = port
        self.agent_card = create_agent_card(host, port)

        # Create executor and handler
        self.agent_executor = InterviewPrepAgentExecutor()
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.agent_executor,
            task_store=InMemoryTaskStore(),
        )

        # Build the A2A app
        self.app = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=self.request_handler,
        )

        logger.info(f"Interview Prep A2A Server initialized on {host}:{port}")

    def run(self, debug: bool = True):
        logger.info("🚀 Starting Interview Prep A2A Server...")
        logger.info(f"📋 Server URL: http://{self.host}:{self.port}")
        logger.info(f"🔍 Agent Card: http://{self.host}:{self.port}/.well-known/agent-card.json")
        logger.info(f"⚡ Skills Available: {len(self.agent_card.skills)}")
        for skill in self.agent_card.skills:
            logger.info(f"  ✓ {skill.name}: {skill.description}")

        uvicorn.run(
            self.app.build(),
            host=self.host,
            port=self.port,
            log_level='info' if debug else 'warning',
        )


def main():
    """Main entry point for the interview prep A2A server."""
    import sys

    host = os.getenv('HOST', 'localhost')
    port = int(os.getenv('PORT', '5002'))

    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        logger.error("❌ No Google AI API key found. Set GOOGLE_API_KEY")
        sys.exit(1)

    try:
        server = InterviewPrepA2AServer(host=host, port=port)
        server.run()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


