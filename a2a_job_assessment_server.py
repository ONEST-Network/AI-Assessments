"""
A2A SDK-based Job Assessment Server

This server uses the official A2A SDK to provide standard A2A protocol compliance
for the job assessment system, replacing the manual Flask-based implementation.
"""

import os
import logging
import uvicorn
from typing import Optional
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TransportProtocol,
)

from job_assessment_executor import JobAssessmentAgentExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_agent_card(host: str = 'localhost', port: int = 5000) -> AgentCard:
    """Create the AgentCard defining job assessment capabilities."""
    
    # Define skills for each assessment type
    skills = [
        AgentSkill(
            id='stitching_assessment',
            name='Stitching Quality Assessment',
            description='Evaluates stitching quality and techniques from uploaded images for tailor positions. Analyzes stitch uniformity, thread tension, seam alignment, and overall craftsmanship.',
            tags=['tailoring', 'stitching', 'craftsmanship', 'quality-evaluation', 'sewing'],
            examples=[
                'Assess my stitching work for tailor role',
                'Evaluate this seam quality',
                'Check my embroidery technique',
                'Rate my fabric joining skills'
            ],
        ),
        AgentSkill(
            id='label_reading_assessment', 
            name='Label Reading Assessment',
            description='Tests ability to accurately read and extract information from product labels through interactive quizzes. Essential for warehouse, logistics positions.',
            tags=['warehouse', 'logistics', 'label-reading', 'information-extraction', 'loader', 'picker'],
            examples=[
                'Start label reading assessment',
                'Test my label reading skills',
                'Evaluate my product information extraction',
                'Check my warehouse skills'
            ],
        ),
        AgentSkill(
            id='electrician_assessment',
            name='Electrician Skills Assessment',
            description='Evaluates entry-level electrician skills including safety, scenario-based knowledge, and basic technical questions.',
            tags=['electrician', 'electrical', 'safety', 'assessment', 'technician'],
            examples=[
                'Assess my electrician skills',
                'Start the electrician assessment',
                'Test my knowledge as an electrician'
            ],
        )
    ]
    
    base_url_env = os.getenv('BASE_URL')
    agent_url = (base_url_env.rstrip('/') + '/') if base_url_env else f'http://{host}:{port}/'

    return AgentCard(
        name='Job Assessment System',
        description='Multi-agent job assessment system that evaluates candidates for blue-collar roles including Tailor, Loader Picker, and Retail Sales positions through specialized skill assessments.',
        url=agent_url,
        version='1.0.0',
        default_input_modes=['text/plain', 'image/jpeg', 'image/png'],
        default_output_modes=['text/plain', 'image/jpeg'],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        skills=skills,
        preferred_transport=TransportProtocol.http_json,
        # Security settings
        security=[{'apiKey': []}] if os.getenv('A2A_API_KEY') else None,
    )


class JobAssessmentA2AServer:
    """A2A SDK-based job assessment server."""
    
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.agent_card = create_agent_card(host, port)
        
        # Create the agent executor
        self.agent_executor = JobAssessmentAgentExecutor()
        
        # Create request handler
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.agent_executor,
            task_store=InMemoryTaskStore(),
        )
        
        # Create the A2A application factory
        self.app = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=self.request_handler,
        )
        
        logger.info(f"Job Assessment A2A Server initialized on {host}:{port}")

    def _build_combined_app(self) -> Starlette:
        """Build Starlette app that serves static files and A2A routes."""
        a2a_app = self.app.build()

        routes = []

        # Mount static directories with absolute paths (more robust to cwd changes)
        label_dir = os.path.abspath('label_dataset')
        if os.path.isdir(label_dir):
            routes.append(Mount('/label-media', StaticFiles(directory=label_dir), name='label-media'))
            try:
                logger.info(f"Static: /label-media -> {label_dir} ({len(os.listdir(label_dir))} files)")
            except Exception:
                logger.info(f"Static: /label-media -> {label_dir}")

        elec_dir = os.path.abspath('ElectricianAssessment')
        if os.path.isdir(elec_dir):
            routes.append(Mount('/ElectricianAssessment', StaticFiles(directory=elec_dir), name='electrical-media'))
            try:
                logger.info(f"Static: /ElectricianAssessment -> {elec_dir} ({len(os.listdir(elec_dir))} files)")
            except Exception:
                logger.info(f"Static: /ElectricianAssessment -> {elec_dir}")

        pres_dir = os.path.abspath('presentation_resources')
        if os.path.isdir(pres_dir):
            routes.append(Mount('/presentation-media', StaticFiles(directory=pres_dir), name='presentation-media'))
            try:
                logger.info(f"Static: /presentation-media -> {pres_dir} ({len(os.listdir(pres_dir))} files)")
            except Exception:
                logger.info(f"Static: /presentation-media -> {pres_dir}")

        # Mount A2A app at root
        routes.append(Mount('/', app=a2a_app))

        return Starlette(routes=routes)

    def run(self, debug: bool = False):
        """Run the A2A server."""
        logger.info(f"🚀 Starting Job Assessment A2A Server...")
        logger.info(f"📋 Server URL: http://{self.host}:{self.port}")
        logger.info(f"🔍 Agent Card: http://{self.host}:{self.port}/.well-known/agent-card.json")
        logger.info(f"⚡ Skills Available: {len(self.agent_card.skills)}")
        
        # Print available skills
        for skill in self.agent_card.skills:
            logger.info(f"  ✓ {skill.name}: {skill.description}")
        
        # Run the server
        uvicorn.run(
            self._build_combined_app(),
            host=self.host,
            port=self.port,
            log_level='info' if debug else 'warning',
        )


def main():
    """Main entry point for the A2A job assessment server."""
    import sys
    
    # Configuration from environment or defaults
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    # debug = '--debug' in sys.argv
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY') 
    vertex_ai = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
    
    if not (google_api_key or gemini_api_key or vertex_ai):
        logger.error("No Google AI API key found. Set GOOGLE_API_KEY, GEMINI_API_KEY, or enable Vertex AI")
        sys.exit(1)
    
    try:
        # Create and run server
        server = JobAssessmentA2AServer(host=host, port=port)
        server.run(debug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()