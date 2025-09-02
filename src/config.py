"""Configuration management for the job assessment system."""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class."""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    A2A_API_KEY = os.getenv('A2A_API_KEY')
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Application Settings
    APP_NAME = "job_assessment_app"
    
    # File Paths
    ASSETS_DIR = "assets"
    LABEL_DATASET_DIR = "label_dataset"
    COMPETENCY_MAP_FILE = "competency_map.json"
    SUB_AGENT_LIBRARY_FILE = "sub_agent_library.json"
    
    # Logging Configuration
    LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present."""
        if not (cls.GOOGLE_API_KEY or cls.GEMINI_API_KEY):
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        
        if cls.GOOGLE_API_KEY and cls.GEMINI_API_KEY:
            # Use Google API key and remove Gemini key to avoid conflicts
            os.environ.pop('GEMINI_API_KEY', None)
            return cls.GOOGLE_API_KEY
        elif cls.GEMINI_API_KEY and not cls.GOOGLE_API_KEY:
            return cls.GEMINI_API_KEY
        
        return cls.GOOGLE_API_KEY or cls.GEMINI_API_KEY

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT
    )
    
    # Suppress verbose logging from external libraries
    logging.getLogger("google.adk").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
