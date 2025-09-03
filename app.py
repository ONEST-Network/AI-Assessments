import asyncio
import json
import logging
import re
import sys
from typing import Optional

from src import Config, setup_logging, StatefulJobAssessmentSystem, A2AServer

logger = setup_logging()


async def run_interactive_mode():
    """Run the assessment system in interactive mode."""
    print("Enhanced Stateful Multi-Agent Job Assessment System")
    print("Built with Google ADK\n")
    
    # Initialize system
    assessment_system = StatefulJobAssessmentSystem()
    
    # Get candidate name only
    candidate_name = input("Enter candidate name: ").strip()
    
    if not candidate_name:
        print("Candidate name is required")
        return
    
    try:
        # Create session without role (will be identified through conversation)
        session_id = await assessment_system.create_candidate_session(candidate_name)
        user_id = session_id.split("_")[1]  # Extract candidate ID
        
        print(f"Session created for {candidate_name}")
        print(f"Session ID: {session_id}")
        print("\nThe system will help identify your target role through natural conversation.")
        print("Type 'exit' to end the session, 'summary' for assessment summary.\n")

        # Start with a simple greeting
        print(f"\nSystem: Hello {candidate_name}! Welcome to our job assessment system. "
              f"I'm here to help evaluate your skills for various positions. "
              f"What kind of work are you interested in today?")
        
        while True:
            user_input = input(f"\n{candidate_name}: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'summary':
                summary = await assessment_system.get_assessment_summary(session_id, user_id)
                print("\nASSESSMENT SUMMARY:")
                print(json.dumps(summary, indent=2, default=str))
                continue
            
            # Extract image path from user input
            image_path = _extract_image_path(user_input)
            if image_path:
                user_input = _clean_image_path_from_input(user_input, image_path)
            
            print("\nProcessing...")
            response = await assessment_system.process_candidate_interaction(
                session_id, user_id, user_input, image_path
            )
            
            print(f"\nSystem: {response}")
        
        print("\nFINAL ASSESSMENT SUMMARY:")
        summary = await assessment_system.get_assessment_summary(session_id, user_id)
        print(json.dumps(summary, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")


def _extract_image_path(user_input: str) -> Optional[str]:
    """Extract image path from user input."""
    # Look for paths in brackets first
    if '[' in user_input and ']' in user_input:
        start = user_input.find('[')
        end = user_input.find(']', start)
        if start != -1 and end != -1:
            return user_input[start+1:end].strip()
    
    # Look for file paths
    path_patterns = [
        r'(/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Unix paths
        r'([A-Za-z]:[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Windows paths
    ]
    
    for pattern in path_patterns:
        matches = re.findall(pattern, user_input, re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    return None


def _clean_image_path_from_input(user_input: str, image_path: str) -> str:
    """Remove image path from user input."""
    # Remove bracketed path
    user_input = re.sub(r'\[[^\]]*\]', '', user_input)
    # Remove the actual path
    user_input = user_input.replace(image_path, '')
    return user_input.strip()


def run_a2a_server():
    """Run the A2A server mode."""
    print("🎯 Initializing Job Assessment System with A2A Support")
    
    # Validate API keys
    try:
        Config.validate_api_keys()
    except ValueError as e:
        logger.error(f"API key validation failed: {e}")
        print(f"Error: {e}")
        return
    
    assessment_system = StatefulJobAssessmentSystem()
    
    print("🚀 Starting A2A Server...")
    a2a_server = A2AServer(assessment_system)
    a2a_server.run()


async def main():
    """Main application entry point."""
    # Validate API keys first
    try:
        Config.validate_api_keys()
    except ValueError as e:
        logger.error(f"API key validation failed: {e}")
        print(f"Error: {e}")
        return
    
    # Check if running as A2A server
    if len(sys.argv) > 1 and sys.argv[1] == "--a2a":
        print("Starting A2A Server Mode")
        run_a2a_server()
        return
    
    # Run interactive mode
    await run_interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
