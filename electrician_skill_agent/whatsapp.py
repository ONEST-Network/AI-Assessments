import os
import uuid
import logging
from typing import Dict, Optional

from fastapi import FastAPI, BackgroundTasks, Form, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from twilio.rest import Client

import sys

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import token tracking
from simple_token_tracker import track_event_usage, get_totals, reset_totals

# Import agent and internals to optionally patch (relative imports)
from electrician_skill_agent import (
    electrician_skill_assessor,
    _get_session_key,
    _QUIZ_SESSIONS,
    score_photo_answer,
    start_knowledge_assessment,
    score_knowledge_answer,
)
from electrician_skill_agent import _analyze_image_safety  # type: ignore
import electrician_skill_agent as agent_mod
# Ensure project root is on sys.path to import sibling packages when running from this folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fitter_skill_agent.fitter_skill_agent import fitter_skill_assessor

# Import interview prep agent
import os

# Import interview prep database tools for user persistence  
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interview-prep-agent'))
    from src.tools.db_tools import get_user_role, load_user_progress, save_user_role
    INTERVIEW_PREP_DB_AVAILABLE = True
except ImportError as e:
    INTERVIEW_PREP_DB_AVAILABLE = False
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "interview-prep-agent"))
try:
    from agent import root_agent as interview_prep_agent
    from google.adk.runners import Runner as InterviewPrepRunner
    INTERVIEW_PREP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Interview prep agent not available: {e}")
    INTERVIEW_PREP_AVAILABLE = False


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Retry utility functions ------------------------------------------------

import asyncio
import time
from typing import Callable, Any, Optional

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    phone_number: Optional[str] = None
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by each retry
        phone_number: User phone number for logging and notifications
    
    Returns:
        Result of the successful function call
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} for {phone_number or 'unknown'} after {delay}s delay")
                # Notify user about retry for transparency
                if phone_number and attempt == 1:  # Only notify on first retry
                    try:
                        twilio_client.send_whatsapp(phone_number, "🔄 Retrying your request...")
                    except Exception:
                        pass  # Don't let notification failure break retry logic
                await asyncio.sleep(delay)
            
            result = await func()
            
            if attempt > 0:
                logger.info(f"Retry successful on attempt {attempt} for {phone_number or 'unknown'}")
                
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Check if this is a retryable error
            is_retryable = (
                "500" in error_msg or "internal" in error_msg or
                "503" in error_msg or "overloaded" in error_msg or "unavailable" in error_msg or
                "504" in error_msg or "deadline" in error_msg or "timeout" in error_msg or
                "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or
                "429" in error_msg
            )
            
            if not is_retryable or attempt >= max_retries:
                logger.error(f"{'Non-retryable error' if not is_retryable else 'Max retries exceeded'} for {phone_number or 'unknown'}: {e}")
                raise e
            
            logger.warning(f"Retryable error on attempt {attempt + 1} for {phone_number or 'unknown'}: {e}")
            
            # Calculate next delay with exponential backoff
            delay = min(delay * backoff_factor, max_delay)
    
    # This should never be reached, but just in case
    raise last_exception


def should_retry_error(error: Exception) -> bool:
    """
    Determine if an error should be retried based on Gemini API documentation.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error should be retried, False otherwise
    """
    error_msg = str(error).lower()
    
    # Retryable errors according to Gemini API docs
    retryable_patterns = [
        "500",           # INTERNAL - unexpected error on Google's side
        "internal",      # INTERNAL error variants
        "503",           # UNAVAILABLE - service temporarily out of capacity  
        "overloaded",    # UNAVAILABLE variants
        "unavailable",   # UNAVAILABLE variants
        "504",           # DEADLINE_EXCEEDED - service unable to finish within deadline
        "deadline",      # DEADLINE_EXCEEDED variants
        "timeout",       # Timeout variants
        "rate",          # Rate limit errors
        "quota",         # Quota exceeded
        "limit",         # Limit exceeded
        "429"            # Too Many Requests
    ]
    
    return any(pattern in error_msg for pattern in retryable_patterns)


app = FastAPI(title="Electrician Agent WhatsApp Webhook", version="1.0.0")


# --- Twilio helper -----------------------------------------------------------

class TwilioClient:
    def __init__(self) -> None:
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        # Twilio Sandbox default if not provided
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

        if not self.account_sid or not self.auth_token:
            raise RuntimeError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN in environment")

        self.client = Client(self.account_sid, self.auth_token)

    def send_whatsapp(self, to_number: str, body: str) -> None:
        to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        from_ = self.from_number if self.from_number.startswith("whatsapp:") else f"whatsapp:{self.from_number}"
        logger.info(f"Twilio send: from={from_} to={to} body_length={len(body)}")
        self.client.messages.create(from_=from_, to=to, body=body)

    def send_media(self, to_number: str, media_url: str, caption: str = "") -> None:
        to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        from_ = self.from_number if self.from_number.startswith("whatsapp:") else f"whatsapp:{self.from_number}"
        logger.info(f"Twilio media: from={from_} to={to} url={media_url}")
        self.client.messages.create(from_=from_, to=to, body=caption or None, media_url=[media_url])


twilio_client = TwilioClient()


# --- Session management ------------------------------------------------------

class SessionEntry:
    def __init__(self, app_name: str, agent_type: str = "skill_assessment", selected_skill: str = "Electrician") -> None:
        self.agent_type = agent_type
        self.selected_skill = selected_skill  # e.g., "Electrician" or "Fitter"
        self.session_service = InMemorySessionService()
        
        if agent_type == "interview_prep" and INTERVIEW_PREP_AVAILABLE:
            self.runner = InterviewPrepRunner(
                agent=interview_prep_agent,
                app_name=app_name,
                session_service=self.session_service,
            )
        else:
            # Skill assessment: choose agent by selected skill
            chosen_agent = electrician_skill_assessor
            try:
                if str(selected_skill).strip().lower() == "fitter":
                    chosen_agent = fitter_skill_assessor
            except Exception:
                chosen_agent = electrician_skill_assessor
            self.runner = Runner(
                agent=chosen_agent,
                app_name=app_name,
                session_service=self.session_service,
            )
        self.session_id: Optional[str] = None
        self.agent_selected: bool = False

    async def ensure(self, user_id: str) -> str:
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            # For interview prep agent, ensure phone number is passed in session state
            session_state = {}
            if self.agent_type == "interview_prep":
                # Store phone number in session state for database tools
                session_state['phone_number'] = user_id
                # Check if this is a new user session to prevent "Welcome back" message
                if hasattr(self, 'new_user_session') and self.new_user_session:
                    session_state['new_user_session'] = True
                    logger.info(f"Creating NEW USER interview prep session with phone number: {user_id}")
                else:
                    logger.info(f"Creating interview prep session with phone number: {user_id}")
            
            await self.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=self.session_id,
                state=session_state,
            )
        return self.session_id


# One session per WhatsApp phone number
SESSIONS: Dict[str, SessionEntry] = {}

# Static hosting for ElectricianAssessment images to allow Twilio to fetch
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ASSESS_FOLDER = os.path.join(PROJECT_ROOT, "ElectricianAssessment")
# Use a mount that matches the rest of the codebase so URLs are consistent
# elsewhere we serve via /ElectricianAssessment
STATIC_MOUNT = "/ElectricianAssessment"
BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g., https://your-domain.com
# Captured dynamically from incoming requests when env is missing
_DYNAMIC_BASE_URL: Optional[str] = None

if os.path.isdir(ASSESS_FOLDER):
    app.mount(STATIC_MOUNT, StaticFiles(directory=ASSESS_FOLDER), name="electrician_static")
else:
    logger.warning(f"ElectricianAssessment folder not found at {ASSESS_FOLDER}")


def local_image_to_public_url(local_path: str) -> Optional[str]:
    """Map various path formats to a public URL Twilio can fetch.

    Accepts:
    - Fully-qualified http(s) URLs: returned as-is
    - Absolute paths inside ElectricianAssessment: mapped to BASE_URL/STATIC_MOUNT/<filename or rel>
    - Relative paths like "ElectricianAssessment/img1.png" or just "img1.png"
    """
    if not local_path:
        return None
    # If already a URL, return directly
    if local_path.startswith("http://") or local_path.startswith("https://"):
        return local_path
    base = (BASE_URL or _DYNAMIC_BASE_URL)
    if not base:
        logger.warning("PUBLIC_BASE_URL not set and no request-derived base; cannot send media over WhatsApp.")
        return None
    try:
        # Normalize incoming
        norm = local_path.replace("\\", "/").lstrip()
        # If path contains ElectricianAssessment anywhere, try to extract the filename or relative portion
        lower_norm = norm.lower()
        if "electricianassessment/" in lower_norm:
            # Keep only the part after ElectricianAssessment/
            idx = lower_norm.rfind("electricianassessment/")
            rel_after = norm[idx + len("ElectricianAssessment/") :]
            rel_after = rel_after.lstrip("/")
            return f"{base.rstrip('/')}{STATIC_MOUNT}/{rel_after}"

        # If absolute path, attempt relativization against ASSESS_FOLDER
        abs_path = os.path.abspath(norm) if os.path.isabs(norm) else os.path.abspath(os.path.join(ASSESS_FOLDER, norm))
        try:
            rel = os.path.relpath(abs_path, ASSESS_FOLDER).replace("\\", "/")
        except Exception:
            # As a fallback, use just the filename to avoid leaking server paths
            rel = os.path.basename(abs_path)
        return f"{base.rstrip('/')}{STATIC_MOUNT}/{rel.lstrip('/')}"
    except Exception:
        logger.exception("Failed to map local path to public URL")
        return None


def get_session(phone_number: str, agent_type: str = "skill_assessment") -> SessionEntry:
    if phone_number not in SESSIONS:
        SESSIONS[phone_number] = SessionEntry(app_name=f"{agent_type}_whatsapp", agent_type=agent_type)
    return SESSIONS[phone_number]


def send_selection_menu(phone_number: str, returning_user: bool = False) -> bool:
    """Send text-based menu to choose between Interview Preparation and Skill Assessment"""
    try:
        from_number = twilio_client.from_number
        if not str(from_number).startswith('whatsapp:'):
            from_number = f"whatsapp:{from_number}"
        if not str(phone_number).startswith('whatsapp:'):
            phone_number = f"whatsapp:{phone_number}"
        
        logger.info(f"Sending selection menu to {phone_number}")
        
        # Use simple text-based menu (templates removed for reliability)
        logger.info("Sending text-based selection menu")
        
        # Customize message for returning users
        if returning_user:
            menu_text = """🔄 Welcome back! Choose your service:

1️⃣ Interview Preparation - Continue learning or practice interviews
2️⃣ Skill Assessment - Get evaluated on your skills

Reply with '1' or '2' to get started.

💡 You can also type 'menu' anytime to see these options again."""
        else:
            menu_text = """🤖 Welcome to Skill Assessment System!

Please choose your service:

1️⃣ Interview Preparation - Learn skills and practice interviews
2️⃣ Skill Assessment - Get evaluated on your skills

Reply with '1' or '2' to get started.

💡 You can type 'menu' anytime to switch between services."""
        
        logger.info(f"Sending text menu to {phone_number}")
        message = twilio_client.client.messages.create(
            from_=from_number,
            to=phone_number,
            body=menu_text
        )
        
        logger.info(f"Text menu sent successfully: {message.sid}, Status: {getattr(message, 'status', 'unknown')}")
        return True
            
    except Exception as e:
        logger.error(f"Failed to send selection menu: {e}")
        return False


# --- Background processing ---------------------------------------------------

async def process_agent_and_reply(phone_number: str, incoming_text: str) -> None:
    try:
        logger.info(f"Incoming from {phone_number}: {incoming_text!r}")
        logger.info(f"Current sessions: {list(SESSIONS.keys())}")
        
        # Check if this is a new user or if they haven't selected an agent yet
        if phone_number not in SESSIONS:
            # All users (new or returning) go directly to Interview Prep with role selection
            logger.info(f"User detected: {phone_number}. Starting Interview Prep flow...")
            SESSIONS[phone_number] = SessionEntry(app_name="interview_prep_whatsapp", agent_type="interview_prep")
            SESSIONS[phone_number].agent_selected = True
            # Initialize session state
            SESSIONS[phone_number]._state = {}
            
            # Send role selection message directly - same for everyone
            welcome_msg = "🎓 Welcome! I help people get ready for jobs.\n\nWhat job are you preparing for?\n1️⃣ Electrician\n2️⃣ Fitter\n\n💡 After selecting your job, you can:\n• Practice for interviews\n• Take skills assessment\n\nReply with the number"
            twilio_client.send_whatsapp(phone_number, welcome_msg)
            return
        
        session = SESSIONS[phone_number]
        logger.info(f"Found existing session for {phone_number}: agent_type={session.agent_type}, agent_selected={session.agent_selected}")
        
        # Handle role selection for Interview Prep users
        if session.agent_type == "interview_prep" and session.agent_selected:
            incoming_lower = incoming_text.lower().strip()
            
            # Check if they want to switch to skill assessment
            if incoming_lower in ['assessment', 'skill assessment', 'test', 'evaluation', 'skills']:
                logger.info(f"User wants to switch to skill assessment: {incoming_text}")
                SESSIONS[phone_number] = SessionEntry(app_name="skill_assessment_whatsapp", agent_type="skill_assessment")
                SESSIONS[phone_number].agent_selected = True
                twilio_client.send_whatsapp(phone_number, "⚡ Switched to Skill Assessment! Let's start your evaluation. Type 'Hi' to begin!")
                await process_agent_and_reply(phone_number, "Hi")
                return
            
            # Check if we're in role selection phase using session state
            # We'll track this in the session to avoid database dependency
            session_state = session.__dict__.get('_state', {})
            role_selected = session_state.get('role_selected', False)
            activity_selected = session_state.get('activity_selected', False)
            
            logger.info(f"Session state check: role_selected={role_selected}, activity_selected={activity_selected}, incoming_lower={incoming_lower}")
            
            if not role_selected:
                # Role selection phase - user hasn't selected a role yet
                if incoming_lower in ['1', '2']:
                    role_map = {
                        '1': 'Electrician',
                        '2': 'Fitter',
                    }
                    selected_role = role_map[incoming_lower]
                    logger.info(f"User selected role: {selected_role}")
                    
                    # Mark role as selected in session state
                    if not hasattr(session, '_state'):
                        session._state = {}
                    session._state['role_selected'] = True
                    session._state['selected_role'] = selected_role
                    # Mark as new user session since they just selected role in this session
                    session._state['new_user_session'] = True
                    
                    # Save the role to database
                    if INTERVIEW_PREP_DB_AVAILABLE:
                        try:
                            result = save_user_role(phone_number, selected_role)
                            logger.info(f"Saved user role: {selected_role}, result: {result}")
                        except Exception as e:
                            logger.error(f"Error saving user role: {e}")
                            logger.exception("Full error details:")
                    
                    # Ask for activity type
                    activity_msg = f"Great! You're preparing for {selected_role}. What would you like to do?\n\n1️⃣ Practice for interviews\n2️⃣ Take skills assessment\n\nReply with '1' or '2'"
                    twilio_client.send_whatsapp(phone_number, activity_msg)
                    return
                else:
                    # Invalid role selection - ask again
                    logger.info(f"Invalid role selection: {incoming_lower}")
                    retry_msg = "Please select a valid option:\n\n1️⃣ Electrician\n2️⃣ Fitter\n\nReply with '1' or '2' only"
                    twilio_client.send_whatsapp(phone_number, retry_msg)
                    return
            else:
                # Activity selection phase - user has a role, now choosing activity
                if not activity_selected and incoming_lower in ['1', '2']:
                    logger.info(f"User selected activity option: {incoming_lower}")
                    
                    # Mark activity as selected
                    if not hasattr(session, '_state'):
                        session._state = {}
                    session._state['activity_selected'] = True
                    session._state['selected_activity'] = 'practice' if incoming_lower == '1' else 'assessment'
                    
                    if incoming_lower == '1':
                        # Practice - check if returning user with progress
                        logger.info("User chose practice - checking for existing progress")
                        
                        # Check if user has previous progress
                        has_progress = False
                        user_progress = {}
                        if INTERVIEW_PREP_DB_AVAILABLE:
                            try:
                                user_data = load_user_progress(phone_number)
                                user_progress = user_data.get('progress', {})
                                has_progress = bool(user_progress)
                                logger.info(f"Progress check: has_progress={has_progress}, modules={list(user_progress.keys())}")
                            except Exception as e:
                                logger.error(f"Error checking user progress: {e}")
                        
                        if has_progress:
                            # Returning user with progress - let interview prep agent handle welcome back
                            session._state['new_user_session'] = False  # Allow "Welcome back" message
                            twilio_client.send_whatsapp(phone_number, "Perfect! Let's continue your learning journey.")
                        else:
                            # New user - prevent "Welcome back" message
                            session._state['new_user_session'] = True
                            session.new_user_session = True
                            twilio_client.send_whatsapp(phone_number, "Perfect! Let's practice interview questions. I'll start with general questions that work for any job.")
                        
                        # Trigger the agent to start learning
                        await process_agent_and_reply(phone_number, "Hi")
                        return
                    elif incoming_lower == '2':
                        # Assessment - ask general vs specific
                        logger.info("User chose assessment - asking general vs specific")
                        assessment_msg = "Great! What type of assessment would you like?\n\n1️⃣ General interview questions\n2️⃣ Technical questions for your role\n\nReply with '1' or '2'"
                        twilio_client.send_whatsapp(phone_number, assessment_msg)
                        # Mark that we're waiting for assessment type selection
                        if not hasattr(session, '_state'):
                            session._state = {}
                        session._state['waiting_for_assessment_type'] = True
                        return
                elif not activity_selected:
                    # Invalid activity selection - ask again
                    logger.info(f"Invalid activity selection: {incoming_lower}")
                    selected_role = session_state.get('selected_role', 'your chosen role')
                    retry_msg = f"Please select a valid option for {selected_role}:\n\n1️⃣ Practice for interviews\n2️⃣ Take skills assessment\n\nReply with '1' or '2' only"
                    twilio_client.send_whatsapp(phone_number, retry_msg)
                    return
                elif activity_selected:
                    # Activity already selected, this should not happen in normal flow
                    logger.warning(f"Activity already selected but received input: {incoming_lower}")
                    # Let the agent handle it
            
            # Check if we're waiting for assessment type selection
            if session_state.get('waiting_for_assessment_type', False):
                if incoming_lower in ['1', '2']:
                    logger.info(f"User selected assessment type: {incoming_lower}")
                    
                    if incoming_lower == '1':
                        # General interview questions assessment - stay with interview prep agent
                        logger.info("User chose general interview questions assessment")
                        twilio_client.send_whatsapp(phone_number, "Perfect! Let's start with general interview questions assessment. I'll test your knowledge on common interview topics.")
                        # Stay with interview prep agent but trigger assessment mode
                        if not hasattr(session, '_state'):
                            session._state = {}
                        session._state['assessment_mode'] = True
                        session._state['assessment_type'] = 'general'
                        session._state['waiting_for_assessment_type'] = False
                        
                        # Pass assessment context to the agent instead of generic "Hi"
                        assessment_start_message = "I want to start a general interview assessment. Begin testing my knowledge with general interview questions."
                        
                        # Let the existing agent routing handle this with proper context
                        incoming_text = assessment_start_message
                        # Fall through to the agent processing below
                    elif incoming_lower == '2':
                        # Technical questions assessment for their role
                        logger.info("User chose technical questions assessment")
                        selected_role = session_state.get('selected_role', 'Electrician')
                        twilio_client.send_whatsapp(phone_number, f"Excellent! Let's test your technical knowledge for the {selected_role} role.")
                        # Switch to skill assessment agent for technical questions (use selected role)
                        SESSIONS[phone_number] = SessionEntry(app_name="skill_assessment_whatsapp", agent_type="skill_assessment", selected_skill=selected_role)
                        SESSIONS[phone_number].agent_selected = True
                        await process_agent_and_reply(phone_number, "Hi")
                        return
                else:
                    # Invalid assessment type selection - ask again
                    logger.info(f"Invalid assessment type selection: {incoming_lower}")
                    retry_msg = "Please select a valid assessment type:\n\n1️⃣ General interview questions\n2️⃣ Technical questions for your role\n\nReply with '1' or '2' only"
                    twilio_client.send_whatsapp(phone_number, retry_msg)
                    return
                
                # Clear the waiting state only if valid selection was made
                if incoming_lower in ['1', '2']:
                    session_state['waiting_for_assessment_type'] = False
            
            # If we get here, let the agent handle the conversation
            logger.info("Passing conversation to Interview Prep agent")
        
        # Check for agent switching commands (works anytime)
        incoming_lower = incoming_text.lower().strip()
        
        # Menu/switching commands
        if incoming_lower in ['menu', 'switch', 'options', 'home', 'back']:
            logger.info(f"User requested menu/switch: {incoming_text}")
            # Reset session to menu mode
            SESSIONS[phone_number] = SessionEntry(app_name="menu_whatsapp", agent_type="menu")
            SESSIONS[phone_number].agent_selected = False
            
            # Check if they're a returning user (has data in database)
            returning_user = False
            if INTERVIEW_PREP_DB_AVAILABLE:
                try:
                    saved_role = get_user_role(phone_number)
                    user_data = load_user_progress(phone_number)
                    returning_user = bool(saved_role or user_data.get('progress', {}))
                except Exception as e:
                    logger.error(f"Error checking returning user status: {e}")
            
            if send_selection_menu(phone_number, returning_user):
                logger.info("Agent switching menu sent successfully")
                return
            else:
                logger.error("Failed to send switching menu")
                twilio_client.send_whatsapp(phone_number, "🔄 Please choose: '1' for Interview Preparation or '2' for Skill Assessment")
                return
        
        # Direct agent switching commands
        elif incoming_lower in ['interview', 'interview prep', 'interview preparation', 'learning', 'study']:
            logger.info(f"Direct switch to Interview Prep requested: {incoming_text}")
            SESSIONS[phone_number] = SessionEntry(app_name="interview_prep_whatsapp", agent_type="interview_prep")
            SESSIONS[phone_number].agent_selected = True
            twilio_client.send_whatsapp(phone_number, "🎓 Switched to Interview Preparation! How can I help you learn today?")
            return
            
        elif incoming_lower in ['assessment', 'skill assessment', 'test', 'evaluation', 'skills']:
            logger.info(f"Direct switch to Skill Assessment requested: {incoming_text}")
            # Default to Electrician unless user has previously chosen a role
            prev = SESSIONS.get(phone_number)
            selected_role = getattr(prev, '_state', {}).get('selected_role', 'Electrician') if prev else 'Electrician'
            SESSIONS[phone_number] = SessionEntry(app_name="skill_assessment_whatsapp", agent_type="skill_assessment", selected_skill=selected_role)
            SESSIONS[phone_number].agent_selected = True
            # Trigger skill assessment immediately
            await process_agent_and_reply(phone_number, "Hi")
            return
        
        # Handle agent selection if not done yet
        if not session.agent_selected:
            logger.info(f"Agent not selected yet for {phone_number}. Processing selection...")
            agent_type = None
            
            # Check for template response or manual selection
            incoming_lower = incoming_text.lower().strip()
            logger.info(f"Checking incoming text '{incoming_lower}' for agent selection")
            
            if incoming_lower in ['1', 'one', 'interview preparation', 'interview prep', 'interview']:
                agent_type = "interview_prep"
                logger.info("Selected: Interview Preparation")
            elif incoming_lower in ['2', 'two', 'skill assessment', 'skill test', 'assessment']:
                agent_type = "skill_assessment"
                logger.info("Selected: Skill Assessment")
            else:
                logger.info(f"No valid selection found in '{incoming_lower}'")
            
            if agent_type:
                logger.info(f"Creating session for agent_type: {agent_type}")
                # Create new session with selected agent type
                SESSIONS[phone_number] = SessionEntry(app_name=f"{agent_type}_whatsapp", agent_type=agent_type)
                session = SESSIONS[phone_number]
                session.agent_selected = True
                
                # Send welcome message based on agent type
                if agent_type == "interview_prep":
                    if INTERVIEW_PREP_AVAILABLE:
                        logger.info("Sending interview prep welcome message")
                        
                        # Check for existing user data after they've selected interview prep
                        welcome_msg = "🎓 Great! You've selected Interview Preparation."
                        if INTERVIEW_PREP_DB_AVAILABLE:
                            try:
                                saved_role = get_user_role(phone_number)
                                user_data = load_user_progress(phone_number)
                                has_progress = bool(user_data.get('progress', {}))
                                skills_worked_on = list(user_data.get('progress', {}).keys())
                                
                                if saved_role and has_progress:
                                    # Returning user with both role and progress
                                    logger.info(f"Returning user: role={saved_role}, skills={skills_worked_on}")
                                    welcome_msg += f" Welcome back! You're preparing for {saved_role} and have worked on: {', '.join(skills_worked_on)}."
                                    welcome_msg += " What would you like to do?\n\n1️⃣ Practice for interviews\n2️⃣ Take skills assessment\n\nReply with '1' or '2'"
                                elif saved_role and not has_progress:
                                    # Has role but no progress yet
                                    logger.info(f"User with role but no progress: role={saved_role}")
                                    welcome_msg += f" I remember you're preparing for {saved_role}."
                                    welcome_msg += " What would you like to do?\n\n1️⃣ Practice for interviews\n2️⃣ Take skills assessment\n\nReply with '1' or '2'"
                                elif not saved_role and has_progress:
                                    # Has progress but no saved role (unusual case)
                                    logger.info(f"User with progress but no role: skills={skills_worked_on}")
                                    welcome_msg += f" I see you've been learning: {', '.join(skills_worked_on)}."
                                    welcome_msg += " What job are you preparing for?\n\n1️⃣ Electrician\n2️⃣ Fitter\n\nReply with the number"
                                else:
                                    # New user - no role, no progress - ASK ROLE FIRST
                                    logger.info("New interview prep user - asking role first")
                                    welcome_msg += " I help people get ready for jobs."
                                    welcome_msg += " What job are you preparing for?\n\n1️⃣ Electrician\n2️⃣ Fitter\n\n💡 After selecting your job, you can:\n• Practice for interviews\n• Take skills assessment\n\nReply with the number"
                            except Exception as e:
                                logger.error(f"Error checking user data: {e}")
                                welcome_msg += " I help people get ready for jobs. What job are you preparing for?\n\n1️⃣ Electrician\n2️⃣ Fitter\n\n💡 After selecting your job, you can:\n• Practice for interviews\n• Take skills assessment\n\nReply with the number"
                        else:
                            welcome_msg += " I help people get ready for jobs. What job are you preparing for?\n\n1️⃣ Electrician\n2️⃣ Fitter\n\n💡 After selecting your job, you can:\n• Practice for interviews\n• Take skills assessment\n\nReply with the number"
                        
                        twilio_client.send_whatsapp(phone_number, welcome_msg)
                        # User will now explicitly choose their path through the improved menu system
                    else:
                        logger.warning("Interview prep not available, switching to skill assessment")
                        twilio_client.send_whatsapp(phone_number, "❌ Interview Preparation is currently unavailable. Switching to Skill Assessment...")
                        agent_type = "skill_assessment"
                        SESSIONS[phone_number] = SessionEntry(app_name="skill_assessment_whatsapp", agent_type="skill_assessment")
                        session = SESSIONS[phone_number]
                        session.agent_selected = True
                
                if agent_type == "skill_assessment":
                    logger.info("Sending skill assessment welcome message")
                    twilio_client.send_whatsapp(phone_number, "⚡ Great! You've selected Skill Assessment. Are you ready to begin your evaluation?")
                
                logger.info("Agent selection completed, returning")
                return
            else:
                # Invalid selection - ask again
                logger.info("Invalid selection, sending prompt again")
                twilio_client.send_whatsapp(phone_number, "Please select a valid option:\n1️⃣ Interview Preparation\n2️⃣ Skill Assessment\n\nReply with '1' or '2'")
                return
        
        session_id = await session.ensure(user_id=phone_number)

        # Build user message content
        message = types.Content(role="user", parts=[types.Part(text=incoming_text or "Start assessment")])

        # Handle different agent types
        if session.agent_type == "interview_prep":
            # Interview prep agent - enhanced with database and phone number context
            try:
                response_text = ""
                
                # Ensure the interview prep runner has access to the phone number for database operations
                logger.info(f"Processing interview prep request for phone: {phone_number}")
                
                conversation_turn = 1  # Track conversation turns
                async for event in session.runner.run_async(user_id=phone_number, session_id=session_id, new_message=message):
                    # Track token usage for interview prep
                    try:
                        track_event_usage(
                            event=event,
                            model="gemini-2.5-flash",
                            agent_type="interview_prep",
                            user_id=phone_number
                        )
                    except Exception as e:
                        logger.warning(f"Failed to track interview prep token usage: {e}")
                    # Log tool usage for debugging
                    if hasattr(event, 'content') and event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                logger.info(f"Interview prep tool call: {getattr(part.function_call, 'name', 'unknown')}")
                            if hasattr(part, 'function_response') and part.function_response:
                                logger.info(f"Interview prep tool response received")
                    
                    if hasattr(event, 'is_final_response') and event.is_final_response():
                        if event.content and event.content.parts:
                            texts = [p.text for p in event.content.parts if getattr(p, 'text', None)]
                            raw_response = "\n".join(t for t in texts if t).strip()
                            
                            # Clean response to remove asterisks and formatting
                            response_text = raw_response.replace("*", "").strip()
                            # Clean up multiple spaces
                            while "  " in response_text:
                                response_text = response_text.replace("  ", " ")
                                
                            logger.info(f"Interview prep response (cleaned): {response_text[:100]}...")
                            break
                
                if response_text:
                    twilio_client.send_whatsapp(phone_number, response_text)
                else:
                    # Better fallback when agent fails to respond
                    logger.warning(f"Interview prep agent returned empty response for {phone_number}")
                    twilio_client.send_whatsapp(phone_number, "Sorry, I had a technical issue. Please repeat your last message or type 'continue' to keep learning.")
                return
                
            except Exception as e:
                # Check if this is a retryable error and implement retry logic
                if should_retry_error(e):
                    logger.warning(f"Retryable error in interview prep for {phone_number}: {e}")
                    
                    # Implement retry with exponential backoff
                    max_retries = 3
                    for retry_attempt in range(max_retries):
                        try:
                            # Notify user about retry on first attempt
                            if retry_attempt == 0:
                                twilio_client.send_whatsapp(phone_number, "🔄 Retrying your request...")
                            
                            # Calculate delay: 1s, 2s, 4s
                            delay = 2 ** retry_attempt
                            logger.info(f"Interview prep retry attempt {retry_attempt + 1}/{max_retries} for {phone_number} after {delay}s delay")
                            await asyncio.sleep(delay)
                            
                            # Retry the request
                            response_text = ""
                            async for event in session.runner.run_async(user_id=phone_number, session_id=session_id, new_message=message):
                                # Track token usage for interview prep
                                try:
                                    track_event_usage(
                                        event=event,
                                        model="gemini-2.5-flash",
                                        agent_type="interview_prep",
                                        user_id=phone_number
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to track interview prep token usage: {e}")
                                
                                if hasattr(event, 'is_final_response') and event.is_final_response():
                                    if event.content and event.content.parts:
                                        texts = [p.text for p in event.content.parts if getattr(p, 'text', None)]
                                        raw_response = "\n".join(t for t in texts if t).strip()
                                        
                                        # Clean response to remove asterisks and formatting
                                        response_text = raw_response.replace("*", "").strip()
                                        while "  " in response_text:
                                            response_text = response_text.replace("  ", " ")
                                        break
                            
                            # If successful, send response and return
                            if response_text:
                                logger.info(f"Interview prep retry successful on attempt {retry_attempt + 1} for {phone_number}")
                                twilio_client.send_whatsapp(phone_number, response_text)
                            else:
                                twilio_client.send_whatsapp(phone_number, "Sorry, I had a technical issue. Please repeat your last message or type 'continue' to keep learning.")
                            return
                            
                        except Exception as retry_error:
                            logger.warning(f"Retry attempt {retry_attempt + 1} failed for {phone_number}: {retry_error}")
                            if retry_attempt == max_retries - 1:
                                # Last retry failed, fall through to error handling
                                e = retry_error
                                break
                            continue
                
                # Handle non-retryable errors or exhausted retries
                error_msg = str(e).lower()
                user_message = ""
                
                if "500" in error_msg or "internal" in error_msg:
                    user_message = "⚠️ I'm experiencing some technical issues right now. Please try again in a moment!"
                    logger.error(f"Gemini API 500 error for {phone_number}: {e}")
                elif "503" in error_msg or "overloaded" in error_msg or "unavailable" in error_msg:
                    user_message = "⚠️ The AI service is busy right now. Please try again in a minute or two!"
                    logger.error(f"Gemini API 503/overloaded error for {phone_number}: {e}")
                elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
                    user_message = "⚠️ Too many requests right now. Please wait a moment and try again!"
                    logger.error(f"Gemini API rate limit error for {phone_number}: {e}")
                elif "timeout" in error_msg:
                    user_message = "⚠️ Request timed out. Please try sending your message again!"
                    logger.error(f"Gemini API timeout error for {phone_number}: {e}")
                else:
                    user_message = "⚠️ I'm having trouble processing your request. Please try again shortly!"
                    logger.error(f"Interview prep agent error for {phone_number}: {e}")
                
                # Send user-friendly error message
                twilio_client.send_whatsapp(phone_number, user_message)
                
                # Suggest alternatives
                help_msg = "💡 In the meantime, you can:\n• Type 'menu' to switch services\n• Type 'assessment' for skill evaluation\n• Try your request again in a few minutes"
                twilio_client.send_whatsapp(phone_number, help_msg)
                return

        # Skill assessment agent - complex processing with images/tools
        # Collect response text and/or tool function response payloads
        response_text = ""
        media_payload: Optional[Dict] = None
        import json as _json
        conversation_turn = 1  # Track conversation turns
        async for event in session.runner.run_async(user_id=phone_number, session_id=session_id, new_message=message):
            # Track token usage for skill assessment
            try:
                track_event_usage(
                    event=event,
                    model="gemini-2.0-flash",
                    agent_type="skill_assessment",
                    user_id=phone_number
                )
            except Exception as e:
                logger.warning(f"Failed to track skill assessment token usage: {e}")
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    try:
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            logger.info(f"Agent tool call: name={getattr(fc, 'name', None)}")
                        fr_dbg = getattr(part, "function_response", None)
                        if fr_dbg is not None:
                            logger.info(f"Agent tool response: name={getattr(fr_dbg, 'name', None)} present={True}")
                        if getattr(part, "text", None):
                            logger.info(f"Agent text part: {(part.text or '')[:200]}")
                    except Exception:
                        pass
                    # Prefer tool/function response payloads when present
                    fr = getattr(part, "function_response", None)
                    if fr is not None:
                        try:
                            resp = getattr(fr, "response", None)
                            logger.info(f"Function response raw: type={type(resp)} content={str(resp)[:500]}")
                            data = None
                            if isinstance(resp, dict):
                                # Check if it has a 'result' key with JSON string
                                if 'result' in resp and isinstance(resp['result'], str):
                                    result_content = resp['result'].strip()
                                    if result_content.startswith('{') and result_content.endswith('}'):
                                        try:
                                            data = _json.loads(result_content)
                                            logger.info(f"Parsed JSON from function response.result: {data}")
                                        except Exception as e:
                                            logger.info(f"Function response.result is not valid JSON, treating as text response")
                                            data = resp  # fallback to original dict
                                    else:
                                        logger.info(f"Function response.result is plain text: {result_content[:100]}...")
                                        data = resp  # fallback to original dict
                                else:
                                    data = resp
                            elif isinstance(resp, str):
                                t = resp.strip()
                                if t.startswith("{") and t.endswith("}"):
                                    try:
                                        data = _json.loads(t)
                                        logger.info(f"Parsed JSON from function response: {data}")
                                    except Exception as e:
                                        logger.info(f"String response is not valid JSON, skipping parsing")
                                        data = None
                                else:
                                    logger.info(f"Function response is plain text: {t[:100]}...")
                                    data = None
                            if isinstance(data, dict):
                                logger.info(f"Function response data keys: {list(data.keys())}")
                                if data.get("image_path") or data.get("imagePath"):
                                    # Normalize keys
                                    media_payload = {
                                        "image_path": data.get("image_path") or data.get("imagePath"),
                                        "question": data.get("question") or data.get("prompt"),
                                    }
                                if media_payload:
                                    logger.info(f"Extracted media payload from function_response: has_image_path={bool(media_payload.get('image_path'))}")
                                # Don't break - continue to collect agent's text response
                            elif isinstance(resp, dict) and 'result' in resp and isinstance(resp['result'], str):
                                # Check if this is a final assessment summary (contains score indicators)
                                result_text = resp['result']
                                if any(indicator in result_text for indicator in ["📊 **Your Results:**", "Assessment Complete", "**Total Score:", "🎯"]):
                                    logger.info("Found assessment summary in tool response, using it as response_text")
                                    response_text = result_text
                                    # Found final assessment, stop processing
                                    break
                        except Exception:
                            logger.exception("Failed to parse function_response payload")
                    # Capture text - prioritize question text over generic intro
                    if getattr(part, "text", None):
                        text_content = (part.text or "").strip()
                        # Prioritize text that contains question words over generic intro text
                        if not response_text or any(word in text_content.lower() for word in ["safe", "unsafe", "what do you think", "question", "look at"]):
                            response_text = text_content
            
            # Break if we found an assessment summary
            if response_text and any(indicator in response_text for indicator in ["📊 **Your Results:**", "Assessment Complete", "**Total Score:", "🎯"]):
                logger.info("Found assessment summary, stopping event processing")
                break
                
            # Only break on final response, not when we have media payload
            # (we want to collect both media payload AND agent's text)
            # If this is the final response and we don't have a media payload, stop
            if hasattr(event, "is_final_response") and callable(event.is_final_response):
                try:
                    if event.is_final_response():
                        break
                except Exception:
                    pass

        if not response_text:
            response_text = "Sorry, I couldn't generate a response right now. Please try again."

        logger.info(f"Final response_text: '{response_text}' has_media_payload: {bool(media_payload)}")

        # Try to extract payload for image sending (prefer tool response payload)
        def extract_json_block(text: str) -> Optional[Dict]:
            t = (text or "").strip()
            try:
                start = t.index("{")
                end = t.rindex("}") + 1
                return _json.loads(t[start:end])
            except Exception:
                return None

        payload = media_payload or extract_json_block(response_text)
        if payload and isinstance(payload, dict) and payload.get("image_path"):
            img_url = local_image_to_public_url(payload["image_path"]) or ""
            # Use agent's generated text as caption (prefer this over any tool response question)
            caption = response_text or payload.get("question") or ""
            if img_url:
                logger.info(f"Sending media via Twilio: url={img_url} caption_len={len(caption)}")
                twilio_client.send_media(phone_number, img_url, caption=caption)
            else:
                twilio_client.send_whatsapp(phone_number, f"{caption}\n(Image unavailable)")
        else:
            # Try to detect Markdown image and send as media
            import re
            md = re.search(r"!\[[^\]]*\]\(([^)]+)\)", response_text)
            if md:
                raw_path = md.group(1).strip()
                img_url = local_image_to_public_url(raw_path)
                caption = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", "", response_text).strip()
                if img_url:
                    twilio_client.send_media(phone_number, img_url, caption=caption)
                else:
                    clean = caption or "Please look at the image."
                    twilio_client.send_whatsapp(phone_number, clean)
                return

            # If the model printed an absolute path in text (e.g., 'Image Path: /...')
            # extract it and send media instead of raw path.
            match = re.search(r"(?i)image\s*path\s*:\s*(\S+)", response_text)
            if match:
                path = match.group(1)
                img_url = local_image_to_public_url(path)
                # Caption: remove the image path line from the text
                caption = re.sub(r"(?i)^.*image\s*path\s*:\s*\S+.*$", "", response_text, flags=re.MULTILINE).strip()
                if img_url:
                    twilio_client.send_media(phone_number, img_url, caption=caption)
                else:
                    # Fallback to clean text without leaking local paths
                    clean = caption or "Please look at the image."
                    twilio_client.send_whatsapp(phone_number, clean)
            else:
                # Twilio outbound reply (text)
                twilio_client.send_whatsapp(phone_number, response_text)

    except Exception as e:
        logger.exception("Error processing agent message")
        
        # Check if this is a retryable error and implement retry logic
        if should_retry_error(e):
            logger.warning(f"Retryable error in skill assessment for {phone_number}: {e}")
            
            # Implement retry with exponential backoff
            max_retries = 3
            for retry_attempt in range(max_retries):
                try:
                    # Notify user about retry on first attempt
                    if retry_attempt == 0:
                        twilio_client.send_whatsapp(phone_number, "🔄 Retrying your request...")
                    
                    # Calculate delay: 1s, 2s, 4s
                    delay = 2 ** retry_attempt
                    logger.info(f"Skill assessment retry attempt {retry_attempt + 1}/{max_retries} for {phone_number} after {delay}s delay")
                    await asyncio.sleep(delay)
                    
                    # Retry the skill assessment processing
                    response_text = ""
                    media_payload: Optional[Dict] = None
                    
                    async for event in session.runner.run_async(user_id=phone_number, session_id=session_id, new_message=message):
                        # Track token usage for skill assessment
                        try:
                            track_event_usage(
                                event=event,
                                model="gemini-2.0-flash",
                                agent_type="skill_assessment",
                                user_id=phone_number
                            )
                        except Exception as track_error:
                            logger.warning(f"Failed to track skill assessment token usage: {track_error}")
                        
                        if getattr(event, "content", None) and getattr(event.content, "parts", None):
                            for part in event.content.parts:
                                # Handle function responses for media payload
                                fr = getattr(part, "function_response", None)
                                if fr is not None:
                                    try:
                                        resp = getattr(fr, "response", None)
                                        if isinstance(resp, dict):
                                            if resp.get("image_path") or resp.get("imagePath"):
                                                media_payload = {
                                                    "image_path": resp.get("image_path") or resp.get("imagePath"),
                                                    "question": resp.get("question") or resp.get("prompt"),
                                                }
                                    except Exception:
                                        pass
                                
                                # Capture text response
                                if getattr(part, "text", None):
                                    text_content = (part.text or "").strip()
                                    if not response_text or any(word in text_content.lower() for word in ["safe", "unsafe", "what do you think", "question", "look at"]):
                                        response_text = text_content
                        
                        # Break on final response
                        if hasattr(event, "is_final_response") and callable(event.is_final_response):
                            try:
                                if event.is_final_response():
                                    break
                            except Exception:
                                pass
                    
                    if not response_text:
                        response_text = "Sorry, I couldn't generate a response right now. Please try again."
                    
                    # Handle media or text response
                    if media_payload and isinstance(media_payload, dict) and media_payload.get("image_path"):
                        img_url = local_image_to_public_url(media_payload["image_path"]) or ""
                        caption = response_text or media_payload.get("question") or ""
                        if img_url:
                            twilio_client.send_media(phone_number, img_url, caption=caption)
                        else:
                            twilio_client.send_whatsapp(phone_number, f"{caption}\n(Image unavailable)")
                    else:
                        twilio_client.send_whatsapp(phone_number, response_text)
                    
                    logger.info(f"Skill assessment retry successful on attempt {retry_attempt + 1} for {phone_number}")
                    return
                    
                except Exception as retry_error:
                    logger.warning(f"Skill assessment retry attempt {retry_attempt + 1} failed for {phone_number}: {retry_error}")
                    if retry_attempt == max_retries - 1:
                        # Last retry failed, fall through to error handling
                        e = retry_error
                        break
                    continue
        
        # Handle non-retryable errors or exhausted retries
        error_msg = str(e).lower()
        user_message = ""
        
        if "500" in error_msg or "internal" in error_msg:
            user_message = "⚠️ I'm experiencing some technical issues right now. Please try again in a moment!"
            logger.error(f"Skill assessment API 500 error for {phone_number}: {e}")
        elif "503" in error_msg or "overloaded" in error_msg or "unavailable" in error_msg:
            user_message = "⚠️ The AI service is busy right now. Please try again in a minute or two!"
            logger.error(f"Skill assessment API 503/overloaded error for {phone_number}: {e}")
        elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
            user_message = "⚠️ Too many requests right now. Please wait a moment and try again!"
            logger.error(f"Skill assessment API rate limit error for {phone_number}: {e}")
        elif "timeout" in error_msg:
            user_message = "⚠️ Request timed out. Please try sending your message again!"
            logger.error(f"Skill assessment API timeout error for {phone_number}: {e}")
        else:
            user_message = "⚠️ I'm having trouble processing your request. Please try again shortly!"
            logger.error(f"Skill assessment agent error for {phone_number}: {e}")
        
        try:
            # Send user-friendly error message
            twilio_client.send_whatsapp(phone_number, user_message)
            
            # Suggest alternatives for skill assessment
            help_msg = "💡 You can try:\n• Type 'menu' to see all options\n• Type 'interview' to switch to learning mode\n• Wait a minute and try again"
            twilio_client.send_whatsapp(phone_number, help_msg)
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


# --- Routes -----------------------------------------------------------------

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


# Simple token usage endpoints
@app.get("/token-totals")
async def get_token_totals():
    """Get current running token totals."""
    try:
        totals = get_totals()
        return totals
    except Exception as e:
        logger.error(f"Error getting token totals: {e}")
        return {"error": str(e)}

@app.post("/token-totals/reset")
async def reset_token_totals():
    """Reset all token totals."""
    try:
        reset_totals()
        return {"message": "Token totals reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting token totals: {e}")
        return {"error": str(e)}


@app.post("/chat")
async def chat_endpoint(request: dict):
    """
    Direct chat endpoint for testing without WhatsApp/Twilio integration.
    
    Expected JSON format:
    {
        "phone_number": "+919101290739",
        "message": "Hi"
    }
    
    Returns the agent's response directly.
    """
    try:
        phone_number = request.get("phone_number", "+919999999999")
        message = request.get("message", "")
        
        if not message:
            return {"error": "Message is required", "status": "error"}
        
        logger.info(f"Direct chat: from={phone_number} message='{message}'")
        
        # Store responses to return to user
        responses = []
        
        # Create a mock response collector
        class MockTwilioClient:
            def __init__(self):
                self.from_number = "whatsapp:+12345678901"  # Mock Twilio number
                self.client = self  # Mock client object
                
            def send_whatsapp(self, to_number: str, body: str) -> None:
                responses.append(body)
                logger.info(f"Mock send: to={to_number} body_length={len(body)}")
            
            def send_media(self, to_number: str, media_url: str, caption: str = "") -> None:
                responses.append(f"[Media: {media_url}] {caption}" if caption else f"[Media: {media_url}]")
                logger.info(f"Mock media: to={to_number} url={media_url}")
            
            def create(self, from_, to, body, media_url=None):
                """Mock Twilio messages.create method"""
                class MockMessage:
                    def __init__(self):
                        self.sid = "mock_message_id_123"
                        self.status = "queued"
                
                if media_url:
                    caption = body or ""
                    responses.append(f"[Media: {media_url[0] if isinstance(media_url, list) else media_url}] {caption}" if caption else f"[Media: {media_url[0] if isinstance(media_url, list) else media_url}]")
                else:
                    responses.append(body)
                logger.info(f"Mock messages.create: from={from_} to={to} body_length={len(body or '')}")
                return MockMessage()
            
            # Make messages a property so it can be accessed as client.messages
            @property  
            def messages(self):
                return self
        
        # Temporarily replace twilio client
        original_client = globals().get('twilio_client')
        globals()['twilio_client'] = MockTwilioClient()
        
        try:
            # Process the message through the same logic as WhatsApp
            await process_agent_and_reply(phone_number, message)
            
            return {
                "status": "success",
                "phone_number": phone_number,
                "user_message": message,
                "agent_responses": responses,
                "response_count": len(responses)
            }
            
        finally:
            # Restore original twilio client
            if original_client:
                globals()['twilio_client'] = original_client
        
    except Exception as e:
        logger.exception("Error in direct chat endpoint")
        return {
            "status": "error", 
            "error": str(e),
            "message": "Failed to process chat message"
        }

@app.post("/twilio-webhook", response_class=PlainTextResponse)
async def twilio_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form("") ,
    Body: str = Form("") ,
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
):
    """Twilio webhook for WhatsApp messages (form-encoded).

    - Respond immediately to meet Twilio timeouts.
    - Process with the agent in a background task, and send reply via Twilio API.
    """
    try:
        phone_number = From.replace("whatsapp:", "") if From else ""
        # Derive base URL dynamically if env isn't set
        global _DYNAMIC_BASE_URL
        if not BASE_URL:
            try:
                # request.base_url includes trailing '/'
                _DYNAMIC_BASE_URL = str(request.base_url).rstrip('/')
            except Exception:
                pass
        num_media = int(NumMedia or "0")
        logger.info(f"Webhook: from={From} body_len={len(Body or '')} media={num_media} type={MediaContentType0}")

        # Handle text input - if empty, check if this is a new user who needs agent selection
        incoming_text = Body or ""
        if not incoming_text and num_media == 0:
            # Empty message from new user - they might need to see the template card
            if phone_number not in SESSIONS:
                incoming_text = "start"  # Will trigger template card
            else:
                session = SESSIONS[phone_number]
                if not session.agent_selected:
                    incoming_text = "start"  # Will trigger selection prompt
                elif session.agent_type == "skill_assessment":
                    incoming_text = "Start assessment"
                else:
                    incoming_text = "Hi"

        if not phone_number:
            return PlainTextResponse("", status_code=200)

        # Kick off background processing so we can ack Twilio immediately
        background_tasks.add_task(process_agent_and_reply, phone_number, incoming_text)

        # Return an empty 200 OK (Twilio will accept empty TwiML for WhatsApp)
        return ""

    except Exception:
        logger.exception("Webhook error")
        # Still return 200 so Twilio doesn't retry aggressively
        return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("whatsapp:app", host="0.0.0.0", port=int(os.getenv("PORT", "5000")))


# ---------------- Startup: pre-analyze images and patch tool ------------------

PRELOADED_PHOTO_QUESTIONS = []


@app.on_event("startup")
async def preload_and_patch() -> None:
    try:
        # Images are now pre-analyzed during module import in electrician_skill_agent.py
        # No need to re-analyze here - just log that we're using the existing analysis
        if hasattr(agent_mod, '_PRE_ANALYZED_ANSWERS') and agent_mod._PRE_ANALYZED_ANSWERS:
            logger.info(f"Agent tool patched to use {len(agent_mod._PRE_ANALYZED_ANSWERS)} preloaded photo questions.")
        else:
            logger.warning("No pre-analyzed answers found in electrician_skill_agent module.")
            return

        # No need to monkey-patch since the agent already has the correct start_photo_assessment function

        # Rebuild tools list explicitly from the module to ensure ADK picks it up
        electrician_skill_assessor.tools = [
            agent_mod.start_photo_assessment,
            agent_mod.score_photo_answer,
            agent_mod.start_knowledge_assessment,
            agent_mod.score_knowledge_answer,
        ]
        logger.info("Agent tool patched to use preloaded photo questions.")

    except Exception:
        logger.exception("Startup pre-analysis/patching failed.")
