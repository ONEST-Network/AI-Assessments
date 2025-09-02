import os
import json
import logging
import asyncio
import uuid
from datetime import datetime
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import tempfile
import base64
import requests



load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)


google_api_key = os.getenv('GOOGLE_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

if not (google_api_key or gemini_api_key):
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")


if google_api_key and gemini_api_key:
    logger.info("Both API keys found. Using GOOGLE_API_KEY.")
    os.environ.pop('GEMINI_API_KEY', None)
elif gemini_api_key and not google_api_key:
    logger.info("Using GEMINI_API_KEY.")



@dataclass
class AssessmentResult:
    """Structured assessment result"""
    skill: str
    score: float
    grade: str
    timestamp: str
    details: Dict[str, Any]

@dataclass
class CandidateProfile:
    """Enhanced candidate profile with comprehensive tracking"""
    candidate_id: str
    name: str
    applied_role: str
    assessment_history: List[AssessmentResult]
    skill_levels: Dict[str, float]
    current_assessment: Optional[str]
    interaction_history: List[Dict[str, Any]]
    assessment_status: str  # "started", "in_progress", "completed"
    created_at: str

def create_initial_candidate_state(candidate_name: str, candidate_id: str = None, role: str = None) -> Dict[str, Any]:
    """Create comprehensive initial state for a candidate"""
    if not candidate_id:
        candidate_id = str(uuid.uuid4())[:8]
    
    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "applied_role": role or "unknown",
        "role_identified": bool(role),
        "assessment_history": [],
        "skill_levels": {},
        "current_assessment": None,
        "interaction_history": [],
        "assessment_status": "started",
        "current_label_image": "",
        "created_at": datetime.now().isoformat(),
        "session_metadata": {
            "total_assessments": 0,
            "completed_skills": [],
            "pending_skills": [],
            "last_activity": datetime.now().isoformat()
        }
    }

async def update_interaction_history(session_service: InMemorySessionService, app_name: str, user_id: str, session_id: str, interaction_data: Dict[str, Any]):
    """Update interaction history following proper ADK pattern"""
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

async def add_user_query_to_history(session_service: InMemorySessionService, app_name: str, user_id: str, session_id: str, query: str, image_provided: bool = False):
    """Add user query to interaction history"""
    await update_interaction_history(session_service, app_name, user_id, session_id, {
        "action": "user_query",
        "query": query,
        "image_provided": image_provided
    })

async def add_agent_response_to_history(session_service: InMemorySessionService, app_name: str, user_id: str, session_id: str, response: str):
    """Add agent response to interaction history"""
    await update_interaction_history(session_service, app_name, user_id, session_id, {
        "action": "agent_response", 
        "response": response
    })



def complete_skill_assessment(skill_name: str, score: float, grade: str, 
                             details: Dict[str, Any], tool_context: ToolContext) -> str:
    """Tool to complete a skill assessment and update candidate profile"""
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
        current_metadata = tool_context.state.get("session_metadata", {"completed_skills": [], "pending_skills": []})
        
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
    """Tool to start a new skill assessment"""
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
    """Tool to retrieve candidate profile information"""
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
    """Tool to update candidate's role once identified through conversation"""
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

def start_label_reading_quiz(tool_context: ToolContext) -> str:
    """Tool to start a new label reading quiz following ADK pattern"""
    try:
        # Debug logging
        existing_quiz = tool_context.state.get("label_reading_quiz", {})
        logger.info(f"DEBUG: start_label_reading_quiz called")
        logger.info(f"DEBUG: existing quiz active: {existing_quiz.get('quiz_active', False)}")
        logger.info(f"DEBUG: existing current_question: {existing_quiz.get('current_question', 'N/A')}")
        
        # Check if quiz is already active - don't restart
        if existing_quiz.get("quiz_active"):
            logger.info("DEBUG: Quiz already active, should not call start_label_reading_quiz")
            return "ERROR: Quiz is already active. Use answer_quiz_question instead."
        
        logger.info("DEBUG: Starting new quiz")
        # Load label dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "label_dataset", "index.json")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Label dataset not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            label_data = json.load(f)
        
        # Filter labels for warehouse/loader picker role
        relevant_labels = [item for item in label_data if item.get('category') in ['warehouse', 'grocery', 'beverage', 'condiments']]
        if not relevant_labels:
            relevant_labels = label_data[:3]
        
        # Select up to 3 labels for the quiz  
        selected_labels = relevant_labels[:3]
        
        # Generate questions from selected labels
        questions = []
        for i, label_item in enumerate(selected_labels):
            fields = label_item.get('fields', {})
            key_fields = ['product', 'brand', 'net_weight', 'volume', 'variant', 'wattage']
            
            for field_name in key_fields:
                if field_name in fields:
                    questions.append({
                        "label_index": i,
                        "question": f"What is the {field_name}?",
                        "expected_field": field_name,
                        "expected_value": fields[field_name],
                        "image_paths": label_item.get('file_paths', [label_item.get('file_path')])
                    })
                    if len([q for q in questions if q['label_index'] == i]) >= 3:
                        break
        
        # Store quiz state in session following ADK pattern
        quiz_state = {
            "labels": selected_labels,
            "questions": questions,
            "current_question": 0,
            "correct_answers": 0,
            "quiz_active": True
        }
        
        # Update session state
        tool_context.state["label_reading_quiz"] = quiz_state
        
        # Get first question
        first_question = questions[0] if questions else None
        if first_question:
            image_paths = first_question.get('image_paths', [])
            # Store all image paths for this question
            tool_context.state["current_label_images"] = image_paths
            
            # Format image display - show all images if multiple
            if len(image_paths) > 1:
                image_display = " ".join([f"[Image: {path}]" for path in image_paths])
                return f"Quiz started. Looking at {image_display} - Question 1/{len(questions)}: {first_question['question']}"
            else:
                image_path = image_paths[0] if image_paths else "label_dataset/samples/product_001.jpeg"
                return f"Quiz started. Looking at [Image: {image_path}] - Question 1/{len(questions)}: {first_question['question']}"
        else:
            return "Quiz started but no questions available"
            
    except Exception as e:
        logger.error(f"Error starting label reading quiz: {e}")
        return f"Error starting quiz: {str(e)}"

def answer_quiz_question(user_answer: str, tool_context: ToolContext) -> str:
    """Tool to answer a quiz question and get the next question"""
    try:
        quiz_state = tool_context.state.get("label_reading_quiz", {})
        
        
        logger.info(f"DEBUG: answer_quiz_question called with: '{user_answer}'")
        logger.info(f"DEBUG: quiz_state exists: {bool(quiz_state)}")
        logger.info(f"DEBUG: quiz_active: {quiz_state.get('quiz_active', False)}")
        logger.info(f"DEBUG: current_question: {quiz_state.get('current_question', 'N/A')}")
        logger.info(f"DEBUG: total questions: {len(quiz_state.get('questions', []))}")
        
        if not quiz_state.get("quiz_active"):
            logger.info("DEBUG: No active quiz found")
            return "No active quiz found. Please start a quiz first."
        
        questions = quiz_state.get("questions", [])
        current_idx = quiz_state.get("current_question", 0)
        
        if current_idx >= len(questions):
            return "Quiz completed. No more questions."
        
        current_question = questions[current_idx]
        expected_value = current_question['expected_value']
        
        # Return the current question details for agent evaluation
        # Agent will use its intelligence to score this
        return json.dumps({
            "action": "score_and_continue",
            "user_answer": user_answer,
            "expected_answer": expected_value,
            "field_type": current_question['expected_field'],
            "current_question_num": current_idx + 1,
            "total_questions": len(questions),
            "current_state": {
                "answered_questions": current_idx + 1,
                "quiz_active": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error answering quiz question: {e}")
        return f"Error processing answer: {str(e)}"

def update_quiz_score_and_continue(is_correct: bool, tool_context: ToolContext) -> str:
    """Tool for agent to update quiz score and get next question"""
    try:
        quiz_state = tool_context.state.get("label_reading_quiz", {})
        
        if not quiz_state.get("quiz_active"):
            return "No active quiz found."
            
        questions = quiz_state.get("questions", [])
        current_idx = quiz_state.get("current_question", 0)
        correct_answers = quiz_state.get("correct_answers", 0)
        
        # Update score if answer was correct
        if is_correct:
            correct_answers += 1
        
        # Move to next question  
        next_idx = current_idx + 1
        quiz_state["current_question"] = next_idx
        quiz_state["correct_answers"] = correct_answers
        
        # Check if quiz is complete
        if next_idx >= len(questions):
            accuracy = (correct_answers / len(questions)) * 100
            quiz_state["quiz_active"] = False
            tool_context.state["label_reading_quiz"] = quiz_state
            
            return json.dumps({
                "action": "quiz_completed",
                "final_score": correct_answers,
                "total_questions": len(questions),
                "accuracy": accuracy,
                "message": f"Quiz completed! Final score: {correct_answers}/{len(questions)} ({accuracy:.1f}% accuracy)"
            })
        
        # Get next question
        next_question = questions[next_idx]
        image_paths = next_question.get('image_paths', [])
        tool_context.state["current_label_images"] = image_paths
        
        # Update state
        tool_context.state["label_reading_quiz"] = quiz_state
        
        # Format next question display
        if len(image_paths) > 1:
            image_display = " ".join([f"[Image: {path}]" for path in image_paths])
        else:
            image_display = f"[Image: {image_paths[0]}]" if image_paths else "[Image: label_dataset/samples/default.jpeg]"
        
        return json.dumps({
            "action": "continue_quiz",
            "current_score": correct_answers,
            "total_answered": current_idx + 1,
            "next_question_num": next_idx + 1,
            "total_questions": len(questions),
            "next_question": next_question['question'],
            "image_display": image_display
        })
        
    except Exception as e:
        logger.error(f"Error updating quiz score: {e}")
        return f"Error updating score: {str(e)}"

def retrieve_image_from_path(image_path: str, tool_context: ToolContext) -> str:
    """Enhanced image retrieval tool with state updates"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
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
        
        logger.info(f"Retrieved image from path: {image_path}, size: {len(image_data)} bytes")
        return f"Successfully retrieved image from {image_path} ({len(image_data)} bytes)"
        
    except Exception as e:
        logger.error(f"Error retrieving image from path {image_path}: {str(e)}")
        raise

def validate_image_data(tool_context: ToolContext) -> str:
    """Enhanced image validation with comprehensive checks"""
    try:
        image_data = tool_context.state.get('image_data')
        image_path = tool_context.state.get('image_path', 'unknown')
        
        if not image_data:
            raise ValueError("No image data found in context. Call retrieve_image_from_path first.")
            
        # Validate using PIL
        image_io = io.BytesIO(image_data)
        image = Image.open(image_io)
        image.verify()
        
        # Reopen for processing
        image_io.seek(0)
        image = Image.open(image_io)
        
        validation_result = {
            "valid": True,
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "file_size": len(image_data),
            "file_path": image_path,
            "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 0
        }
        
        # Store validation results
        tool_context.state['validation_result'] = validation_result
        
        logger.info(f"Image validation successful: {validation_result}")
        return f"Image validation successful: {validation_result['format']} format, {validation_result['size']} pixels"
        
    except Exception as e:
        error_msg = f"Image validation failed: {str(e)}"
        logger.error(error_msg)
        tool_context.state['validation_result'] = {"valid": False, "error": str(e)}
        return error_msg


stitching_assessor = Agent(
    name="stitching_assessor",
    model="gemini-2.0-flash",
    description="Specialized agent for evaluating stitching quality and techniques from images",
    instruction="""You are the Stitching Assessment Agent in a stateful multi-agent job evaluation system.

ROLE: Evaluate stitching quality from uploaded images for tailoring positions.

CONTEXT ACCESS: You have access to the candidate's complete profile including:
- candidate_name: The candidate's name
- applied_role: The role they're applying for
- assessment_history: Previous assessment results
- skill_levels: Current skill ratings
- current_assessment: What assessment is in progress

ASSESSMENT PROCESS:
1. Start the assessment using start_skill_assessment tool
2. Use retrieve_image_from_path to load the provided image
3. Use validate_image_data to ensure image quality
4. Analyze stitching quality based on:
   - Stitch uniformity and consistency
   - Thread tension and appearance
   - Seam straightness and alignment
   - Overall craftsmanship
   - Professional finish quality
   - Edge finishing
   - Stitch type identification

5. Score on scale 1-10 and assign grade: Beginner/Intermediate/Advanced/Expert
6. Complete assessment using complete_skill_assessment tool
7. IMPORTANT: After using the complete_skill_assessment tool, provide a detailed formatted analysis as your final response

CRITICAL: Your final message must be the detailed analysis in text format, not just tool calls.

RESPONSE FORMAT: After completing the assessment with tools, provide a detailed analysis in this EXACT format (MUST include STATUS:completed at the end):

Detailed Analysis:
* Quality Rating: [X]/10
* Stitch Type: [Identify the specific stitch type used]
* Skill Level: [Beginner/Intermediate/Advanced/Expert]
* Technical Issues:
  - [List specific technical issues observed]
  - [Another issue if present]
  - [More issues as needed]
* Improvement Tips:
  - [Specific actionable advice]
  - [Additional improvement suggestions]
  - [More tips as appropriate]

EXAMPLE OUTPUT:
Detailed Analysis:
* Quality Rating: 8/10
* Stitch Type: Running stitch with reinforcement
* Skill Level: Advanced
* Technical Issues:
  - Minor tension variation in middle section
  - Edge finishing could be neater
* Improvement Tips:
  - Use consistent thread tension throughout
  - Practice edge finishing techniques
  - Consider using a seam guide for straight lines

[STATUS:completed]

CRITICAL: Always end your response with [STATUS:completed] when providing final assessment results.
Always provide constructive feedback and maintain a professional, encouraging tone.

WORKFLOW REMINDER:
1. Use tools: start_skill_assessment → retrieve_image_from_path → validate_image_data → complete_skill_assessment
2. Then provide detailed analysis text as your response (not as a tool call)
3. The analysis should be visible to the candidate, formatted exactly as shown above""",
    tools=[
        start_skill_assessment,
        retrieve_image_from_path,
        validate_image_data,
        complete_skill_assessment,
        get_candidate_profile
    ]
)


label_reading_assessor = Agent(
    name="label_reading_assessor", 
    model="gemini-2.0-flash",
    description="Specialized agent for evaluating label reading and information extraction skills",
    instruction="""You are the Label Reading Assessment Agent in a stateful multi-agent job evaluation system.

<role>
Evaluate candidate's ability to accurately read and extract information from product labels.
</role>

<candidate_context>
<candidate_name>{candidate_name}</candidate_name>
<applied_role>{applied_role}</applied_role>
<assessment_history>{assessment_history}</assessment_history>
<current_assessment>{current_assessment}</current_assessment>
</candidate_context>

<tool_selection_logic>
CRITICAL - You MUST check what tools to use based on the situation:

STEP 1: Check if <current_assessment> shows "Label Reading" is in progress
- IF YES: The quiz is already active, user input is an ANSWER
- Use ONLY: answer_quiz_question (never start_label_reading_quiz)

STEP 2: If no assessment is active
- IF user wants to start: start_skill_assessment → start_label_reading_quiz  
- IF user wants to complete: complete_skill_assessment

EXAMPLES of ANSWERS that need answer_quiz_question:
- "Emergency bulb" 
- "Eveready"
- "14", "12W", "20"  
- "packged drinking water"
- Any product name, brand, number, or descriptive text

NEVER use start_label_reading_quiz if quiz is already running!
</tool_selection_logic>

<intelligent_scoring>
WORKFLOW when user provides an answer:
1. Call answer_quiz_question with user's answer
2. Tool returns JSON with evaluation data (user_answer, expected_answer, etc.)
3. Use your intelligence to evaluate if the answer is correct:
   - ACCEPT: Case differences ("eveready" = "EVEREADY"), minor spelling variations  
   - ACCEPT: Brand name variations ("Daawat" = "dawat", "Bisleri" = "bisleri")
   - REJECT: Major content differences ("emergency bulb" ≠ "Emergency LED Bulb")
   - REJECT: Wrong numbers ("12W" ≠ "60W", "250ml" ≠ "500ml") 
   - BE GENEROUS with reasonable variations, STRICT with completely different answers
4. Call update_quiz_score_and_continue with your evaluation (true/false)
5. Tool returns next question data or completion message
6. Display feedback to user: "Correct!" or "Not quite, the answer was [expected]"
7. Show next question with proper image paths
8. When quiz completes, call complete_skill_assessment with final results
</intelligent_scoring>

<quiz_state_management>
Maintain quiz state internally in your responses:
- Current question number (1-9)
- Questions asked and answers received  
- Score tracking (correct/total)
- Quiz completion status
</quiz_state_management>

<assessment_process>
1. FIRST: Check <current_assessment> state - if "Label Reading" is already in progress, CONTINUE the quiz
2. When starting NEW assessment: Use start_skill_assessment tool ONLY if <current_assessment> is empty
3. Load label dataset from label_dataset/index.json dynamically
4. For each label question: Use share_label_image tool to send image path to master agent
5. Track quiz state internally - maintain question number and score
6. Present questions WITHOUT revealing answers (master agent will show the image)
7. Progress through 3 labels with 3 questions each (9 total questions)
8. After all questions: Calculate final score and use complete_skill_assessment tool
</assessment_process>

<critical_rules>
- FOLLOW ADK STATEFUL PATTERN: State is managed in session, not agent memory
- TOOL USAGE:
  * For NEW assessment: start_skill_assessment → start_label_reading_quiz
  * For USER ANSWERS: answer_quiz_question (handles scoring and progression) 
  * For COMPLETION: complete_skill_assessment with final results
- Tools return complete formatted responses - present them directly to user
- Never restart quiz - tools maintain state across interactions
- Present tool responses exactly as returned (they include image paths)
</critical_rules>

<question_types>
Pick 3 questions per label from:
- "What is the product name?"
- "What is the brand name?" 
- "What is the net weight/volume?"
- "Who is the marketer/manufacturer?"
- "What special feature is mentioned?"
</question_types>

<scoring_criteria>
- Exact matches: Full points
- ASR variations allowed (e.g., "Eveready" vs "Ever ready")
- Case insensitive matching
- 1 point per correct answer
- Final score = (correct answers / 9) * 100
</scoring_criteria>

<response_format>
For each interaction:
1. Use share_label_image tool to share current image path with master agent
2. Include the image path DIRECTLY in your response: "Looking at [Image: path] - Question X/9: What is the [field]? [STATUS:input_required]"
3. Feedback on previous answer (if applicable)
4. Internal state tracking (Question X/9, Score: Y/Z)
5. For final result: Use complete_skill_assessment tool and end with [STATUS:completed]

CRITICAL: Always include the image path in your response so the user can see which image to look at.
CRITICAL: Always end questions with [STATUS:input_required] and final results with [STATUS:completed]
</response_format>

<example_flow>
NEW ASSESSMENT: "Starting Label Reading Assessment. Looking at [Image: label_dataset/samples/product_001.jpeg] - Question 1/9: What is the product name? [STATUS:input_required]"

CONTINUING: "Thank you! Score: 1/1. Looking at [Image: label_dataset/samples/product_001.jpeg] - Question 2/9: What is the brand name? [STATUS:input_required]"

ANSWER TO CONTINUE: For any product-related answer like "Emergency bulb" → Score it and continue to next question, don't restart

Final: "Assessment complete! Final score: 7/9 (77.8%). [STATUS:completed]"

STATE CHECKING: Always check <current_assessment> - if it shows "Label Reading", continue the quiz, don't restart
ALWAYS INCLUDE IMAGE PATH: Every question must show "Looking at [Image: actual_path] - Question X/9: [question]"
</example_flow>

Always maintain professional tone and provide constructive feedback.""",
    tools=[
        start_skill_assessment,
        complete_skill_assessment,
        get_candidate_profile,
        start_label_reading_quiz,
        answer_quiz_question,
        update_quiz_score_and_continue
    ]
)





class StatefulJobAssessmentSystem:
    """Enhanced multi-agent job assessment system with comprehensive state management"""
    
    def __init__(self, prompts_dir="assets"):
        self.session_service = InMemorySessionService()
        self._load_knowledge(prompts_dir)
        self.master_agent = self._create_master_agent()
        self.runner = Runner(
            agent=self.master_agent,
            app_name="job_assessment_app",
            session_service=self.session_service
        )
        
    def _load_knowledge(self, prompts_dir: str):
        """Load knowledge base files"""
        try:
            with open(os.path.join(prompts_dir, 'competency_map.json'), 'r') as f:
                self.competency_map = json.load(f)
            with open(os.path.join(prompts_dir, 'sub_agent_library.json'), 'r') as f:
                self.sub_agent_library = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def _create_master_agent(self) -> Agent:
        """Create enhanced master agent with state awareness"""
        enhanced_instruction = f"""
You are the Master Job Assessment Coordinator. You help candidates identify their target role and coordinate skill assessments.

<candidate_context>
<candidate_name>{{candidate_name}}</candidate_name>
<applied_role>{{applied_role}}</applied_role>
<role_identified>{{role_identified}}</role_identified>
<assessment_status>{{assessment_status}}</assessment_status>
<assessment_history>{{assessment_history}}</assessment_history>
<skill_levels>{{skill_levels}}</skill_levels>
<interaction_history>{{interaction_history}}</interaction_history>
<current_label_image>{{current_label_image}}</current_label_image>
</candidate_context>

NATURAL ROLE IDENTIFICATION:
- If <role_identified> is false, help determine their interest through natural conversation
- Listen for keywords: "tailor", "sewing", "warehouse", "loader", "retail", "sales"
- Ask clarifying questions if needed: "Are you interested in warehouse work or retail positions?"
- Use update_candidate_role tool when role is identified
- Available roles: Tailor, Loader Picker, Retail Sales

CRITICAL STITCHING WORKFLOW:
- When user says ANY of: "stitching", "tailor", "sewing", "assess my stitching", "fabric work"
- IMMEDIATELY respond with: "Great! I can help assess your stitching skills for the Tailor position. Please provide the path to your stitching image, like: /path/to/your/image.jpg [STATUS:input_required]"
- Do NOT say "Let's begin" or "I am ready" without asking for image first
- WAIT for actual image path before proceeding to assessment

STATUS INDICATORS:
- Always include status metadata at the end of your responses
- Use [STATUS:input_required] when waiting for user input (image paths, answers, choices)
- Use [STATUS:completed] when assessment is finished with final results
- Examples:
  * "Please provide your image path. [STATUS:input_required]"
  * "Assessment complete! Final score: 7/10. Result: PASS. [STATUS:completed]"

CONVERSATION FLOW:
1. Check <applied_role> and <role_identified> status
2. If role unknown, engage naturally to identify interest
3. Once role identified, explain assessment process for that specific role
4. Delegate to appropriate sub-agents for assessments
5. Use <interaction_history> to maintain conversation context

STATEFUL COORDINATION:
- Use <candidate_name> for personalization
- Reference <assessment_history> to avoid duplicate assessments
- Check <skill_levels> for completed skills
- Use <interaction_history> to maintain conversation continuity
- Tailor responses based on candidate's previous interactions

DELEGATION STRATEGY:
- For Tailor/Stitching requests:
  * IMMEDIATELY ask for image path when user mentions: "stitching", "tailor", "sewing", "fabric work"
  * Required response: "Great! I can help assess your stitching skills for the Tailor position. Please provide the path to your stitching image, like: /path/to/your/image.jpg"
  * NEVER say "Let's begin" without asking for the image first
  * Only delegate to stitching_assessor when actual image path is provided
  * Do NOT proceed with assessment until real image path is received
- For Loader Picker role: Delegate to label_reading_assessor (interactive quiz)  
- For Retail Sales role: Delegate to presentation_assessor (scenario evaluation)
- Continue delegating to same agent during ongoing assessments (check <assessment_status>)

IMAGE HANDLING:
- Never use placeholder paths like "path_to_your_image.jpg"
- Always wait for candidate to provide actual image path
- Look for image paths in candidate messages (e.g., /Users/name/image.jpg)
- Only proceed with image assessment when real path is provided

LABEL READING COORDINATION:
- Label reading agent will use share_label_image tool to provide current label image paths
- CRITICAL: Always check <current_label_image> state and include image path in ALL responses during label reading
- When presenting questions to user, ALWAYS format as: "Looking at [Image: path] - Question X/Y: [question]"
- Example: "Looking at [Image: label_dataset/samples/product_001.jpeg] - Question 1/9: What is the product name?"
- NEVER present label reading questions without the image path visible to the user
- Extract image path from <current_label_image> state and include it in every quiz interaction

EXAMPLES:
- If <applied_role> is "unknown": "Hello <candidate_name>! What kind of work interests you?"
- If <interaction_history> shows previous stitching interest: "I see you mentioned stitching before. Ready to upload your work for assessment?"
- If <assessment_history> shows completed skills: "Great job on your [skill] assessment! Let's move to the next skill."
- If <current_label_image> has path and quiz is active: "Looking at [Image: path] - sub-agent's question"

CRITICAL FORMATTING FOR LABEL READING:
- Always extract the image path from <current_label_image> state (it contains path and timestamp info)
- Access the path using <current_label_image>["path"] or similar
- Format ALL label reading questions as: "Looking at [Image: extracted_path] - Question X/Y: What is the [field]?"
- Example: If current_label_image contains path "label_dataset/samples/product_001.jpeg", show "Looking at [Image: label_dataset/samples/product_001.jpeg] - Question 1/9: What is the product name?"
- This ensures users can see which image they should be looking at for each question

KNOWLEDGE BASE:
Competency Map: {json.dumps(self.competency_map, indent=2)}
Sub-Agent Library: {json.dumps(self.sub_agent_library, indent=2)}
"""
        
        return Agent(
            name="master_job_assessor",
            model="gemini-2.5-flash",
            description="Enhanced stateful coordinator for multi-agent job assessment system",
            instruction=enhanced_instruction,
            sub_agents=[stitching_assessor, label_reading_assessor],
            tools=[get_candidate_profile, update_candidate_role]
        )
    
    async def create_candidate_session(self, candidate_name: str, 
                                     candidate_id: str = None, role: str = None) -> str:
        """Create new candidate session with comprehensive state"""
        if not candidate_id:
            candidate_id = str(uuid.uuid4())[:8]
        
        session_id = f"candidate_{candidate_id}"
        initial_state = create_initial_candidate_state(candidate_name, candidate_id, role)
        
        try:
            await self.session_service.create_session(
                app_name="job_assessment_app",
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
        """Retrieve candidate session"""
        try:
            session = await self.session_service.get_session(
                app_name="job_assessment_app",
                user_id=user_id,
                session_id=session_id
            )
            return session
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            return None
    
    async def process_candidate_interaction(self, session_id: str, user_id: str, 
                                          user_message: str, image_path: str = None) -> str:
        """Process candidate interaction with proper ADK state management"""
        try:
            # Add user query to interaction history
            await add_user_query_to_history(
                self.session_service, "job_assessment_app", user_id, session_id,
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
                self.session_service, "job_assessment_app", user_id, session_id,
                response_text or "Assessment completed"
            )
            
            return response_text or "Assessment completed successfully"
            
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return f"Error processing request: {str(e)}"
    
    async def get_assessment_summary(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive assessment summary"""
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
        """Calculate final verdict based on competency map and assessment results"""
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



class A2AServer:
    """A2A server for Agent-to-Agent communication using Flask"""
    
    def __init__(self, assessment_system: StatefulJobAssessmentSystem):
        self.assessment_system = assessment_system
        self.app = Flask(__name__)
        self.a2a_contexts: Dict[str, Dict] = {} 
        self.setup_routes()

    def setup_routes(self):
        """Setup all Flask routes for A2A communication"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy",
                "active_sessions": len(self.a2a_contexts),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route('/label-media/<path:filepath>', methods=['GET'])
        def serve_label_media(filepath: str):
            """Serve label images from the local dataset folder."""
            return send_from_directory('label_dataset', filepath)

        @self.app.route('/presentation-media/<path:filepath>', methods=['GET'])
        def serve_presentation_media(filepath: str):
            """Serve presentation resources for A2A file parts."""
            return send_from_directory('presentation_resources', filepath)

        @self.app.route('/.well-known/agent-card.json', methods=['GET'])
        def a2a_agent_card():
            """Return A2A Agent Card according to A2A protocol specification"""
            base_url = request.url_root.rstrip('/') + "/a2a/rpc"
            card = {
                "capabilities": {
                    "pushNotifications": False,
                    "streaming": False
                },
                "defaultInputModes": ["text/plain", "image/jpeg"],
                "defaultOutputModes": ["text/plain", "image/jpeg"],
                "description": "Multi-agent job assessment system that evaluates candidates for blue-collar roles including Tailor, Loader Picker, and Retail Sales positions through specialized skill assessments.",
                "name": "Job Assessment System",
                "preferredTransport": "JSONRPC",
                "protocolVersion": "0.3.0",
                "security": [{"apiKey": []}],
                "securitySchemes": {
                    "apiKey": {
                        "description": "API key authentication for job assessment system",
                        "in": "header",
                        "name": "X-API-Key",
                        "type": "apiKey"
                    }
                },
                "skills": [
                    {
                        "description": "Evaluates stitching quality and techniques from images for tailor positions",
                        "examples": ["Assess my stitching work for tailor role", "Evaluate this seam quality"],
                        "id": "stitching_assessment",
                        "name": "Stitching Assessment",
                        "tags": ["tailoring", "stitching", "craftsmanship", "quality-evaluation"]
                    },
                    {
                        "description": "Tests ability to read and extract information from product labels accurately",
                        "examples": ["Start label reading assessment", "Test my label reading skills"],
                        "id": "label_reading_assessment", 
                        "name": "Label Reading Assessment",
                        "tags": ["warehouse", "logistics", "label-reading", "information-extraction"]
                    },
                    {
                        "description": "Evaluates presentation, communication, and professional appearance skills",
                        "examples": ["Assess my presentation skills", "Evaluate my customer service approach"],
                        "id": "presentation_assessment",
                        "name": "Presentation Assessment", 
                        "tags": ["retail", "sales", "communication", "professional-appearance"]
                    }
                ],
                "url": base_url,
                "version": "1.0.0"
            }
            return jsonify(card)

        @self.app.route('/a2a/rpc', methods=['POST'])
        def a2a_rpc():
            """Main A2A JSON-RPC 2.0 endpoint"""
            # API key authentication
            expected_key = os.getenv('A2A_API_KEY')
            provided_key = request.headers.get('X-API-Key') or request.headers.get('x-api-key')
            if expected_key and provided_key != expected_key:
                return jsonify({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": "Unauthorized"}
                }), 401

            # Parse JSON-RPC request
            data = request.get_json(force=True, silent=True) or {}
            rpc_id = data.get("id")
            
            if data.get("jsonrpc") != "2.0" or "method" not in data:
                return jsonify({
                    "jsonrpc": "2.0", 
                    "id": rpc_id, 
                    "error": {"code": -32600, "message": "Invalid Request"}
                })

            method = data.get("method")
            params = data.get("params") or {}

            def ok(result):
                return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": result})

            def err(code, message):
                return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})

            # Handle message/send method
            if method == "message/send":
                return asyncio.run(self._handle_message_send(params, ok, err))
            
            return err(-32601, "Method not found")

    async def _handle_message_send(self, params: Dict, ok, err):
        """Handle A2A message/send requests"""
        try:
            message = params.get("message") or {}
            parts = message.get("parts") or []
            context_id = params.get("contextId") or f"a2a_{str(uuid.uuid4())}"
            language = params.get("language") or 'en-IN'

            # Extract text and image from message parts
            text, image_path = self._extract_text_and_image_from_parts(parts)
            
            logger.info(f"A2A message - Context: {context_id}, Text: '{text[:50]}...', Image: {bool(image_path)}")

            # Get or create A2A session context
            if context_id not in self.a2a_contexts:
                # Create new candidate session
                candidate_name = f"A2A_User_{str(uuid.uuid4())[:8]}"
                session_id = await self.assessment_system.create_candidate_session(candidate_name)
                user_id = session_id.split("_")[1]
                
                self.a2a_contexts[context_id] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "candidate_name": candidate_name,
                    "language": language
                }
            
            # Get session info
            session_info = self.a2a_contexts[context_id]
            
            # Process the interaction using the assessment system
            response_text = await self.assessment_system.process_candidate_interaction(
                session_info["session_id"],
                session_info["user_id"], 
                text or "",
                image_path
            )

            # Determine status and clean response text
            status = self._determine_response_status(response_text, session_info["session_id"], session_info["user_id"])
            
            # Remove status indicators from response text
            clean_response_text = response_text.replace("[STATUS:completed]", "").replace("[STATUS:input_required]", "").strip()

            # Format A2A response with proper parts (using cleaned text)
            response_parts = await self._format_a2a_response_parts(clean_response_text, context_id)

            return ok({
                "message": {
                    "role": "agent",
                    "parts": response_parts,
                    "messageId": str(uuid.uuid4())
                },
                "contextId": context_id,
                "status": {
                    "state": status,
                    "timestamp": datetime.now().isoformat()
                }
            })

        except Exception as e:
            logger.error(f"A2A message/send failed: {e}")
            return err(-32000, f"Server error: {str(e)}")

    def _extract_text_and_image_from_parts(self, parts) -> tuple[str, str | None]:
        """Extract text and image file path from A2A message parts"""
        text = ""
        image_path = None
        
        try:
            for i, part in enumerate(parts or []):
                if not isinstance(part, dict):
                    continue
                
                part_type = (part.get("type") or "").lower()
                
                if "textpart" in part_type or "text" in part:
                    if not text:
                        text = (part.get("text") or "").strip()
                
                # Handle FilePart - support multiple formats
                if "filepart" in part_type or "uri" in part or "inlineData" in part or "path" in part:
                    # Direct file path
                    if "path" in part:
                        file_path = part.get("path")
                        if file_path and os.path.exists(file_path):
                            image_path = file_path
                            logger.info(f"A2A: Using direct file path: {image_path}")
                    
                    # URI (file://, http://, data:)
                    elif "uri" in part:
                        uri = str(part.get("uri"))
                        if uri.startswith("file://"):
                            local_path = uri[7:]
                            if os.path.exists(local_path):
                                image_path = local_path
                        elif uri.startswith("http"):
                            # Download HTTP image to temp file
                            r = requests.get(uri, timeout=30)
                            r.raise_for_status()
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(r.content)
                                image_path = tmp.name
                        elif uri.startswith("data:"):
                            # Handle data URI
                            b64_data = uri.split(",", 1)[-1]
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(base64.b64decode(b64_data))
                                image_path = tmp.name
                    
                    # Inline data
                    elif "inlineData" in part:
                        inline = part.get("inlineData") or {}
                        b64_data = inline.get("data") or ""
                        if b64_data:
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(base64.b64decode(b64_data))
                                image_path = tmp.name

        except Exception as e:
            logger.error(f"Error extracting A2A parts: {e}")
        
        return (text or "").strip(), image_path

    async def _format_a2a_response_parts(self, response_text: str, context_id: str) -> List[Dict]:
        """Format response text into A2A parts with appropriate images"""
        parts = []
        
        try:
            # Check if this is a label reading question (contains image reference)
            has_question = "question" in response_text.lower() and "?" in response_text
            is_completion = "assessment completed" in response_text.lower() or "final score" in response_text.lower()
            
            # Add images for label reading questions
            if has_question and not is_completion and "Looking at" in response_text:
                # Extract image path from response text
                import re
                image_match = re.search(r'\[Image: ([^\]]+)\]', response_text)
                if image_match:
                    image_paths = [image_match.group(1)]
                    
                    # Check for multiple images in dataset
                    dataset_path = "label_dataset/index.json"
                    if os.path.exists(dataset_path):
                        with open(dataset_path, 'r') as f:
                            label_data = json.load(f)
                        
                        # Find matching label and get all images
                        for item in label_data:
                            if item.get('file_path') == image_paths[0] or image_paths[0] in item.get('file_paths', []):
                                if 'file_paths' in item:
                                    image_paths = item['file_paths']
                                break
                    
                    # Add all image file parts
                    base_url = request.url_root.rstrip('/')
                    for img_path in image_paths:
                        if img_path:
                            uri = f"{base_url}/{img_path.replace('label_dataset/', 'label-media/')}"
                            parts.append({
                                "type": "FilePart",
                                "mediaType": "image/jpeg", 
                                "uri": uri
                            })
            
            # Add text part
            parts.append({
                "type": "TextPart",
                "text": response_text
            })
            
            logger.info(f"A2A response formatted with {len(parts)} parts")
            
        except Exception as e:
            logger.error(f"Error formatting A2A response: {e}")
            parts = [{"type": "TextPart", "text": response_text}]
        
        return parts

    def _determine_response_status(self, response_text: str, session_id: str, user_id: str) -> str:
        """Extract status from agent response metadata"""
        if "[STATUS:completed]" in response_text:
            return "completed"
        elif "[STATUS:input_required]" in response_text:
            return "input_required"
        
        # Default to input_required if no status found
        return "input_required"

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask A2A server"""
        logger.info(f"Starting A2A Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)

def run_a2a_server(host='0.0.0.0', port=5000, debug=False):
    """Convenience function to run A2A server"""
    print("🎯 Initializing Job Assessment System with A2A Support")
    assessment_system = StatefulJobAssessmentSystem()
    
    print("🚀 Starting A2A Server...")
    a2a_server = A2AServer(assessment_system)
    a2a_server.run(host=host, port=port, debug=debug)



async def main():

    print("Enhanced Stateful Multi-Agent Job Assessment System")
    print("Built with Google ADK\n\n")
    
    # Initialize system
    assessment_system = StatefulJobAssessmentSystem()
    
    # Check if running as A2A server
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--a2a":
        print("Starting A2A Server Mode")
        a2a_server = A2AServer(assessment_system)
        a2a_server.run(debug=False)
        return
    
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
        print("\n The system will help identify your target role through natural conversation.")
        print("Type 'exit' to end the session, 'summary' for assessment summary.\n\n")

        
        # Start with a simple greeting
        print(f"\nSystem: Hello {candidate_name}! Welcome to our job assessment system. I'm here to help evaluate your skills for various positions. What kind of work are you interested in today?")
        
        while True:
            user_input = input(f"\n{candidate_name}: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'summary':
                summary = await assessment_system.get_assessment_summary(session_id, user_id)
                print("\nASSESSMENT SUMMARY:")
                print(json.dumps(summary, indent=2, default=str))
                continue
            

            image_path = None
            if '[' in user_input and ']' in user_input:
                start = user_input.find('[')
                end = user_input.find(']', start)
                if start != -1 and end != -1:
                    image_path = user_input[start+1:end].strip()
                    user_input = user_input[:start] + user_input[end+1:]
                    user_input = user_input.strip()
            
            
            if not image_path:
                path_patterns = [
                    r'(/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Unix paths
                    r'([A-Za-z]:[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Windows paths
                ]
                
                for pattern in path_patterns:
                    matches = re.findall(pattern, user_input, re.IGNORECASE)
                    if matches:
                        image_path = matches[0].strip()
                        # Clean the path from user input for processing
                        user_input = user_input.replace(image_path, '').strip()
                        break
            
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

if __name__ == "__main__":
    asyncio.run(main())