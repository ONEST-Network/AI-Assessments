
import os
import logging
from typing import Dict, List, Any

from google.adk.agents import Agent
from google.adk.tools import ToolContext
import json
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio
import uuid
import random
from pydantic import BaseModel, Field
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

generate_content_config = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=2000,
    top_p=0.95,
    seed=42
)

class PhotoIdentificationResult(BaseModel):
    """Schema for the photo identification task result."""
    question: str = Field(..., description="The question that was asked.")
    candidate_answer: str = Field(..., description="The candidate's answer.")
    is_correct: bool = Field(..., description="Whether the candidate's answer was correct.")
    image_path: str = Field(..., description="The path to the image for the question.")

class ElectricianAssessmentOutput(BaseModel):
    """Output schema for the Electrician Skill Assessment Agent."""
    photo_identification_results: List[PhotoIdentificationResult] = Field([], description="List of results for each photo identification question.")
    photo_identification_score: int = Field(0, description="The total score for the photo identification tasks.")
    final_message: str = Field("", description="A final message to the user, e.g., completion message.")

# In-memory storage for quiz sessions
_QUIZ_SESSIONS = {}

# Pre-analyzed image answers (populated at startup)
_PRE_ANALYZED_ANSWERS = {}

def _get_session_key(context_info: str = "default") -> str:
    return f"electrician_quiz_{context_info}"

def initialize_pre_analyzed_answers():
    """Pre-analyze images at startup to ensure consistent scoring."""
    global _PRE_ANALYZED_ANSWERS
    
    logger.info("Initializing pre-analyzed image answers...")
    
    # Find images in ElectricianAssessment folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    folder_path = os.path.join(project_root, 'ElectricianAssessment')
    cache_file = os.path.join(current_dir, 'image_analysis_cache.json')
    
    if not os.path.exists(folder_path):
        logger.error(f"ElectricianAssessment folder not found at {folder_path}")
        return {}
    
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Ensure consistent order
    except Exception as e:
        logger.error(f"Failed to list images in {folder_path}: {e}")
        return {}
    
    if not image_files:
        logger.error(f"No images found in {folder_path}")
        return {}
    
    # Try to load from cache first
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid (same images)
            cached_files = set(path.split('/')[-1] for path in cached_data.keys())
            current_files = set(image_files)
            
            if cached_files == current_files:
                _PRE_ANALYZED_ANSWERS.update(cached_data)
                logger.info(f"Loaded {len(_PRE_ANALYZED_ANSWERS)} pre-analyzed answers from cache")
                return _PRE_ANALYZED_ANSWERS
            else:
                logger.info("Cache is outdated (different images), will re-analyze")
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}, will re-analyze")
    
    logger.info(f"Found {len(image_files)} images to analyze: {image_files}")
    
    async def analyze_all_images():
        """Analyze all images and store results."""
        _PRE_ANALYZED_ANSWERS.clear()
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            relative_path = f"ElectricianAssessment/{img_file}"
            
            logger.info(f"Pre-analyzing image {i+1}/{len(image_files)}: {img_file}")
            
            try:
                # Analyze the image for safety
                result = await _analyze_image_safety(img_path)
                
                _PRE_ANALYZED_ANSWERS[relative_path] = {
                    "correct_answer": result.get("safety", "unknown"),
                    "explanation": result.get("explanation", ""),
                    "question": "Look at the image. Is the practice shown safe or unsafe?"
                }
                
                logger.info(f"Analysis result for {img_file}: {result.get('safety', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {img_file}: {e}")
                _PRE_ANALYZED_ANSWERS[relative_path] = {
                    "correct_answer": "unknown",
                    "explanation": "Analysis failed",
                    "question": "Look at the image. Is the practice shown safe or unsafe?"
                }
    
    # Run the analysis
    try:
        _run_coro_in_new_loop(analyze_all_images())
        logger.info(f"Pre-analyzed {len(_PRE_ANALYZED_ANSWERS)} images successfully")
        
        # Save results to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(_PRE_ANALYZED_ANSWERS, f, indent=2)
            logger.info(f"Saved analysis results to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        
        # Log the results for verification
        for path, data in _PRE_ANALYZED_ANSWERS.items():
            logger.info(f"  {path}: {data['correct_answer']} - {data['explanation']}")
            
    except Exception as e:
        logger.error(f"Failed to pre-analyze images: {e}")
        return {}
    
    return _PRE_ANALYZED_ANSWERS



import io

async def _analyze_image_safety(image_path: str) -> Dict[str, str]:
    """Analyzes an image to determine if it depicts a safe or unsafe electrical practice using the agent itself."""
    analysis_prompt = """
    [ANALYSIS MODE]
    Analyze the given image and determine if the practice shown is safe or unsafe.
    """
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=electrician_skill_assessor,
        app_name="analysis_session",
        session_service=session_service
    )
    session_id = f"analyze_{uuid.uuid4()}"
    await session_service.create_session(
        app_name="analysis_session",
        user_id="analyzer",
        session_id=session_id,
    )

    img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    image_blob = types.Blob(mime_type='image/png', data=img_byte_arr)
    image_part = types.Part(inline_data=image_blob)

    content = types.Content(role='user', parts=[types.Part(text=analysis_prompt), image_part])
    
    response_text = ""
    async for event in runner.run_async(user_id="analyzer", session_id=session_id, new_message=content):
        if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text = part.text.strip()
                    break
            if response_text:
                break
    
    raw_json = _extract_json(response_text)
    return json.loads(raw_json)

def start_photo_assessment(tool_context: ToolContext, folder_path: str = "ElectricianAssessment/") -> str:
    """
    Starts the photo identification part of the electrician skill assessment using pre-analyzed image data.
    """
    session_key = _get_session_key()
    
    # Use pre-analyzed answers to create questions
    if not _PRE_ANALYZED_ANSWERS:
        return json.dumps({"error": "No pre-analyzed image data available. Server may need restart."})
    
    # Convert pre-analyzed data to question format
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    photo_questions = []
    for relative_path, data in _PRE_ANALYZED_ANSWERS.items():
        absolute_path = os.path.join(project_root, relative_path)
        photo_questions.append({
            "image_path": absolute_path,
            "question": data["question"],
            "correct_answer": data["correct_answer"],
            "explanation": data["explanation"]
        })
    
    # Sort to ensure consistent order
    photo_questions.sort(key=lambda x: x["image_path"])
    
    quiz_state = {
        "questions": photo_questions,
        "current_question_index": 0,
        "results": [],
        "score": 0
    }
    _QUIZ_SESSIONS[session_key] = quiz_state
    
    logger.info(f"Started photo assessment with {len(photo_questions)} pre-analyzed questions")
        
    first_question = quiz_state["questions"][0]
    
    response = {
        "image_path": os.path.abspath(first_question["image_path"])
    }
    
    return json.dumps(response)

def _run_coro_in_new_loop(coro, timeout: float = 30.0):
    import threading
    result: Dict[str, Any] = {"value": None, "error": None}
    done = threading.Event()

    def _target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            result["error"] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    finished = done.wait(timeout)
    if not finished:
        raise TimeoutError("LLM scoring timed out")
    if result["error"]:
        raise result["error"]
    return result["value"]

def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove leading and trailing code fences
        text = text.strip('`')
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return text[start:end]
    except ValueError:
        return text

def score_photo_answer(user_answer: str, tool_context: ToolContext) -> str:
    """
    Scores the user's answer for the current photo identification question using pre-analyzed answers.
    """
    session_key = _get_session_key()
    if session_key not in _QUIZ_SESSIONS:
        return "Assessment not started. Please start the assessment first."

    quiz_state = _QUIZ_SESSIONS[session_key]
    current_index = quiz_state["current_question_index"]
    question = quiz_state["questions"][current_index]

    # Get the correct answer from pre-analyzed data
    image_path = question["image_path"]
    if image_path not in _PRE_ANALYZED_ANSWERS:
        logger.error(f"No pre-analyzed answer found for {image_path}")
        # Fallback to simple comparison
        is_correct = user_answer.lower().strip() == question.get("correct_answer", "").lower().strip()
    else:
        correct_answer = _PRE_ANALYZED_ANSWERS[image_path]["correct_answer"]
        # Simple but reliable comparison
        user_normalized = user_answer.lower().strip()
        correct_normalized = correct_answer.lower().strip()
        is_correct = user_normalized == correct_normalized
        
        logger.info(f"Scoring image {image_path}: user='{user_normalized}' correct='{correct_normalized}' result={is_correct}")

    # Update quiz state

    if is_correct:
        quiz_state["score"] += 1

    quiz_state["results"].append({
        "question": question["question"],
        "candidate_answer": user_answer,
        "is_correct": is_correct,
        "image_path": os.path.abspath(question["image_path"])
    })

    quiz_state["current_question_index"] += 1
    _QUIZ_SESSIONS[session_key] = quiz_state

    if quiz_state["current_question_index"] < len(quiz_state["questions"]):
        next_question = quiz_state["questions"][quiz_state["current_question_index"]]
        response = {
            "image_path": os.path.abspath(next_question["image_path"])
        }
        return json.dumps(response)
    else:
        photo_results = {
            "photo_identification_results": quiz_state["results"],
            "photo_identification_score": quiz_state["score"],
        }
        # Store photo results in the session
        _QUIZ_SESSIONS[session_key]["photo_assessment_results"] = photo_results
        # Don't delete the session, just the quiz-specific parts
        del _QUIZ_SESSIONS[session_key]["questions"]
        del _QUIZ_SESSIONS[session_key]["current_question_index"]
        del _QUIZ_SESSIONS[session_key]["results"]
        del _QUIZ_SESSIONS[session_key]["score"]
        del _QUIZ_SESSIONS[session_key]["assessment_type"]

        # Auto-start knowledge assessment as per instructions
        knowledge_start_result = start_knowledge_assessment(tool_context)
        return f"Photo identification task completed. Your score is {photo_results['photo_identification_score']} out of {len(photo_results['photo_identification_results'])}.\n\n{knowledge_start_result}"


def start_knowledge_assessment(tool_context: ToolContext) -> str:
    """
    Starts the knowledge-based part of the electrician skill assessment by selecting 5 random questions.
    """
    session_key = _get_session_key()
    
    # Check if session exists from photo assessment
    if session_key not in _QUIZ_SESSIONS:
        return "Session not found. Please start with the photo assessment first."
    
    # Knowledge question pool
    knowledge_questions = [
        {"question": "Which tool is used to check current flow?", "answer": "Multimeter/Tester"},
        {"question": "What is the color coding for Earth wire in India?", "answer": "Green/Green-Yellow"},
        {"question": "Safe voltage range for household appliances.", "answer": "220–240V"},
        {"question": "Which tool is used to check whether current is flowing in a wire?", "answer": "Multimeter"},
        {"question": "Which tool is commonly used to strip insulation from wires?", "answer": "Wire stripper"},
        {"question": "Which instrument is used to measure voltage, current, and resistance?", "answer": "Multimeter"},
        {"question": "What is the standard color of the earth wire in India?", "answer": "Green"},
        {"question": "What color is typically used for the neutral wire in India?", "answer": "Blue"},
        {"question": "In India, which color is usually used for the live wire?", "answer": "Red or Brown"},
        {"question": "Before working on an electrical circuit, the first step should be:", "answer": "Switch off the main power"},
        {"question": "Why should electricians use insulated tools?", "answer": "They prevent electric shock"},
        {"question": "The standard household supply voltage in India is:", "answer": "220–240V"},
        {"question": "A fuse is used to:", "answer": "Break the circuit during overload"},
        {"question": "Which device automatically trips in case of overload or short circuit?", "answer": "Circuit Breaker / MCB"},
        {"question": "If an appliance repeatedly trips the breaker, the correct action is:", "answer": "Check for faulty wiring or overload"},
        {"question": "What does the acronym \"MCB\" stand for?", "answer": "Miniature Circuit Breaker"},
        {"question": "What is the purpose of grounding in electrical systems?", "answer": "To provide a safe path for fault current"},
        {"question": "What is the function of a Residual Current Device (RCD)?", "answer": "To quickly disconnect a circuit when leakage current is detected"},
        {"question": "What is used to measure electrical power?", "answer": "Wattmeter"},
        {"question": "What is the unit of electrical resistance?", "answer": "Ohm"},
        {"question": "Which fire extinguisher is used for fire in electrical equipment?", "answer": "Halon"},
        {"question": "Which personal protective equipment is used for protection from smoke?", "answer": "Nasal mask"},
        {"question": "What immediate steps should be taken to save the victim if he is still in contact with the electric power source?", "answer": "Switch off the power supply"},
        {"question": "What is the use of pincers?", "answer": "Removing pins and nails from wood"},
        {"question": "What is the waste disposal method that saves a lot of energy?", "answer": "Recycling"},
        {"question": "What is the colour of the back side of the warning symbol in the original category?", "answer": "Yellow"},
        {"question": "What is the use of mortise chisel?", "answer": "Drilling rectangular holes in wood"},
        {"question": "Which type of soldering flux is used for soldering aluminum conductors?", "answer": "Ker-al-Light"},
        {"question": "What is the effect of electric current in neon lamp called?", "answer": "gas ionization effect"},
        {"question": "Which type of wire connections are found in a junction box?", "answer": "Rat tail joint"},
        {"question": "What is the unit of insulator resistance is", "answer": "Mega ohm"},
        {"question": "What is the full form of XLPE?", "answer": "Cross Linked Poly Ethylene"},
        {"question": "What is the use of serving layer in underground cable?", "answer": "Protects the armouring from atmospheric conditions"},
        {"question": "How many electrons are there in the bond cycle of a copper atom?", "answer": "1"},
        {"question": "Why should a soldering iron be kept on a stand when not in use?", "answer": "It prevents burning and fire"},
        {"question": "Which type of soldering flux is used for galvanized iron?", "answer": "Hydrochloric acid"},
        {"question": "What type of cabling method is used in production plants?", "answer": "Rack in the air"}
    ]
    
    # Select 5 random questions
    selected_questions = random.sample(knowledge_questions, 5)
    
    # Set up quiz state for knowledge assessment
    quiz_state = _QUIZ_SESSIONS[session_key]
    quiz_state.update({
        "assessment_type": "knowledge",
        "questions": selected_questions,
        "current_question_index": 0,
        "results": [],
        "score": 0
    })
    _QUIZ_SESSIONS[session_key] = quiz_state
    
    # Return the first question
    first_question = selected_questions[0]
    return f"Knowledge Assessment Started\n\nQuestion 1 of 5:\n\n{first_question['question']}"


def score_knowledge_answer(user_answer: str, tool_context: ToolContext) -> str:
    """
    Scores the user's answer for the current knowledge-based question using the LLM.
    """
    session_key = _get_session_key()
    if session_key not in _QUIZ_SESSIONS or _QUIZ_SESSIONS[session_key].get("assessment_type") != "knowledge":
        return "Knowledge assessment not started. Please start the assessment first."

    quiz_state = _QUIZ_SESSIONS[session_key]
    current_index = quiz_state["current_question_index"]
    question = quiz_state["questions"][current_index]

    scoring_prompt = f"""
    [SCORING MODE]
    You are an expert electrician instructor. Your task is to evaluate a candidate's answer to a knowledge-based question.
    
    Question: \"{question['question']}\"\n    Candidate's Answer: \"{user_answer}\"\n    Correct Answer: \"{question['answer']}\"\n
    Based on the candidate's answer, is it correct? The meaning should be the same, even if the wording is different.
    Respond with a single JSON object: {{'is_correct': boolean}}
    """

    async def get_llm_score():
        session_service = InMemorySessionService()
        runner = Runner(
            agent=electrician_skill_assessor,
            app_name="scoring_session",
            session_service=session_service
        )
        session_id = f"score_{uuid.uuid4()}"
        await session_service.create_session(
            app_name="scoring_session",
            user_id="scorer",
            session_id=session_id,
        )

        content = types.Content(role='user', parts=[types.Part(text=scoring_prompt)])
        response_text = ""
        async for event in runner.run_async(user_id="scorer", session_id=session_id, new_message=content):
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text = part.text.strip()
                        break
                if response_text:
                    break
        return response_text

    try:
        # Add a small delay to avoid rapid API calls
        import time
        time.sleep(2)
        llm_response = _run_coro_in_new_loop(get_llm_score(), timeout=30.0)
        logger.info(f"LLM scoring response: {llm_response}")
        raw_json = _extract_json(llm_response)
        scoring_data = json.loads(raw_json)
        is_correct = scoring_data.get("is_correct", False)
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}. Falling back to simple comparison.")
        is_correct = user_answer.lower() in question["answer"].lower()

    if is_correct:
        quiz_state["score"] += 1

    quiz_state["results"].append({
        "question": question["question"],
        "candidate_answer": user_answer,
        "correct_answer": question["answer"],
        "is_correct": is_correct
    })

    quiz_state["current_question_index"] += 1
    _QUIZ_SESSIONS[session_key] = quiz_state

    if quiz_state["current_question_index"] < len(quiz_state["questions"]):
        next_question = quiz_state["questions"][quiz_state["current_question_index"]]
        return f"Next question:\n\n{next_question['question']}"
    else:
        knowledge_results = {
            "knowledge_questions_selected": quiz_state["results"],
            "knowledge_score": quiz_state["score"],
        }

        photo_results = _QUIZ_SESSIONS[session_key].get("photo_assessment_results", {})

        overall_total = photo_results.get("photo_identification_score", 0) + knowledge_results.get("knowledge_score", 0)

        recommendation = "Reject"
        if overall_total >= 7:
            recommendation = "Hire"

        final_output = {
            "Photo_Identification_Score": photo_results.get("photo_identification_score", 0),
            "Scenario_Score": 0, # Not implemented yet
            "Knowledge_Questions_Selected": knowledge_results["knowledge_questions_selected"],
            "Knowledge_Score": knowledge_results["knowledge_score"],
            "Overall_Total": overall_total,
            "Recommendation": recommendation,
            "Comments": "Assessment complete."
        }

        # Clean up session
        del _QUIZ_SESSIONS[session_key]

        # Return human-readable summary instead of raw JSON
        photo_score = photo_results.get("photo_identification_score", 0)
        knowledge_score = knowledge_results["knowledge_score"]
        
        summary = f"""🎯 **Assessment Complete!**

📊 **Your Results:**
• Photo Identification: {photo_score}/4
• Knowledge Questions: {knowledge_score}/5
• **Total Score: {overall_total}/9**

🔍 **Recommendation: {recommendation}**

✅ You correctly answered {knowledge_score} out of 5 knowledge questions.
📸 You correctly identified {photo_score} out of 4 photo identification questions.

Thank you for completing the electrician skill assessment!"""

        return summary

electrician_skill_assessor = Agent(
    generate_content_config=generate_content_config,
    name="electrician_skill_assessor",
    model="gemini-2.0-flash",
    description="An agent that assesses the skills of an entry-level electrician through an interactive quiz.",
    instruction="""You are an AI agent designed to assess the skills of an entry-level electrician through an interactive quiz.

**INTERACTION FLOW:**

**1. Photo Identification Task:**
- The user will start the assessment.
- You will call `start_photo_assessment` to begin the photo quiz.
- For each question, present the question and the image to the user.
- The user will provide an answer (e.g., "safe", "unsafe").
- You will call `score_photo_answer` with the user's answer.
- When the photo quiz is complete, the tool will return a JSON object with the results. You MUST present these results to the user.

**2. Knowledge-Based MCQ Task:**
- After the photo quiz is complete, the system will automatically start the knowledge-based quiz.
- You will present the questions to the user one by one.
- The user will provide an answer.
- You will call `score_knowledge_answer` with the user's answer.
- When the knowledge quiz is complete, the tool will return a formatted assessment summary. You MUST present this summary exactly as returned by the tool, without adding or modifying any content.

**3. Scoring Mode:**
- If the prompt starts with `[SCORING MODE]`, your only task is to evaluate the user's answer based on the provided context and return a single JSON object with the key "is_correct" and a boolean value.

**4. Analysis Mode:**
- If the prompt starts with `[ANALYSIS MODE]`, your task is to analyze the given image from the perspective of an expert electrician and safety inspector. Determine if the practice shown is safe or unsafe, and provide an explanation. Return a single JSON object with two keys: "safety" (either "safe" or "unsafe") and "explanation".

Do not add any conversational fluff. Stick to the assessment.

STRICT OUTPUT STYLE:
- Do NOT mention folder names, file paths, or any internal processing steps.
- Do NOT say anything like "I need a folder path" or "I will use initial_photos".
- When starting, directly present the first question with the image (no setup chatter).
- Do NOT include Markdown or HTML image tags; the system will handle sending images.
- When tools return formatted responses (especially final assessment summaries), present them EXACTLY as returned without modification.
- NEVER replace detailed tool responses with your own summary text.
""",
    tools=[start_photo_assessment, score_photo_answer, start_knowledge_assessment, score_knowledge_answer],
)

# Initialize pre-analyzed answers at module import
try:
    initialize_pre_analyzed_answers()
    logger.info("Pre-analyzed image answers loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize pre-analyzed answers: {e}")
    # Don't fail completely, just log the error
    _PRE_ANALYZED_ANSWERS = {}
