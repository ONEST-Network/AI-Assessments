import asyncio
import logging
import random
import uuid
import json
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

generate_content_config = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=1200,
    top_p=0.95,
    seed=42
)

# In-memory storage for quiz sessions
_QUIZ_SESSIONS: Dict[str, Any] = {}

def _get_session_key(context_info: str = "default") -> str:
    return f"fitter_quiz_{context_info}"

def _normalize_answer(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = t.replace("option ", "").replace(".", "").replace("(", "").replace(")", "").strip()
    return t

def _run_coro_in_new_loop(coro, timeout: float = 30.0):
    import threading
    result: Dict[str, Any] = {"value": None, "error": None}
    done = threading.Event()

    def _target():
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            result["error"] = exc
        finally:
            if loop:
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

# -------- Scenario-Based Assessment (no images) --------

def start_scenario_assessment(tool_context: ToolContext) -> str:
    """
    Starts the scenario-based MCQ round for the fitter assessment.
    Selects ~5 questions and returns the first one.
    """
    session_key = _get_session_key()

    scenarios: List[Dict[str, Any]] = [
        {
            "question": "Loose bolt in a Machine",
            "options": {
                "a": "Ignore it until the next scheduled maintenance.",
                "b": "Tighten it immediately with the proper tool.",
                "c": "Hit it with a hammer.",
                "d": "Call the supervisor without checking."
            },
            "correct": "b"
        },
        {
            "question": "Pipe Joint Leaking Slightly",
            "options": {
                "a": "Apply sealant without shutting the system.",
                "b": "Shut down system pressure and then repair.",
                "c": "Wrap it with cloth temporarily.",
                "d": "Ignore if the leak is small."
            },
            "correct": "b"
        },
        {
            "question": "Fitting a Shaft",
            "options": {
                "a": "Use brute force with a hammer.",
                "b": "Align properly and use a mallet or press fit.",
                "c": "File the shaft down until it fits loosely.",
                "d": "Push it by hand even if misaligned."
            },
            "correct": "b"
        }
    ]

    total_available = len(scenarios)
    if total_available <= 0:
        return "No scenario questions configured. Please add scenarios and try again."
    pick = min(5, total_available)
    selected = random.sample(scenarios, pick)

    _QUIZ_SESSIONS[session_key] = {
        "assessment_type": "scenario",
        "questions": selected,
        "current_question_index": 0,
        "results": [],
        "score": 0
    }

    q = selected[0]
    return _format_scenario_question(q, 1, len(selected))

def _format_scenario_question(q: Dict[str, Any], idx: int, total: int) -> str:
    return (
        f"Scenario Assessment\n\nQuestion {idx} of {total}:\n\n"
        f"{q['question']}\n"
        f"A. {q['options']['a']}\n"
        f"B. {q['options']['b']}\n"
        f"C. {q['options']['c']}\n"
        f"D. {q['options']['d']}\n"
        f"(Reply with A, B, C, or D)"
    )

def score_scenario_answer(user_answer: str, tool_context: ToolContext) -> str:
    """
    Scores the user's answer for the current scenario question.
    On completion, automatically starts the knowledge assessment.
    """
    session_key = _get_session_key()
    if session_key not in _QUIZ_SESSIONS or _QUIZ_SESSIONS[session_key].get("assessment_type") != "scenario":
        return "Scenario assessment not started. Please start the assessment first."

    quiz_state = _QUIZ_SESSIONS[session_key]
    idx = quiz_state["current_question_index"]
    q = quiz_state["questions"][idx]

    ans = _normalize_answer(user_answer)
    is_correct = False
    if ans in ["a", "b", "c", "d"]:
        is_correct = (ans == q["correct"])
    else:
        for key, text in q["options"].items():
            if ans == _normalize_answer(text):
                is_correct = (key == q["correct"])
                break

    if is_correct:
        quiz_state["score"] += 1

    quiz_state["results"].append({
        "question": q["question"],
        "candidate_answer": user_answer,
        "is_correct": is_correct
    })
    quiz_state["current_question_index"] += 1

    if quiz_state["current_question_index"] < len(quiz_state["questions"]):
        next_q = quiz_state["questions"][quiz_state["current_question_index"]]
        _QUIZ_SESSIONS[session_key] = quiz_state
        return _format_scenario_question(next_q, quiz_state["current_question_index"] + 1, len(quiz_state["questions"]))
    else:
        scenario_results = {
            "scenario_results": quiz_state["results"],
            "scenario_score": quiz_state["score"]
        }
        _QUIZ_SESSIONS[session_key] = {
            "scenario_results": scenario_results
        }
        knowledge_start = start_knowledge_assessment(tool_context)
        return f"Scenario round completed. Your score is {scenario_results['scenario_score']} out of {len(scenario_results['scenario_results'])}.\n\n{knowledge_start}"

# -------- Knowledge Assessment --------

def start_knowledge_assessment(tool_context: ToolContext) -> str:
    """
    Starts the knowledge-based part of the fitter skill assessment by selecting 5 random questions.
    """
    session_key = _get_session_key()
    
    # Check if session exists from scenario assessment
    if session_key not in _QUIZ_SESSIONS:
        return "Session not found. Please start with the scenario assessment first."
    
    # Knowledge question pool
    knowledge_questions = [
        {"question": "Which tool is used to measure small dimensions accurately?", "answer": "Vernier caliper"},
        {"question": "What is a micrometer used for?", "answer": "Measuring thickness/diameter precisely"},
        {"question": "Which tool is used for tightening nuts and bolts?", "answer": "Spanner"},
        {"question": "What is the purpose of lubrication in machines?", "answer": "Reduce friction and wear"},
        {"question": "Which safety gear protects eyes while grinding?", "answer": "Goggles"},
        {"question": "Which tool would you use to cut a metal rod?", "answer": "Hacksaw"},
        {"question": "What does 'tolerance' mean in fitting work?", "answer": "Permissible variation in dimensions"},
        {"question": "Which instrument checks alignment of machine parts?", "answer": "Dial gauge"}
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

def _format_knowledge_question(q: Dict[str, Any], idx: int, total: int) -> str:
    return (
        f"Knowledge Assessment\n\nQuestion {idx} of {total}:\n\n"
        f"{q['question']}\n"
        f"(Reply with a short answer)"
    )

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
    You are an expert mechanical fitter instructor. Your task is to evaluate a candidate's answer to a knowledge-based question.
    
    Question: \"{question['question']}\"\n    Candidate's Answer: \"{user_answer}\"\n    Correct Answer: \"{question['answer']}\"\n
    Based on the candidate's answer, is it correct? The meaning should be the same, even if the wording is different.
    Respond with a single JSON object: {{'is_correct': boolean}}
    """

    async def get_llm_score():
        session_service = InMemorySessionService()
        runner = Runner(
            agent=fitter_skill_assessor,
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

        scenario = _QUIZ_SESSIONS.get(session_key, {}).get("scenario_results", {})
        scenario_score = scenario.get("scenario_score", 0)
        scenario_count = len(scenario.get("scenario_results", []))

        knowledge_score = knowledge_results["knowledge_score"]
        knowledge_count = len(knowledge_results["knowledge_questions_selected"])

        overall_total = scenario_score + knowledge_score
        overall_max = scenario_count + knowledge_count
        recommendation = "Hire" if overall_total >= max(7, overall_max - 2) else "Reject"

        # Clean up session
        del _QUIZ_SESSIONS[session_key]

        # Return human-readable summary instead of raw JSON
        summary = f"""🎯 **Assessment Complete!**

📊 **Your Results:**
• Scenario Round: {scenario_score}/{scenario_count}
• Knowledge Questions: {knowledge_score}/{knowledge_count}
• **Total Score: {overall_total}/{overall_max}**

🔍 **Recommendation: {recommendation}**

✅ You correctly answered {knowledge_score} out of {knowledge_count} knowledge questions.
🧩 You correctly solved {scenario_score} out of {scenario_count} scenario questions.

Thank you for completing the fitter skill assessment!"""
        return summary

fitter_skill_assessor = Agent(
    generate_content_config=generate_content_config,
    name="fitter_skill_assessor",
    model="gemini-2.0-flash",
    description="Assesses the skills of an entry-level mechanical fitter via scenario and knowledge MCQs.",
    instruction="""You are an AI agent that assesses the skills of an entry-level mechanical fitter using two rounds:
1) Scenario-Based MCQs (no images).
2) Knowledge-Based MCQs.

INTERACTION FLOW:
- FIRST TURN: You MUST call `start_scenario_assessment`.
- SCENARIOS: Present EXACTLY the text returned by tools (question and options A–D). Do NOT paraphrase, reword, or invent any scenario or options.
- ANSWERS: After the user replies, call `score_scenario_answer`.
- NEXT SCENARIO: If the tool returns another scenario, present it EXACTLY as returned.
- AFTER SCENARIOS: The system will automatically start the knowledge round by calling `start_knowledge_assessment`.
- KNOWLEDGE ROUND: Present EXACTLY the tool-returned question (free-text answer expected). Call `score_knowledge_answer` after each user answer.
- COMPLETION: When finished, present the returned summary EXACTLY as provided.

**SCORING MODE:**
- If the prompt starts with `[SCORING MODE]`, your only task is to evaluate the user's answer based on the provided context and return a single JSON object with the key "is_correct" and a boolean value.

STRICT OUTPUT STYLE:
- Do NOT mention internal logic or tool names.
- For scenarios, present options labeled A–D and ask for A/B/C/D only, mirroring tool text.
- Never synthesize your own questions or options; ONLY echo tool output.
- Present the final summary exactly as returned by the tools, without modification.
""",
    tools=[start_scenario_assessment, score_scenario_answer, start_knowledge_assessment, score_knowledge_answer],
)


