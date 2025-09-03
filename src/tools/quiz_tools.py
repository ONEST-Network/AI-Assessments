"""Quiz tools for label reading assessment."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def start_label_reading_quiz(tool_context: ToolContext) -> str:
    """Tool to start a new label reading quiz following ADK pattern."""
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
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(current_dir, "label_dataset", "index.json")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Label dataset not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            label_data = json.load(f)
        
        # Filter labels for warehouse/loader picker role
        relevant_labels = [
            item for item in label_data 
            if item.get('category') in ['warehouse', 'grocery', 'beverage', 'condiments']
        ]
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
                return (f"Quiz started. Looking at {image_display} - "
                       f"Question 1/{len(questions)}: {first_question['question']}")
            else:
                image_path = image_paths[0] if image_paths else "label_dataset/samples/product_001.jpeg"
                return (f"Quiz started. Looking at [Image: {image_path}] - "
                       f"Question 1/{len(questions)}: {first_question['question']}")
        else:
            return "Quiz started but no questions available"
            
    except Exception as e:
        logger.error(f"Error starting label reading quiz: {e}")
        return f"Error starting quiz: {str(e)}"


def answer_quiz_question(user_answer: str, tool_context: ToolContext) -> str:
    """Tool to answer a quiz question and get the next question."""
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
    """Tool for agent to update quiz score and get next question."""
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
                "message": (f"Quiz completed! Final score: {correct_answers}/{len(questions)} "
                           f"({accuracy:.1f}% accuracy)")
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
            image_display = (f"[Image: {image_paths[0]}]" if image_paths 
                           else "[Image: label_dataset/samples/default.jpeg]")
        
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
