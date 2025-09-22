"""Enhanced learning tools with SQLite database persistence and phone number integration."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from google.adk.tools import ToolContext

from .learning_tools import _load_learning_kb
from .db_tools import save_module_progress, load_user_progress, clean_agent_response, save_user_role, get_user_role, save_user_personal_info, get_user_personal_info

def _get_phone_number_from_context(tool_context: ToolContext) -> str:
    """Extract phone number from tool context for database operations."""
    # Try different ways to get the user identifier
    user_id = getattr(tool_context, 'user_id', None)
    
    # For WhatsApp integration, the user_id might be the phone number
    if user_id and (user_id.startswith('+') or user_id.startswith('whatsapp:')):
        phone_number = user_id.replace('whatsapp:', '').replace('+', '').strip()
        return f"+{phone_number}" if not phone_number.startswith('+') else phone_number
    
    # Fallback: check if it's stored in state
    phone_number = tool_context.state.get('phone_number') or tool_context.state.get('user_phone')
    if phone_number:
        return phone_number
    
    # Default fallback
    return user_id or "unknown_user"

def db_list_skills(tool_context: ToolContext) -> str:
    """List available skills from the learning KB (database-enhanced version)."""
    kb = _load_learning_kb()
    skills = [item['skill'] for item in kb]
    
    # Store phone number in state for subsequent calls
    phone_number = _get_phone_number_from_context(tool_context)
    tool_context.state['phone_number'] = phone_number
    
    # Load user's progress from database
    user_data = load_user_progress(phone_number)
    user_progress = user_data.get('progress', {})
    
    # Get user's role for context
    user_role = tool_context.state.get('user_role') or get_user_role(phone_number)
    
    # Categorize skills and add progress info
    skills_with_info = []
    for skill in skills:
        skill_info = {
            "name": skill,
            "modules_completed": 0,
            "modules_total": 0,
            "has_progress": False
        }
        
        # Check if user has progress in this skill
        if skill in user_progress:
            skill_progress = user_progress[skill]
            skill_info["modules_completed"] = sum(1 for m in skill_progress.values() if m.get('status') == 'completed')
            skill_info["modules_total"] = len(skill_progress)
            skill_info["has_progress"] = True
        
        # Add skill category context
        if skill == "General Interview Skills":
            skill_info["category"] = "general"
            skill_info["description"] = "Interview skills that work for any job"
        elif user_role and skill.lower() == user_role.lower():
            skill_info["category"] = "role_specific"
            skill_info["description"] = f"Technical skills for {user_role} role"
        else:
            skill_info["category"] = "other_role"
            skill_info["description"] = f"Technical skills for {skill} role"
        
        skills_with_info.append(skill_info)
    
    # Update tool context state with database data
    learning = tool_context.state.get('learning', {})
    learning.update({
        'skills': skills,
        'skills_with_info': skills_with_info,
        'progress': user_progress,
        'session': user_data.get('session', {}),
        'user_role': user_role
    })
    tool_context.state['learning'] = learning
    
    return json.dumps({
        "skills": skills,
        "skills_with_progress": skills_with_info,
        "user_role": user_role
    })

def db_list_modules(skill: str, tool_context: ToolContext) -> str:
    """List modules for a given skill with database-persisted status."""
    kb = _load_learning_kb()
    modules: List[Dict[str, Any]] = []
    for item in kb:
        if item['skill'].lower() == skill.lower():
            modules = [
                {"id": m['id'], "title": m['title'], "summary": m.get('summary', '')}
                for m in item['modules']
            ]
            break

    # Get phone number and load progress from database
    phone_number = _get_phone_number_from_context(tool_context)
    user_data = load_user_progress(phone_number)
    
    # Annotate modules with database status
    skill_progress = user_data.get('progress', {}).get(skill, {})
    for m in modules:
        module_data = skill_progress.get(m['id'], {})
        m['status'] = module_data.get('status', 'pending')
        if module_data.get('started_at'):
            m['started_at'] = module_data['started_at']
        if module_data.get('completed_at'):
            m['completed_at'] = module_data['completed_at']

    # Update tool context state
    learning = tool_context.state.get('learning', {})
    learning.update({
        'current_skill': skill,
        'available_modules': modules,
        'progress': user_data.get('progress', {}),
        'session': user_data.get('session', {})
    })
    tool_context.state['learning'] = learning
    
    return json.dumps({"skill": skill, "modules": modules})

def db_set_module_status(skill: str, module_id: str, status: str, tool_context: ToolContext) -> str:
    """Set a module's status with database persistence."""
    status = (status or '').strip().lower()
    if status not in {'pending', 'in_progress', 'completed'}:
        return json.dumps({"error": "invalid_status", "allowed": ['pending', 'in_progress', 'completed']})

    # Get phone number and save to database
    phone_number = _get_phone_number_from_context(tool_context)
    result = save_module_progress(phone_number, skill, module_id, status)
    
    # Also update in-memory state for immediate consistency
    learning = tool_context.state.get('learning', {})
    progress = learning.get('progress', {})
    
    if skill not in progress:
        progress[skill] = {}
    if module_id not in progress[skill]:
        progress[skill][module_id] = {}
    
    progress[skill][module_id]['status'] = status
    now = datetime.now().isoformat()
    
    if status == 'in_progress':
        if 'started_at' not in progress[skill][module_id]:
            progress[skill][module_id]['started_at'] = now
        learning['current_skill'] = skill
        learning['current_module'] = module_id
    elif status == 'completed':
        progress[skill][module_id]['completed_at'] = now
        if learning.get('current_skill') == skill and learning.get('current_module') == module_id:
            learning.pop('current_module', None)
    elif status == 'pending':
        progress[skill][module_id].pop('started_at', None)
        progress[skill][module_id].pop('completed_at', None)
        if learning.get('current_skill') == skill and learning.get('current_module') == module_id:
            learning.pop('current_module', None)
    
    learning['progress'] = progress
    tool_context.state['learning'] = learning
    
    return result

def db_get_learning_progress(tool_context: ToolContext) -> str:
    """Return the user's progress from database with current active module info and skill recommendations."""
    phone_number = _get_phone_number_from_context(tool_context)
    user_data = load_user_progress(phone_number)
    
    progress = user_data.get('progress', {})
    session = user_data.get('session', {})
    
    # Get user role for context
    user_role = tool_context.state.get('user_role') or get_user_role(phone_number)
    
    # Find current active module and skill
    current_skill = session.get('current_skill')
    current_module = session.get('current_module')
    
    # If no session data, find the most recent in_progress module
    if not current_skill or not current_module:
        for skill_name, skill_progress in progress.items():
            for module_id, module_data in skill_progress.items():
                if module_data.get('status') == 'in_progress':
                    current_skill = skill_name
                    current_module = module_id
                    break
            if current_skill and current_module:
                break
    
    # Analyze their progress and make recommendations
    has_general_progress = "General Interview Skills" in progress
    has_role_progress = user_role and user_role in progress
    skills_worked_on = list(progress.keys())
    
    # Determine user's learning status
    learning_status = "new"  # new, has_progress, multi_skill
    if len(skills_worked_on) == 0:
        learning_status = "new"
    elif len(skills_worked_on) == 1:
        learning_status = "single_skill"
    else:
        learning_status = "multi_skill"
    
    # Generate recommendations
    recommendations = []
    if current_skill and current_module:
        recommendations.append(f"Continue {current_module} in {current_skill}")
    
    if user_role and not has_role_progress:
        recommendations.append(f"Start {user_role} technical skills")
    
    if not has_general_progress:
        recommendations.append("Try General Interview Skills (works for any job)")
    
    if has_general_progress and has_role_progress:
        recommendations.append("Explore other skills or continue current progress")
    
    # Prepare summary for the agent
    summary = {
        "user_role": user_role,
        "learning_status": learning_status,
        "total_skills_worked_on": len(progress),
        "current_skill": current_skill,
        "current_module": current_module,
        "skills_worked_on": skills_worked_on,
        "recommendations": recommendations,
        "has_general_progress": has_general_progress,
        "has_role_progress": has_role_progress,
        "progress": progress
    }
    
    # Add skill-specific summaries
    for skill_name, skill_progress in progress.items():
        completed = sum(1 for m in skill_progress.values() if m.get('status') == 'completed')
        in_progress = sum(1 for m in skill_progress.values() if m.get('status') == 'in_progress')
        total_modules = len(skill_progress)
        
        summary[f"{skill_name.replace(' ', '_')}_summary"] = {
            "skill": skill_name,
            "completed": completed,
            "in_progress": in_progress,
            "total": total_modules,
            "status": "completed" if completed == total_modules else "in_progress" if in_progress > 0 else "not_started"
        }
    
    # Update tool context state
    learning = tool_context.state.get('learning', {})
    learning.update({
        'progress': progress,
        'session': session,
        'current_skill': current_skill,
        'current_module': current_module,
        'user_role': user_role,
        'recommendations': recommendations
    })
    tool_context.state['learning'] = learning
    
    return json.dumps(summary)

def save_progress_and_respond(message: str, tool_context: ToolContext) -> str:
    """Save current progress and return a cleaned response message."""
    # Clean the message first
    cleaned_message = clean_agent_response(message)
    
    # Get current learning state and ensure it's saved
    learning = tool_context.state.get('learning', {})
    phone_number = _get_phone_number_from_context(tool_context)
    
    # If there's a current skill/module in progress, make sure it's saved
    current_skill = learning.get('current_skill')
    current_module = learning.get('current_module')
    
    if current_skill and current_module:
        # Ensure the current module is marked as in_progress in database
        save_module_progress(phone_number, current_skill, current_module, 'in_progress')
    
    return json.dumps({
        "message": cleaned_message,
        "progress_saved": True,
        "phone_number": phone_number
    })

def set_user_role(role: str, tool_context: ToolContext) -> str:
    """Set and save the user's role/job title to avoid asking again."""
    phone_number = _get_phone_number_from_context(tool_context)
    
    # Save role to database
    result = save_user_role(phone_number, role.strip())
    
    # Also store in current context state for immediate use
    tool_context.state['user_role'] = role.strip()
    
    # Update learning context
    learning = tool_context.state.get('learning', {})
    learning['user_role'] = role.strip()
    tool_context.state['learning'] = learning
    
    return result

def get_user_role_tool(tool_context: ToolContext) -> str:
    """Get the user's previously saved role from database."""
    phone_number = _get_phone_number_from_context(tool_context)
    
    # Check if this is a new user session (just selected role in WhatsApp flow)
    is_new_user_session = tool_context.state.get('new_user_session', False)
    
    # First check current context state
    current_role = tool_context.state.get('user_role')
    if current_role:
        return json.dumps({
            "role": current_role,
            "source": "context_state",
            "is_new_user_session": is_new_user_session
        })
    
    # Check database
    saved_role = get_user_role(phone_number)
    if saved_role:
        # Store in context for future use in this session
        tool_context.state['user_role'] = saved_role
        learning = tool_context.state.get('learning', {})
        learning['user_role'] = saved_role
        tool_context.state['learning'] = learning
        
        return json.dumps({
            "role": saved_role,
            "source": "database",
            "is_new_user_session": is_new_user_session
        })
    
    # No role found
    return json.dumps({
        "role": None,
        "source": "not_found",
        "is_new_user_session": is_new_user_session
    })

def gather_basic_info(question_type: str, tool_context: ToolContext) -> str:
    """Ask basic questions to personalize responses. question_type: 'graduation', 'experience', 'family', 'all'"""
    phone_number = _get_phone_number_from_context(tool_context)
    
    questions = {
        "graduation": "When did you finish your training or school? (Example: 2022, 2023)",
        "experience": "Do you have any work experience? If yes, what kind of work?",
        "family": "How many people are in your family? What do your parents do for work?",
        "all": "Let me know about you:\n1. When did you finish training/school?\n2. Do you have work experience?\n3. How many people in your family?\n4. What do your parents do?"
    }
    
    question = questions.get(question_type, questions["all"])
    
    # Store that we're gathering info
    learning = tool_context.state.get('learning', {})
    learning['gathering_info'] = True
    learning['info_type'] = question_type
    tool_context.state['learning'] = learning
    
    return json.dumps({
        "question": question,
        "info_type": question_type,
        "instructions": "Ask this question to the user and wait for their answer before providing examples."
    })

def save_user_info(info_type: str, user_answer: str, tool_context: ToolContext) -> str:
    """Save user's basic information for personalized responses."""
    phone_number = _get_phone_number_from_context(tool_context)
    
    # Store in context for this session
    learning = tool_context.state.get('learning', {})
    user_info = learning.get('user_info', {})
    user_info[info_type] = user_answer
    learning['user_info'] = user_info
    tool_context.state['learning'] = learning
    
    # Also save to database based on info type
    kwargs = {}
    if info_type == "graduation" or info_type == "graduation_year":
        kwargs["graduation_year"] = user_answer
    elif info_type == "training":
        kwargs["training_info"] = user_answer
    elif info_type == "experience" or info_type == "work_experience":
        kwargs["work_experience"] = user_answer
    elif info_type == "family" or info_type == "family_size":
        kwargs["family_size"] = user_answer
    elif info_type == "family_background":
        kwargs["family_background"] = user_answer
    elif info_type == "all":
        # Parse comprehensive answer for all fields
        kwargs = {
            "graduation_year": user_answer,  # Store full answer for now
            "training_info": user_answer,
            "work_experience": user_answer,
            "family_size": user_answer
        }
    
    if kwargs:
        save_result = save_user_personal_info(phone_number, **kwargs)
    
    return json.dumps({
        "saved": True,
        "info_type": info_type,
        "answer": user_answer,
        "saved_to_db": bool(kwargs),
        "message": f"Got it! I saved your {info_type} information."
    })

def get_personal_info(tool_context: ToolContext) -> str:
    """Get user's personal information from database."""
    phone_number = _get_phone_number_from_context(tool_context)
    
    # Get personal info from database
    personal_info = get_user_personal_info(phone_number)
    
    # Also update context state for immediate use
    learning = tool_context.state.get('learning', {})
    learning['user_info'] = personal_info
    tool_context.state['learning'] = learning
    
    return json.dumps(personal_info)
