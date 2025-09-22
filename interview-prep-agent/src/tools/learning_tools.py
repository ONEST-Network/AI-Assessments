"""Learning tools for skills/modules listing and in-memory progress tracking.

This simplified toolset assumes the LLM generates all teaching content. We only:
- read skills/modules metadata from assets/learning_modules.json
- track per-session module status in memory: pending | in_progress | completed
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from google.adk.tools import ToolContext

def _load_learning_kb() -> List[Dict[str, Any]]:
    # Resolve assets path relative to the package root to avoid cwd issues
    package_root = Path(__file__).resolve().parents[2]
    kb_path = package_root / 'assets' / 'learning_modules.json'
    with kb_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def list_skills(tool_context: ToolContext) -> str:
    """List available skills from the learning KB."""
    kb = _load_learning_kb()
    skills = [item['skill'] for item in kb]
    learning = tool_context.state.get('learning')
    if not isinstance(learning, dict):
        learning = {}
        tool_context.state['learning'] = learning
    learning['skills'] = skills
    return json.dumps({"skills": skills})

def list_modules(skill: str, tool_context: ToolContext) -> str:
    """List modules for a given skill, annotated with user's current status.

    Status per module: pending | in_progress | completed
    """
    kb = _load_learning_kb()
    modules: List[Dict[str, Any]] = []
    for item in kb:
        if item['skill'].lower() == skill.lower():
            modules = [
                {"id": m['id'], "title": m['title'], "summary": m.get('summary', '')}
                for m in item['modules']
            ]
            break

    # annotate with status from in-memory state
    learning = tool_context.state.get('learning')
    if not isinstance(learning, dict):
        learning = {}
        tool_context.state['learning'] = learning
    user_progress = learning.get('progress') if isinstance(learning.get('progress'), dict) else {}
    skill_progress: Dict[str, Any] = user_progress.get(skill, {}) if isinstance(user_progress.get(skill, {}), dict) else {}
    for m in modules:
        status = skill_progress.get(m['id'], {}).get('status', 'pending')
        m['status'] = status

    learning['current_skill'] = skill
    learning['available_modules'] = modules
    return json.dumps({"skill": skill, "modules": modules})

def set_module_status(skill: str, module_id: str, status: str, tool_context: ToolContext) -> str:
    """Set a module's status for the current session in memory.

    status: 'pending' | 'in_progress' | 'completed'
    """
    status = (status or '').strip().lower()
    if status not in {'pending', 'in_progress', 'completed'}:
        return json.dumps({"error": "invalid_status", "allowed": ['pending', 'in_progress', 'completed']})

    learning = tool_context.state.get('learning')
    if not isinstance(learning, dict):
        learning = {}
        tool_context.state['learning'] = learning

    progress = learning.get('progress')
    if not isinstance(progress, dict):
        progress = {}
        learning['progress'] = progress

    skill_map: Dict[str, Any] = progress.get(skill)
    if not isinstance(skill_map, dict):
        skill_map = {}
        progress[skill] = skill_map

    entry = skill_map.get(module_id)
    if not isinstance(entry, dict):
        entry = {}
        skill_map[module_id] = entry
    entry['status'] = status
    now = datetime.now().isoformat()
    if status == 'in_progress' and 'started_at' not in entry:
        entry['started_at'] = now
    if status == 'completed':
        entry['completed_at'] = now
    if status == 'pending':
        # Keep minimal trace but clear timestamps if present
        entry.pop('started_at', None)
        entry.pop('completed_at', None)

    # Maintain lightweight "current" pointers to help the LLM resume
    if status == 'in_progress':
        learning['current_skill'] = skill
        learning['current_module'] = module_id
    elif status in {'completed', 'pending'}:
        if learning.get('current_skill') == skill and learning.get('current_module') == module_id:
            # Clear current module if the same module is completed or reset
            learning.pop('current_module', None)

    return json.dumps({"status": "ok", "skill": skill, "module_id": module_id, "new_status": status})

def get_learning_progress(tool_context: ToolContext) -> str:
    """Return the user's in-memory progress across all skills/modules."""
    learning = tool_context.state.get('learning', {})
    return json.dumps(learning.get('progress', {}))
