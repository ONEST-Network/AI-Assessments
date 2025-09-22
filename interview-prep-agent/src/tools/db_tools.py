"""SQLite database tools for persistent user progress tracking."""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Database path relative to the interview-prep-agent directory
DB_PATH = Path(__file__).resolve().parents[2] / "user_progress.db"
logger.info(f"Database path: {DB_PATH}")

def _init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT UNIQUE NOT NULL,
                role TEXT,
                graduation_year TEXT,
                training_info TEXT,
                work_experience TEXT,
                family_size TEXT,
                family_background TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_progress table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                skill TEXT NOT NULL,
                module_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed')),
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE (user_id, skill, module_id)
            )
        """)
        
        # Create user_sessions table to track current learning state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                current_skill TEXT,
                current_module TEXT,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE (user_id)
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def _get_or_create_user(phone_number: str) -> int:
    """Get existing user ID or create new user for the phone number."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        
        # Try to get existing user
        cursor.execute("SELECT id FROM users WHERE phone_number = ?", (phone_number,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Create new user
        cursor.execute(
            "INSERT INTO users (phone_number) VALUES (?)", 
            (phone_number,)
        )
        user_id = cursor.lastrowid
        conn.commit()
        
        logger.info(f"Created new user with ID {user_id} for phone {phone_number}")
        return user_id
        
    except Exception as e:
        logger.error(f"Failed to get/create user: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def save_module_progress(phone_number: str, skill: str, module_id: str, status: str) -> str:
    """Save module progress to database."""
    try:
        _init_database()
        user_id = _get_or_create_user(phone_number)
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            # Prepare timestamps
            now = datetime.now().isoformat()
            started_at = None
            completed_at = None
            
            if status == 'in_progress':
                # Check if we already have a started_at timestamp
                cursor.execute(
                    "SELECT started_at FROM user_progress WHERE user_id = ? AND skill = ? AND module_id = ?",
                    (user_id, skill, module_id)
                )
                result = cursor.fetchone()
                if result and result[0]:
                    started_at = result[0]
                else:
                    started_at = now
            elif status == 'completed':
                completed_at = now
            
            # Insert or update progress
            cursor.execute("""
                INSERT OR REPLACE INTO user_progress 
                (user_id, skill, module_id, status, started_at, completed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, skill, module_id, status, started_at, completed_at, now))
            
            # Update session tracking
            if status == 'in_progress':
                cursor.execute("""
                    INSERT OR REPLACE INTO user_sessions 
                    (user_id, current_skill, current_module, last_accessed)
                    VALUES (?, ?, ?, ?)
                """, (user_id, skill, module_id, now))
            elif status == 'completed':
                # Clear current module if completed
                cursor.execute("""
                    UPDATE user_sessions 
                    SET current_module = NULL, last_accessed = ?
                    WHERE user_id = ? AND current_skill = ? AND current_module = ?
                """, (now, user_id, skill, module_id))
            
            conn.commit()
            logger.info(f"Saved progress: user={phone_number}, skill={skill}, module={module_id}, status={status}")
            
            return json.dumps({
                "status": "ok", 
                "skill": skill, 
                "module_id": module_id, 
                "new_status": status,
                "saved_to_db": True
            })
            
        except Exception as e:
            logger.error(f"Database error saving progress: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to save module progress: {e}")
        return json.dumps({
            "error": "database_error",
            "message": str(e)
        })

def load_user_progress(phone_number: str) -> Dict[str, Any]:
    """Load user's complete progress from database."""
    try:
        _init_database()
        user_id = _get_or_create_user(phone_number)
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            # Load all progress
            cursor.execute("""
                SELECT skill, module_id, status, started_at, completed_at, updated_at
                FROM user_progress 
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
            
            progress = {}
            for row in cursor.fetchall():
                skill, module_id, status, started_at, completed_at, updated_at = row
                
                if skill not in progress:
                    progress[skill] = {}
                
                progress[skill][module_id] = {
                    "status": status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "updated_at": updated_at
                }
            
            # Load current session info
            cursor.execute("""
                SELECT current_skill, current_module, last_accessed
                FROM user_sessions 
                WHERE user_id = ?
            """, (user_id,))
            
            session_data = cursor.fetchone()
            current_info = {}
            if session_data:
                current_skill, current_module, last_accessed = session_data
                if current_skill:
                    current_info["current_skill"] = current_skill
                if current_module:
                    current_info["current_module"] = current_module
                if last_accessed:
                    current_info["last_accessed"] = last_accessed
            
            logger.info(f"Loaded progress for user {phone_number}: {len(progress)} skills")
            
            return {
                "progress": progress,
                "session": current_info
            }
            
        except Exception as e:
            logger.error(f"Database error loading progress: {e}")
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to load user progress: {e}")
        return {"progress": {}, "session": {}}

def save_user_role(phone_number: str, role: str) -> str:
    """Save user's role to database."""
    try:
        _init_database()
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Update user's role (create user if doesn't exist)
            cursor.execute("""
                INSERT INTO users (phone_number, role, created_at, updated_at) 
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phone_number) DO UPDATE SET
                role = excluded.role,
                updated_at = excluded.updated_at
            """, (phone_number, role, now, now))
            
            conn.commit()
            logger.info(f"Saved role '{role}' for user {phone_number}")
            
            return json.dumps({
                "status": "ok",
                "phone_number": phone_number,
                "role": role,
                "saved_to_db": True
            })
            
        except Exception as e:
            logger.error(f"Database error saving role: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to save user role: {e}")
        return json.dumps({
            "error": "database_error",
            "message": str(e)
        })

def get_user_role(phone_number: str) -> Optional[str]:
    """Get user's role from database."""
    try:
        _init_database()
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT role FROM users WHERE phone_number = ?", (phone_number,))
            result = cursor.fetchone()
            
            if result and result[0]:
                logger.info(f"Retrieved role '{result[0]}' for user {phone_number}")
                return result[0]
            else:
                logger.info(f"No role found for user {phone_number}")
                return None
                
        except Exception as e:
            logger.error(f"Database error getting role: {e}")
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to get user role: {e}")
        return None

def save_user_personal_info(phone_number: str, graduation_year: str = None, training_info: str = None, 
                           work_experience: str = None, family_size: str = None, family_background: str = None) -> str:
    """Save user's personal information to database."""
    try:
        _init_database()
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Get existing user data first
            cursor.execute("""
                SELECT graduation_year, training_info, work_experience, family_size, family_background 
                FROM users WHERE phone_number = ?
            """, (phone_number,))
            
            existing_data = cursor.fetchone()
            
            # Prepare update values (keep existing if new value is None)
            if existing_data:
                final_graduation = graduation_year if graduation_year is not None else existing_data[0]
                final_training = training_info if training_info is not None else existing_data[1]
                final_experience = work_experience if work_experience is not None else existing_data[2]
                final_family_size = family_size if family_size is not None else existing_data[3]
                final_family_bg = family_background if family_background is not None else existing_data[4]
            else:
                final_graduation = graduation_year
                final_training = training_info
                final_experience = work_experience
                final_family_size = family_size
                final_family_bg = family_background
            
            # Update user's personal info (create user if doesn't exist)
            cursor.execute("""
                INSERT INTO users (phone_number, graduation_year, training_info, work_experience, 
                                 family_size, family_background, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(phone_number) DO UPDATE SET
                graduation_year = excluded.graduation_year,
                training_info = excluded.training_info,
                work_experience = excluded.work_experience,
                family_size = excluded.family_size,
                family_background = excluded.family_background,
                updated_at = excluded.updated_at
            """, (phone_number, final_graduation, final_training, final_experience, 
                  final_family_size, final_family_bg, now, now))
            
            conn.commit()
            logger.info(f"Saved personal info for user {phone_number}")
            
            return json.dumps({
                "status": "ok",
                "phone_number": phone_number,
                "saved_fields": {
                    "graduation_year": final_graduation,
                    "training_info": final_training,
                    "work_experience": final_experience,
                    "family_size": final_family_size,
                    "family_background": final_family_bg
                },
                "saved_to_db": True
            })
            
        except Exception as e:
            logger.error(f"Database error saving personal info: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to save personal info: {e}")
        return json.dumps({
            "error": "database_error",
            "message": str(e)
        })

def get_user_personal_info(phone_number: str) -> Dict[str, Any]:
    """Get user's personal information from database."""
    try:
        _init_database()
        
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT graduation_year, training_info, work_experience, family_size, family_background
                FROM users WHERE phone_number = ?
            """, (phone_number,))
            
            result = cursor.fetchone()
            
            if result:
                graduation_year, training_info, work_experience, family_size, family_background = result
                
                personal_info = {
                    "graduation_year": graduation_year,
                    "training_info": training_info, 
                    "work_experience": work_experience,
                    "family_size": family_size,
                    "family_background": family_background,
                    "has_personal_info": any([graduation_year, training_info, work_experience, family_size, family_background])
                }
                
                logger.info(f"Retrieved personal info for user {phone_number}")
                return personal_info
            else:
                logger.info(f"No personal info found for user {phone_number}")
                return {
                    "graduation_year": None,
                    "training_info": None,
                    "work_experience": None,
                    "family_size": None,
                    "family_background": None,
                    "has_personal_info": False
                }
                
        except Exception as e:
            logger.error(f"Database error getting personal info: {e}")
            raise
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to get personal info: {e}")
        return {
            "graduation_year": None,
            "training_info": None,
            "work_experience": None,
            "family_size": None,
            "family_background": None,
            "has_personal_info": False
        }

def clean_agent_response(response: str) -> str:
    """Clean agent response by removing unwanted characters like asterisks."""
    if not response:
        return response
    
    # Remove standalone asterisks and asterisk formatting
    cleaned = response.replace("*", "")
    
    # Clean up any double spaces that might result
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    logger.debug(f"Cleaned response: '{response[:50]}...' -> '{cleaned[:50]}...'")
    return cleaned

# Initialize database on module import
try:
    _init_database()
except Exception as e:
    logger.error(f"Failed to initialize database on import: {e}")
