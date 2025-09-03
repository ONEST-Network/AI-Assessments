"""A2A server for Agent-to-Agent communication using Flask."""

import os
import json
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, send_from_directory

from ..config import Config
from ..utils import extract_text_and_image_from_parts
from ..assessment_system import StatefulJobAssessmentSystem

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A server for Agent-to-Agent communication using Flask."""
    
    def __init__(self, assessment_system: StatefulJobAssessmentSystem):
        self.assessment_system = assessment_system
        self.app = Flask(__name__)
        self.a2a_contexts: Dict[str, Dict] = {} 
        self.setup_routes()

    def setup_routes(self):
        """Setup all Flask routes for A2A communication."""
        
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
            """Return A2A Agent Card according to A2A protocol specification."""
            base_url = request.url_root.rstrip('/') + "/a2a/rpc"
            card = {
                "capabilities": {
                    "pushNotifications": False,
                    "streaming": False
                },
                "defaultInputModes": ["text/plain", "image/jpeg"],
                "defaultOutputModes": ["text/plain", "image/jpeg"],
                "description": ("Multi-agent job assessment system that evaluates candidates for "
                              "blue-collar roles including Tailor, Loader Picker, and Retail Sales "
                              "positions through specialized skill assessments."),
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
                        "description": ("Evaluates stitching quality and techniques from images "
                                      "for tailor positions"),
                        "examples": ["Assess my stitching work for tailor role", 
                                   "Evaluate this seam quality"],
                        "id": "stitching_assessment",
                        "name": "Stitching Assessment",
                        "tags": ["tailoring", "stitching", "craftsmanship", "quality-evaluation"]
                    },
                    {
                        "description": ("Tests ability to read and extract information from "
                                      "product labels accurately"),
                        "examples": ["Start label reading assessment", 
                                   "Test my label reading skills"],
                        "id": "label_reading_assessment", 
                        "name": "Label Reading Assessment",
                        "tags": ["warehouse", "logistics", "label-reading", "information-extraction"]
                    },
                    {
                        "description": ("Evaluates presentation, communication, and "
                                      "professional appearance skills"),
                        "examples": ["Assess my presentation skills", 
                                   "Evaluate my customer service approach"],
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
            """Main A2A JSON-RPC 2.0 endpoint."""
            # API key authentication
            expected_key = Config.A2A_API_KEY
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
                return jsonify({
                    "jsonrpc": "2.0", 
                    "id": rpc_id, 
                    "error": {"code": code, "message": message}
                })

            # Handle message/send method
            if method == "message/send":
                import asyncio
                return asyncio.run(self._handle_message_send(params, ok, err))
            
            return err(-32601, "Method not found")

    async def _handle_message_send(self, params: Dict, ok, err):
        """Handle A2A message/send requests."""
        try:
            message = params.get("message") or {}
            parts = message.get("parts") or []
            context_id = params.get("contextId") or f"a2a_{str(uuid.uuid4())}"
            language = params.get("language") or 'en-IN'

            # Extract text and image from message parts
            text, image_path = extract_text_and_image_from_parts(parts)
            
            logger.info(f"A2A message - Context: {context_id}, Text: '{text[:50]}...', "
                       f"Image: {bool(image_path)}")

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
            status = self._determine_response_status(
                response_text, session_info["session_id"], session_info["user_id"]
            )
            
            # Remove status indicators from response text
            clean_response_text = (response_text.replace("[STATUS:completed]", "")
                                 .replace("[STATUS:input_required]", "").strip())

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

    async def _format_a2a_response_parts(self, response_text: str, context_id: str) -> List[Dict]:
        """Format response text into A2A parts with appropriate images."""
        parts = []
        
        try:
            # Check if this is a label reading question (contains image reference)
            has_question = "question" in response_text.lower() and "?" in response_text
            is_completion = ("assessment completed" in response_text.lower() or 
                           "final score" in response_text.lower())
            
            # Add images for label reading questions
            if has_question and not is_completion and "Looking at" in response_text:
                # Extract image path from response text
                image_match = re.search(r'\[Image: ([^\]]+)\]', response_text)
                if image_match:
                    image_paths = [image_match.group(1)]
                    
                    # Check for multiple images in dataset
                    dataset_path = os.path.join(Config.LABEL_DATASET_DIR, "index.json")
                    if os.path.exists(dataset_path):
                        with open(dataset_path, 'r') as f:
                            label_data = json.load(f)
                        
                        # Find matching label and get all images
                        for item in label_data:
                            file_path = item.get('file_path')
                            file_paths = item.get('file_paths', [])
                            if (file_path == image_paths[0] or 
                                image_paths[0] in file_paths):
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
        """Extract status from agent response metadata."""
        if "[STATUS:completed]" in response_text:
            return "completed"
        elif "[STATUS:input_required]" in response_text:
            return "input_required"
        
        # Default to input_required if no status found
        return "input_required"

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the Flask A2A server."""
        host = host or Config.HOST
        port = port or Config.PORT
        debug = debug if debug is not None else Config.DEBUG
        
        logger.info(f"Starting A2A Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
