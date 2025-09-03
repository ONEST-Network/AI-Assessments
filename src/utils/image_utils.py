"""Image processing utilities."""

import os
import io
import logging
import tempfile
import base64
import requests
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


def validate_image(image_data: bytes, image_path: str = "unknown") -> dict:
    """Enhanced image validation with comprehensive checks."""
    try:
        if not image_data:
            raise ValueError("No image data provided")
            
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
        
        logger.info(f"Image validation successful: {validation_result}")
        return validation_result
        
    except Exception as e:
        error_msg = f"Image validation failed: {str(e)}"
        logger.error(error_msg)
        return {"valid": False, "error": str(e)}


def load_image_from_path(image_path: str) -> bytes:
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    logger.info(f"Retrieved image from path: {image_path}, size: {len(image_data)} bytes")
    return image_data


def extract_image_from_uri(uri: str) -> Optional[str]:
    """Extract image from various URI formats and save to temp file."""
    try:
        if uri.startswith("file://"):
            local_path = uri[7:]
            if os.path.exists(local_path):
                return local_path
        elif uri.startswith("http"):
            # Download HTTP image to temp file
            r = requests.get(uri, timeout=30)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                tmp.write(r.content)
                return tmp.name
        elif uri.startswith("data:"):
            # Handle data URI
            b64_data = uri.split(",", 1)[-1]
            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                tmp.write(base64.b64decode(b64_data))
                return tmp.name
    except Exception as e:
        logger.error(f"Error extracting image from URI {uri}: {e}")
    
    return None


def extract_text_and_image_from_parts(parts) -> Tuple[str, Optional[str]]:
    """Extract text and image file path from A2A message parts."""
    text = ""
    image_path = None
    
    try:
        for part in parts or []:
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
                    image_path = extract_image_from_uri(uri)
                
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
