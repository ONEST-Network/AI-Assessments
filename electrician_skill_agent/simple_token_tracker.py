"""
Simple token usage tracking - just print usage metadata and running totals.
"""

import logging

logger = logging.getLogger(__name__)

class SimpleTokenTracker:
    """Simple token tracker that just prints usage and keeps running totals."""
    
    def __init__(self):
        # Running totals
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_calls = 0
        
        # By model totals
        self.by_model = {}
    
    def _calculate_cost(self, model: str, prompt_tokens: int, candidates_tokens: int, has_images: bool = False) -> float:
        """Calculate cost based on current Gemini pricing."""
        
        # Current pricing (USD per million tokens)
        pricing = {
            "gemini-2.0-flash": {
                "input_text": 0.10,
                "input_image": 0.10,
                "output": 0.40
            },
            "gemini-2.5-flash": {
                "input_text": 0.075,
                "input_image": 0.075,
                "output": 0.30
            }
        }
        
        if model not in pricing:
            logger.warning(f"Unknown model {model}, using gemini-2.0-flash pricing")
            model = "gemini-2.0-flash"
        
        rates = pricing[model]
        
        # Calculate costs
        input_rate = rates["input_image"] if has_images else rates["input_text"]
        input_cost = (prompt_tokens / 1_000_000) * input_rate
        output_cost = (candidates_tokens / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
    
    def track_and_print(self, event, model: str, agent_type: str, user_id: str):
        """Extract ALL usage metadata fields, calculate cost, and print with running totals."""
        
        if not hasattr(event, 'usage_metadata') or not event.usage_metadata:
            print("No usage metadata in event")
            return
        
        usage_meta = event.usage_metadata
        
        # Extract ALL the specific fields from GenerateContentResponseUsageMetadata
        cache_tokens_details = getattr(usage_meta, 'cache_tokens_details', None)
        cached_content_token_count = getattr(usage_meta, 'cached_content_token_count', 0) or 0
        candidates_token_count = getattr(usage_meta, 'candidates_token_count', 0) or 0
        candidates_tokens_details = getattr(usage_meta, 'candidates_tokens_details', None)
        prompt_token_count = getattr(usage_meta, 'prompt_token_count', 0) or 0
        prompt_tokens_details = getattr(usage_meta, 'prompt_tokens_details', None)
        thoughts_token_count = getattr(usage_meta, 'thoughts_token_count', 0) or 0
        tool_use_prompt_token_count = getattr(usage_meta, 'tool_use_prompt_token_count', 0) or 0
        tool_use_prompt_tokens_details = getattr(usage_meta, 'tool_use_prompt_tokens_details', None)
        total_token_count = getattr(usage_meta, 'total_token_count', 0) or 0
        traffic_type = getattr(usage_meta, 'traffic_type', None)
        
        # Check for images (simple detection)
        has_images = False
        if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
            for part in event.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    if 'image' in getattr(part.inline_data, 'mime_type', ''):
                        has_images = True
                        break
        
        # Calculate cost
        cost = self._calculate_cost(model, prompt_token_count, candidates_token_count, has_images)
        
        # Update running totals
        self.total_prompt_tokens += prompt_token_count
        self.total_candidates_tokens += candidates_token_count
        self.total_tokens += total_token_count
        self.total_cost += cost
        self.total_calls += 1
        
        # Update by model
        if model not in self.by_model:
            self.by_model[model] = {
                'calls': 0,
                'prompt_tokens': 0,
                'candidates_tokens': 0,
                'total_tokens': 0,
                'cost': 0.0
            }
        
        self.by_model[model]['calls'] += 1
        self.by_model[model]['prompt_tokens'] += prompt_token_count
        self.by_model[model]['candidates_tokens'] += candidates_token_count
        self.by_model[model]['total_tokens'] += total_token_count
        self.by_model[model]['cost'] += cost
        
        # Print current call usage - ALL FIELDS
        print(f"\n🔢 Token Usage - {agent_type} ({model})")
        print(f"   User: {user_id}")
        print(f"   Prompt tokens: {prompt_token_count:,}")
        print(f"   Candidates tokens: {candidates_token_count:,}")
        print(f"   Total tokens: {total_token_count:,}")
        
        # Optional fields - only print if they have values
        if cached_content_token_count > 0:
            print(f"   Cached content tokens: {cached_content_token_count:,}")
        if thoughts_token_count > 0:
            print(f"   Thoughts tokens: {thoughts_token_count:,}")
        if tool_use_prompt_token_count > 0:
            print(f"   Tool use prompt tokens: {tool_use_prompt_token_count:,}")
        if traffic_type:
            print(f"   Traffic type: {traffic_type}")
        
        # Modality details - print if available
        if cache_tokens_details:
            print(f"   Cache tokens details: {cache_tokens_details}")
        if candidates_tokens_details:
            print(f"   Candidates tokens details: {candidates_tokens_details}")
        if prompt_tokens_details:
            print(f"   Prompt tokens details: {prompt_tokens_details}")
        if tool_use_prompt_tokens_details:
            print(f"   Tool use prompt tokens details: {tool_use_prompt_tokens_details}")
        
        print(f"   Has images: {has_images}")
        print(f"   Cost: ${cost:.6f}")
        
        # Print running aggregates
        print(f"\n📊 Running Totals (After {self.total_calls} calls)")
        print(f"   Total prompt tokens: {self.total_prompt_tokens:,}")
        print(f"   Total candidates tokens: {self.total_candidates_tokens:,}")
        print(f"   Total tokens: {self.total_tokens:,}")
        print(f"   Total cost: ${self.total_cost:.6f}")
        
        # Print by model breakdown
        if len(self.by_model) > 1:
            print(f"\n🧠 By Model:")
            for model_name, stats in self.by_model.items():
                print(f"   {model_name}: {stats['calls']} calls, {stats['total_tokens']:,} tokens, ${stats['cost']:.6f}")
        
        print("-" * 60)

# Global tracker instance
_tracker = SimpleTokenTracker()

def track_event_usage(event, model: str, agent_type: str, user_id: str):
    """Track and print token usage from event."""
    global _tracker
    _tracker.track_and_print(event, model, agent_type, user_id)

def get_totals():
    """Get current running totals."""
    global _tracker
    return {
        'total_calls': _tracker.total_calls,
        'total_prompt_tokens': _tracker.total_prompt_tokens,
        'total_candidates_tokens': _tracker.total_candidates_tokens,
        'total_tokens': _tracker.total_tokens,
        'total_cost': _tracker.total_cost,
        'by_model': _tracker.by_model.copy()
    }

def reset_totals():
    """Reset all running totals."""
    global _tracker
    _tracker = SimpleTokenTracker()
    print("✅ Token usage totals reset")
