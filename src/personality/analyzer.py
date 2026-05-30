"""Message analyzer for context and emotional tone detection"""
import json
import re
import logging
from typing import Dict
from src.core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class MessageAnalyzer:
    """Analyze context and emotional tone of messages"""

    CATEGORIES = ['joke', 'empathy', 'casual', 'helpful', 'excited', 'skeptical', 'greeting', 'question', 'mature']
    TONES = ['neutral', 'positive', 'negative', 'sad', 'angry', 'excited', 'confused']
    TYPES = ['question', 'joke', 'complaint', 'help_request', 'conversation', 'news_sharing', 'greeting']

    def __init__(self):
        self.llm = LLMInterface()
        logger.info("Message analyzer initialized")

    def analyze(self, message: str, context: str = "") -> Dict[str, str]:
        """
        Full message analysis

        Returns:
            {
                'tone': emotional tone,
                'type': message type,
                'response_style': recommended response style,
                'complexity': simple/medium/complex
            }
        """
        # Simplified English prompt for faster tokenization
        analysis_prompt = f"""Analyze this message and return JSON only:

Message: "{message}"
Context: {context[:100]}

JSON format (NO explanations):
{{
  "tone": "neutral|positive|negative|sad|angry|excited|confused",
  "type": "question|joke|complaint|help_request|conversation|news_sharing|greeting",
  "response_style": "casual|empathy|helpful|joke|excited|skeptical|greeting|question|mature",
  "complexity": "simple|medium|complex"
}}

Use 'mature' style if message contains: sex, violence, politics, illegal topics, profanity.
Respond ONLY with valid JSON."""

        result = self.llm.analyze_short(analysis_prompt)

        try:
            # Clean up response artifacts
            # Remove </think> blocks (from reasoning models)
            cleaned = re.sub(r'</think>', '', result)
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            # Remove any leading/trailing whitespace
            cleaned = cleaned.strip()

            # Aggressive JSON extraction: find the LAST complete JSON object
            # This handles cases like: "{</think>{...}" or "{{\n..."
            # Try to find all potential JSON objects and parse the last valid one
            json_candidates = re.findall(r'\{[^{]*"tone".*?\}', cleaned, re.DOTALL)

            if json_candidates:
                # Try parsing from last to first candidate
                for candidate in reversed(json_candidates):
                    try:
                        analysis = json.loads(candidate)
                        # Validate response_style
                        if analysis.get('response_style') not in self.CATEGORIES:
                            analysis['response_style'] = 'casual'
                        logger.debug(f"Analysis result: {analysis}")
                        return analysis
                    except json.JSONDecodeError:
                        continue

        except (AttributeError, Exception) as e:
            logger.warning(f"Failed to parse analysis JSON: {e}. Raw: {result[:200]}")

        # Fallback to defaults
        default_analysis = {
            'tone': 'neutral',
            'type': 'conversation',
            'response_style': 'casual',
            'complexity': 'medium'
        }
        logger.debug(f"Using default analysis: {default_analysis}")
        return default_analysis

    def categorize_message(self, message: str) -> str:
        """Quick categorization of message for database storage"""
        prompt = f"""Определи категорию сообщения одним словом из списка:
{', '.join(self.CATEGORIES)}

Сообщение: "{message}"

Категория:"""

        result = self.llm.analyze_short(prompt).strip().lower()

        category = result if result in self.CATEGORIES else 'casual'
        logger.debug(f"Categorized '{message[:50]}...' as: {category}")
        return category
