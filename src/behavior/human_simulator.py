"""Human behavior simulation"""
import random
import asyncio
import re
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class HumanBehaviorSimulator:
    """Simulate natural human behavior in messaging"""

    @staticmethod
    def should_skip_message() -> bool:
        """Decide whether to skip a message (simulating being busy)"""
        skip = random.random() < Config.behavior.skip_message_prob
        if skip:
            logger.debug("Decided to skip message (simulating busy)")
        return skip

    @staticmethod
    def should_respond_immediately() -> bool:
        """Decide whether to respond immediately or with delay"""
        immediate = random.random() < Config.behavior.immediate_response_prob
        logger.debug(f"Response mode: {'immediate' if immediate else 'delayed'}")
        return immediate

    @staticmethod
    def get_busy_delay() -> float:
        """Get delay when 'busy'"""
        delay = random.uniform(
            Config.behavior.min_busy_delay,
            Config.behavior.max_busy_delay
        )
        logger.debug(f"Busy delay: {delay:.1f}s")
        return delay

    @staticmethod
    def get_typing_delay(text_length: int) -> float:
        """
        Calculate realistic typing delay

        Args:
            text_length: Text length in characters

        Returns:
            Delay in seconds
        """
        words = text_length / 5  # Approx chars per word
        wpm = Config.behavior.typing_speed_wpm

        base_delay = (words / wpm) * 60  # Base typing time
        variance = random.uniform(-0.3, 0.5) * base_delay  # ±30-50% variance

        total_delay = base_delay + variance

        final_delay = max(
            Config.behavior.min_typing_delay,
            min(total_delay, Config.behavior.max_typing_delay)
        )

        logger.debug(f"Typing delay for {text_length} chars: {final_delay:.1f}s")
        return final_delay

    @staticmethod
    async def simulate_reading(message_length: int):
        """Simulate reading a message"""
        reading_time = min(message_length * 0.05, 3.0)
        jitter = random.uniform(-0.2, 0.5)
        delay = max(0.5, reading_time + jitter)
        logger.debug(f"Reading delay: {delay:.1f}s")
        await asyncio.sleep(delay)

    @staticmethod
    def add_natural_imperfections(text: str) -> str:
        """
        Add natural elements to text

        - Replace words with colloquial variants
        - Rare typos
        - Abbreviations
        """
        if random.random() > Config.behavior.add_imperfections_prob:
            return text

        replacements = [
            (r'\bчто\b', 'чо'),
            (r'\bвообще\b', 'вобще'),
            (r'\bсейчас\b', 'щас'),
            (r'\bкороче\b', 'кароч'),
            (r'\bконечно\b', 'конечн'),
            (r'\bнаверное\b', 'наверн'),
            (r'\bпотому что\b', 'потому чо'),
            (r'\bкак будто\b', 'как бы'),
        ]

        # Apply random replacement
        pattern, replacement = random.choice(replacements)
        modified_text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

        if modified_text != text:
            logger.debug(f"Added imperfection: {pattern} -> {replacement}")

        return modified_text
