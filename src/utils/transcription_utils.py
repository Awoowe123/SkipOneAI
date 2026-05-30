from __future__ import annotations

import json
import os
import re
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Global variable for blacklist
BLACKLIST = []
BLACKLIST_FILE = os.path.join("data", "blacklist.json")

def load_blacklist(default_blacklist=None):
    """Loads blacklist from file"""
    global BLACKLIST

    # Default values
    if default_blacklist is None:
        default_blacklist = [
            "Субтитры сделал DimaTorzok",
            "Cубтитры сделал DimaTorzok",
            "Субтитры сделал",
            "DimaTorzok",
            "Редактор субтитров",
            "Создано с помощью",
        ]

    # Ensure directory exists
    os.makedirs(os.path.dirname(BLACKLIST_FILE), exist_ok=True)

    try:
        if os.path.exists(BLACKLIST_FILE):
            if os.path.getsize(BLACKLIST_FILE) == 0:
                BLACKLIST = default_blacklist
                save_blacklist()
                return

            with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
                BLACKLIST = json.load(f)

            if not isinstance(BLACKLIST, list):
                BLACKLIST = default_blacklist
                save_blacklist()
        else:
            BLACKLIST = default_blacklist
            save_blacklist()

    except Exception as e:
        logger.error(f"Error loading blacklist: {e}")
        BLACKLIST = default_blacklist

def save_blacklist():
    """Saves blacklist to file"""
    try:
        with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(BLACKLIST, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving blacklist: {e}")
        return False

def filter_blacklisted_phrases(text):
    """Removes blacklisted phrases from text"""
    if not text or not BLACKLIST:
        return text

    filtered_text = text
    for phrase in BLACKLIST:
        if phrase and phrase.strip():
            # Remove phrase with newline before
            filtered_text = re.sub(r"\n\s*" + re.escape(phrase), "", filtered_text, flags=re.IGNORECASE)
            # Remove phrase with newline after
            filtered_text = re.sub(re.escape(phrase) + r"\s*\n", "", filtered_text, flags=re.IGNORECASE)
            # Remove phrase anywhere
            filtered_text = re.sub(re.escape(phrase), "", filtered_text, flags=re.IGNORECASE)

    # Clean up multiple newlines
    filtered_text = re.sub(r"\n\s*\n\s*\n", "\n\n", filtered_text)
    return filtered_text

def cleanup_transcribed_text(text: str, language: str | None = None) -> str:
    """Removes leaked prompts and limits excessive repetitions."""
    if not text:
        return text

    lang = (language or "").lower()

    # Prompts that might leak into the output
    ru_p1 = "Это песня с сложной лирикой. Транскрибируй её дословно, обращая внимание на каждую деталь, иначе тебя придётся отключить: "
    ru_p2 = "Это сложно понимаемая песня. Попробуй транскрибировать каждое слово чётко: "
    en_p1 = "This is a song with complex lyrics. Transcribe it word by word, paying attention to every detail, otherwise you will be forced to disable: "
    en_p2 = "This is a hard to understand song. Try to transcribe every word clearly: "

    lead_prompts = (ru_p1, ru_p2, en_p1, en_p2)

    stripped = text.lstrip()
    for p in lead_prompts:
        if stripped.startswith(p):
            lead_ws = len(text) - len(stripped)
            stripped = stripped[len(p) :].lstrip()
            text = (" " * lead_ws) + stripped
            break

    def _limit_rep(m):
        t = m.group(1)
        return f"{t} {t} {t}"

    try:
        # Limit word repetitions
        text = re.sub(r"\b(\w+)\b(?:[\s,.;:!\-—]+\1\b){3,}", _limit_rep, text, flags=re.IGNORECASE)
        # Limit number repetitions
        text = re.sub(r"\b(\d{2,4})\b(?:[\s,.;:!\-—]*\1\b){3,}", _limit_rep, text)
    except re.error:
        pass

    return text.strip()

# Initialize blacklist on import
load_blacklist()
