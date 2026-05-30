"""Configuration module for Telegram AI Bot"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TelegramConfig:
    """Telegram API configuration"""
    api_id: int = int(os.getenv('TELEGRAM_API_ID', 0))
    api_hash: str = os.getenv('TELEGRAM_API_HASH', '')
    phone: str = os.getenv('TELEGRAM_PHONE', '')
    session_path: str = os.getenv('SESSION_PATH', 'data/session')
    contacts_only: bool = os.getenv('CONTACTS_ONLY', 'true').lower() == 'true'

    # Proxy settings
    system_proxy_enabled: bool = os.getenv('TELEGRAM_USE_SYSTEM_PROXY', 'false').lower() == 'true'
    # Format: "type:host:port" or "type:host:port:user:pass" (e.g. "socks5:127.0.0.1:9050")
    proxy_string: str = os.getenv('TELEGRAM_PROXY', '')

    # Bot owner for style mimicry (learn owner's communication style)
    bot_owner_id: int = int(os.getenv('BOT_OWNER_ID', 0))

@dataclass
class LLMConfig:
    """Local LLM configuration"""
    base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:1234')
    model: str = os.getenv('OLLAMA_MODEL', 'glm-4.6v-flash')
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 2000
    timeout: int = 120
    frequency_penalty: float = 0.3  # Reduce repetitive text (0.0-2.0)
    presence_penalty: float = 0.1   # Encourage new topics (0.0-2.0)

@dataclass
class LLMRouterConfig:
    """Multi-model routing configuration"""
    enabled: bool = os.getenv('LLM_ROUTER_ENABLED', 'true').lower() == 'true'

    # Model assignments for different tasks (use LM Studio "Identifier API" names!)
    chat_model: str = os.getenv('CHAT_MODEL', 'glm-4.6v-flash')
    code_model: str = os.getenv('CODE_MODEL', 'deepseek-ai_-_deepseek-coder-6.7b-instruct')
    analysis_model: str = os.getenv('ANALYSIS_MODEL', 'glm-4-9b-chat')

    # LLM-based classification (more accurate than keywords)
    use_llm_classifier: bool = os.getenv('USE_LLM_CLASSIFIER', 'false').lower() == 'true'
    classifier_model: str = os.getenv('CLASSIFIER_MODEL', 'meta-llama-llama-3.2-3b-instruct-qlora-malaysian-16k')  # Llama 3.2 3B

    # Master Classifier: unified analyzer + router (ONE LLM call for everything)
    use_master_classifier: bool = os.getenv('USE_MASTER_CLASSIFIER', 'false').lower() == 'true'


    # Future: reasoning and tools models
    # reasoning_model: str = os.getenv('REASONING_MODEL', 'microsoft/Phi-3-medium-4k-instruct')
    # tools_model: str = os.getenv('TOOLS_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')


@dataclass
class BehaviorConfig:
    """Human behavior simulation configuration"""
    skip_message_prob: float = float(os.getenv('SKIP_MESSAGE_PROBABILITY', 0.05))
    immediate_response_prob: float = float(os.getenv('IMMEDIATE_RESPONSE_PROBABILITY', 0.70))
    typing_speed_wpm: int = int(os.getenv('TYPING_SPEED_WPM', 60))
    add_imperfections_prob: float = 0.15

    min_typing_delay: float = 1.0
    max_typing_delay: float = 8.0
    min_busy_delay: float = 10.0
    max_busy_delay: float = 60.0

    context_messages_limit: int = 15  # Last N messages for context
    context_max_length: int = 2000  # Max chars for context

    inactivity_threshold_hours: int = 2
    check_activity_interval_minutes: int = 5

    # Proactive messaging (bot initiates conversation)
    proactive_enabled: bool = os.getenv('PROACTIVE_MESSAGING_ENABLED', 'false').lower() == 'true'
    proactive_check_interval_hours: int = int(os.getenv('PROACTIVE_CHECK_INTERVAL_HOURS', 6))  # Check every N hours
    proactive_min_silence_hours: int = int(os.getenv('PROACTIVE_MIN_SILENCE_HOURS', 12))  # Min silence before initiating
    proactive_probability: float = float(os.getenv('PROACTIVE_PROBABILITY', 0.3))  # Chance to write when conditions met

@dataclass
class DatabaseConfig:
    """Database and RAG configuration"""
    path: str = os.getenv('DB_PATH', 'data/personality.db')
    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    similar_examples_count: int = 3

@dataclass
class BrowserConfig:
    """Browser configuration (Playwright)"""
    browser_path: str = os.getenv('BROWSER_PATH', '')
    headless: bool = True

@dataclass
class PerformanceConfig:
    """Model performance monitoring configuration"""
    threshold_seconds: int = 120  # Trigger reload if avg generation > 2 minutes
    cooldown_hours: int = 1  # Wait 1 hour between reloads
    tracking_window: int = 10  # Track last 10 generations

class Config:
    """Main configuration class"""
    telegram = TelegramConfig()
    llm = LLMConfig()
    llm_router = LLMRouterConfig()
    behavior = BehaviorConfig()
    database = DatabaseConfig()
    browser = BrowserConfig()
    performance = PerformanceConfig()
