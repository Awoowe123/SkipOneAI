"""Dynamic LLM Router - switches between specialized models based on task type"""
import logging
import asyncio
import torch
from typing import Optional, Dict, Any, Literal
from src.config import Config

logger = logging.getLogger(__name__)

TaskType = Literal['chat', 'code', 'reasoning', 'tools', 'analysis']

class DynamicModelRouter:
    """Routes requests to specialized models, managing VRAM dynamically"""

    def __init__(self, llm_interface, model_manager=None):
        """
        Initialize router with connection to LLM interface

        Args:
            llm_interface: Main LLMInterface instance (handles API requests)
            model_manager: Optional ModelManager for dynamic loading
        """
        self.llm = llm_interface
        self.model_manager = model_manager
        self.current_model = None

        # Model configuration from config
        self.models = {
            'chat': Config.llm_router.chat_model,
            'code': Config.llm_router.code_model,
            'analysis': Config.llm_router.analysis_model,
        }

        # Track which model is currently loaded (for future VRAM management)
        self.active_model_name = self.models['chat']

        logger.info(f"LLM Router initialized with models: {self.models}")

        # Reasoning effort levels for GPT-OSS models
        self.reasoning_levels = {
            'chat': 'low',       # Brief thinking for better chat responses
            'code': 'medium',    # Balanced code generation (better quality, ~2-3sec thinking)
            'analysis': 'high',  # Deep reasoning for complex tasks
            'tools': 'low',      # Fast tool execution
            'reasoning': 'high'  # Explicit reasoning tasks
        }

    async def route(
        self,
        task: TaskType,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs  # Accept additional arguments like tools, tool_choice
    ) -> str:
        """
        Route request to appropriate model based on task type

        Args:
            task: Type of task ('chat', 'code', 'analysis', etc.)
            prompt: User prompt
            system_prompt: Optional system message
            options: Generation options (temperature, max_tokens, etc.)
            timeout: Optional custom timeout
            **kwargs: Additional LLM arguments (tools, tool_choice, etc.)

        Returns:
            Generated text from selected model
        """
        # Select model for task
        model_name = self.models.get(task, self.models['chat'])

        # Log model selection
        if model_name != self.active_model_name:
            logger.info(f"[ROUTER] Switching model: {self.active_model_name} → {model_name} (task: {task})")

            # Perform model swap via ModelManager
            if self.model_manager:
                logger.info(f"[ROUTER] Performing smart swap...")
                await self.model_manager.smart_swap(
                    from_model=self.active_model_name,
                    to_model=model_name
                )

            self.active_model_name = model_name
        else:
            logger.debug(f"[ROUTER] Using active model: {model_name} (task: {task})")

        # Ensure model is loaded (if manager available)
        if self.model_manager:
            await self.model_manager.ensure_loaded(model_name)

        # Update model in LLM interface
        original_model = self.llm.model
        self.llm.model = model_name

        try:
            # Get reasoning effort for this task/model combination
            reasoning_effort = self._get_reasoning_effort(task, model_name)
            if reasoning_effort:
                logger.debug(f"[ROUTER] Using reasoning_effort={reasoning_effort} for {task}")
                kwargs['reasoning_effort'] = reasoning_effort

            # Generate response (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    options=options,
                    timeout=timeout,
                    **kwargs
                )
            )

            # Auto-unload cold models after response (keep hot model always loaded)
            if self.model_manager and model_name != self.model_manager.hot_model:
                logger.info(f"[ROUTER] Auto-unloading cold model: {model_name}")
                asyncio.create_task(
                    self.model_manager.unload_model(model_name)
                )

            return response
        finally:
            # Restore original model
            self.llm.model = original_model

    def _get_reasoning_effort(self, task: TaskType, model_name: str) -> Optional[str]:
        """Get reasoning effort level for task and model

        Args:
            task: Task type
            model_name: Model identifier

        Returns:
            Reasoning effort level ('low', 'medium', 'high') or None
        """
        # Only apply reasoning effort to GPT-OSS models
        if 'gpt-oss' not in model_name.lower():
            return None

        return self.reasoning_levels.get(task)

    def classify_task(self, message: str, has_tools: bool = False) -> TaskType:
        """
        Classify task type based on message content

        Args:
            message: User message
            has_tools: Whether tools are available in context

        Returns:
            TaskType for routing
        """
        # Check if LLM classification is enabled
        if hasattr(Config.llm_router, 'use_llm_classifier') and Config.llm_router.use_llm_classifier:
            return self._classify_with_llm(message, has_tools)
        else:
            return self._classify_with_keywords(message, has_tools)

    def _classify_with_llm(self, message: str, has_tools: bool = False) -> TaskType:
        """Use lightweight LLM to classify task type"""

        # Prompt for classification
        classification_prompt = f"""Classify this user message into ONE category:

Categories:
- "code" - programming tasks (create/write/program files, HTML/CSS/JS, debugging, errors)
- "analysis" - analysis commands (/style, /bio)
- "chat" - casual conversation, questions

Examples:
- "создай HTML файл" → code
- "напиши функцию" → code
- "как дела?" → chat
- "/style analyze" → analysis

Message: "{message[:200]}"

Reply with ONLY ONE WORD: code, analysis, or chat"""

        try:
            # Use classifier model (lightweight, on CPU)
            original_model = self.llm.model
            self.llm.model = Config.llm_router.classifier_model

            result = self.llm.generate(
                prompt=classification_prompt,
                options={'temperature': 0.1, 'max_tokens': 5}
            )

            # Restore model
            self.llm.model = original_model

            # Parse result
            result_lower = result.strip().lower()
            if 'code' in result_lower or 'программ' in result_lower:
                task = 'code'
            elif 'analysis' in result_lower or 'анализ' in result_lower:
                task = 'analysis'
            else:
                task = 'chat'

            logger.info(f"[ROUTER] LLM classified as: {task} (raw: '{result.strip()}')")
            return task

        except Exception as e:
            logger.warning(f"[ROUTER] LLM classification failed: {e}, falling back to keywords")
            return self._classify_with_keywords(message, has_tools)

    def _classify_with_keywords(self, message: str, has_tools: bool = False) -> TaskType:
        """Fallback: keyword-based classification"""
        message_lower = message.lower()

        # Code detection (expanded keywords)
        code_keywords = [
            # General
            'код', 'функци', 'python', 'javascript', 'ошибка', 'баг', 'debug',
            'fix', 'исправ', 'программ', 'скрипт', 'class', 'def ', 'import ',
            # Web dev
            'html', 'css', 'спрограмм', 'сдела', 'сайт', 'верстк', 'react', 'vue',
            # Specific
            'файлик', 'файл с', '<div', '<html', 'console.log', 'function(',
            # Actions
            'напиш', 'созда', 'сгенер', 'компил'
        ]
        if any(kw in message_lower for kw in code_keywords):
            logger.info(f"[ROUTER] Keyword classified as CODE (message: '{message[:50]}...')")
            return 'code'

        # Analysis detection (bio, style commands)
        if '/style' in message or '/bio' in message:
            logger.info(f"[ROUTER] Keyword classified as ANALYSIS (command detected)")
            return 'analysis'

        # Tools detection
        if has_tools:
            return 'tools'


        # Default: casual chat
        logger.info(f"[ROUTER] Keyword classified as CHAT (message: '{message[:50]}...')")
        return 'chat'

    def get_active_model(self) -> str:
        """Get currently active model name"""
        return self.active_model_name
