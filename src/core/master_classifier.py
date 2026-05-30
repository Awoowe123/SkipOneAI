"""Master Classifier - uses lightweight LLM to analyze message and determine routing + tone"""
import logging
import json
from typing import Dict, Any, Literal
from src.config import Config

logger = logging.getLogger(__name__)

TaskType = Literal['chat', 'code', 'reasoning', 'tools', 'analysis']

class MasterClassifier:
    """
    Unified message analyzer using lightweight LLM (Qwen 0.5B)

    Single LLM call determines:
    - task: which model to route to (chat/code/analysis)
    - tone: message tone (neutral/excited/angry/etc)
    - type: request type (conversation/help_request/question)
    - complexity: response complexity (simple/medium/complex)
    """

    def __init__(self, llm_interface):
        """
        Args:
            llm_interface: Main LLMInterface instance
        """
        self.llm = llm_interface
        self.classifier_model = Config.llm_router.classifier_model
        logger.info(f"Master Classifier initialized with model: {self.classifier_model}")

    def analyze(self, message: str) -> Dict[str, Any]:
        """
        Analyze message with single LLM call

        Args:
            message: User message

        Returns:
            Dict with keys: task, tone, type, response_style, complexity
        """

        # Ultra-strict prompt for Granite - exact format required
        analysis_prompt = f"""Message: "{message[:200]}"

Return JSON (exact format):
{{"task":"chat","tone":"neutral","type":"conversation","response_style":"casual","complexity":"simple","audio_intent":"none","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}

task: "chat" OR "code" OR "analysis"
tone: "neutral" OR "friendly" OR "serious"
type: "conversation" OR "help_request"
response_style: "casual" OR "helpful"
complexity: "simple" OR "medium"
audio_intent: "transcribe" OR "reply" OR "none" (ONLY if referring to audio/voice)
asr_model: "gigaam" OR "whisper" OR "auto" (which ASR model to use)
asr_language: "ru" OR "en" OR "uk" OR "kk" OR null (null = auto-detect)
use_bs_roformer: true OR false (extract vocals before transcription)
vocals_only: true OR false (only extract vocals, no transcription)

Examples:
"создай HTML" -> {{"task":"code","tone":"neutral","type":"help_request","response_style":"helpful","complexity":"simple","audio_intent":"none","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}
"как дела?" -> {{"task":"chat","tone":"friendly","type":"conversation","response_style":"casual","complexity":"simple","audio_intent":"none","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}
"расшифруй" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"transcribe","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}
"расшифруй Giga" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"transcribe","asr_model":"gigaam","asr_language":"ru","use_bs_roformer":false,"vocals_only":false}}
"расшифруй ру Whisper" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"transcribe","asr_model":"whisper","asr_language":"ru","use_bs_roformer":false,"vocals_only":false}}
"расшифруй en Whisper" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"transcribe","asr_model":"whisper","asr_language":"en","use_bs_roformer":false,"vocals_only":false}}
"очисти и расшифруй Giga" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"transcribe","asr_model":"gigaam","asr_language":"ru","use_bs_roformer":true,"vocals_only":false}}
"очисти аудиозапись" -> {{"task":"chat","tone":"neutral","type":"helper","response_style":"helpful","complexity":"simple","audio_intent":"none","asr_model":"auto","asr_language":null,"use_bs_roformer":true,"vocals_only":true}}
"что он сказал?" -> {{"task":"chat","tone":"neutral","type":"question","response_style":"casual","complexity":"simple","audio_intent":"transcribe","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}
"ответь ему нет" -> {{"task":"chat","tone":"serious","type":"conversation","response_style":"casual","complexity":"simple","audio_intent":"reply","asr_model":"auto","asr_language":null,"use_bs_roformer":false,"vocals_only":false}}

Rules:
- If message asks to transcribe/read audio -> audio_intent="transcribe"
- If message asks to reply/answer to audio -> audio_intent="reply"
- If mentions "Giga" OR "GigaAM" -> asr_model="gigaam", asr_language="ru"
- If mentions "Whisper" -> asr_model="whisper"
- If mentions language (ру/ru/en/uk) -> set asr_language accordingly
- If mentions "очисти"/"clean" with audio -> use_bs_roformer=true
- If ONLY "очисти аудиозапись" with no transcribe request -> vocals_only=true
- Otherwise -> defaults (auto, null, false, false)

JSON:"""

        try:
            # Use classifier model
            original_model = self.llm.model
            self.llm.model = self.classifier_model

            result = self.llm.generate(
                prompt=analysis_prompt,
                options={'temperature': 0, 'max_tokens': 100, 'skip_prefix': True}  # Simple deterministic output
            )

            # Restore model
            self.llm.model = original_model

            # Parse JSON response with aggressive cleaning
            result_clean = result.strip()

            # Method 1: Remove markdown code blocks
            if '```json' in result_clean:
                result_clean = result_clean.split('```json')[1].split('```')[0]
            elif '```' in result_clean:
                result_clean = result_clean.split('```')[1].split('```')[0]

            # Method 2: Extract JSON using regex (find first {...})
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_clean)
            if json_match:
                result_clean = json_match.group(0)
            else:
                # Method 3: Manual search for { }
                start_idx = result_clean.find('{')
                end_idx = result_clean.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    result_clean = result_clean[start_idx:end_idx+1]

            # Clean up common issues
            result_clean = result_clean.replace('\n', ' ')  # Remove newlines
            result_clean = re.sub(r',\s*}', '}', result_clean)  # Remove trailing commas

            # Try to parse
            analysis = json.loads(result_clean)

            # Validate required keys
            required_keys = ['task', 'tone', 'type', 'response_style', 'complexity', 'audio_intent',
                           'asr_model', 'asr_language', 'use_bs_roformer', 'vocals_only']
            if not all(k in analysis for k in required_keys):
                # Backwards compatibility / fix missing key
                if 'audio_intent' not in analysis:
                    analysis['audio_intent'] = 'none'
                if 'asr_model' not in analysis:
                    analysis['asr_model'] = 'auto'
                if 'asr_language' not in analysis:
                    analysis['asr_language'] = None
                if 'use_bs_roformer' not in analysis:
                    analysis['use_bs_roformer'] = False
                if 'vocals_only' not in analysis:
                    analysis['vocals_only'] = False

                # Re-check
                if not all(k in analysis for k in required_keys):
                     raise ValueError(f"Missing keys in analysis: {analysis}")

            logger.info(f"[MASTER] Analyzed message: {analysis}")
            return analysis

        except Exception as e:
            logger.warning(f"[MASTER] Analysis failed: {e}, falling back to defaults")
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """Fallback: keyword-based analysis"""
        message_lower = message.lower()

        # Determine task
        code_keywords = ['код', 'функци', 'html', 'создай', 'напиш', 'программ', 'файл']
        if any(kw in message_lower for kw in code_keywords):
            task = 'code'
        elif '/style' in message or '/bio' in message:
            task = 'analysis'
        else:
            task = 'chat'

        # Determine audio intent keyword fallback
        audio_intent = 'none'
        if any(k in message_lower for k in ['расшифруй', 'читай', 'текст', 'что сказал']):
             audio_intent = 'transcribe'
        elif any(k in message_lower for k in ['ответь', 'скажи', 'передай']):
             audio_intent = 'reply'

        # Determine ASR model
        asr_model = 'auto'
        asr_language = None
        use_bs_roformer = False
        vocals_only = False

        if 'giga' in message_lower or 'gigaam' in message_lower:
            asr_model = 'gigaam'
            asr_language = 'ru'  # GigaAM is Russian only
        elif 'whisper' in message_lower:
            asr_model = 'whisper'
            # Detect language
            if 'ру' in message_lower or ' ru' in message_lower:
                asr_language = 'ru'
            elif 'en' in message_lower:
                asr_language = 'en'
            elif 'uk' in message_lower:
                asr_language = 'uk'

        # Detect BS-Roformer usage
        if 'очисти' in message_lower or 'clean' in message_lower:
            use_bs_roformer = True
            # Check if vocal extraction only
            if 'аудиозапись' in message_lower and 'расшифруй' not in message_lower:
                vocals_only = True
                audio_intent = 'none'

        # Simple defaults
        return {
            'task': task,
            'tone': 'neutral',
            'type': 'conversation',
            'response_style': 'casual',
            'complexity': 'medium',
            'audio_intent': audio_intent,
            'asr_model': asr_model,
            'asr_language': asr_language,
            'use_bs_roformer': use_bs_roformer,
            'vocals_only': vocals_only
        }
