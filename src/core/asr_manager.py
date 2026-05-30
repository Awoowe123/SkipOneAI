import os
import logging
import asyncio
import gc
import torch

logger = logging.getLogger(__name__)

class ASRManager:
    def __init__(self, model_size: str = "deepdml/faster-whisper-large-v3-turbo-ct2", device: str = None):
        """
        Manages Faster-Whisper ASR model.
        """
        self.model_size = model_size
        self.device = device
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self._is_loaded = False

        # --- Patch environment for ctranslate2 (Windows primarily) ---
        if os.name == "nt":
            self._register_nvidia_libs()

    def _register_nvidia_libs(self):
        """Adds NVIDIA pip packages to PATH/DLL search for ctranslate2."""
        import sys
        import glob

        # Check standard site-packages locations
        site_packages = [p for p in sys.path if "site-packages" in p]

        libs = ["cublas", "cudnn"]

        for sp in site_packages:
            nvidia_base = os.path.join(sp, "nvidia")
            if not os.path.isdir(nvidia_base):
                continue

            for lib in libs:
                lib_bin = os.path.join(nvidia_base, lib, "bin")
                if os.path.isdir(lib_bin):
                    # Add to PATH
                    if lib_bin not in os.environ["PATH"]:
                        os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]
                        logger.info(f"[ASR] Added NVIDIA lib to PATH: {lib_bin}")

                    # Add to Python DLL search path
                    try:
                        os.add_dll_directory(lib_bin)
                    except Exception as e:
                        logger.warning(f"[ASR] Failed to add_dll_directory for {lib_bin}: {e}")


    async def load_model(self):
        """Loads the model into memory."""
        if self._is_loaded:
            return

        try:
            logger.info(f"[ASR] Loading Faster-Whisper model {self.model_size} on {self.device}...")

            # Run blocking load in thread
            def _load():
                from faster_whisper import WhisperModel

                # Use local models directory if available
                download_root = os.path.join(os.getcwd(), "models")
                if not os.path.exists(download_root):
                    os.makedirs(download_root, exist_ok=True)

                return WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8",
                    download_root=download_root
                )

            self.model = await asyncio.to_thread(_load)
            self._is_loaded = True
            logger.info("[ASR] Model loaded successfully.")

        except Exception as e:
            logger.error(f"[ASR] Failed to load model: {e}", exc_info=True)
            raise

    async def unload_model(self):
        """Unloads model and clears VRAM."""
        if not self._is_loaded:
            return

        logger.info("[ASR] Unloading model...")
        del self.model
        self.model = None
        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("[ASR] Model unloaded.")

    async def transcribe(self, audio_path: str, task: str = "transcribe", language: str = None, prompt: str = None, is_song: bool = False) -> str:
        """
        Transcribes the given audio file using Faster-Whisper.
        Supports specialized 'Song Mode' for better lyrics accuracy.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        auto_loaded = False
        if not self._is_loaded:
            await self.load_model()
            auto_loaded = True

        try:
            logger.info(f"[ASR] Transcribing {audio_path} (Song Mode: {is_song})...")

            # --- Helper to run blocking transcription in thread ---
            def _run_inference():
                import re
                from src.utils.transcription_utils import cleanup_transcribed_text, filter_blacklisted_phrases


                if is_song:
                    ru_prompt = "Это песня с сложной лирикой. Транскрибируй её дословно, обращая внимание на каждую деталь, иначе тебя придётся отключить: "
                    en_prompt = "This is a song with complex lyrics. Transcribe it word by word, paying attention to every detail, otherwise you will be forced to disable: "

                    initial_prompt = ru_prompt if (language or "").lower() == "ru" else en_prompt

                    # First pass: high beam size for accuracy
                    logger.info("[ASR] Starting 1st pass (Song Mode)...")
                    segments, info = self.model.transcribe(
                        audio_path,
                        beam_size=5,
                        language=language,
                        task=task,
                        initial_prompt=initial_prompt,
                        word_timestamps=True,
                        vad_filter=False,             # Disable VAD for songs
                        condition_on_previous_text=False,
                        patience=2.0,
                        length_penalty=1.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.4,
                        temperature=0.0
                    )
                    segments = list(segments)

                    segments = list(segments)

                    total_chars = sum(len((s.text or "").strip()) for s in segments)
                    logger.info(f"[ASR] 1st pass result: {len(segments)} segments, {total_chars} chars.")

                    # Retry logic
                    if len(segments) < 2 or (len(segments) < 5 and total_chars < 10):
                        logger.info("[ASR] First pass yielded poor results. Retrying with alternative prompt (2nd pass)...")
                        ru_prompt2 = "Это сложно понимаемая песня. Попробуй транскрибировать каждое слово чётко: "
                        en_prompt2 = "This is a hard to understand song. Try to transcribe every word clearly: "
                        initial_prompt = ru_prompt2 if (language or "").lower() == "ru" else en_prompt2

                        segments, info = self.model.transcribe(
                            audio_path,
                            beam_size=1, # Lower beam for retry
                            language=language,
                            task=task,
                            initial_prompt=initial_prompt,
                            word_timestamps=True,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            patience=2.0,
                            length_penalty=1.0,
                            compression_ratio_threshold=2.4,
                            no_speech_threshold=0.4,
                            temperature=0.0
                        )
                        segments = list(segments)
                        logger.info(f"[ASR] 2nd pass result: {len(segments)} segments.")

                # 2. Regular Mode Strategy (Speech)
                else:
                    logger.info("[ASR] Starting Regular Mode (Voice Message)...")
                    segments, info = self.model.transcribe(
                        audio_path,
                        beam_size=1,
                        language=language,
                        task=task,
                        initial_prompt=prompt,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=100, speech_pad_ms=100, threshold=0.4),
                        condition_on_previous_text=False,
                        patience=0.8,
                        suppress_blank=True,
                        length_penalty=1.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.05,
                        temperature=0.0
                    )
                    segments = list(segments)
                    logger.info(f"[ASR] Regular pass result: {len(segments)} segments.")

                # --- Post-Processing ---
                result_text_list = []
                for seg in segments:
                    text = (seg.text or "").strip()
                    if not text:
                        continue

                    # Legacy behavior: Cleanup then normalize spaces
                    cleaned_text = cleanup_transcribed_text(text, language)
                    if cleaned_text:
                        normalized = re.sub(r"\s+", " ", cleaned_text)
                        if normalized:
                            result_text_list.append(normalized)

                # Legacy behavior: join with newlines
                combined_text = "\n".join(result_text_list)

                # Apply Blacklist
                final_text = filter_blacklisted_phrases(combined_text)

                if len(final_text) != len(combined_text):
                    logger.info(f"[ASR] Blacklist filtering applied. Reduced chars from {len(combined_text)} to {len(final_text)}.")

                return final_text.strip()

            # Run in thread
            text = await asyncio.to_thread(_run_inference)

            logger.info(f"[ASR] Transcription complete: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"[ASR] Transcription failed: {e}", exc_info=True)
            raise

        finally:
            if auto_loaded:
                await self.unload_model()
