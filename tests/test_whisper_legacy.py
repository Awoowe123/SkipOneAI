import asyncio
import logging
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.asr_manager import ASRManager

# Configure logging to see ASRManager output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def main():
    # File to valid
    audio_file = r"path/to/audio.mp3"

    if not os.path.exists(audio_file):
        logger.error(f"Test file not found: {audio_file}")
        # Try to find any mp3 in current dir or Downloads
        return

    logger.info(f"Testing with parameters from 'whis' (Legacy Config) on: {audio_file}")

    # Initialize Manager (auto-loads 'large-v3-turbo' from local 'models/' dir)
    manager = ASRManager(model_size="deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda")

    try:
        # 'is_song=True' triggers the logic from whisper_processor.py:
        # - beam_size=5
        # - patience=2.0
        # - compression_ratio_threshold=2.4
        # - specific prompts
        # - NO VAD (vad_filter=False) for songs
        text_song = await manager.transcribe(
            audio_file,
            language="ru",
            is_song=True
        )

        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT (Song Mode):")
        print("="*50)
        print(text_song)
        print("="*50 + "\n")

        text_regular = await manager.transcribe(
            audio_file,
            language="ru",
            is_song=False # Regular Mode
        )

        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT (Regular Mode):")
        print("="*50)
        print(text_regular)
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
    finally:
        await manager.unload_model()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
