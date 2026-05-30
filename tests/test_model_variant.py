import asyncio
import logging
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.asr_manager import ASRManager

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    audio_file = r"path/to/audio.mp3"

    # Path to antony66 model in legacy folder
    model_path = r"D:\Whisper 2\Worker\models\ct2\models--antony66--whisper-large-v3-russian"

    logger.info(f"Testing with model: {model_path}")

    if not os.path.exists(model_path):
        logger.error("Model path not found!")
        return

    # Initialize Manager with this specific model
    # Note: device index 0 assumed
    manager = ASRManager(model_size=model_path, device="cuda")

    try:
        text = await manager.transcribe(
            audio_file,
            language="ru",
            is_song=True
        )

        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT (Antony66):")
        print("="*50)
        print(text)
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
    finally:
        await manager.unload_model()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
