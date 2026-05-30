import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from src.core.asr_manager import ASRManager

logging.basicConfig(level=logging.INFO)

async def test_song_transcription():
    audio_file = r"path/to/audio.flac"

    # Use the same model as in the reference Worker project
    model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"

    manager = ASRManager(model_size=model_size, device="cuda")

    print(f"--- Testing Faster Whisper {model_size} (Song Mode) ---")
    try:
        text = await manager.transcribe(
            audio_file,
            language="ru",
            is_song=True # Enable the new song mode
        )
        print("\n=== TRANSCRIPTION RESULT ===")
        print(text)
        print("============================")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await manager.unload_model()

if __name__ == "__main__":
    asyncio.run(test_song_transcription())
