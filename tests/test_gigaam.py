"""
GigaAM-v3 Test Script (Official API)
Tests GigaAM-v3 model on Russian audio using official gigaam package.
"""

import os
import time
from gigaam import load_model, load_audio


def test_gigaam_transcription(audio_path: str, model_name: str = "v3_e2e_rnnt"):
    """
    Test GigaAM-v3 on a single audio file using official gigaam package.

    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        model_name: Model name (v3_e2e_rnnt, v3_e2e_ctc, v3_rnnt, v3_ctc, v3_ssl)
    """
    print(f"\n{'='*80}")
    print(f"Testing GigaAM {model_name}")
    print(f"{'='*80}\n")

    print(f"📂 Audio file: {audio_path}")
    print(f"🧠 Model: {model_name}")

    # Load model
    print(f"\n⏳ Loading GigaAM model...")
    load_start = time.time()

    model = load_model(
        model_name=model_name,
        fp16_encoder=True,
        device="cuda"
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Transcribe (model loads audio internally)
    print(f"\n� Transcribing...")
    transcribe_start = time.time()

    result = model.transcribe_longform(audio_path)

    transcribe_time = time.time() - transcribe_start

    # Get audio duration for RTF calculation
    # Load audio just to get duration
    from gigaam import load_audio
    audio_tensor = load_audio(audio_path)
    audio_duration = len(audio_tensor) / 16000

    rtf = transcribe_time / audio_duration

    # Extract text from result (might be a dict or string)
    if isinstance(result, dict):
        transcription = result.get('text', str(result))
    else:
        transcription = str(result)

    # Results
    print(f"\n{'='*80}")
    print(f"✅ RESULTS")
    print(f"{'='*80}\n")
    print(f"📝 Transcription:\n{transcription}\n")
    print(f"⏱️  Transcription time: {transcribe_time:.2f}s")
    print(f"📊 Real-Time Factor (RTF): {rtf:.3f}")
    print(f"🚀 Speed: {1/rtf:.1f}x real-time")
    print(f"📏 Text length: {len(transcription)} chars")

    return {
        "transcription": transcription,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "text_length": len(transcription)
    }


if __name__ == "__main__":
    import sys

    # Default test file
    default_file = r"path/to/audio.mp3"

    audio_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    model_variant = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        print(f"\nUsage: python test_gigaam.py [audio_file] [model_name]")
        print(f"\nAvailable models:")
        print(f"  v3_e2e_rnnt  - GigaAM-v3 RNNT with punctuation (default)")
        print(f"  v3_e2e_ctc   - GigaAM-v3 CTC with punctuation")
        print(f"  v3_rnnt      - GigaAM-v3 RNNT (no punctuation)")
        print(f"  v3_ctc       - GigaAM-v3 CTC (no punctuation)")
        print(f"  v3_ssl       - GigaAM-v3 SSL (encoder only)")
        sys.exit(1)

    try:
        result = test_gigaam_transcription(audio_file, model_variant)
        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
