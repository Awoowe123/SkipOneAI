"""
GigaAM-v3 Docker Test Script (Native VAD with transcribe_longform)
Runs GigaAM-v3 inference in Docker container using VAD-based segmentation
This requires torchcodec==0.7.0 for proper audio decoding
"""

import os
import sys
import time
from gigaam import load_model


def test_gigaam_vad(audio_path: str, model_name: str = "v3_e2e_rnnt"):
    """Test GigaAM-v3 using native transcribe_longform with VAD segmentation"""

    print(f"\n{'='*80}")
    print(f"GigaAM-v3 Docker Test (Native VAD)")
    print(f"{'='*80}\n")

    print(f"📂 Audio: {audio_path}")
    print(f"🧠 Model: {model_name}")
    print(f"🐳 Running in Docker (Linux)")
    print(f"⚙️  Method: transcribe_longform() with VAD")
    print(f"🎯 Segmentation: Automatic by speech pauses")

    # Load model
    print(f"\n⏳ Loading GigaAM-v3...")
    load_start = time.time()

    model = load_model(
        model_name=model_name,
        fp16_encoder=False,  # Use full Float32 precision for maximum accuracy
        device="cuda"
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Transcribe using native longform with VAD
    print(f"\n🚀 Transcribing with VAD segmentation...")
    print(f"   (This may take longer due to VAD processing)\n")
    transcribe_start = time.time()

    try:
        # transcribe_longform returns list of utterances with boundaries
        utterances = model.transcribe_longform(audio_path)

        transcribe_time = time.time() - transcribe_start

        # Extract and display utterances
        print(f"✅ Transcription completed!")
        print(f"\n{'='*80}")
        print(f"📝 UTTERANCES (segmented by VAD)")
        print(f"{'='*80}\n")

        full_transcription = []
        for i, utt in enumerate(utterances):
            text = utt["transcription"]
            start, end = utt["boundaries"]
            duration = end - start

            print(f"[{i+1}] [{start:.1f}s - {end:.1f}s] ({duration:.1f}s)")
            print(f"    {text}\n")

            full_transcription.append(text)

        # Combine all utterances
        full_text = " ".join(full_transcription)

        # Calculate metrics
        # Get audio duration from last utterance
        audio_duration = utterances[-1]["boundaries"][1] if utterances else 0
        rtf = transcribe_time / audio_duration if audio_duration > 0 else 0

        # Results summary
        print(f"\n{'='*80}")
        print(f"✅ SUMMARY")
        print(f"{'='*80}\n")
        print(f"📝 Full Transcription:\n{full_text}\n")
        print(f"⏱️  Load time: {load_time:.2f}s")
        print(f"⏱️  Transcription time: {transcribe_time:.2f}s")
        print(f"📊 Audio duration: {audio_duration:.2f}s")
        print(f"📊 RTF: {rtf:.3f}")
        print(f"🚀 Speed: {1/rtf:.1f}x real-time")
        print(f"📏 Text length: {len(full_text)} chars")
        print(f"🎯 Utterances (VAD segments): {len(utterances)}")
        print(f"⚙️  Precision: Float32")

        # Save result
        os.makedirs("/workspace/results", exist_ok=True)
        result_file = "/workspace/results/gigaam_vad.txt"

        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"GigaAM-v3 Transcription Result (Native VAD)\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Audio file: {audio_path}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Method: transcribe_longform() with VAD\n")
            f.write(f"Duration: {audio_duration:.2f}s\n")
            f.write(f"Utterances: {len(utterances)}\n")
            f.write(f"Precision: Float32 (fp16_encoder=False)\n")
            f.write(f"Load time: {load_time:.2f}s\n")
            f.write(f"Transcription time: {transcribe_time:.2f}s\n")
            f.write(f"RTF: {rtf:.3f}\n")
            f.write(f"Speed: {1/rtf:.1f}x real-time\n\n")

            f.write(f"Utterances:\n")
            f.write(f"{'='*80}\n\n")
            for i, utt in enumerate(utterances):
                text = utt["transcription"]
                start, end = utt["boundaries"]
                f.write(f"[{i+1}] [{start:.1f}s - {end:.1f}s]: {text}\n\n")

            f.write(f"\nFull Transcription:\n")
            f.write(f"{'='*80}\n")
            f.write(f"{full_text}\n")

        print(f"\n📁 Result saved to: {result_file}")

        return {
            "transcription": full_text,
            "utterances": utterances,
            "load_time": load_time,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": rtf
        }

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        print(f"\nThis might be due to torchcodec compatibility issues.")
        print(f"Check that torchcodec==0.7.0 is properly installed.")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python test_gigaam_vad.py <audio_file>")
        print("   Example: python test_gigaam_vad.py /audio/file.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_variant = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    try:
        result = test_gigaam_vad(audio_file, model_variant)
        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
