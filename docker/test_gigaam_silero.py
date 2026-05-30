"""
GigaAM-v3 Docker Test Script (Silero VAD Smart Chunking)
Runs GigaAM-v3 inference using Silero VAD to detect speech pauses
Creates optimal chunk boundaries without cutting words mid-sentence
"""

import os
import sys
import time
import torch
import torchaudio
from gigaam import load_model


def detect_speech_segments(audio_path, min_silence_duration=0.5, speech_threshold=0.3):
    """
    Detect speech segments using Silero VAD

    Args:
        audio_path: Path to audio file
        min_silence_duration: Minimum silence duration to consider as boundary (seconds)
        speech_threshold: Threshold for speech detection (0-1, lower = more sensitive)

    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    print(f"🎯 Running Silero VAD (threshold={speech_threshold})...")

    # Load Silero VAD model (lightweight, ~1MB)
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True  # Suppress trust warning
    )

    (get_speech_timestamps, _, read_audio, *_) = utils

    # Load audio
    wav = read_audio(audio_path, sampling_rate=16000)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=speech_threshold,
        min_silence_duration_ms=int(min_silence_duration * 1000),
        return_seconds=True
    )

    print(f"✅ Found {len(speech_timestamps)} speech segments")

    return speech_timestamps


def merge_short_segments(segments, min_chunk_duration=10, max_chunk_duration=22):
    """
    Merge short speech segments into optimal chunks

    Args:
        segments: List of (start, end) tuples from VAD
        min_chunk_duration: Minimum chunk duration in seconds
        max_chunk_duration: Maximum chunk duration (GigaAM limit: 25s)

    Returns:
        List of merged (start, end) chunks
    """
    if not segments:
        return []

    merged_chunks = []
    current_start = segments[0]['start']
    current_end = segments[0]['end']

    for i in range(1, len(segments)):
        seg = segments[i]
        gap = seg['start'] - current_end
        potential_duration = seg['end'] - current_start

        # Merge if:
        # 1. Gap is small (< 2s of silence)
        # 2. Total duration doesn't exceed max limit
        if gap < 2.0 and potential_duration <= max_chunk_duration:
            current_end = seg['end']
        else:
            # Save current chunk
            merged_chunks.append((current_start, current_end))
            current_start = seg['start']
            current_end = seg['end']

    # Add last chunk
    merged_chunks.append((current_start, current_end))

    print(f"📦 Merged into {len(merged_chunks)} optimal chunks")
    return merged_chunks


def extract_chunk(waveform, sample_rate, start_time, end_time):
    """Extract audio chunk by time boundaries"""
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return waveform[:, start_sample:end_sample]


def test_gigaam_silero(audio_path: str, model_name: str = "v3_e2e_rnnt"):
    """Test GigaAM-v3 with Silero VAD smart chunking"""

    print(f"\n{'='*80}")
    print(f"GigaAM-v3 Docker Test (Silero VAD Smart Chunking)")
    print(f"{'='*80}\n")

    print(f"📂 Audio: {audio_path}")
    print(f"🧠 Model: {model_name}")
    print(f"🐳 Running in Docker (Linux)")
    print(f"⚙️  Precision: Float32")
    print(f"🎯 VAD: Silero (lightweight, CPU)")

    # Detect speech segments with Silero VAD
    speech_segments = detect_speech_segments(audio_path, speech_threshold=0.3)

    # Merge into optimal chunks
    chunks = merge_short_segments(speech_segments)

    if not chunks:
        print(f"\n❌ No speech detected! Falling back to 22s fixed chunks...")
        # Fallback to fixed chunking
        waveform_temp, sr_temp = torchaudio.load(audio_path)
        if sr_temp != 16000:
            resampler = torchaudio.transforms.Resample(sr_temp, 16000)
            waveform_temp = resampler(waveform_temp)
            sr_temp = 16000
        duration = waveform_temp.shape[1] / sr_temp
        num_chunks = int(duration / 22) + 1
        chunks = [(i * 22, min((i + 1) * 22, duration)) for i in range(num_chunks)]

    print(f"\n📊 Chunking strategy:")
    for i, (start, end) in enumerate(chunks):
        duration = end - start
        print(f"  Chunk {i+1}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")

    # Load model
    print(f"\n⏳ Loading GigaAM-v3...")
    load_start = time.time()

    model = load_model(
        model_name=model_name,
        fp16_encoder=False,
        device="cuda"
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Load audio
    print(f"\n📥 Loading audio...")
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    audio_duration = waveform.shape[1] / sample_rate
    print(f"📊 Audio duration: {audio_duration:.2f}s")

    # Transcribe each VAD-detected chunk
    print(f"\n🚀 Transcribing {len(chunks)} VAD chunks...")
    transcribe_start = time.time()

    transcriptions = []
    for i, (start_time, end_time) in enumerate(chunks):
        duration = end_time - start_time
        print(f"  [{i+1}/{len(chunks)}] {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)...")

        # Extract chunk
        chunk = extract_chunk(waveform, sample_rate, start_time, end_time)

        # Save to temp file
        temp_chunk_path = f"/tmp/vad_chunk_{i}.wav"
        torchaudio.save(temp_chunk_path, chunk, sample_rate)

        # Transcribe
        result = model.transcribe(temp_chunk_path)

        # Clean up
        os.remove(temp_chunk_path)

        # Extract text
        if isinstance(result, dict):
            text = result.get('text', str(result))
        else:
            text = str(result)

        transcriptions.append(text.strip())

    transcribe_time = time.time() - transcribe_start
    full_transcription = " ".join(transcriptions)
    rtf = transcribe_time / audio_duration

    # Results
    print(f"\n{'='*80}")
    print(f"✅ RESULTS")
    print(f"{'='*80}\n")
    print(f"📝 Transcription:\n{full_transcription}\n")
    print(f"⏱️  Load time: {load_time:.2f}s")
    print(f"⏱️  Transcription time: {transcribe_time:.2f}s")
    print(f"📊 Audio duration: {audio_duration:.2f}s")
    print(f"📊 RTF: {rtf:.3f}")
    print(f"🚀 Speed: {1/rtf:.1f}x real-time")
    print(f"📏 Text length: {len(full_transcription)} chars")
    print(f"🎯 VAD chunks: {len(chunks)}")
    print(f"⚙️  Precision: Float32")

    # Save result
    os.makedirs("/workspace/results", exist_ok=True)
    result_file = "/workspace/results/gigaam_silero_vad.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"GigaAM-v3 Transcription Result (Silero VAD)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"VAD: Silero (threshold=0.3, speech pause detection)\n")
        f.write(f"Duration: {audio_duration:.2f}s\n")
        f.write(f"Chunks: {len(chunks)} (VAD-optimized)\n")
        f.write(f"Precision: Float32\n")
        f.write(f"Load time: {load_time:.2f}s\n")
        f.write(f"Transcription time: {transcribe_time:.2f}s\n")
        f.write(f"RTF: {rtf:.3f}\n")
        f.write(f"Speed: {1/rtf:.1f}x\n\n")
        f.write(f"Chunk boundaries:\n")
        for i, (start, end) in enumerate(chunks):
            f.write(f"  [{i+1}] {start:.1f}s - {end:.1f}s ({end-start:.1f}s)\n")
        f.write(f"\nTranscription:\n{full_transcription}\n")

    print(f"\n📁 Result saved to: {result_file}")

    return {
        "transcription": full_transcription,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "chunks": len(chunks)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python test_gigaam_silero.py <audio_file>")
        print("   Example: python test_gigaam_silero.py /audio/file.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_variant = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    try:
        result = test_gigaam_silero(audio_file, model_variant)
        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
