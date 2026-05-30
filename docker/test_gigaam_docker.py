"""
GigaAM-v3 Docker Test Script (FP32 Precision)
Runs GigaAM-v3 inference in Docker container (Linux)
Uses manual chunking with Float32 precision for maximum accuracy
"""

import os
import sys
import time
import torch
import torchaudio
from gigaam import load_model


def chunk_audio(audio_tensor, sample_rate, chunk_duration=22):
    """
    Split audio into chunks without overlap

    Args:
        audio_tensor: Audio tensor (channels, samples)
        sample_rate: Sample rate in Hz
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of (audio chunk, start_time) tuples
    """
    chunk_samples = chunk_duration * sample_rate
    total_samples = audio_tensor.shape[1]

    chunks = []
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio_tensor[:, start:end]
        start_time = start / sample_rate
        chunks.append((chunk, start_time))
        start += chunk_samples

    return chunks


def test_gigaam(audio_path: str, model_name: str = "v3_e2e_rnnt"):
    """Test GigaAM-v3 on audio file with FP32 precision"""

    print(f"\n{'='*80}")
    print(f"GigaAM-v3 Docker Test (FP32 Precision)")
    print(f"{'='*80}\n")

    print(f"📂 Audio: {audio_path}")
    print(f"🧠 Model: {model_name}")
    print(f"🐳 Running in Docker (Linux)")
    print(f"⚙️  Precision: Float32 (fp16_encoder=False)")

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

    # Load audio with torchaudio (no torchcodec dependency)
    print(f"\n📥 Loading audio...")
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    audio_duration = waveform.shape[1] / sample_rate
    print(f"📊 Audio duration: {audio_duration:.2f}s")
    print(f"📊 Sample rate: {sample_rate}Hz")

    # Split into chunks
    print(f"\n✂️  Splitting audio into 22-second chunks...")
    chunks = chunk_audio(waveform, sample_rate, chunk_duration=22)
    print(f"📦 Total chunks: {len(chunks)}")

    # Transcribe each chunk
    print(f"\n🚀 Transcribing chunks...")
    transcribe_start = time.time()

    transcriptions = []
    for i, (chunk, chunk_start) in enumerate(chunks):
        chunk_duration_actual = chunk.shape[1] / sample_rate
        chunk_end = chunk_start + chunk_duration_actual
        print(f"  [{i+1}/{len(chunks)}] Processing {chunk_start:.1f}s - {chunk_end:.1f}s...")

        # Save chunk to temporary file (GigaAM expects file path, not tensor)
        temp_chunk_path = f"/tmp/chunk_{i}.wav"
        torchaudio.save(temp_chunk_path, chunk, sample_rate)

        # Transcribe chunk
        result = model.transcribe(temp_chunk_path)

        # Clean up temp file
        os.remove(temp_chunk_path)

        # Extract text
        if isinstance(result, dict):
            text = result.get('text', str(result))
        else:
            text = str(result)

        transcriptions.append(text.strip())

    transcribe_time = time.time() - transcribe_start

    # Combine all transcriptions
    full_transcription = " ".join(transcriptions)

    # Calculate metrics
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
    print(f"📦 Chunks processed: {len(chunks)}")
    print(f"⚙️  Precision: Float32")

    # Save result
    os.makedirs("/workspace/results", exist_ok=True)
    result_file = "/workspace/results/gigaam_fp32.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"GigaAM-v3 Transcription Result (FP32 Precision)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Duration: {audio_duration:.2f}s\n")
        f.write(f"Chunks: {len(chunks)} (22s each)\n")
        f.write(f"Precision: Float32 (fp16_encoder=False)\n")
        f.write(f"Load time: {load_time:.2f}s\n")
        f.write(f"Transcription time: {transcribe_time:.2f}s\n")
        f.write(f"RTF: {rtf:.3f}\n")
        f.write(f"Speed: {1/rtf:.1f}x real-time\n")
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
        print("❌ Usage: python test_gigaam_docker.py <audio_file>")
        print("   Example: python test_gigaam_docker.py /audio/file.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_variant = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    try:
        result = test_gigaam(audio_file, model_variant)
        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
