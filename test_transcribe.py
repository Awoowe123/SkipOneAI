#!/usr/bin/env python3
"""
Simple ASR transcription test script for Qwen3-ASR.
Usage: python test_transcribe.py path/to/audio.mp3
"""

import sys
import logging
from pathlib import Path
import torch
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
import librosa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def transcribe_with_chunking(model, audio_path: str, language=None):
    """Transcribe audio with automatic chunking for long files (reduces VRAM)."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr

    logger.info(f"Audio duration: {duration:.1f}s")

    # If audio is short (<2 min), process normally
    MAX_CHUNK_DURATION = 120  # 2 minutes per chunk (safe for 10GB VRAM)

    if duration <= MAX_CHUNK_DURATION:
        logger.info("Processing in single pass...")
        return model.transcribe(
            audio=(audio, sr),
            language=language
        )

    # Long audio: split into chunks
    logger.info(f"Long audio detected. Splitting into {int(duration/MAX_CHUNK_DURATION)+1} chunks...")

    chunk_samples = int(MAX_CHUNK_DURATION * sr)
    all_results = []

    for i in range(0, len(audio), chunk_samples):
        chunk_num = i // chunk_samples + 1
        chunk = audio[i:i + chunk_samples]
        logger.info(f"  Processing chunk {chunk_num}...")

        chunk_result = model.transcribe(
            audio=(chunk, sr),
            language=language
        )
        all_results.extend(chunk_result)

    return all_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_transcribe.py <audio_file_path>")
        print("Example: python test_transcribe.py my_audio.mp3")
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Loading Qwen3-ASR model...")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16,  # Use FP16 to halve memory usage
        device_map="auto",  # Auto CUDA/CPU
        max_inference_batch_size=4  # Process 4 chunks at a time (reduces VRAM)
    )

    # Apply speed optimizations
    logger.info("Applying speed optimizations...")

    try:
        # 1. BetterTransformer (20-30% faster)
        from optimum.bettertransformer import BetterTransformer
        model.model = BetterTransformer.transform(model.model)
        logger.info("✓ BetterTransformer enabled")
    except Exception as e:
        logger.warning(f"BetterTransformer not available: {e}")

    try:
        # 2. torch.compile (PyTorch 2.0+, additional 20-40% speedup)
        from torch import _dynamo
        _dynamo.config.suppress_errors = True
        model.model = torch.compile(model.model, mode="reduce-overhead")
        logger.info("✓ torch.compile() enabled")
    except Exception as e:
        logger.warning(f"torch.compile() not available: {e}")

    logger.info("✓ Model loaded!")

    logger.info(f"Transcribing: {audio_path.name}")
    results = transcribe_with_chunking(model, str(audio_path), language=None)

    # results is a list of ASRTranscription objects
    if results and len(results) > 0:
        print("\n" + "="*60)
        print("📝 TRANSCRIPTION RESULT")
        print("="*60)

        # Combine all chunks
        all_text = ""
        all_langs = set()

        for i, result in enumerate(results):
            if result.language:
                all_langs.add(result.language)
            all_text += result.text + " "

        print(f"Language(s): {', '.join(all_langs) or 'Unknown'}")
        print(f"Chunks processed: {len(results)}")
        print(f"\nText:\n{all_text.strip()}")
        print("="*60 + "\n")
    else:
        logger.warning("No transcription result returned.")

if __name__ == "__main__":
    main()
