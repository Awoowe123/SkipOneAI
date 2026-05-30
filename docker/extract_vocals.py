#!/usr/bin/env python3
"""
Advanced Vocal Extraction Script
Supports multiple state-of-the-art models:
- BS-Roformer (best quality, slower)
- Demucs v4 (fast, good quality)
- MDX23C (balanced)
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


MODELS = {
    "bs-roformer": {
        "name": "BS-Roformer ViperX",
        "description": "🏆 Best quality (gold standard)",
        "speed": "🐌 Slow",
        "vram": "8-12GB"
    },
    "demucs": {
        "name": "Demucs v4 (htdemucs_ft)",
        "description": "⚡ Fast with good quality",
        "speed": "🚀 Fast",
        "vram": "4-6GB"
    },
    "mdx23c": {
        "name": "MDX23C InstVoc HQ",
        "description": "⚖️ Balanced quality/speed",
        "speed": "⏱️ Medium",
        "vram": "6-8GB"
    }
}


def extract_vocals_demucs(audio_path: str, output_dir: str, verbose: bool = True):
    """Extract vocals using Demucs v4."""
    import subprocess

    if verbose:
        print(f"\n⏳ Extracting vocals with Demucs v4 (htdemucs_ft)...")

    demucs_cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs_ft",
        "-o", "/tmp/demucs_output",
        audio_path
    ]

    try:
        if verbose:
            subprocess.run(demucs_cmd, check=True)
        else:
            subprocess.run(demucs_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs failed: {e}")

    vocals_path = f"/tmp/demucs_output/htdemucs_ft/{Path(audio_path).stem}/vocals.wav"

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Demucs output not found: {vocals_path}")

    return vocals_path


def extract_vocals_separator(audio_path: str, output_dir: str, model_name: str, verbose: bool = True):
    """Extract vocals using audio-separator (BS-Roformer, MDX23C, etc.)."""
    from audio_separator.separator import Separator

    if verbose:
        model_info = MODELS.get(model_name, {})
        print(f"\n⏳ Extracting vocals with {model_info.get('name', model_name)}...")

    # Map model names to audio-separator model files
    model_mapping = {
        "bs-roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "mdx23c": "MDX23C-8KFFT-InstVoc_HQ.ckpt"
    }

    separator_model = model_mapping.get(model_name)
    if not separator_model:
        raise ValueError(f"Model {model_name} not supported via audio-separator")

    # Initialize separator
    separator = Separator(
        model_file_dir="/root/.cache/audio-separator-models",
        output_dir="/tmp/separator_output",
        output_format="wav"
    )

    # Load model
    separator.load_model(model_filename=separator_model)

    # Separate vocals
    output_files = separator.separate(audio_path)

    # Find vocals file (audio-separator returns filenames, need to build full paths)
    vocals_path = None
    for filename in output_files:
        if "Vocals" in filename or "vocals" in filename:
            # Build full path
            full_path = os.path.join("/tmp/separator_output", filename)
            if os.path.exists(full_path):
                vocals_path = full_path
                break

    if not vocals_path:
        raise FileNotFoundError(f"Vocals output not found in {output_files}")

    return vocals_path


def extract_vocals(
    audio_path: str,
    output_dir: str = "/workspace/results/audio",
    model: str = "demucs",
    verbose: bool = True
):
    """
    Extract vocals from audio using the specified model.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save extracted vocals
        model: Model to use ('bs-roformer', 'demucs', 'mdx23c')
        verbose: Print progress messages

    Returns:
        Path to extracted vocals file
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get filename
    audio_filename = Path(audio_path).stem

    if verbose:
        model_info = MODELS.get(model, {})
        print("\n" + "="*80)
        print(f"Advanced Vocal Extraction - {model_info.get('name', model)}")
        print("="*80)
        print(f"\n📂 Input: {audio_path}")
        print(f"📁 Output: {output_dir}/")
        print(f"\n🎯 Model: {model_info.get('description', model)}")
        print(f"⚡ Speed: {model_info.get('speed', 'N/A')}")
        print(f"💾 VRAM: {model_info.get('vram', 'N/A')}")

    start_time = time.time()

    # Extract vocals using appropriate method
    try:
        if model == "demucs":
            vocals_path = extract_vocals_demucs(audio_path, output_dir, verbose)
        elif model in ["bs-roformer", "mdx23c"]:
            vocals_path = extract_vocals_separator(audio_path, output_dir, model, verbose)
        else:
            raise ValueError(f"Unknown model: {model}. Use one of: {list(MODELS.keys())}")
    except Exception as e:
        raise RuntimeError(f"Vocal extraction failed: {e}")

    # Copy to output directory
    output_path = os.path.join(output_dir, f"{audio_filename}_vocals.wav")
    shutil.copy(vocals_path, output_path)

    # Cleanup temporary files
    shutil.rmtree("/tmp/demucs_output", ignore_errors=True)
    shutil.rmtree("/tmp/separator_output", ignore_errors=True)

    extraction_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

    if verbose:
        print(f"✅ Vocals extracted in {extraction_time:.1f}s")
        print(f"\n📁 Saved to: {output_path}")
        print(f"📊 File size: {file_size:.1f} MB")
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocals from audio using state-of-the-art models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Models:
-----------------
{chr(10).join([f"  {k}: {v['description']} - {v['speed']}" for k, v in MODELS.items()])}

Examples:
  # Best quality (slow)
  python extract_vocals.py "song.mp3" --model bs-roformer

  # Fast with good quality (default)
  python extract_vocals.py "song.mp3"

  # Balanced
  python extract_vocals.py "song.mp3" --model mdx23c
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to audio file"
    )

    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()),
        default="demucs",
        help="Model to use (default: demucs)"
    )

    parser.add_argument(
        "-o", "--output",
        default="/workspace/results/audio",
        help="Output directory (default: /workspace/results/audio)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        extract_vocals(
            audio_path=args.audio_file,
            output_dir=args.output,
            model=args.model,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
