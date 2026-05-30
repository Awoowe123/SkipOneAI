#!/usr/bin/env python3
"""
Transcription Benchmark Script
Compares 5 different transcription methods:
1. Whisper on original MP3
2. Whisper on BS-Roformer vocals
3. GigaAM on original MP3
4. GigaAM on BS-Roformer vocals
5. BS-Roformer + GigaAM (Demucs pipeline)
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, cwd=None, description=""):
    """Run shell command and return output."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            check=True
        )
        duration = time.time() - start_time

        # Decode stdout
        try:
            stdout = result.stdout.decode('utf-8')
        except:
            try:
                stdout = result.stdout.decode('cp1251')
            except:
                stdout = result.stdout.decode('utf-8', errors='replace')

        # Decode stderr
        stderr = ""
        if result.stderr:
            try:
                stderr = result.stderr.decode('utf-8')
            except:
                try:
                    stderr = result.stderr.decode('cp1251')
                except:
                    stderr = result.stderr.decode('utf-8', errors='replace')

        print(stdout)
        if stderr:
            print(stderr)

        return {
            "success": True,
            "duration": duration,
            "output": stdout,
            "error": None
        }
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time

        # Decode error outputs
        try:
            err_stdout = e.stdout.decode('utf-8')
        except:
            try:
                err_stdout = e.stdout.decode('cp1251')
            except:
                err_stdout = e.stdout.decode('utf-8', errors='replace')

        try:
            err_stderr = e.stderr.decode('utf-8')
        except:
            try:
                err_stderr = e.stderr.decode('cp1251')
            except:
                err_stderr = e.stderr.decode('utf-8', errors='replace')

        print(f"❌ Error: {e}")
        print(err_stdout)
        print(err_stderr)

        return {
            "success": False,
            "duration": duration,
            "output": err_stdout,
            "error": str(e)
        }


async def run_whisper_test(audio_path: str):
    """Run Whisper transcription via test_faster_whisper_song.py."""
    # Modify test file to use correct audio path
    test_file = str(Path(__file__).resolve().parent / "tests" / "test_faster_whisper_song.py")

    # Read original
    with open(test_file, 'r', encoding='utf-8') as f:
        original = f.read()

    # Replace audio path
    modified = original
    for line in original.split('\n'):
        if 'audio_file =' in line and line.strip().startswith('audio_file'):
            # Replace with new path
            modified = original.replace(
                line,
                f'    audio_file = r"{audio_path}"'
            )
            break

    # Write modified
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(modified)

    # Run test
    start_time = time.time()
    process = await asyncio.create_subprocess_exec(
        sys.executable, test_file,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    duration = time.time() - start_time

    # Restore original
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(original)

    # Decode with error handling for non-UTF-8 bytes (Windows console)
    try:
        output = stdout.decode('utf-8')
    except UnicodeDecodeError:
        # Fallback to cp1251 (Windows Cyrillic) or replace errors
        try:
            output = stdout.decode('cp1251')
        except:
            output = stdout.decode('utf-8', errors='replace')

    # Extract transcription from output
    transcription = ""
    if "=== TRANSCRIPTION RESULT ===" in output:
        # Parse between === TRANSCRIPTION RESULT === and ============================
        parts = output.split("=== TRANSCRIPTION RESULT ===")
        if len(parts) > 1:
            result_part = parts[1].split("============================")[0]
            transcription = result_part.strip()

    # Decode stderr with same error handling
    error_msg = None
    if stderr:
        try:
            error_msg = stderr.decode('utf-8')
        except UnicodeDecodeError:
            try:
                error_msg = stderr.decode('cp1251')
            except:
                error_msg = stderr.decode('utf-8', errors='replace')

    return {
        "success": process.returncode == 0,
        "duration": duration,
        "output": output,
        "transcription": transcription,
        "error": error_msg
    }


def extract_vocals_bs_roformer(audio_path: str):
    """Extract vocals using BS-Roformer."""
    docker_dir = str(Path(__file__).resolve().parent / "docker")
    audio_basename = os.path.basename(audio_path)

    result = run_command(
        ["bash", "./extract_vocals.sh", audio_basename, "bs-roformer"],
        cwd=docker_dir,
        description="Step 0: Extracting vocals with BS-Roformer"
    )

    # Find vocals file
    audio_stem = Path(audio_path).stem
    project_root = Path(__file__).resolve().parent
    vocals_path = str(project_root / "results" / "audio" / f"{audio_stem}_vocals.wav")

    if os.path.exists(vocals_path):
        result["vocals_path"] = vocals_path

    return result


def transcribe_gigaam(audio_path: str):
    """Transcribe audio using GigaAM via Docker."""
    docker_dir = str(Path(__file__).resolve().parent / "docker")
    audio_basename = os.path.basename(audio_path)

    # If vocals file from results/audio, copy to Downloads first
    if "results\\audio" in audio_path or "results/audio" in audio_path:
        downloads_path = str(Path.home() / "Downloads")
        temp_audio = os.path.join(downloads_path, audio_basename)
        if not os.path.exists(temp_audio):
            shutil.copy(audio_path, temp_audio)
        # Use temp file for docker
        docker_audio_path = temp_audio
    else:
        docker_audio_path = audio_path

    # Use basename from docker_audio_path
    docker_basename = os.path.basename(docker_audio_path)

    result = run_command(
        ["bash", "./transcribe.sh", docker_basename],
        cwd=docker_dir,
        description=f"Transcribing with GigaAM: {audio_basename}"
    )

    # Extract transcription from output (GigaAM outputs to results/*.txt)
    audio_stem = Path(audio_path).stem
    project_root = Path(__file__).resolve().parent
    result_file = str(project_root / "results" / f"{audio_stem}.txt")

    transcription = ""
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        result["transcription"] = transcription
        result["result_file"] = result_file

    return result


async def benchmark_transcription(audio_path: str):
    """Run full benchmark comparing all methods."""
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        return

    print("\n" + "="*80)
    print("🎯 TRANSCRIPTION BENCHMARK")
    print("="*80)
    print(f"\n📂 Audio file: {audio_path}")
    print(f"📊 Running 5 transcription methods...\n")

    results = {}

    # Step 0: Extract vocals with BS-Roformer
    print("\n" + "🎵 "*30)
    vocals_result = extract_vocals_bs_roformer(audio_path)
    results["bs_roformer_extraction"] = vocals_result

    if not vocals_result["success"] or "vocals_path" not in vocals_result:
        print("❌ Failed to extract vocals, skipping vocal-based tests")
        vocals_path = None
    else:
        vocals_path = vocals_result["vocals_path"]
        print(f"✅ Vocals extracted: {vocals_path}")

    # Step 1: Whisper on original MP3
    print("\n" + "🎵 "*30)
    print("\n📝 Method 1: Whisper on original MP3")
    results["whisper_original"] = await run_whisper_test(audio_path)

    # Step 2: Whisper on BS-Roformer vocals
    if vocals_path:
        print("\n" + "🎵 "*30)
        print("\n📝 Method 2: Whisper on BS-Roformer vocals")
        results["whisper_vocals"] = await run_whisper_test(vocals_path)

    # Step 3: GigaAM on original MP3
    print("\n" + "🎵 "*30)
    print("\n📝 Method 3: GigaAM on original MP3")
    results["gigaam_original"] = transcribe_gigaam(audio_path)

    # Step 4: GigaAM on BS-Roformer vocals
    if vocals_path:
        print("\n" + "🎵 "*30)
        print("\n📝 Method 4: GigaAM on BS-Roformer vocals")
        results["gigaam_vocals"] = transcribe_gigaam(vocals_path)

    # Print summary
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*80)

    methods = [
        ("BS-Roformer Extraction", "bs_roformer_extraction"),
        ("Whisper (original)", "whisper_original"),
        ("Whisper (BS-Roformer vocals)", "whisper_vocals"),
        ("GigaAM (original)", "gigaam_original"),
        ("GigaAM (BS-Roformer vocals)", "gigaam_vocals"),
    ]

    print(f"\n{'Method':<35} {'Status':<10} {'Duration':<12} {'Transcription Preview'}")
    print("-" * 120)

    for name, key in methods:
        if key not in results:
            continue

        result = results[key]
        status = "✅ Success" if result["success"] else "❌ Failed"
        duration = f"{result['duration']:.1f}s"

        transcription = result.get("transcription", "")
        preview = (transcription[:60] + "...") if len(transcription) > 60 else transcription

        print(f"{name:<35} {status:<10} {duration:<12} {preview}")

    # Print full transcriptions
    print("\n" + "="*80)
    print("📝 FULL TRANSCRIPTIONS")
    print("="*80)

    for name, key in methods:
        if key not in results or key == "bs_roformer_extraction":
            continue

        result = results[key]
        transcription = result.get("transcription", "N/A")

        print(f"\n{'─'*80}")
        print(f"🎯 {name}")
        print(f"{'─'*80}")
        print(transcription)

    # Print vocals path
    if vocals_path:
        print("\n" + "="*80)
        print("📁 VOCALS FILE")
        print("="*80)
        print(f"\n✅ Vocals saved to: {vocals_path}")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE!")
    print("="*80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transcription methods: Whisper vs GigaAM with/without BS-Roformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_transcription.py "path/to/song.mp3"
  python benchmark_transcription.py "/audio/Alex Masht - Свет.mp3"
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to audio file (MP3)"
    )

    args = parser.parse_args()

    # Run benchmark
    asyncio.run(benchmark_transcription(args.audio_file))


if __name__ == "__main__":
    main()
