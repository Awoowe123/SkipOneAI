#!/bin/bash
# Wrapper for benchmark_transcription.py
# Usage: ./benchmark.sh "song.mp3"

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "$1" ]; then
    echo "Usage: $0 <audio_file>"
    echo ""
    echo "This will run 5 transcription methods and compare results:"
    echo "  1. Whisper on original MP3"
    echo "  2. Whisper on BS-Roformer vocals"
    echo "  3. GigaAM on original MP3"
    echo "  4. GigaAM on BS-Roformer vocals"
    echo "  0. BS-Roformer vocal extraction"
    echo ""
    echo "Examples:"
    echo "  $0 \"song.mp3\""
    echo "  $0 \"path/to/song.mp3\""
    exit 1
fi

AUDIO_FILE="$1"

# Convert to absolute path if needed
if [[ ! "$AUDIO_FILE" =~ ^[A-Z]:\\ ]] && [[ ! "$AUDIO_FILE" =~ ^/ ]]; then
    # Relative path - convert to absolute
    AUDIO_FILE="$(cd "$(dirname "$AUDIO_FILE")" && pwd)/$(basename "$AUDIO_FILE")"
fi

echo "Running benchmark for: $AUDIO_FILE"
python "$SCRIPT_DIR/benchmark_transcription.py" "$AUDIO_FILE"
