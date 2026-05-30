#!/bin/bash
# Advanced vocal extraction wrapper
# Supports BS-Roformer, Demucs v4, and MDX23C
# Usage: ./extract_vocals.sh "song.mp3" [model]

set -e

AUDIO_FILE="$1"
MODEL="${2:-demucs}"  # Default to demucs

if [ -z "$AUDIO_FILE" ]; then
    echo "Usage: $0 <audio_file> [model]"
    echo ""
    echo "Models:"
    echo "  bs-roformer  - 🏆 Best quality (gold standard, slow)"
    echo "  demucs       - ⚡ Fast with good quality (default)"
    echo "  mdx23c       - ⚖️ Balanced quality/speed"
    echo ""
    echo "Examples:"
    echo "  $0 \"song.mp3\"                  # Use default (demucs)"
    echo "  $0 \"song.mp3\" bs-roformer      # Best quality"
    echo "  $0 \"song.mp3\" mdx23c           # Balanced"
    exit 1
fi

# Run Docker container
docker-compose run --rm gigaam python /workspace/extract_vocals.py "/audio/$(basename "$AUDIO_FILE")" --model "$MODEL"

# Extract filename without extension for output message
FILENAME=$(basename "$AUDIO_FILE")
BASENAME="${FILENAME%.*}"

echo ""
echo "✅ Vocals saved to: ../results/audio/${BASENAME}_vocals.wav"
