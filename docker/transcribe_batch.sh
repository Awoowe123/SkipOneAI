#!/bin/bash
# Batch transcription - process multiple files at once
# Usage: ./transcribe_batch.sh file1.mp3 file2.mp3 file3.mp3

set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./transcribe_batch.sh <file1.mp3> <file2.mp3> ..."
    echo "Example: ./transcribe_batch.sh song1.mp3 song2.mp3 podcast.wav"
    exit 1
fi

echo "🎯 GigaAM-v3 Batch Transcription"
echo "📊 Files to process: $#"
echo ""

TOTAL=$#
CURRENT=0

for file in "$@"; do
    CURRENT=$((CURRENT + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 Processing [$CURRENT/$TOTAL]: $file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    docker-compose run --rm gigaam python gigaam_production.py "/audio/$file" --quiet

    FILENAME="${file%.*}"
    echo "✅ Saved: ../results/${FILENAME}.txt"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All files processed! ($TOTAL files)"
echo "📁 Results in: ../results/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
