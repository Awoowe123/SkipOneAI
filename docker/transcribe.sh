#!/bin/bash
# GigaAM-v3 Transcription Helper Script
# Usage: ./transcribe.sh "filename.mp3"
# Example: ./transcribe.sh "Моя Песня.mp3"

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if filename provided
if [ -z "$1" ]; then
    echo "Usage: ./transcribe.sh <audio_filename>"
    echo "Example: ./transcribe.sh \"Моя Песня.mp3\""
    echo ""
    echo "Audio files should be in your configured AUDIO_PATH (see docker-compose.yml)"
    exit 1
fi

AUDIO_FILE="$1"

echo -e "${BLUE}🎯 GigaAM-v3 Transcription${NC}"
echo -e "${BLUE}📂 File: ${AUDIO_FILE}${NC}"
echo ""

# Run transcription
docker-compose run --rm gigaam python gigaam_production.py "/audio/${AUDIO_FILE}"

# Extract filename without extension
FILENAME="${AUDIO_FILE%.*}"
RESULT_FILE="../results/${FILENAME}.txt"

echo ""
echo -e "${GREEN}✅ Done! Result saved to:${NC}"
echo -e "${GREEN}   ${RESULT_FILE}${NC}"
