# GigaAM-v3 Quick Reference

## 🚀 Easy Usage (Recommended)

### Single file:
```bash
cd docker
./transcribe.sh "filename.mp3"
```

### Multiple files:
```bash
./transcribe_batch.sh "file1.mp3" "file2.mp3" "file3.mp3"
```

---

## 📋 Manual Commands

### Process one file:
```bash
docker-compose run --rm gigaam python gigaam_production.py "/audio/file.mp3"
```

### With options:
```bash
docker-compose run --rm gigaam python gigaam_production.py \
  "/audio/file.mp3" \
  --output "/workspace/results/custom.txt" \
  --model v3_e2e_ctc \
  --quiet
```

### Edit default file (docker-compose.yml):
```yaml
command: python gigaam_production.py /audio/YourFile.mp3
```
Then: `docker-compose up`

---

## 📂 File Locations

**Input:** `~/Downloads/` → mounted as `/audio/`
**Output:** `./results/`

---

## 🎯 Examples

```bash
# Single file
./transcribe.sh "Лжедмитрий IV feat. LeanJe - Путь в рай.mp3"

# Batch processing
./transcribe_batch.sh \
  "song1.mp3" \
  "song2.mp3" \
  "podcast.wav"

# Manual with quiet mode
docker-compose run --rm gigaam python gigaam_production.py \
  "/audio/interview.mp3" --quiet
```

---

## ⚡ Performance

- First run: ~125s (model download)
- Cached: ~4s load + ~5s transcription
- Speed: 30-40x real-time
- Accuracy: ~87%
