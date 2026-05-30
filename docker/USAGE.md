# GigaAM-v3 Production Usage Guide

## Quick Start

```bash
# Start Docker service
cd docker
docker-compose up
```

Results will be saved to `./results/`

---

## Custom Usage

### Run specific audio file
```bash
docker-compose run --rm gigaam python gigaam_production.py "/audio/your_file.mp3"
```

### With custom output path
```bash
docker-compose run --rm gigaam python gigaam_production.py \
  "/audio/input.mp3" --output "/workspace/results/output.txt"
```

### Use CTC model instead of RNNT
```bash
docker-compose run --rm gigaam python gigaam_production.py \
  "/audio/input.mp3" --model v3_e2e_ctc
```

### Quiet mode (no progress output)
```bash
docker-compose run --rm gigaam python gigaam_production.py \
  "/audio/input.mp3" --quiet
```

---

## Performance Expectations

**Audio:** 3 minutes (~180s)
- Load time: ~4s (first run) / ~2s (cached)
- Transcription: ~6s
- Total: ~10s
- Speed: ~18-30x real-time

**Accuracy:** 85-90% for Russian speech/songs

---

## Supported Formats

- MP3
- WAV
- FLAC
- M4A
- OGG
- Any format supported by FFmpeg

---

## Model Variants

| Model | Best For | Accuracy | Speed |
|-------|----------|----------|-------|
| `v3_e2e_rnnt` | Songs, music | Higher | Slightly slower |
| `v3_e2e_ctc` | Clean speech | Slightly lower | Slightly faster |

**Recommendation:** Use `v3_e2e_rnnt` (default) for best results.

---

## Troubleshooting

### Models keep redownloading
Cache volumes configured - models should persist after first run.

### GPU not detected
Ensure NVIDIA Container Toolkit is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

### Out of memory
Reduce audio file size or use CPU:
- Edit docker-compose.yml: remove GPU section
- Slower but works on any hardware

---

## Files

- **Production:** `gigaam_production.py`
- **Dockerfile:** `Dockerfile.gigaam`
- **Compose:** `docker-compose.yml`
- **Results:** `./results/`
