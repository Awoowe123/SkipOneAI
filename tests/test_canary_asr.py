import os
import sys
import logging
import torch
import subprocess
import tempfile
import soundfile as sf

# CRITICAL: Import patch BEFORE nemo to fix Windows incompatibility
# Add parent directory to path so we can import lhotse_patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lhotse_patch  # noqa

# Suppress heavy NeMo logs
os.environ["NEMO_LOG_LEVEL"] = "ERROR"

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("CRITICAL: NeMo toolkit not found. Install it with: pip install nemo_toolkit[asr]")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(input_path):
    """Converts audio to 16kHz mono WAV using ffmpeg."""
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            temp_path
        ]
        print(f"Converting {input_path} to WAV...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path
    except Exception as e:
        print(f"Warning: ffmpeg conversion failed: {e}")
        return input_path

def test_canary():
    """Test Canary using patched lhotse."""
    print("--- Starting Canary-1B Test (with lhotse patch) ---")

    # Check CLI args
    target_file = None
    original_file = None
    temp_wav = None

    if len(sys.argv) > 1:
        target_file = sys.argv[1].strip('"').strip("'")
        if not os.path.exists(target_file):
            print(f"ERROR: File not found: {target_file}")
            return
        original_file = target_file
    else:
        print("Usage: python tests/test_canary_asr.py path/to/audio.mp3")
        return

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # Load Model (v2 supports more languages including Russian)
        print("Loading model 'nvidia/canary-1b-v2'...")
        asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        asr_model = asr_model.to(device)
        asr_model.eval()
        print("Model loaded successfully!")

        print(f"Testing with provided file: {original_file}")

        # Convert if not WAV
        if not target_file.lower().endswith(".wav"):
            print("Input is not .wav, attempting conversion...")
            temp_wav = convert_to_wav(target_file)
            target_file = temp_wav

        # Transcribe using patched lhotse
        print(f"Transcribing {target_file}...")

        try:
            # Canary is multilingual - we need to specify the language
            # For Russian: source_lang='ru', target_lang='ru'
            # For English: source_lang='en', target_lang='en'
            transcriptions = asr_model.transcribe(
                audio=[target_file],
                batch_size=1,
                num_workers=0,
                # Try Russian first (based on filename)
                source_lang='ru',
                target_lang='ru'
            )

            print("\n" + "="*60)
            print("TRANSCRIPTION RESULT:")
            print("="*60)
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                print(transcriptions[0])
            else:
                print(transcriptions)
            print("="*60)

        except Exception as e:
            print(f"\nTranscription failed: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_canary()
