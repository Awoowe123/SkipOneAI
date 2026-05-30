import os
import sys
import json
import logging
import torch
import subprocess
import tempfile
import soundfile as sf

# CRITICAL: Import patch BEFORE nemo to fix Windows incompatibility
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

def create_manifest(audio_path, output_manifest, language='ru'):
    """Create a JSONL manifest file with proper metadata for Canary."""
    # Get audio duration
    try:
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
    except:
        duration = None

    manifest_entry = {
        "audio_filepath": audio_path,
        "duration": duration,
        "taskname": "asr",
        "source_lang": language,  # ru for Russian
        "target_lang": language,  # Same as source for ASR
        "pnc": "yes",  # Enable punctuation and capitalization
        "answer": "na"
    }

    with open(output_manifest, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest_entry) + '\n')

    print(f"Created manifest: {output_manifest}")
    return output_manifest

def test_canary_optimized():
    """Test Canary with optimized decoding parameters."""
    print("--- Canary-1B-v2 Test (OPTIMIZED) ---")

    # Check CLI args
    target_file = None
    original_file = None
    temp_wav = None
    manifest_file = None

    if len(sys.argv) > 1:
        target_file = sys.argv[1].strip('"').strip("'")
        if not os.path.exists(target_file):
            print(f"ERROR: File not found: {target_file}")
            return
        original_file = target_file
    else:
        print("Usage: python tests/test_canary_optimized.py path/to/audio.mp3")
        return

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # Load Model
        print("Loading model 'nvidia/canary-1b-v2'...")
        asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        asr_model = asr_model.to(device)
        asr_model.eval()

        # OPTIMIZE DECODING PARAMETERS
        print("Configuring optimized decoding strategy...")
        decode_cfg = asr_model.cfg.decoding

        # Print available beam parameters for debugging
        print(f"Available beam config: {decode_cfg.beam}")

        decode_cfg.beam.beam_size = 5  # Beam width 5 (NVIDIA recommendation)
        decode_cfg.beam.len_pen = 1.0  # Length penalty (correct param name!)
        asr_model.change_decoding_strategy(decode_cfg)

        print("Model loaded and configured successfully!")

        print(f"Testing with: {original_file}")

        # Convert if not WAV
        if not target_file.lower().endswith(".wav"):
            print("Converting to WAV...")
            temp_wav = convert_to_wav(target_file)
            target_file = temp_wav

        # Create JSONL manifest
        fd, manifest_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        create_manifest(target_file, manifest_file, language='ru')

        # Transcribe using manifest with patched chunking
        print(f"Transcribing (beam_size=5, len_pen=1.0, chunking=ON [patched])...")

        # Import proper config type and helper classes for patching
        from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig
        # Fixed import path based on grep search
        from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
        from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextLhotseDataset

        # --- MONKEY PATCH TO FIX HARDCODED 3600s CHUNKING ---
        def patched_setup_dataloader_from_config(self, config):
            # Verify we are using Lhotse
            # assert config.get("use_lhotse", False) # Skipping assert to be safe

            global_rank = config.get("global_rank", self.global_rank)
            world_size = config.get("world_size", self.world_size)
            enable_chunking = config.get("enable_chunking", False)

            if enable_chunking:
                # FORCE 40s duration (Canary max) instead of default 3600s
                print("[PATCH] Forcing cut_into_windows_duration = 40.0")
                config.cut_into_windows_duration = 40.0
                config.cut_into_windows_hop = 20.0 # 50% overlap

            return get_lhotse_dataloader_from_config(
                config,
                global_rank=global_rank,
                world_size=world_size,
                dataset=PromptedAudioToTextLhotseDataset(
                    tokenizer=self.tokenizer,
                    prompt=self.prompt,
                    enable_chunking=enable_chunking,
                ),
                tokenizer=self.tokenizer,
            )

        # Apply patch to the model instance
        import types
        asr_model._setup_dataloader_from_config = types.MethodType(patched_setup_dataloader_from_config, asr_model)
        # ----------------------------------------------------

        # Create override config
        override_cfg = MultiTaskTranscriptionConfig(
            batch_size=1,
            num_workers=0,
        )

        # Manually enable chunking flag (it's not in init args for some versions)
        try:
            override_cfg.enable_chunking = True
        except:
            print("Warning: Could not set enable_chunking on config")

        try:
            transcriptions = asr_model.transcribe(
                manifest_file,
                override_config=override_cfg
            )

            print("\n" + "="*60)
            print("TRANSCRIPTION RESULT (OPTIMIZED):")
            print("="*60)
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                result = transcriptions[0]
                if hasattr(result, 'text'):
                    print(result.text)
                else:
                    print(result)
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

        if manifest_file and os.path.exists(manifest_file):
            try:
                os.remove(manifest_file)
            except:
                pass

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_canary_optimized()
