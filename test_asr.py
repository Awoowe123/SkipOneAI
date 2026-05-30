import logging
import torch
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "Qwen/Qwen3-ASR-1.7B"

try:
    logger.info("Attempting to load model with Qwen3ASRModel (qwen-asr package)...")

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True
        # device_map="cpu" # For testing. In production use self.device
    )
    logger.info("Model loaded successfully!")
    logger.info(f"Processor loaded internal: {model.processor}")

    # Test transcribe?? (Requires audio file)
    logger.info("Skipping inference test due to no audio file.")

except Exception as e:
    logger.error(f"Failed: {e}", exc_info=True)
