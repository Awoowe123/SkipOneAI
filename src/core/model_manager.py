"""Model management for LM Studio - automatic reload when performance degrades"""
import requests
import time
import logging
import asyncio
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages LM Studio model lifecycle (load/unload/swap)"""

    def __init__(self, base_url: str, model: str):
        """
        Initialize model manager

        Args:
            base_url: LM Studio base URL (e.g., http://localhost:1234)
            model: Default model identifier/path
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = model
        self.last_reload_time = 0
        self.loaded_models = set()  # Track loaded models
        self.hot_model = "qwen2.5-0.5b-instruct"  # Always keep this loaded

    async def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded models from LM Studio

        Returns:
            List of loaded model names
        """
        try:
            # Use OpenAI-compatible endpoint for checking loaded models
            url = f"{self.base_url}/v1/models"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, timeout=10)
            )

            if response.status_code == 200:
                data = response.json()
                models = [m['id'] for m in data.get('data', [])]
                self.loaded_models = set(models)
                logger.debug(f"[MODEL_MANAGER] Loaded models: {models}")
                return models
            else:
                logger.warning(f"[MODEL_MANAGER] Failed to get models: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"[MODEL_MANAGER] Error getting models: {e}")
            return []

    async def load_model(self, model_name: str, wait_ready: bool = True) -> bool:
        """
        Load a specific model into memory with optimized settings

        Args:
            model_name: Model identifier to load
            wait_ready: If True, wait for model to be fully loaded

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already loaded
            loaded = await self.get_loaded_models()
            if model_name in loaded:
                logger.info(f"[MODEL_MANAGER] Model {model_name} already loaded")
                return True

            logger.info(f"[MODEL_MANAGER] Loading model: {model_name}")

            load_url = f"{self.base_url}/api/v1/models/load"  # Correct API path
            load_payload = {
                "model": model_name,
                "context_length": 32768,  # 32K context
                "flash_attention": True,
                "offload_kv_cache_to_gpu": True,
                "eval_batch_size": 512,
                "echo_load_config": True
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(load_url, json=load_payload, timeout=120)
            )

            if response.status_code not in [200, 201]:
                logger.error(f"[MODEL_MANAGER] Load failed {response.status_code}: {response.text}")
                return False

            # Log load config
            try:
                result = response.json()
                logger.info(f"[MODEL_MANAGER] Load time: {result.get('load_time_seconds', 'N/A')}s")
                if 'load_config' in result:
                    logger.debug(f"[MODEL_MANAGER] Config: {result['load_config']}")
            except:
                pass

            if wait_ready:
                # Wait for model to be ready (poll every 2s, max 60s)
                for _ in range(30):
                    await asyncio.sleep(2)
                    loaded = await self.get_loaded_models()
                    if model_name in loaded:
                        logger.info(f"[MODEL_MANAGER] ✅ {model_name} loaded successfully")
                        return True

                logger.warning(f"[MODEL_MANAGER] {model_name} load timeout")
                return False

            logger.info(f"[MODEL_MANAGER] ✅ {model_name} load initiated")
            return True

        except Exception as e:
            logger.error(f"[MODEL_MANAGER] Load error: {e}")
            return False

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from memory

        Args:
            model_name: Model identifier to unload

        Returns:
            True if successful, False otherwise
        """
        try:
            # Never unload the hot model
            if model_name == self.hot_model:
                logger.warning(f"[MODEL_MANAGER] Refusing to unload hot model: {model_name}")
                return False

            logger.info(f"[MODEL_MANAGER] Unloading model: {model_name}")

            unload_url = f"{self.base_url}/api/v1/models/unload"  # Correct API path
            unload_payload = {"instance_id": model_name}  # Correct field name

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(unload_url, json=unload_payload, timeout=30)
            )

            if response.status_code not in [200, 201, 204]:
                logger.warning(f"[MODEL_MANAGER] Unload returned {response.status_code}: {response.text}")
                # Not critical if unload fails

            # Remove from tracking
            self.loaded_models.discard(model_name)
            logger.info(f"[MODEL_MANAGER] ✅ {model_name} unloaded")
            return True

        except Exception as e:
            logger.error(f"[MODEL_MANAGER] Unload error: {e}")
            return False

    async def ensure_loaded(self, model_name: str) -> bool:
        """
        Ensure a model is loaded, loading it if necessary

        Args:
            model_name: Model to ensure is loaded

        Returns:
            True if model is now loaded, False otherwise
        """
        loaded = await self.get_loaded_models()

        if model_name in loaded:
            logger.debug(f"[MODEL_MANAGER] {model_name} already loaded")
            return True

        logger.info(f"[MODEL_MANAGER] Ensuring {model_name} is loaded...")
        return await self.load_model(model_name, wait_ready=True)

    async def smart_swap(self, from_model: Optional[str], to_model: str) -> bool:
        """
        Smart model swap: unload old model, load new one

        Keeps hot model (qwen) always loaded

        Args:
            from_model: Model to unload (None = don't unload anything)
            to_model: Model to load

        Returns:
            True if swap successful
        """
        logger.info(f"[MODEL_MANAGER] Smart swap: {from_model} -> {to_model}")

        # Load new model first (ensures we don't lose all models)
        success = await self.ensure_loaded(to_model)
        if not success:
            logger.error(f"[MODEL_MANAGER] Failed to load {to_model}, aborting swap")
            return False

        # Unload old model (if specified and not hot)
        if from_model and from_model != self.hot_model and from_model != to_model:
            # Check if model is actually loaded before trying to unload
            loaded = await self.get_loaded_models()
            if from_model in loaded:
                await self.unload_model(from_model)
            else:
                logger.debug(f"[MODEL_MANAGER] {from_model} already unloaded, skipping")

        return True

    def reload_model(self) -> bool:
        """
        Legacy sync method - now wraps async version
        Unload and reload default model to clear memory/cache

        Returns:
            True if reload successful, False otherwise
        """
        try:
            logger.info(f"[MODEL_RELOAD] Starting reload for {self.default_model}")

            # Step 1: Unload model
            unload_url = f"{self.base_url}/v1/models/unload"
            unload_payload = {"model": self.default_model}

            logger.debug(f"[MODEL_RELOAD] Unloading via {unload_url}")
            unload_response = requests.post(
                unload_url,
                json=unload_payload,
                timeout=30
            )

            if unload_response.status_code not in [200, 201, 204]:
                logger.warning(f"[MODEL_RELOAD] Unload returned {unload_response.status_code}: {unload_response.text}")

            # Step 2: Wait for memory cleanup
            logger.debug("[MODEL_RELOAD] Waiting 5s for memory cleanup")
            time.sleep(5)

            # Step 3: Load model back
            load_url = f"{self.base_url}/v1/models/load"
            load_payload = {"model": self.default_model}

            logger.debug(f"[MODEL_RELOAD] Loading via {load_url}")
            load_response = requests.post(
                load_url,
                json=load_payload,
                timeout=60
            )

            if load_response.status_code not in [200, 201]:
                logger.error(f"[MODEL_RELOAD] Load failed {load_response.status_code}: {load_response.text}")
                return False

            # Success!
            self.last_reload_time = time.time()
            logger.info(f"[MODEL_RELOAD] ✅ Successfully reloaded {self.default_model}")
            return True

        except requests.exceptions.Timeout:
            logger.error("[MODEL_RELOAD] Request timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[MODEL_RELOAD] Request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[MODEL_RELOAD] Unexpected error: {e}")
            return False

    def can_reload(self, cooldown_seconds: int) -> bool:
        """
        Check if enough time has passed since last reload

        Args:
            cooldown_seconds: Minimum seconds between reloads

        Returns:
            True if reload is allowed, False if in cooldown
        """
        if self.last_reload_time == 0:
            return True

        elapsed = time.time() - self.last_reload_time
        return elapsed >= cooldown_seconds
