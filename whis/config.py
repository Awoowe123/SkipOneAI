from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import psutil
import torch
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

TEMP_DIR = Path("temp_audio")
MODELS_DIR = Path("models")

# Создаем директории, если они не существуют
TEMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Название модели Whisper
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3-turbo")

# Тип вычислений (автоматически выбирается в зависимости от устройства, но может быть переопределен)
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto")

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

# Единые директории для моделей
# Можно переопределить через ENV:
#   WHISPER_MODELS_DIR — корень для скачивания/хранения моделей Faster-Whisper
#   CT2_MODELS_ROOT   — корень для конвертированных CT2 моделей (реестр)
WHISPER_MODELS_DIR = Path(os.getenv("WHISPER_MODELS_DIR", str(BASE_DIR / "models")))
CT2_MODELS_ROOT = Path(os.getenv("CT2_MODELS_ROOT", str(WHISPER_MODELS_DIR / "ct2")))

# Создаем директории, если они не существуют
for directory in [TEMP_DIR, LOG_DIR, WHISPER_MODELS_DIR, CT2_MODELS_ROOT]:
    directory.mkdir(exist_ok=True)

# Аудио MIME типы -> расширения файлов (для TaskProcessor._prepare_audio_to_temp)
AUDIO_MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/mp4": "m4a",
}

# Настройки Whisper
DEVICE = os.getenv("DEVICE", "cuda")  # cpu, cuda, npu, rocm
# MODEL_SIZE определен выше на строке 18

# Дополнительные модели
AVAILABLE_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",  # Стандартные модели
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",  # Версии large
    "large-v3-int8",  # Квантизованные модели
    "large-v2-ru",  # Специализированные для русского
    "distil-medium",
    "distil-large-v2",  # Дистиллированные модели
]

# Предпочтительные модели для различных устройств
DEVICE_MODEL_PREFERENCE = {
    "cpu": ["tiny", "base", "small", "distil-medium", "large-v3-int8"],
    "cuda": ["large-v3-turbo", "large-v3", "large-v2", "large-v3-int8"],
    "rocm": ["large-v3-turbo", "large-v3", "large-v2", "large-v3-int8"],
    "npu": ["small", "base", "tiny", "distil-medium", "large-v3-int8"],
}

# Настройка расширенных параметров для моделей
MODEL_CONFIG = {
    # Настройки для large-v3-turbo (оптимизированы для музыки)
    "large-v3-turbo": {
        "beam_size": 5,
        "patience": 2.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.05,
    },
    # Настройки для интегрированных NPU (AMD Ryzen AI)
    "npu_amd": {"compute_type": "int8", "cpu_threads": 16, "num_workers": 8},
}

# Настройки сети
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8765"))
WORKER_CONNECTION_TIMEOUT = 30  # увеличьте тайм-аут соединения
WORKER_HEARTBEAT_INTERVAL = 2  # секунд (уменьшено с 30 до 5 и потом до 2)
TASK_TIMEOUT = 3600  # 60 минут для очень больших файлов

# Network timeouts (секунды)
HTTP_REQUEST_TIMEOUT = int(os.getenv("HTTP_REQUEST_TIMEOUT", "30"))
WS_SEND_TIMEOUT = int(os.getenv("WS_SEND_TIMEOUT", "10"))
WS_CONNECT_TIMEOUT = int(os.getenv("WS_CONNECT_TIMEOUT", "60"))

# Buffering (байты)
AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "8192"))
WARMUP_AUDIO_SAMPLES = int(os.getenv("WARMUP_AUDIO_SAMPLES", "16000"))

# Настройки производительности
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))  # Количество потоков для обработки

# Настройки памяти
MAX_VRAM_USAGE = float(os.getenv("MAX_VRAM_USAGE", "0.0"))  # В ГБ, 0 - без ограничений

# Лимиты на размеры файлов (защита от DoS)
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "500"))  # Максимальный размер аудио файла в MB
MAX_WS_MESSAGE_SIZE_MB = int(os.getenv("MAX_WS_MESSAGE_SIZE_MB", "300"))  # Максимальный размер WebSocket сообщения

# Пороги загрузки
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "90.0"))  # %
GPU_THRESHOLD = float(os.getenv("GPU_THRESHOLD", "90.0"))  # %

# Пороги для повторной попытки распознавания песни
# Если сегментов меньше MIN_SEGMENTS — пробуем вторую попытку.
# Если сегментов меньше RELAXED_SEGMENTS и суммарная длина текста меньше MIN_CHARS_SUM — тоже пробуем повторно.
MIN_SEGMENTS = int(os.getenv("MIN_SEGMENTS", "2"))
RELAXED_SEGMENTS = int(os.getenv("RELAXED_SEGMENTS", "5"))
MIN_CHARS_SUM = int(os.getenv("MIN_CHARS_SUM", "10"))

# Черный список фраз для фильтрации из распознанного текста
BLACKLIST = []


# Функция для загрузки черного списка
def load_blacklist():
    global BLACKLIST
    try:
        blacklist_path = BASE_DIR / "data" / "blacklist.json"
        if blacklist_path.exists():
            with open(blacklist_path, encoding="utf-8") as f:
                BLACKLIST = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Ошибка при загрузке черного списка: {e}")


# Функция для получения информации о доступных моделях
def get_model_info():
    try:
        from whisper_module.whisper_models import get_available_models

        return {
            "available_models": get_available_models(),
            "selected_model": MODEL_SIZE,
            "device": DEVICE,
        }
    except ImportError:
        return {
            "available_models": AVAILABLE_MODELS,
            "selected_model": MODEL_SIZE,
            "device": DEVICE,
        }


# Функция для выбора наиболее подходящей модели для текущего устройства
def select_optimal_model():
    global MODEL_SIZE

    try:
        from whisper_module.whisper_models import (
            get_model_for_requirements,
        )

        # Определяем доступную память
        available_vram = 0

        if DEVICE == "cuda" and torch.cuda.is_available():
            # Для NVIDIA GPU
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_vram = total_vram * 0.9  # Используем 90% доступной памяти
        elif DEVICE == "rocm" and hasattr(torch, "hip"):
            # Для AMD GPU с ROCm
            available_vram = 8.0  # Предполагаем 8 ГБ для AMD GPU
        elif DEVICE == "npu":
            # Для разных NPU
            available_vram = 4.0  # Предполагаем 4 ГБ для NPU
        else:
            # Для CPU используем доступную системную память
            system_ram = psutil.virtual_memory().total / (1024**3)
            available_vram = system_ram * 0.5  # Используем половину системной памяти

        # Если пользователь указал ограничение памяти, используем его
        if MAX_VRAM_USAGE > 0:
            available_vram = min(available_vram, MAX_VRAM_USAGE)

        # Определяем минимальное необходимое качество
        min_quality = 0.7  # По умолчанию хорошее качество

        # Выбираем оптимальную модель
        model_name, model_info = get_model_for_requirements(
            min_quality=min_quality, max_vram=available_vram, device=DEVICE
        )

        # Если модель отличается от выбранной пользователем, выводим предупреждение
        if MODEL_SIZE not in (model_name, "auto"):
            print(f"Внимание: выбранная модель {MODEL_SIZE} заменена на {model_name} из-за ограничений оборудования.")

        # Устанавливаем выбранную модель
        if MODEL_SIZE == "auto":
            MODEL_SIZE = model_name
            print(f"Автоматически выбрана модель: {MODEL_SIZE} ({model_info.description})")

        return MODEL_SIZE
    except ImportError:
        # Если модуль недоступен, используем простой алгоритм
        if MODEL_SIZE == "auto":
            for model in DEVICE_MODEL_PREFERENCE.get(DEVICE, ["tiny"]):
                MODEL_SIZE = model
                break
            print(f"Автоматически выбрана модель: {MODEL_SIZE}")
        return MODEL_SIZE


# Загружаем черный список при инициализации
load_blacklist()

# Если указана автоматическая модель, выбираем оптимальную
if MODEL_SIZE == "auto":
    MODEL_SIZE = select_optimal_model()

# Настройка уровня логирования
DEBUG = os.getenv("DEBUG", "False") == "True"
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO

# Подавление сообщений при импорте модулей (полезно для unit-тестов)
QUIET_IMPORTS = os.getenv("QUIET_IMPORTS", "True") == "True"

# --- Расширяемые параметры для распознавания (VAD/beam/penalties) ---
# Параметры по умолчанию соответствуют текущему поведению кода
VAD_FILTER = os.getenv("VAD_FILTER", "False") == "True"
CONDITION_ON_PREVIOUS_TEXT = os.getenv("CONDITION_ON_PREVIOUS_TEXT", "False") == "True"
BEAM_SIZE_FIRST = int(os.getenv("BEAM_SIZE_FIRST", "5"))
BEAM_SIZE_SECOND = int(os.getenv("BEAM_SIZE_SECOND", "1"))
PATIENCE = float(os.getenv("PATIENCE", "2.0"))
LENGTH_PENALTY = float(os.getenv("LENGTH_PENALTY", "1.0"))
COMPRESSION_RATIO_THRESHOLD = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "2.4"))
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.05"))

# --- Валидация входного аудио ---
MIN_DURATION_SECONDS = float(os.getenv("MIN_DURATION_SECONDS", "0.2"))
SILENCE_DBFS_THRESHOLD = float(os.getenv("SILENCE_DBFS_THRESHOLD", "-45.0"))
