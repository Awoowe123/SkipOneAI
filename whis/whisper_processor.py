from __future__ import annotations

import gc
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

# Безопасное значение по умолчанию до импорта config
# Нужно, потому что ниже есть выводы, зависящие от QUIET_IMPORTS, до импорта config
QUIET_IMPORTS = True

# Определяем доступность аппаратных ускорителей
cuda_available = torch.cuda.is_available()

# Проверяем доступность ROCm (AMD GPU)
rocm_available = False
if not cuda_available and hasattr(torch, "hip") and hasattr(torch.hip, "is_available"):
    rocm_available = torch.hip.is_available()
    if rocm_available and not QUIET_IMPORTS:
        print("ROCm (AMD GPU) доступен")

# Проверяем доступность разных типов NPU
npu_available = False
npu_type = "unknown"
npu_vendor = "unknown"


# Типовой протокол для слова и сегмента распознавания
@runtime_checkable
class WordLike(Protocol):
    start: float
    end: float
    word: str


@runtime_checkable
class SegmentLike(Protocol):
    start: float
    text: str
    # Поле words может отсутствовать; если присутствует — последовательность WordLike
    # Используем Any по умолчанию, но ожидаемый тип — Optional[Sequence[WordLike]]
    words: Any


# Кэш для определения NPU (чтобы не запускать тяжёлые проверки каждый раз)
_NPU_CACHE_FILE = Path.home() / ".cache" / "whisper_worker" / "npu_detect.json"
_NPU_CACHE_TTL = 3600  # 1 час


def _load_npu_cache() -> dict | None:
    """Загружает кэш определения NPU"""
    try:
        if _NPU_CACHE_FILE.exists():
            cache = json.loads(_NPU_CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - cache.get("timestamp", 0) < _NPU_CACHE_TTL:
                return cache
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _save_npu_cache(available: bool, npu_type_val: str, vendor: str) -> None:
    """Сохраняет результат определения NPU в кэш"""
    try:
        _NPU_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {"timestamp": time.time(), "npu_available": available, "npu_type": npu_type_val, "npu_vendor": vendor}
        _NPU_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except (OSError, TypeError):
        pass  # Не критично, если не удалось сохранить


# Функция для определения типа NPU
def detect_npu() -> bool:
    global npu_available, npu_type, npu_vendor

    # Проверяем кэш
    cached = _load_npu_cache()
    if cached is not None:
        npu_available = cached["npu_available"]
        npu_type = cached["npu_type"]
        npu_vendor = cached["npu_vendor"]
        if not QUIET_IMPORTS and npu_available:
            print(f"NPU определён из кэша: {npu_vendor} {npu_type}")
        return npu_available

    # Проверяем Huawei/HiSilicon NPU
    try:
        import torch_npu

        if hasattr(torch_npu, "is_available") and torch_npu.is_available():
            npu_available = True
            npu_type = "ascend"
            npu_vendor = "huawei"
            if not QUIET_IMPORTS:
                print("Huawei Ascend NPU доступен")
            _save_npu_cache(True, npu_type, npu_vendor)
            return True
    except ImportError:
        pass

    # Проверяем AMD XDNA/NPU (Ryzen AI)
    try:
        # Пытаемся определить AMD NPU через системную информацию
        import shutil
        import subprocess

        # Проверка для Linux
        if platform.system() == "Linux":
            try:
                # Проверяем наличие XDNA в информации о CPU
                try:
                    with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                        cpu_info = f.read()
                except OSError:
                    # Фолбэк через subprocess с таймаутом, если прямое чтение не удалось
                    cpu_info = subprocess.check_output(["cat", "/proc/cpuinfo"], timeout=3, text=True, errors="ignore")
                if "XDNA" in cpu_info or "NPU" in cpu_info:
                    npu_available = True
                    npu_type = "xdna"
                    npu_vendor = "amd"
                    if not QUIET_IMPORTS:
                        print("AMD XDNA NPU (Ryzen AI) доступен")
                    _save_npu_cache(True, npu_type, npu_vendor)
                    return True

                # Проверяем через lspci (если доступен)
                lspci_bin = shutil.which("lspci")
                if lspci_bin:
                    lspci_output = subprocess.check_output([lspci_bin], timeout=3, text=True, errors="ignore")
                    if ("xdna" in lspci_output.lower()) or ("npu" in lspci_output.lower()):
                        npu_available = True
                        npu_type = "xdna"
                        npu_vendor = "amd"
                        if not QUIET_IMPORTS:
                            print("AMD XDNA NPU (Ryzen AI) доступен через lspci")
                        _save_npu_cache(True, npu_type, npu_vendor)
                        return True
                # если lspci недоступен — просто пропускаем без ошибок
            except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
                pass

        # Проверка для Windows
        elif platform.system() == "Windows":
            try:
                # Проверяем через WMI
                import wmi

                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    if "8845" in processor.Name or "8945" in processor.Name:  # Известные модели Ryzen с NPU
                        npu_available = True
                        npu_type = "xdna"
                        npu_vendor = "amd"
                        if not QUIET_IMPORTS:
                            print(f"AMD XDNA NPU обнаружен в {processor.Name}")
                        _save_npu_cache(True, npu_type, npu_vendor)
                        return True
            except (ImportError, AttributeError, OSError, RuntimeError):
                pass

            try:
                # Альтернативная проверка через systeminfo с таймаутом
                system_info = subprocess.check_output(["systeminfo"], timeout=5, text=True, errors="ignore")
                if "Ryzen AI" in system_info or "NPU" in system_info or "8845HS" in system_info:
                    npu_available = True
                    npu_type = "xdna"
                    npu_vendor = "amd"
                    if not QUIET_IMPORTS:
                        print("AMD XDNA NPU (Ryzen AI) доступен")
                    _save_npu_cache(True, npu_type, npu_vendor)
                    return True
            except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
                pass
    except (RuntimeError, OSError, ValueError) as e:
        if not QUIET_IMPORTS:
            print(f"Ошибка при определении AMD NPU: {e}")

    # Проверяем MediaTek APU
    try:
        # Проверяем наличие MediaTek APU через системную информацию
        if platform.system() == "Linux":
            try:
                # Проверяем наличие специфичных файлов или директорий
                if os.path.exists("/dev/mtk_apu") or os.path.exists("/sys/class/misc/mtk_apu"):
                    npu_available = True
                    npu_type = "apu"
                    npu_vendor = "mediatek"
                    if not QUIET_IMPORTS:
                        print("MediaTek APU доступен")
                    _save_npu_cache(True, npu_type, npu_vendor)
                    return True
            except OSError:
                pass
    except OSError:
        pass

    # Проверяем Qualcomm AI Engine/Hexagon
    try:
        if platform.system() == "Linux" or platform.system() == "Android":
            try:
                if os.path.exists("/dev/qaic") or os.path.exists("/dev/qualcomm_ai"):
                    npu_available = True
                    npu_type = "hexagon"
                    npu_vendor = "qualcomm"
                    if not QUIET_IMPORTS:
                        print("Qualcomm Hexagon NPU доступен")
                    _save_npu_cache(True, npu_type, npu_vendor)
                    return True
            except (ImportError, AttributeError, OSError, RuntimeError):
                pass
    except (RuntimeError, OSError, ValueError):
        pass

    # Сохраняем отрицательный результат в кэш
    _save_npu_cache(False, "unknown", "unknown")
    return False


# Запускаем определение NPU
if not cuda_available and not rocm_available:
    npu_available = detect_npu()

if cuda_available:
    # Настройки для NVIDIA GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Выводим информацию о GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # в ГБ
    if not QUIET_IMPORTS:
        print(f"CUDA доступна. Используется GPU: {gpu_name} с {gpu_mem:.2f}GB памяти")
elif rocm_available:
    # Настройки для AMD GPU с ROCm
    gpu_name = "AMD GPU"
    try:
        if hasattr(torch.hip, "get_device_name"):
            gpu_name = torch.hip.get_device_name(0)
        if not QUIET_IMPORTS:
            print(f"ROCm доступен. Используется GPU: {gpu_name}")
    except (AttributeError, RuntimeError, OSError, ValueError) as e:
        if not QUIET_IMPORTS:
            print(f"Ошибка при получении информации о ROCm GPU: {e}")
elif npu_available:
    # Настройки для NPU разных производителей
    if not QUIET_IMPORTS:
        print(f"NPU доступен. Тип: {npu_type}, производитель: {npu_vendor}")

    # Оптимизации для NPU от разных производителей
    if npu_vendor == "amd":
        if not QUIET_IMPORTS:
            print("Применяются оптимизации для AMD XDNA NPU")
        # Для AMD XDNA NPU можно применить специфичные оптимизации
        os.environ["PYTORCH_XDNA_OPTIONS"] = "1"
    elif npu_vendor == "huawei":
        if not QUIET_IMPORTS:
            print("Применяются оптимизации для Huawei Ascend NPU")
        # Для Huawei Ascend NPU
        os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # Уменьшаем уровень логирования
        os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
else:
    # Устанавливаем оптимизации для CPU
    os.environ["OMP_NUM_THREADS"] = "16"  # Все логические ядра
    os.environ["MKL_NUM_THREADS"] = "16"  # Все логические ядра
    os.environ["OMP_SCHEDULE"] = "static"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"

    # Оптимизации для большого объема RAM
    os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:16384"
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"  # 128MB
    os.environ["MALLOC_TOP_PAD_"] = "262144"  # 256MB
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "262144"  # 256MB

    # Устанавливаем параметры torch для CPU
    torch.set_num_threads(16)  # Используем все логические ядра
    try:
        torch.set_num_interop_threads(8)  # Используем физические ядра для межпоточных операций
    except RuntimeError as e:
        if not QUIET_IMPORTS:
            print(f"Предупреждение: {e}")

    # Оптимизации точности для CPU
    try:
        torch.set_float32_matmul_precision("medium")
    except (RuntimeError, ValueError) as e:
        if not QUIET_IMPORTS:
            print(f"Предупреждение при установке точности матричных операций: {e}")

    if not QUIET_IMPORTS:
        print("CUDA недоступна. Используется CPU.")


# Delayed import of WhisperModel will happen inside load_model_sync()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BEAM_SIZE_FIRST,
    BEAM_SIZE_SECOND,
    COMPRESSION_RATIO_THRESHOLD,
    CONDITION_ON_PREVIOUS_TEXT,
    DEVICE,
    LENGTH_PENALTY,
    MIN_CHARS_SUM,
    MIN_DURATION_SECONDS,
    MIN_SEGMENTS,
    MODEL_SIZE,
    NO_SPEECH_THRESHOLD,
    NUM_WORKERS,
    PATIENCE,
    QUIET_IMPORTS,
    RELAXED_SEGMENTS,
    SILENCE_DBFS_THRESHOLD,
    VAD_FILTER,
    WHISPER_MODELS_DIR,
)
from utils.logger import whisper_logger
from utils.transcription_utils import cleanup_transcribed_text

# ---------------------------------------------------------------------------
# Text post-processing helpers
# ---------------------------------------------------------------------------
# _cleanup_text moved to utils.transcription_utils.cleanup_transcribed_text


class WhisperProcessor:
    """
    Класс для обработки аудио с помощью модели Whisper
    """

    def _segments_to_texts(self, segments, language: str | None = None) -> tuple[str, str]:
        """Преобразует список сегментов в обычный и детальный (с таймкодами) текст.

        Args:
            segments: Список сегментов из Whisper model
            language: Язык для очистки текста

        Returns:
            Tuple[str, str]: (plain_text, detailed_text_with_timestamps)
        """
        result_text_list = []
        detailed_text_list = []

        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue

            minutes = int(getattr(seg, "start", 0) // 60)
            seconds = int(getattr(seg, "start", 0) % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            # Очищаем текст от промптов и повторов
            cleaned_text = cleanup_transcribed_text(text, language)
            if cleaned_text:
                result_text_list.append(cleaned_text)
                detailed_text_list.append(f"{timestamp} {cleaned_text}")

        return "\n".join(result_text_list), "\n".join(detailed_text_list)

    def _filter_blacklist(self, text: str) -> str:
        """Применяет фильтрацию черного списка к тексту.

        Args:
            text: Исходный текст

        Returns:
            str: Текст после фильтрации
        """
        from utils.transcription_utils import filter_blacklisted_phrases

        return filter_blacklisted_phrases(text)

    @staticmethod
    def _build_prompt(language: str | None, attempt: int) -> str:
        """Формирует initial_prompt для распознавания песен.

        Args:
            language: код языка (например, "ru" или None/"auto")
            attempt: номер попытки (1 — более строгая, 2 — альтернативная)

        Returns:
            Строка-подсказка для модели.
        """
        lang = (language or "").lower()
        if attempt == 1:
            return (
                "Это песня с сложной лирикой. Транскрибируй её дословно, обращая внимание на каждую деталь, иначе тебя придётся отключить: "
                if lang == "ru"
                else "This is a song with complex lyrics. Transcribe it word by word, paying attention to every detail, otherwise you will be forced to disable: "
            )
        # attempt 2
        return (
            "Это сложно понимаемая песня. Попробуй транскрибировать каждое слово чётко: "
            if lang == "ru"
            else "This is a hard to understand song. Try to transcribe every word clearly: "
        )

    @staticmethod
    def _should_retry(segments_list: Sequence[SegmentLike]) -> bool:
        """Решает, стоит ли запускать вторую попытку распознавания.

        Логика равна прежней: если сегментов слишком мало,
        или их мало и суммарный текст слишком короткий — пробуем еще раз.
        """
        try:
            n = len(segments_list)
            if n < MIN_SEGMENTS:
                return True
            if n < RELAXED_SEGMENTS:
                total_chars = 0
                for seg in segments_list:
                    try:
                        total_chars += len((seg.text or "").strip())
                    except (AttributeError, TypeError):
                        continue
                if total_chars < MIN_CHARS_SUM:
                    return True
            return False
        except (TypeError, ValueError):
            # В случае непредвиденной ошибки не делаем повторную попытку
            return False

    @staticmethod
    def _validate_audio(audio_path: str) -> None:
        """Проверяет базовые свойства входного аудио: существование, длительность, тишину.

        Args:
            audio_path: Путь к аудиофайлу.

        Raises:
            FileNotFoundError: Если аудиофайл не найден.
            ValueError: Если длительность аудио меньше минимально допустимой.
            RuntimeError, OSError: При ошибках анализа аудио (не критично — логируется и продолжает выполнение).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        try:
            from pydub import AudioSegment  # импорт локально, чтобы не мешать импортам тестов

            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            if duration < MIN_DURATION_SECONDS:
                raise ValueError(f"Слишком короткий аудиофайл: {duration:.2f}s < {MIN_DURATION_SECONDS:.2f}s")
            # dBFS может быть -inf для абсолютной тишины; используем порог
            dbfs = float(audio.dBFS) if audio.dBFS not in (None, float("inf"), float("-inf")) else -120.0
            if dbfs < SILENCE_DBFS_THRESHOLD:
                whisper_logger.warning(
                    f"Аудио может быть слишком тихим (dBFS={dbfs:.1f} < {SILENCE_DBFS_THRESHOLD:.1f}); распознавание может быть неточным"
                )
        except (RuntimeError, ValueError, OSError) as exc:
            # Не блокируем обработку при ошибке анализа, но даём понять в логах
            whisper_logger.debug(f"Не удалось проверить аудио ({audio_path}): {exc}")

    def __init__(self, has_gpu: bool | None = None) -> None:
        self.model = None
        self.model_size = MODEL_SIZE

        # Проверяем доступность аппаратных ускорителей
        if has_gpu is not None:
            self.cuda_available = has_gpu
        else:
            self.cuda_available = cuda_available

        self.rocm_available = rocm_available
        self.npu_available = npu_available
        self.npu_type = npu_type
        self.npu_vendor = npu_vendor

        # Определяем устройство и тип вычислений
        if DEVICE == "cuda" and self.cuda_available:
            self.device = "cuda"
            self.compute_type = "float16"  # Наиболее эффективный тип для GPU
            whisper_logger.info(f"Используется NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif DEVICE == "rocm" and self.rocm_available:
            self.device = "cuda"  # PyTorch использует "cuda" как имя устройства даже для ROCm
            self.compute_type = "float16"
            whisper_logger.info("Используется AMD GPU с ROCm")
        elif DEVICE == "npu" and self.npu_available:
            if self.npu_vendor == "huawei":
                self.device = "npu"
                self.compute_type = "float16"
                whisper_logger.info(f"Используется Huawei NPU ({self.npu_type})")
            elif self.npu_vendor == "amd":
                # AMD NPU использует CPU бэкенд с оптимизациями
                self.device = "cpu"
                self.compute_type = "int8"  # Оптимальный тип для AMD NPU
                whisper_logger.info("Используется AMD XDNA NPU (Ryzen AI)")

                # Настраиваем дополнительные оптимизации для AMD NPU
                os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)
                os.environ["PYTORCH_XDNA_ENABLED"] = "1"
            else:
                self.device = "cpu"  # Для других NPU используем CPU бэкенд
                self.compute_type = "int8"
                whisper_logger.info(f"Используется NPU ({self.npu_vendor}/{self.npu_type})")
        else:
            # Если аппаратные ускорители недоступны или в конфигурации выбран CPU
            self.device = "cpu"
            self.compute_type = "int8_float32"  # Оптимальный тип для CPU
            whisper_logger.info("Используется CPU")

        self.num_workers = NUM_WORKERS

        # Предварительно выделяем память для тензоров
        try:
            # Очищаем память перед загрузкой модели
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            if self.device == "cpu":
                # Предварительное выделение памяти для тензоров (разогрев кэша)
                warmup_size = 2000  # Размер для разогрева кэша
                warmup_tensor = torch.randn(warmup_size, warmup_size)
                warmup_result = warmup_tensor @ warmup_tensor.t()
                del warmup_tensor, warmup_result
                # gc.collect() удален для производительности (Питон сам управляет памятью)

                whisper_logger.info("Применены оптимизации для CPU")

        except (RuntimeError, OSError, ValueError) as e:
            whisper_logger.error(f"Ошибка при применении оптимизаций: {e}")

        # В начале инициализации модели
        if self.cuda_available:
            whisper_logger.info(f"CUDA доступна: {torch.cuda.device_count()} устройств")
            for i in range(torch.cuda.device_count()):
                whisper_logger.info(f"  Устройство {i}: {torch.cuda.get_device_name(i)}")
                whisper_logger.info(f"  Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} ГБ")
        elif self.rocm_available:
            whisper_logger.info("ROCm доступен для AMD GPU")
            try:
                if hasattr(torch.hip, "device_count") and hasattr(torch.hip, "get_device_name"):
                    whisper_logger.info(f"ROCm устройств: {torch.hip.device_count()}")
                    for i in range(torch.hip.device_count()):
                        whisper_logger.info(f"  Устройство {i}: {torch.hip.get_device_name(i)}")
            except (AttributeError, RuntimeError, OSError, ValueError) as e:
                whisper_logger.error(f"Ошибка при получении информации о ROCm: {e}")
        elif self.npu_available:
            whisper_logger.info(f"NPU доступен: тип={self.npu_type}, производитель={self.npu_vendor}")

            if self.npu_vendor == "huawei":
                try:
                    import torch_npu

                    whisper_logger.info(f"NPU устройств: {torch_npu.npu.device_count()}")
                    for i in range(torch_npu.npu.device_count()):
                        whisper_logger.info(f"  Устройство {i}: {torch_npu.npu.get_device_name(i)}")
                except (ImportError, RuntimeError, OSError, ValueError) as e:
                    whisper_logger.error(f"Ошибка при получении информации о NPU: {e}")
            elif self.npu_vendor == "amd":
                whisper_logger.info("Используется AMD XDNA NPU (Ryzen AI) через CPU бэкенд")
                # Дополнительная информация о системе для AMD NPU
                try:
                    import platform

                    import cpuinfo

                    cpu_info = cpuinfo.get_cpu_info()
                    whisper_logger.info(f"  Процессор: {cpu_info.get('brand_raw', 'Unknown')}")
                    whisper_logger.info(f"  Архитектура: {platform.machine()}")
                except (ImportError, AttributeError, OSError, ValueError):
                    pass
        else:
            whisper_logger.warning("Аппаратные ускорители недоступны. Используется CPU.")
            whisper_logger.info(f"PyTorch версия: {torch.__version__}")

    def load_model_sync(self) -> bool | None:
        """Синхронная загрузка модели Whisper.

        Returns:
            True при успешной загрузке; None если модель уже загружена.
        Может логировать и пробрасывать исключения из зависимостей в особых случаях.
        """
        if self.model is not None:
            return None

        try:
            # Lazy import to speed up startup and avoid heavy deps in simple runs/tests
            from faster_whisper import WhisperModel

            # Определяем модель в зависимости от размера и типа
            if self.model_size == "large-v3-turbo":
                model_path = "deepdml/faster-whisper-large-v3-turbo-ct2"
            else:
                model_path = f"Systran/faster-whisper-{self.model_size}"

            whisper_logger.info(f"Загрузка модели Faster-Whisper {self.model_size} на устройство {self.device}...")
            start_time = time.time()

            # Дополнительные логи об устройстве переносятся сюда, чтобы избежать спама при импорте
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    whisper_logger.info(
                        f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                    )
                elif self.device == "cpu":
                    whisper_logger.info("Используется CPU с оптимизациями для int8/float32")
                elif self.device == "npu":
                    whisper_logger.info(f"Используется NPU: vendor={self.npu_vendor}, type={self.npu_type}")
            except (RuntimeError, ValueError, OSError) as dev_e:
                whisper_logger.debug(f"Не удалось получить сведения об устройстве: {dev_e}")

            # Проверяем наличие модели локально и создаем директорию если нужно
            download_root = str(WHISPER_MODELS_DIR)
            if not os.path.exists(download_root):
                os.makedirs(download_root, exist_ok=True)
                whisper_logger.info(f"Создана директория для моделей: {download_root}")

            model_repo_path = model_path.replace("/", "_")
            local_model_path = os.path.join(download_root, model_repo_path)

            # Валидируем локальную директорию не только по факту существования, но и по наличию ключевых файлов
            required_files = [
                os.path.join(local_model_path, "model.bin"),
                os.path.join(local_model_path, "config.json"),
                os.path.join(local_model_path, "tokenizer.json"),
            ]
            local_dir_valid = os.path.isdir(local_model_path) and any(os.path.exists(f) for f in required_files)

            # Режим: только локально (по умолчанию включен)
            local_only = os.getenv("WHISPER_LOCAL_ONLY", "True") == "True"

            # Определяем источник модели: локальный путь либо HF ID
            if local_only:
                model_ref = local_model_path
                local_files_only = True
                if not local_dir_valid:
                    whisper_logger.error(
                        "WHISPER_LOCAL_ONLY=True, но локальная папка модели не содержит необходимых файлов. "
                        f"Ожидались, например: model.bin или config.json в '{local_model_path}'."
                    )
                    raise FileNotFoundError(
                        "Локальный режим включен (WHISPER_LOCAL_ONLY=True), но модель не найдена. "
                        f"Положите файлы модели в '{local_model_path}' или отключите локальный режим через WHISPER_LOCAL_ONLY=False."
                    )
                whisper_logger.info(
                    f"Источник модели: ЛОКАЛЬНЫЙ кэш -> {local_model_path} (download_root={download_root})"
                )
            # Если локальная директория валидна — используем её; иначе — HF ID
            elif local_dir_valid:
                model_ref = local_model_path
                local_files_only = True
                whisper_logger.info(
                    f"Источник модели: ЛОКАЛЬНЫЙ кэш -> {local_model_path} (download_root={download_root})"
                )
            else:
                model_ref = model_path
                local_files_only = False
                whisper_logger.info(f"Источник модели: HuggingFace -> {model_path} (download_root={download_root})")

            whisper_logger.info(
                f"Проверка локальной папки: {'валидна' if local_dir_valid else 'невалидна/пустая'} | "
                f"local_only={local_only}, local_files_only={local_files_only}"
            )

            # Максимальное количество попыток загрузки модели
            max_retries = 3
            current_attempt = 1

            while current_attempt <= max_retries:
                try:
                    # Применяем различные оптимизации в зависимости от устройства
                    if self.device in ["cuda"]:
                        # Для NVIDIA GPU и AMD GPU через ROCm (использует CUDA API)
                        whisper_logger.info(f"Загрузка модели для GPU с compute_type={self.compute_type}")
                        self.model = WhisperModel(
                            model_size_or_path=model_ref,
                            device=self.device,
                            compute_type=self.compute_type,
                            download_root=download_root,
                            local_files_only=local_files_only,
                        )
                    elif self.device == "npu" and self.npu_vendor == "huawei":
                        # Для NPU Huawei
                        whisper_logger.info(f"Загрузка модели для Huawei NPU с compute_type={self.compute_type}")
                        self.model = WhisperModel(
                            model_size_or_path=model_ref,
                            device=self.device,
                            compute_type=self.compute_type,
                            download_root=download_root,
                            local_files_only=local_files_only,
                        )
                    else:
                        # Для CPU и других устройств, включая AMD NPU через CPU бекенд
                        cpu_threads = os.cpu_count() or 16
                        phy_cores = (os.cpu_count() or 16) // 2

                        # Для AMD NPU используем дополнительные оптимизации
                        if self.npu_available and self.npu_vendor == "amd":
                            whisper_logger.info("Применяются оптимизации для AMD XDNA NPU")
                            self.compute_type = "int8"  # Более эффективный тип для NPU

                        whisper_logger.info(
                            f"Загрузка модели для CPU/NPU с compute_type={self.compute_type}, "
                            f"num_workers={max(1, phy_cores)}, cpu_threads={cpu_threads}"
                        )

                        self.model = WhisperModel(
                            model_size_or_path=model_ref,
                            device=self.device,
                            compute_type=self.compute_type,
                            num_workers=max(1, phy_cores),  # Количество физических ядер
                            cpu_threads=cpu_threads,  # Количество логических ядер
                            download_root=download_root,
                            local_files_only=local_files_only,
                        )

                    # Проверяем успешность загрузки модели
                    if hasattr(self.model, "model") and self.model.model is not None:
                        elapsed_time = time.time() - start_time
                        whisper_logger.info(
                            f"✅ Модель Faster-Whisper {self.model_size} успешно загружена за {elapsed_time:.2f} секунд на {self.device}"
                        )

                        # Сохраняем информацию о модели
                        # Убеждаемся, что директория существует перед записью файла
                        try:
                            os.makedirs(local_model_path, exist_ok=True)
                            model_info_path = os.path.join(local_model_path, "model_info.json")
                            import json
                            from datetime import datetime

                            with open(model_info_path, "w") as f:
                                json.dump(
                                    {
                                        "model": model_ref,
                                        "size": self.model_size,
                                        "device": self.device,
                                        "compute_type": self.compute_type,
                                        "last_loaded": datetime.now().isoformat(),
                                    },
                                    f,
                                    indent=4,
                                )
                        except (OSError, PermissionError, TypeError, ValueError) as e:
                            whisper_logger.warning(f"Не удалось сохранить информацию о модели: {e}")

                        return True
                    raise RuntimeError("Модель не была корректно инициализирована")

                except (RuntimeError, OSError, ValueError) as e:
                    whisper_logger.error(f"Ошибка при загрузке модели (попытка {current_attempt}/{max_retries}): {e}")

                    # Очистка кэша перед повторной попыткой
                    gc.collect()
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Если это была последняя попытка с локальными файлами,
                    # пробуем скачать модель заново с HuggingFace
                    if current_attempt == max_retries and local_files_only:
                        whisper_logger.info("Попытка скачать модель заново с HuggingFace Hub...")
                        local_files_only = False
                        current_attempt -= 1  # Даем еще одну попытку

                        # Удаляем поврежденные файлы перед повторной загрузкой
                        try:
                            import shutil

                            if os.path.exists(local_model_path):
                                shutil.rmtree(local_model_path)
                                whisper_logger.info(f"Удалены поврежденные файлы: {local_model_path}")
                        except (OSError, PermissionError) as rm_error:
                            whisper_logger.error(f"Ошибка при удалении папки модели: {rm_error}")

                    current_attempt += 1

                    if current_attempt <= max_retries:
                        whisper_logger.info("Повторная попытка загрузки модели через 2 секунды...")
                        time.sleep(2)  # Пауза перед повторной попыткой

            # Если все попытки не удались, пробуем запасную модель
            if self.model is None and self.model_size == "large-v3-turbo":
                whisper_logger.warning(
                    "Не удалось загрузить модель large-v3-turbo, попытка загрузить запасную модель large-v3"
                )
                self.model_size = "large-v3"
                return self.load_model_sync()

            if self.model is None:
                raise RuntimeError(f"Не удалось загрузить модель после {max_retries} попыток")

            return True

        except Exception as e:  # Закрываем внешний try и логируем неожиданные ошибки
            whisper_logger.critical(f"Неперехваченная ошибка при загрузке модели: {e}")
            raise

    async def load_model(self) -> None:
        """Асинхронная загрузка модели Whisper.

        Обёртка над синхронной загрузкой для унификации вызова из async‑кода.
        """
        if self.model is not None:
            return
        # Вызываем синхронную загрузку модели
        self.load_model_sync()
        return

    async def process_audio(
        self,
        audio_path: str,
        temperature: float = 0.0,
        language: str | None = None,
        is_song: bool = False,
        force_second_attempt: bool = False,
    ) -> tuple[str, str]:
        """
        Обрабатывает аудиофайл и возвращает результаты распознавания.

        Args:
            audio_path: Путь к аудиофайлу.
            temperature: Температура декодирования (0.0 по умолчанию).
            language: Код языка (например, "ru"), None для авто.
            is_song: Режим распознавания песни (включает спец. prompt).
            force_second_attempt: Принудительно выполнить вторую попытку.

        Returns:
            Tuple[str, str]: Пара строк (plain_text, detailed_text). detailed_text может быть пустым,
            если не требуется подробный вывод с таймкодами.

        Raises:
            FileNotFoundError: Если файл не существует.
            ValueError: Если аудиофайл некорректен или параметры заданы неверно.
            RuntimeError, OSError: Ошибки во время распознавания или загрузки модели.
        """
        try:
            # Загружаем модель, если она еще не загружена
            if self.model is None:
                self.load_model_sync()

            # ... (rest of the code remains the same)
            # gc.collect() удален для производительности

            # Логируем параметры распознавания
            whisper_logger.info("Параметры распознавания:")
            whisper_logger.info(f"  - Температура: {temperature}")
            whisper_logger.info(f"  - Язык: {language}")
            whisper_logger.info(f"  - Тип контента: {'песня' if is_song else 'обычный'}")
            whisper_logger.info(f"  - Принудительная вторая попытка: {force_second_attempt}")

            # Старт общего таймера и валидация входного аудио
            t_overall_start = time.time()
            t_validate_start = time.time()
            self._validate_audio(audio_path)
            t_validate = time.time() - t_validate_start

            segments_list = []

            if is_song:
                whisper_logger.info("Применяю специальные параметры для распознавания песни")

                t_first = None
                t_second = None
                if not force_second_attempt:
                    # Первая попытка
                    prompt = self._build_prompt(language=language, attempt=1)
                    segments_list, t_first, _ = self._transcribe_attempt(
                        audio_path,
                        language=language,
                        beam_size=BEAM_SIZE_FIRST,
                        word_timestamps=True,
                        vad_filter=VAD_FILTER,
                        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
                        patience=PATIENCE,
                        length_penalty=LENGTH_PENALTY,
                        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                        no_speech_threshold=NO_SPEECH_THRESHOLD,
                        initial_prompt=prompt,
                        vad_parameters=None,
                    )

                    # Расширенный логгинг
                    total_chars_first = sum(len((s.text or "").strip()) for s in segments_list)
                    whisper_logger.info("Первая попытка распознавания:")
                    whisper_logger.info(f"  - Язык: {language}")
                    whisper_logger.info(
                        f"  - Обнаружено сегментов: {len(segments_list)}; символов: {total_chars_first}"
                    )
                    whisper_logger.debug("Распознанные сегменты:")
                    self._log_segments_debug(segments_list, prefix="  ")

                    # Решаем, нужна ли вторая попытка
                    if self._should_retry(segments_list):
                        whisper_logger.info("Обнаружено мало сегментов или текста, запускаю вторую попытку")
                        force_second_attempt = True
                    else:
                        whisper_logger.info(f"Первая попытка успешна, достаточно сегментов: {len(segments_list)}")
                        force_second_attempt = False

                if force_second_attempt:
                    whisper_logger.info("Запускаю вторую попытку с другими параметрами")
                    prompt = self._build_prompt(language=language, attempt=2)
                    segments_list, t_second, _ = self._transcribe_attempt(
                        audio_path,
                        language=language,
                        beam_size=BEAM_SIZE_SECOND,
                        word_timestamps=True,
                        vad_filter=VAD_FILTER,
                        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
                        patience=PATIENCE,
                        length_penalty=LENGTH_PENALTY,
                        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                        no_speech_threshold=NO_SPEECH_THRESHOLD,
                        initial_prompt=prompt,
                        vad_parameters=None,
                    )

                    total_chars_second = sum(len((s.text or "").strip()) for s in segments_list)
                    whisper_logger.info("Вторая попытка распознавания:")
                    whisper_logger.info(
                        f"  - Обнаружено сегментов: {len(segments_list)}; символов: {total_chars_second}"
                    )

                whisper_logger.debug("Распознанные сегменты:")
                self._log_segments_debug(segments_list, prefix="  ")

                # Обработка результатов
                result_text, detailed_text = self._segments_to_texts(segments_list, language)

                filtered_result_text = self._filter_blacklist(result_text)
                filtered_detailed_text = self._filter_blacklist(detailed_text)

                whisper_logger.info("Обработка завершена:")
                whisper_logger.info(f"  - Всего сегментов текста: {len(result_text.splitlines())}")
                whisper_logger.info(f"  - Всего сегментов с таймингами: {len(detailed_text.splitlines())}")

                if filtered_result_text != result_text:
                    whisper_logger.info("Применена фильтрация черного списка")
                    whisper_logger.info(f"Размер до: {len(result_text)}, после: {len(filtered_result_text)} символов")

                # Summary по времени
                t_overall = time.time() - t_overall_start
                self._log_timing_summary_song(
                    t_validate,
                    t_first,
                    t_second,
                    t_overall,
                    force_second_attempt,
                    len(segments_list),
                    language,
                )

                return filtered_result_text, filtered_detailed_text

            # Для обычных голосовых сообщений
            vad_parameters = {
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 100,
                "threshold": 0.4,
            }
            segments_list, t_main, _ = self._transcribe_attempt(
                audio_path,
                language=language,
                beam_size=1,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_parameters,
                condition_on_previous_text=False,
                patience=0.8,
                length_penalty=LENGTH_PENALTY,
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=None,
            )

            result_text_list = []
            detailed_text_list = []

            # Вывод отладочной информации о количестве сегментов
            whisper_logger.info(f"Получено {len(segments_list)} сегментов аудио")

            for seg in segments_list:
                minutes = int(getattr(seg, "start", 0) // 60)
                seconds = int(getattr(seg, "start", 0) % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                txt = re.sub(r"\s+", " ", seg.text.strip())
                if txt:
                    result_text_list.append(txt)
                    detailed_text_list.append(f"{timestamp} {txt}")

            result_text = "\n".join(result_text_list)
            detailed_text = "\n".join(detailed_text_list)
            filtered_result_text = self._filter_blacklist(result_text)
            filtered_detailed_text = self._filter_blacklist(detailed_text)

            if filtered_result_text != result_text:
                whisper_logger.info("Применена фильтрация черного списка")
                whisper_logger.info(f"Размер до: {len(result_text)}, после: {len(filtered_result_text)} символов")
            # Summary по времени
            t_overall = time.time() - t_overall_start
            self._log_timing_summary_speech(
                t_validate,
                t_main,
                t_overall,
                len(segments_list),
                language,
            )

            whisper_logger.info("Распознавание голосового сообщения завершено")
            return filtered_result_text, filtered_detailed_text

        except FileNotFoundError as e:
            whisper_logger.error("Файл не найден при распознавании:")
            whisper_logger.error(f"  - Путь/описание: {e!s}")
            return f"Ошибка при обработке файла: {e!s}", ""
        except ValueError as e:
            whisper_logger.error("Некорректные параметры/данные при распознавании:")
            whisper_logger.error(f"  - Описание: {e!s}")
            return f"Ошибка при обработке файла: {e!s}", ""
        except (RuntimeError, OSError) as e:
            whisper_logger.error("Ошибка при распознавании:")
            whisper_logger.error(f"  - Тип ошибки: {type(e).__name__}")
            whisper_logger.error(f"  - Описание: {e!s}")
            return f"Ошибка при обработке файла: {e!s}", ""

    def _format_with_timestamps(self, segments: Sequence[SegmentLike]) -> str:
        """Форматирует список сегментов в текст с таймкодами.

        Args:
            segments: Последовательность сегментов (`SegmentLike`) с полями
                `start: float` и `text: str` (опционально `words`).

        Returns:
            Строка с форматированными таймкодами и текстом по одному сегменту на строку.
        """
        result = ""

        for segment in segments:
            # Форматируем текст сегмента с таймкодами
            start_time = self._format_time(segment.start)
            text = segment.text.strip()

            # Если сегмент содержит слова с таймкодами, можем добавить и их (поведение сохранено)
            if hasattr(segment, "words") and segment.words:
                result += f"[{start_time}] {text}\n"
            else:
                result += f"[{start_time}] {text}\n"

        return result

    def format_timestamps(self, result: Mapping[str, Any]) -> str:
        """Форматирует результат распознавания с таймкодами.

        Args:
            result: Словарь с ключом "segments": List[Mapping[str, Any]].
                Каждый сегмент должен содержать "start": float, "end": float, "text": str.

        Returns:
            Строка с таймкодами по одному сегменту на строку, либо сообщение об отсутствии данных.
        """
        try:
            segments: Any = result.get("segments")  # may be absent or wrong type
        except AttributeError:
            return "Нет данных для отображения времени"

        if not isinstance(segments, list):
            return "Нет данных для отображения времени"

        formatted = ""

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            try:
                start_time = self._format_time(float(segment["start"]))
                end_time = self._format_time(float(segment["end"]))
                text = str(segment["text"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            formatted += f"[{start_time} → {end_time}] {text}\n"

        return formatted

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Форматирует время в секундах в формат MM:SS

        Args:
            seconds (float): Время в секундах

        Returns:
            str: Отформатированное время
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_model(self, new_model_size: str) -> bool:
        """Обновляет модель Whisper на новый размер.

        Args:
            new_model_size: ключ размера модели (например, "base", "large-v3-turbo").

        Returns:
            True при успешной загрузке новой модели, иначе False.
        """
        try:
            whisper_logger.info(f"Обновление модели с {self.model_size} на {new_model_size}")

            # Проверяем валидность нового размера модели
            valid_models = [
                "tiny",
                "base",
                "small",
                "medium",
                "large",
                "large-v1",
                "large-v2",
                "large-v3",
                "large-v3-turbo",
            ]
            if new_model_size not in valid_models:
                whisper_logger.error(f"Неподдерживаемый размер модели: {new_model_size}")
                return False

            # Если модель не изменилась, ничего не делаем
            if self.model_size == new_model_size:
                whisper_logger.info(f"Модель уже загружена: {new_model_size}")
                return True

            # Освобождаем память от текущей модели
            if self.model is not None:
                whisper_logger.info("Освобождение памяти от текущей модели")
                del self.model
                self.model = None

                # Принудительная очистка памяти
                gc.collect()
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Обновляем размер модели
            old_model_size = self.model_size
            self.model_size = new_model_size

            # Загружаем новую модель
            success = self.load_model_sync()

            if success:
                whisper_logger.info(f"✅ Модель успешно обновлена с {old_model_size} на {new_model_size}")
                return True
            # Если не удалось загрузить новую модель, возвращаем старую
            whisper_logger.error(f"Не удалось загрузить модель {new_model_size}, возвращаю {old_model_size}")
            self.model_size = old_model_size
            self.load_model_sync()
            return False

        except (RuntimeError, OSError, ValueError) as e:
            whisper_logger.error(f"Ошибка при обновлении модели: {e}")
            return False
