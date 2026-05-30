# GigaAM-v3 Docker Test

Запуск GigaAM-v3 в Docker контейнере (Linux) для избежания проблем с зависимостями на Windows.

## Требования

- Docker Desktop с WSL2 backend
- NVIDIA Container Toolkit (для GPU поддержки)
- HuggingFace токен (для загрузки pyannote/segmentation-3.0)

## Быстрый старт

### 1. Установить HF_TOKEN

В WSL2 терминале:

```bash
export HF_TOKEN="твой_hf_токен_здесь"
```

Или создать файл `.env` в `docker/`:

```bash
HF_TOKEN=твой_hf_токен_здесь
```

### 2. Запустить

```bash
cd docker

# Сборка и запуск
docker-compose up --build

# Или с произвольным файлом
docker-compose run --rm gigaam python test_gigaam_docker.py /audio/your_file.mp3
```

### 3. Результат

Транскрипция сохранится в `./results/gigaam_result.txt`

## Структура

```
docker/
├── Dockerfile.gigaam       # Docker образ с GigaAM-v3
├── docker-compose.yml       # Конфигурация запуска
├── test_gigaam_docker.py    # Тест-скрипт
└── README.md               # Инструкция
```

## Модели

Доступные варианты:
- `v3_e2e_rnnt` (по умолчанию) - RNNT с пунктуацией
- `v3_e2e_ctc` - CTC с пунктуацией
- `v3_rnnt` - RNNT без пунктуации
- `v3_ctc` - CTC без пунктуации

Пример с другой моделью:

```bash
docker-compose run --rm gigaam python test_gigaam_docker.py /audio/file.mp3 v3_e2e_ctc
```

## Troubleshooting

### GPU не обнаружена

Проверь NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Не находит HF_TOKEN

Убедись, что переменная установлена:

```bash
echo $HF_TOKEN
```

### Файл не найден

Проверь путь к файлу (внутри контейнера это `/audio/`):

```bash
docker-compose run --rm gigaam ls -la /audio/
```
