# 🎯 Multi-Model LLM Router - Setup Guide

## Что это?

Система маршрутизации запросов между специализированными моделями для оптимальной производительности на RTX 3080 10GB.

## 🏗️ Архитектура

```
┌─────────────────────┐
│   User Message      │
└──────────┬──────────┘
           │
           ▼
    ┌─────────────┐
    │  Classifier  │ (classify_task)
    └──────┬──────┘
           │
           ├──► "chat"     ──► Qwen/Qwen2.5-7B-Instruct
           ├──► "code"     ──► DeepSeek-Coder-6.7B
           ├──► "analysis" ──► THUDM/glm-4-9b-chat
           └──► "tools"    ──► (future)
```

## 📦 Модели

| Задача | Модель | Размер | Описание |
|--------|--------|--------|----------|
| **Chat** | `Qwen/Qwen2.5-7B-Instruct` | ~4.5GB Q4 | Casual chat, стиль, общение |
| **Code** | `deepseek-ai/DeepSeek-Coder-6.7B-Instruct` | ~4GB Q4 | Код, дебаг, программирование |
| **Analysis** | `THUDM/glm-4-9b-chat` | ~5.5GB Q4 | Bio/Style анализ, суммаризация |

## ⚙️ Установка

### 1. Обновите `.env`:

Файл уже обновлён с конфигурацией:
```bash
# Multi-Model Router
LLM_ROUTER_ENABLED=true
CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
CODE_MODEL=deepseek-ai/DeepSeek-Coder-6.7B-Instruct
ANALYSIS_MODEL=THUDM/glm-4-9b-chat
```

### 2. Скачайте модели в LM Studio:

**Способ 1: Через GUI LM Studio**
1. Откройте LM Studio
2. Перейдите в раздел "Search" (🔍)
3. Скачайте каждую модель (выбирайте **Q4_K_M** квантизацию):
   - `Qwen/Qwen2.5-7B-Instruct` (Q4_K_M)
   - `deepseek-ai/DeepSeek-Coder-6.7B-Instruct` (Q4_K_M)
   - `THUDM/glm-4-9b-chat` (Q4_K_M или Q4_0)

**Способ 2: Через CLI (если есть lms)**
```bash
lms download Qwen/Qwen2.5-7B-Instruct --quantization Q4_K_M
lms download deepseek-ai/DeepSeek-Coder-6.7B-Instruct --quantization Q4_K_M
lms download THUDM/glm-4-9b-chat --quantization Q4_K_M
```

### 3. Проверьте, что LM Studio работает:

1. Запустите LM Studio
2. Перейдите в "Local Server"
3. Загрузите **любую** модель (например, Qwen2.5-7B)
4. Нажмите "Start Server"
5. Убедитесь, что сервер запущен на `http://localhost:1234`

**Важно:** Роутер **автоматически** переключает модели через API, вам не нужно вручную менять модель в LM Studio!

### 4. Перезапустите бота:

```bash
# Остановите текущий процесс (Ctrl+C)
python main.py
```

## 🔍 Как это работает

### Классификация задач

Роутер автоматически определяет тип задачи:

```python
# Примеры классификации:
"напиши функцию на python" → code
"как исправить эту ошибку?" → code
"/style 500 -1002445805841" → analysis
"/bio analyze @username" → analysis
"привет, как дела?" → chat
```

### Переключение моделей

При смене задачи роутер:
1. Обновляет параметр `model` в API запросе
2. LM Studio **автоматически** выгружает текущую модель и загружает нужную
3. Генерирует ответ
4. (опционально) возвращает основную модель

**Время переключения:** ~2-5 секунд

## 📊 Логи

Следите за работой роутера в логах:

```
[ROUTER] Task classified as: code
[ROUTER] Switching model: Qwen2.5-7B → DeepSeek-Coder-6.7B (task: code)
```

## 🛠️ Настройка

### Отключить роутер:

Если хотите вернуться к одной модели:

```bash
# В .env
LLM_ROUTER_ENABLED=false
```

### Изменить модели:

```bash
# Например, использовать Llama вместо Qwen для chat:
CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Добавить новые задачи:

Отредактируйте `src/core/llm_router.py`:

```python
def classify_task(self, message: str, has_tools: bool = False) -> TaskType:
    # Добавьте свои ключевые слова
    if 'перевод' in message_lower or 'translate' in message_lower:
        return 'translation'  # Новая задача
```

И добавьте модель в `config.py`:

```python
translation_model: str = os.getenv('TRANSLATION_MODEL', 'your-model-here')
```

## ❓ FAQ

**Q: Нужно ли держать LM Studio открытым?**
A: Да, LM Studio должен быть запущен с активным сервером на `localhost:1234`.

**Q: Модель не переключается?**
A: Убедитесь, что:
1. Все модели скачаны в LM Studio
2. Названия моделей в `.env` точно совпадают с названиями в LM Studio
3. Включён роутер (`LLM_ROUTER_ENABLED=true`)

**Q: Можно ли использовать Ollama вместо LM Studio?**
A: Да, но нужно адаптировать `llm_router.py` для Ollama API (модели переключаются через `/api/pull`).

**Q: Что если модель не влезает в VRAM?**
A: Используйте более агрессивную квантизацию (Q3_K_M вместо Q4_K_M) или меньшую модель.

## 🚀 Будущие улучшения

- [ ] Автоматическое скачивание моделей при старте
- [ ] Кеширование моделей в RAM для быстрого свапа
- [ ] Reasoning модель (Phi-3 или QwQ)
- [ ] Tools модель (Llama 3.2 3B)
- [ ] Поддержка Ollama
- [ ] Мониторинг VRAM usage

## 📝 Примеры использования

```python
# В боте автоматически:
User: "как написать сортировку на питоне?"
→ [ROUTER] Task: code → DeepSeek-Coder-6.7B

User: "/style 1000 -1002445805841"
→ [ROUTER] Task: analysis → glm-4-9b-chat

User: "привет! как дела?"
→ [ROUTER] Task: chat → Qwen2.5-7B
```

## 📞 Поддержка

Если что-то не работает:
1. Проверьте логи бота
2. Убедитесь, что LM Studio сервер запущен
3. Проверьте, что модели скачаны
4. Откройте issue с логами
