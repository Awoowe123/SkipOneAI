# Quick Start Guide

## Пошаговая инструкция для первого запуска

### Шаг 1: Установка зависимостей

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать (Windows)
venv\Scripts\activate

# Установить пакеты
pip install -r requirements.txt
```

### Шаг 2: Настройка LM Studio

1. Скачайте LM Studio с https://lmstudio.ai/
2. Запустите и загрузите модель GLM 4.7 Flash через интерфейс
3. Перейдите в раздел "Local Server"
4. Нажмите "Start Server" (по умолчанию порт 1234)
5. Убедитесь что сервер запущен (должна быть зелёная иконка)

### Шаг 3: Получение Telegram API

1. Откройте https://my.telegram.org
2. Войдите с вашим номером телефона
3. Перейдите в "API development tools"
4. Заполните форму для нового приложения:
   - App title: любое название (например "My Bot")
   - Short name: короткое имя
   - Platform: выберите Other
5. Нажмите "Create application"
6. Сохраните `api_id` и `api_hash`

### Шаг 4: Настройка .env

```bash
# Скопировать шаблон
copy .env.example .env

# Открыть в редакторе и заполнить
notepad .env
```

Пример заполненного `.env`:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
TELEGRAM_PHONE=+79991234567

OLLAMA_BASE_URL=http://localhost:1234
OLLAMA_MODEL=glm-4-9b-chat

DB_PATH=data/personality.db
SESSION_PATH=data/session

SKIP_MESSAGE_PROBABILITY=0.05
IMMEDIATE_RESPONSE_PROBABILITY=0.70
TYPING_SPEED_WPM=60
```

### Шаг 5: Заполнение базы данных

**Рекомендуется**: Начните с ручных примеров

```bash
python scripts\populate_database.py
```

**Опционально**: Соберите свои сообщения из Telegram

```bash
python scripts\collect_messages.py
```

⚠️ Это займёт некоторое время и потребует авторизации в Telegram

### Шаг 6: Первый запуск

```bash
python -m src.main
```

**При первом запуске**:
1. Вам придёт код авторизации в Telegram
2. Введите его в консоль
3. Если у вас включена двухфакторная аутентификация, введите пароль
4. Сессия сохранится в `data/session/` и больше код не потребуется

### Шаг 7: Тестирование

1. Отправьте себе сообщение в Telegram (Saved Messages)
2. Бот должен ответить через несколько секунд
3. Проверьте логи в `logs/bot.log`

## Troubleshooting

### Ошибка "No module named 'src'"

```bash
# Убедитесь что вы в корневой директории проекта
cd SkipOneAI

# И что виртуальное окружение активировано
venv\Scripts\activate
```

### LM Studio: Connection refused

1. Убедитесь что LM Studio запущен
2. Проверьте что сервер активен (зелёная иконка)
3. Проверьте порт в настройках (должен быть 1234)
4. Попробуйте:

```bash
curl http://localhost:1234/v1/models
```

### Telegram: API ID invalid

- Проверьте что скопировали правильные значения
- `api_id` должен быть числом (без кавычек)
- `api_hash` должен быть строкой из 32 символов

### База данных пустая

Если при запуске видите ошибки о пустой БД:

```bash
python scripts\populate_database.py
```

## Проверка что всё работает

### 1. Проверка LM Studio

```bash
# Должен вернуть список моделей
curl http://localhost:1234/v1/models
```

### 2. Проверка базы данных

```bash
# Запустить Python REPL
python

# В REPL:
>>> from src.personality.database import PersonalityDatabase
>>> db = PersonalityDatabase()
>>> print(db.get_category_stats())
>>> db.close()
```

Должен показать статистику по категориям.

### 3. Проверка конфигурации

```bash
python

>>> from src.config import Config
>>> print(Config.telegram.api_id)
>>> print(Config.llm.base_url)
```

Должны отобразиться ваши настройки.

## Рекомендации

### Для начала

1. Используйте низкий `SKIP_MESSAGE_PROBABILITY` (0.05)
2. Высокий `IMMEDIATE_RESPONSE_PROBABILITY` (0.70)
3. Начните с тестирования в Saved Messages

### Для продакшена

1. Настройте задержки под свой стиль
2. Соберите больше примеров из своих сообщений (500-1000)
3. Настройте проактивное поведение
4. Используйте Docker для постоянной работы

## Следующие шаги

После успешного запуска:

1. Соберите больше ваших сообщений для обучения стиля
2. Настройте параметры задержек под себя
3. Добавьте обработку групповых чатов (в `telegram_client.py`)
4. Настройте мониторинг каналов
5. Разверните в Docker для 24/7 работы
