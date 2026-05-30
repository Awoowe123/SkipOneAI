# ✅ Конфигурация - 3 модели (Qwen + DeepSeek + GLM)

## 📋 Финальная конфигурация `.env`:

```bash
# LLM Configuration (LM Studio)
OLLAMA_BASE_URL=http://localhost:1234

# Default/Fallback model (when router is disabled)
OLLAMA_MODEL=qwen2.5-7b-instruct

# Multi-Model Router (3 models - RECOMMENDED)
LLM_ROUTER_ENABLED=true

# Model identifiers (MUST match "Identifier API" in LM Studio!)
CHAT_MODEL=qwen2.5-7b-instruct
CODE_MODEL=deepseek-ai - deepseek-coder-6.7b-instruct
ANALYSIS_MODEL=glm-4-9b-chat
```

---

## 🎯 Роли моделей:

| Модель | Identifier в LM Studio | Роль | Когда используется |
|--------|------------------------|------|-------------------|
| **Qwen 2.5 7B** | `qwen2.5-7b-instruct` | Chat | Casual chat, стиль, общение |
| **DeepSeek Coder 6.7B** | `deepseek-ai - deepseek-coder-6.7b-instruct` | Code | Код, программирование, дебаг |
| **GLM-4 9B** | `glm-4-9b-chat` | Analysis | `/bio analyze`, `/style` |

---

## ⚙️ Настройки в LM Studio:

### 1. **Qwen 2.5 7B** (основная, всегда в памяти)
```yaml
Context Length: 8192
GPU Layers: 28
Keep in Memory: ✅ ON
```

### 2. **DeepSeek Coder 6.7B** (по требованию)
```yaml
Context Length: 8192
GPU Layers: 32
Keep in Memory: ❌ OFF
```

### 3. **GLM-4 9B** (по требованию)
```yaml
Context Length: 8192
GPU Layers: 26
Keep in Memory: ❌ OFF
```

---

## 🚀 Как проверить работу:

### 1. Скачай модели в LM Studio:
- Qwen/Qwen2.5-7B-Instruct (Q4_K_M)
- deepseek-ai/DeepSeek-Coder-6.7B-Instruct (Q4_K_M)
- THUDM/glm-4-9b-chat (уже есть)

### 2. Запусти LM Studio сервер:
- Загрузи **любую** модель (например, Qwen)
- Start Server на `localhost:1234`
- Роутер автоматически переключит модели!

### 3. Запусти бота:
```bash
python main.py
```

### 4. Проверь логи:
```
[ROUTER] Task classified as: chat
[ROUTER] Using active model: qwen2.5-7b-instruct
```

**Если пишешь про код:**
```
[ROUTER] Task classified as: code
[ROUTER] Switching model: qwen2.5-7b-instruct → deepseek-ai - deepseek-coder-6.7b-instruct
```

---

## ❓ FAQ:

**Q: Почему в .env написано `deepseek-ai - deepseek-coder-6.7b-instruct` с пробелами?**
A: Это точное название из LM Studio "Identifier API". Копируй как есть!

**Q: GLM захардкорена?**
A: Да, но правильно:
- Как **fallback** (если роутер выключен) - теперь Qwen
- Как **analysis model** (для `/style`) - GLM остаётся

**Q: Можно вернуться к одной модели?**
A: Да! Установи `LLM_ROUTER_ENABLED=false` и будет использоваться только `OLLAMA_MODEL`.

**Q: Как узнать правильный Identifier?**
A: В LM Studio → выбери модель → скопируй из поля "Identifier API".

---

## 📊 Потребление VRAM (Context 8K):

```
Qwen 2.5 7B:      ~5.5 GB  ✅
DeepSeek Coder:   ~4.8 GB  ✅
GLM-4 9B:         ~6.7 GB  ✅

Одновременно: 1 модель (остальные в RAM)
```

---

## ✅ Готово!

Конфигурация настроена. Перезапусти бота и наслаждайся!
