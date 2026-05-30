<p align="center">
  <img src="docs/logo.jpg" alt="SkipOneAI" />
</p>

<h1 align="center">SkipOneAI</h1>

<p align="center">
  <strong>Autonomous AI Agent for Telegram</strong><br/>
  A digital twin that fully replicates the account owner's communication style
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Telethon-1.x-2196F3?style=flat-square&logo=telegram&logoColor=white" alt="Telethon" />
  <img src="https://img.shields.io/badge/LM_Studio-Local_LLM-8B5CF6?style=flat-square" alt="LM Studio" />
  <img src="https://img.shields.io/badge/Faster_Whisper-ASR-06B6D4?style=flat-square" alt="Faster Whisper" />
  <img src="https://img.shields.io/badge/Docker-Supported-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/CUDA-GPU-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA" />
</p>

<p align="center">
  <a href="#about">About</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#tools">Tools</a> ·
  <a href="#speech-recognition">Speech Recognition</a> ·
  <a href="#docker">Docker</a>
</p>

---

## About

SkipOneAI is an autonomous Telegram agent that operates on behalf of your account. The agent learns from your real messages via a RAG system (Retrieval-Augmented Generation) and carries on conversations in your style — including slang, humor patterns, and formality level tailored to each contact.

All inference runs locally through LM Studio. No data ever leaves your machine.

### Key Features

| Component | Description |
|---|---|
| **Digital Clone** | RAG-based style mimicry. Learns from real messages, adapts to each contact |
| **Local LLM** | Full privacy. GLM-4, DeepSeek, Llama and any model via LM Studio |
| **Speech Recognition** | Faster Whisper with GPU acceleration. Two modes: voice messages and songs (two-pass) |
| **Tool-Use** | 9 tools: web search, terminal, browser, file ops, GIF, stickers, summarization |
| **Human-like Behavior** | Response delays, typing simulation, message skipping, typos, sticker reactions |
| **Routing** | LLM Router selects the right model. Master Classifier handles analysis in one call |
| **Vocal Extraction** | Docker: BS-Roformer, Demucs v4, MDX23C for separating vocals from music |
| **Contacts** | Automatic relationship analysis via LLM. Auto-bio after 50+ messages |
| **Access Control** | Contact whitelist, owner privileges via `BOT_OWNER_ID` |

---

## Architecture

```
src/
├── main.py                        # Entry point
├── config.py                      # Configuration (.env)
├── core/
│   ├── telegram_client.py         # Telethon client
│   ├── event_processor.py         # Main message processing pipeline
│   ├── llm_interface.py           # LM Studio interface (OpenAI-compatible API)
│   ├── llm_router.py              # Multi-model routing
│   ├── master_classifier.py       # Unified classifier (analysis + routing in 1 call)
│   ├── model_manager.py           # Performance monitoring and auto-reload
│   ├── asr_manager.py             # Faster Whisper ASR (CUDA/CPU)
│   ├── command_parser.py          # Command parser
│   ├── context_manager.py         # XML context serialization
│   ├── permissions.py             # Access control (whitelist)
│   └── sticker_manager.py         # Sticker indexing and sending
├── personality/
│   ├── database.py                # Vector DB (SQLite + embeddings)
│   ├── analyzer.py                # Message style analyzer
│   └── contacts_manager.py        # Contact CRM with auto-bio
├── behavior/
│   ├── human_simulator.py         # Delays, typos, skipping
│   └── proactive_agent.py         # Proactive messaging
├── tools/
│   ├── tools_manager.py           # Tool registry and dispatcher
│   ├── web_search.py              # Web search (DuckDuckGo)
│   ├── browser.py                 # Browser (Playwright)
│   ├── terminal.py                # Terminal (sandboxed)
│   ├── file_ops.py                # File operations
│   ├── gif_sender.py              # GIF search and sending
│   ├── message_search.py          # Message history search
│   ├── contact_info.py            # Contact information
│   └── summarizer.py              # Chat summarization
└── utils/
    ├── logging_config.py          # Logging
    └── transcription_utils.py     # Transcription cleanup
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/)
- [Telegram API credentials](https://my.telegram.org)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/SkipOneAI.git
cd SkipOneAI

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/macOS

pip install -r requirements.txt
```

### Configuration

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Fill in `.env`:

```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+1234567890
BOT_OWNER_ID=your_telegram_id_here

OLLAMA_BASE_URL=http://localhost:1234
OLLAMA_MODEL=glm-4-9b-chat
```

### Setting up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: **GLM-4 9B Chat** or **GLM-4.7V Flash**)
3. Go to *Local Server* → *Start Server* (port 1234)

### Database initialization

```bash
python scripts/populate_database.py        # pre-made examples
python scripts/collect_messages.py         # or collect your own messages
```

### Run

```bash
python -m src.main
```

First launch requires an SMS code for Telegram authorization.

---

## Tools

| Tool | Description |
|---|---|
| `search_web` | Web search (exchange rates, weather, news) |
| `search_messages` | Message history search |
| `summarize_chat` | Chat summary |
| `send_gif` | GIF animations |
| `get_contact_info` | Contact information |
| `open_browser` | Pages in Playwright with JS |
| `run_terminal` | Command execution (sandboxed) |
| `read_file` / `write_file` | File read/write |
| `download_file` / `send_file` | Download and send files |

---

## Speech Recognition

Built-in system powered by **Faster Whisper** (`deepdml/faster-whisper-large-v3-turbo-ct2`).

### Voice Message Mode

```
beam_size=1, vad_filter=True, condition_on_previous_text=False
```

### Song Mode

Two-pass system:

1. **First pass** — `beam_size=5`, no VAD, complex lyrics prompt
2. **Retry** (if < 2 segments or < 10 chars) — `beam_size=1`, alternative prompt

### Post-processing

- Transcription artifact cleanup
- Blacklist filtering

### Vocal Extraction (Docker)

```bash
cd docker
./extract_vocals.sh "song.mp3" bs-roformer   # best quality
./extract_vocals.sh "song.mp3" demucs         # fast
./extract_vocals.sh "song.mp3" mdx23c         # balanced
```

---

## Docker

```bash
cd docker
docker-compose build
docker-compose run --rm gigaam python gigaam_production.py /audio/song.mp3
```

GPU acceleration requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## Commands

| Command | Description |
|---|---|
| `/help` | Command list |
| `/bio Name: description` | Create contact bio |
| `/bio analyze @username` | Auto-analyze relationships via LLM |
| `/bio show @username` | Show bio |
| `/bio list` | List contacts |
| `/style` | Current style analysis |
| `/style update` | Update analysis |
| `/reset` | Reset context |
| `/load_stickers` | Load stickers |

---

## Security

- Never publish `.env` — it contains your credentials
- Session files (`data/session/`) grant full account access
- All data is stored locally
- `BOT_OWNER_ID` defines the bot owner — make sure it's set to your ID

---

## License

MIT
