# VoiceDub Pro 🎬🎙️

An AI-powered video dubbing and audio processing platform with speaker identification, voice cloning, and real-time translation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20Web%20Framework-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 🌟 Features

### Core Capabilities
- **🎥 Video Dubbing**: Automatically identify speakers, translate speech, and generate dubbed audio using voice cloning
- **🗣️ Speaker Diarization**: AI-powered speaker identification using WavLM embeddings and agglomerative clustering
- **🔄 Real-time Translation**: Integrated with Lollms for high-quality neural translation
- **🎙️ Voice Cloning**: F5-TTS and FishSpeech engines for realistic voice synthesis
- **🎵 Background Separation**: Optional Demucs integration to isolate speech from music/noise

### Additional Tools
- **📝 Transcription**: Whisper-powered speech-to-text with timestamps
- **📊 AI Analysis**: Meeting summarization and report generation
- **🎤 Voice Recorder**: In-browser audio/video recording
- **📺 YouTube Support**: Direct video download and processing

### Technical Highlights
- **Resumable Pipelines**: Crash recovery with SQLite persistence—resume interrupted tasks after app restart
- **8GB GPU Optimized**: Chunked processing, model caching with LRU eviction, and aggressive VRAM management
- **WebSocket Real-time**: Live progress updates, logs, and interactive speaker validation
- **Cross-platform**: Windows, Linux, and macOS support with portable FFmpeg

## 🏗️ Architecture

```
VoiceDub Pro
├── core/                 # Shared infrastructure
│   ├── database.py       # SQLite task persistence with JSON fields
│   └── resources.py      # Singleton GPU resource manager (Whisper, F5-TTS, etc.)
├── modules/              # Feature modules
│   ├── translate_video/  # Main dubbing pipeline (3 phases)
│   │   ├── pipeline/     # Phase 2 (translation) & Phase 3 (recomposition)
│   │   ├── diarization.py # Speaker clustering with WavLM
│   │   └── task_manager.py # Lifecycle management with resume support
│   ├── transcribe/       # STT endpoint
│   ├── analysis/         # LLM-powered meeting analysis
│   ├── recorder/         # Browser media capture
│   └── tts/              # Voice cloning endpoint
├── static/js/views/      # Modular frontend (ES6 modules)
└── templates/            # Jinja2 HTML templates
```

### Processing Pipeline

| Phase | Description | Output |
|-------|-------------|--------|
| **1. Identification** | Extract audio → VAD → diarization → transcription | Speaker samples + transcribed segments |
| **2. Translation** | Translate text → TTS synthesis with voice cloning | Dubbed audio segments |
| **3. Recomposition** | Mix speech with background → merge with video | Final dubbed video |

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **CUDA-capable GPU** (8GB+ VRAM recommended, CPU fallback available)
- **Git** (for installing transformers from source)

### Installation

#### Windows (PowerShell)
```powershell
# 1. Clone and enter directory
git clone <repository-url>
cd VoiceDub-Pro

# 2. Run automated installer
.\install.bat

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Start server
python server.py
```

#### Linux/macOS (Bash)
```bash
# 1. Clone and enter directory
git clone <repository-url>
cd VoiceDub-Pro

# 2. Run automated installer
chmod +x install.sh
./install.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Start server
python server.py
```

#### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows: venv\Scripts\activate, Linux/Mac: source venv/bin/activate)
pip install --upgrade pip

# Install dependencies (note: transformers from git for SeamlessM4T)
pip install -r requirements.txt

# Download FFmpeg (optional - installer handles this)
# Windows: https://github.com/BtbN/FFmpeg-Builds/releases
# Linux: https://johnvansickle.com/ffmpeg/
# Extract ffmpeg/ffmpeg.exe to project root
```

### Configuration

Create `.env` file for optional services:

```env
# Lollms for translation (default: http://localhost:9600)
LOLLMS_URL=http://localhost:9600
LOLLMS_MODEL_NAME=mistral

# FishSpeech API for alternative TTS (default: http://localhost:8080/v1/tts)
FISH_SPEECH_API_URL=http://localhost:8080/v1/tts
```

## 🖥️ Usage

### Web Interface
Open http://localhost:8000 in your browser.

#### Video Dubbing Workflow
1. **Upload**: Select video file or paste YouTube URL
2. **Configure**: Choose source/target languages, TTS engine, background separation
3. **Review**: Validate detected speakers, rename them, mark for removal
4. **Process**: Watch real-time translation and synthesis progress
5. **Download**: Get your fully dubbed video

#### Keyboard Shortcuts
- **Dashboard**: View all projects, resume interrupted tasks
- **Transcribe**: Quick STT with optional translation
- **Analysis**: Meeting summarization with markdown output
- **Recorder**: Browser-based audio/video capture

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload video for dubbing |
| `/api/youtube/download` | POST | Download YouTube video |
| `/api/projects` | GET | List all projects |
| `/api/projects/{id}` | GET | Project state + transcription |
| `/api/projects/{id}/resume` | POST | Resume interrupted task |
| `/api/projects/{id}/restart` | POST | Restart from specific phase |
| `/api/projects/{id}/validate` | POST | Submit speaker configuration |
| `/ws/{id}` | WebSocket | Real-time progress & logs |
| `/transcribe/` | POST | Speech-to-text |
| `/tts/generate` | POST | Voice cloning |
| `/analysis/` | POST | LLM analysis |

### Python API Example

```python
import requests

# Upload video for dubbing
with open("meeting.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload",
        files={"file": f},
        data={
            "src_lang": "en",      # Source language
            "tgt_lang": "es",      # Target language (Spanish)
            "tts_engine": "f5",    # F5-TTS or fishspeech
            "separate_audio": "false"
        }
    )
    
task_id = response.json()["task_id"]

# Connect to WebSocket for real-time updates
import websocket
ws = websocket.create_connection(f"ws://localhost:8000/ws/{task_id}")

# Wait for speaker validation prompt, then submit config
requests.post(
    f"http://localhost:8000/api/projects/{task_id}/validate",
    json={"speakers": {"0": {"name": "Alice", "action": "dub"}}}
)
```

## 🧠 AI Models Used

| Component | Model | Purpose | VRAM |
|-----------|-------|---------|------|
| **STT** | Whisper Large v2 | Speech recognition | ~3GB |
| **Diarization** | WavLM Base+ | Speaker embeddings | ~1GB |
| **VAD** | Silero VAD | Voice activity detection | ~100MB |
| **TTS** | F5-TTS Base | Voice cloning | ~2GB |
| **Separation** | Demucs HTDemucs | Background isolation | ~2GB |

**Total with caching**: ~6-7GB active, designed for 8GB GPUs with LRU model eviction.

## 🔧 Advanced Configuration

### Memory Optimization
Edit `core/resources.py` to adjust:
```python
torch.cuda.set_per_process_memory_fraction(0.85)  # 85% of GPU memory
self._max_cached_models = 3  # Concurrent loaded models
```

### Chunk Sizes
Edit pipeline files for your GPU:
```python
# modules/translate_video/pipeline/phase2_translation.py
chunk_size = 3  # Segments per GPU batch (lower = less VRAM)
```

### Custom FFmpeg Path
Set environment variable:
```bash
export FFMPEG_PATH=/path/to/ffmpeg  # Linux/Mac
set FFMPEG_PATH=C:\path\to\ffmpeg.exe  # Windows
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `WinError 87` on Windows | Use `python server.py` directly, not through IDE terminals |
| CUDA out of memory | Reduce `chunk_size` in pipeline files, enable `separate_audio: false` |
| Whisper pipeline fails | Ensure `chunk_length_s=30` and `return_timestamps=False` in resources.py |
| FishSpeech connection refused | Start FishSpeech server separately or switch to F5-TTS |
| Resume fails repeatedly | Check `resume_attempts` in database; manually restart from dashboard |

### Logs & Debug
- **Server logs**: Console output with colored formatting
- **Task logs**: Persisted in `projects_db/{task_id}.json`
- **Database**: `tasks.db` (SQLite, view with any SQLite browser)

## 🤝 Contributing

This project uses a modular architecture to enable easy extension:

1. **New TTS Engine**: Add to `modules/tts/logic.py` and update `core/resources.py`
2. **New Pipeline Phase**: Follow pattern in `modules/translate_video/pipeline/`
3. **Frontend Views**: Add ES6 module in `static/js/views/`

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Whisper](https://github.com/openai/whisper) by OpenAI
- [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid
- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Lollms](https://github.com/ParisNeo/lollms) for local LLM integration

---

**Made with ❤️ for breaking language barriers in video content.**