# Stark-Jr.v0.5

A childhood idea turned into a working local AI system.  Stark-Jr is a real-time, fully offline voice assistant powered by Faster-Whisper, Piper TTS, and local LLMs running through Ollama. It listens, thinks, and talks — all without touching the internet. 
external servers.

# Features

Completely offline — STT, LLM inference, and TTS run locally.

Fast, low-latency STT using Faster-Whisper.

Streaming LLM responses with sentence-level real-time playback.

Smart Markdown cleaning so the voice does not read symbols, bold markers, or code fences.

Instant interruption — talk anytime and JARVIS stops speaking immediately.

Selectable LLM brain at startup.

Noise-sensitive listening using RMS thresholds + silence detection.

# System Architecture

The assistant works as three cooperating subsystems:

**JarvisEars (Speech-to-Text)**

Handles microphone input, voice activity detection, and Whisper transcription.
It captures audio in small chunks, detects when speech starts, and stops when silence crosses a set threshold.

**JarvisMouth (Text-to-Speech)**

Uses Piper TTS with chunk-based audio playback.
Can be interrupted instantly.
Also cleans Markdown, removes formatting symbols, and summarizes code blocks before speaking.

**Brain (Local LLM via Ollama)**

Ollama streams model outputs token-by-token.
As soon as a complete sentence appears (ending with ., !, or ?), it is sent to TTS for speaking.

This allows low-latency, sentence-level real-time responses.

# Requirements/Installation Guide

**✅ Full pip installation**
Install all required Python packages
```bash
pip install \
    sounddevice \
    numpy \
    faster-whisper \
    ollama \
    piper-tts
```

🧩 If you want a single copy-paste one-liner:
```bash
pip install sounddevice numpy faster-whisper ollama piper-tts
```

**⚙️ System Dependencies (Important!)**

Ubuntu / Debian
```bash
sudo apt install python3-dev portaudio19-dev ffmpeg
```

Arch Linux
```bash
sudo pacman -S portaudio ffmpeg python-pip
```

**🗂️ Model Installation Section**
Whisper model (handled automatically by faster-whisper)

No manual download needed.

Piper TTS model
```bash
mkdir -p models/piper
wget -O models/piper/en_US-lessac-medium.onnx \
  https://github.com/rhasspy/piper/releases/download/v1.0.0/en_US-lessac-medium.onnx
```

Ollama model (example)
```bash
ollama pull llama3.2:3b
```

⚡ Optional: GPU acceleration for Whisper
```bash
pip install faster-whisper[cuda]
```

# Why This Exists

Growing up, Tony Stark wasn’t just a character — he was a blueprint.
Building the arc reactor in cardboard was the first iteration.

This is the adult one.

The goal wasn’t simply “voice commands.”

The goal was a local AI companion that feels present, responsive, and private.

# 🛠️ Requirements

Python 3.10+

GPU recommended (NVIDIA for CUDA acceleration)

ollama installed locally with at least one model pulled

Piper TTS model (en_US-lessac-medium.onnx)

Faster-Whisper

Python packages:

```pip install sounddevice numpy faster-whisper ollama piper-tts```

**🏃 Running the Assistant**

```python3 jarvis.py```

On startup, you’ll be prompted to select a model installed in Ollama.

JARVIS becomes fully interactive from that point onward.

# 🧠 Internals

**Speech Detection**

RMS thresholding + timed silence window lets the system know when you stop speaking.

Markdown Cleaning.

LLMs love outputting **bold**, *italics*, and code blocks.

The assistant strips these before speech so the voice output sounds natural.

**Sentence-Splitting Playback**

Incoming LLM tokens accumulate until a complete sentence is formed.

This sentence is immediately spoken, reducing perceived latency.

**Interrupt-Mechanism**

Speech is delivered in chunks (2048 samples).
Every chunk checks whether the user interrupted by talking or pressing Ctrl+C.

# 🗺️ Roadmap

Hotword activation (“Jarvis”)

Multilingual STT + TTS

Agent modes (system control, reminders, file ops)

Memory + persistent context

# 📄 License

MIT License
