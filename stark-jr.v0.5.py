#!/usr/bin/env python3
import os
import sys
import queue
import re
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Audio Input
    SAMPLE_RATE = 16000
    CHANNELS = 1
    THRESHOLD = 0.03      
    SILENCE_LIMIT = 2.0   
    
    # Audio Output
    VOLUME = 0.8          # Master volume (0.0 - 1.0)
    CHUNK_SIZE = 2048     # Smaller = faster interrupt, High CPU. 2048 is sweet spot.
    
    # Paths & Models
    PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx" 
    WHISPER_SIZE = "base.en" 
    DEVICE = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"

# ==========================================
# 1. THE EARS (STT)
# ==========================================
class JarvisEars:
    def __init__(self):
        print(f"[{Config.DEVICE.upper()}] Loading Faster-Whisper...")
        try:
            self.model = WhisperModel(Config.WHISPER_SIZE, device=Config.DEVICE, compute_type="float16" if Config.DEVICE == "cuda" else "int8")
        except Exception as e:
            print(f"[ERROR] Whisper load failed: {e}")
            sys.exit(1)

    def listen(self):
        print("\n[LISTENING] ... (Ctrl+C to interrupt)")
        q = queue.Queue()

        def callback(indata, frames, time, status):
            if status: print(status, file=sys.stderr)
            q.put(indata.copy())

        try:
            with sd.InputStream(samplerate=Config.SAMPLE_RATE, channels=Config.CHANNELS, callback=callback):
                audio_data = []
                silence_start = None
                speaking_started = False
                
                while True:
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    rms = np.sqrt(np.mean(data**2))
                    
                    if rms > Config.THRESHOLD:
                        speaking_started = True
                        silence_start = None
                    elif speaking_started:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > Config.SILENCE_LIMIT:
                            break 

                    if speaking_started:
                        audio_data.append(data)
        except KeyboardInterrupt:
            print("\n[SYSTEM] Manual Exit.")
            sys.exit(0)

        if not audio_data: return None
            
        print("[PROCESSING] Transcribing...")
        concatenated = np.concatenate(audio_data, axis=0)
        if concatenated.dtype != np.float32:
            concatenated = concatenated.astype(np.float32)

        segments, _ = self.model.transcribe(concatenated.flatten(), beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        
        if not text or text.lower() in ["you", "and", "thanks", "okay", "mbc", "thank you"]:
            return None

        print(f"[USER]: {text}")
        return text

# ==========================================
# 2. THE MOUTH (Instant Interrupt)
# ==========================================
class JarvisMouth:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.piper = None
        self.active = True
        
        try:
            from piper.voice import PiperVoice
            self.piper = PiperVoice.load(Config.PIPER_MODEL)
            print("[SYSTEM] Piper TTS Loaded.")
            
            # Audio Warmup
            print("[SYSTEM] Audio Warmup...")
            sd.sleep(100) # Short sleep to let audio subsystems settle
            
        except Exception as e:
            print(f"[ERROR] Piper init failed: {e}")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop_talking(self):
        """Instantly stops audio loop and clears queue"""
        self.active = False
        with self.queue.mutex:
            self.queue.queue.clear()
        # Give the worker a moment to break the loop
        time.sleep(0.05) 
        self.active = True

    def speak(self, text):
        cleaned = self._clean(text)
        if cleaned:
            self.queue.put(cleaned)

    def wait(self):
        self.queue.join()

    def _clean(self, text):
        text = re.sub(r'\*\*|__|\*', '', text)
        text = re.sub(r'```.*?```', 'Code block omitted.', text, flags=re.DOTALL)
        return re.sub(r'\s+', ' ', text).strip()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.5)
                if self.active and self.piper:
                    self._play_chunked(text)
                elif self.active:
                    print(f"[FALLBACK]: {text}")
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS ERROR] {e}")

    def _play_chunked(self, text):
        """
        Feeds audio to the stream in chunks. 
        Checks `self.active` every few milliseconds to allow instant stop.
        """
        try:
            stream = self.piper.synthesize(text)
            audio_buffer = b""

            # 1. Generate full audio for the sentence (fast)
            for chunk in stream:
                if not self.active: return 
                audio_buffer += chunk.audio_int16_bytes
            
            if not audio_buffer or not self.active: return

            # 2. Process Audio
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_float32 = audio_float32 * Config.VOLUME
            
            # 3. Play in interruptible loop
            # using OutputStream allows us to write manually
            with sd.OutputStream(samplerate=self.piper.config.sample_rate, channels=1, dtype='float32') as s:
                for i in range(0, len(audio_float32), Config.CHUNK_SIZE):
                    if not self.active: 
                        break # BREAK LOOP INSTANTLY
                    
                    # Get chunk
                    chunk = audio_float32[i : i + Config.CHUNK_SIZE]
                    # Pad last chunk if needed
                    if len(chunk) < Config.CHUNK_SIZE:
                        chunk = np.pad(chunk, (0, Config.CHUNK_SIZE - len(chunk)))
                    
                    s.write(chunk)

        except Exception as e:
            print(f"[PLAYBACK ERROR] {e}")

# ==========================================
# 3. MAIN
# ==========================================
def select_model():
    print("\n=== Select Brain ===")
    try:
        models = ollama.list()
        model_list = [m['model'] for m in models.get('models', [])]
    except Exception as e:
        print(f"[ERROR] Ollama check failed: {e}")
        return "qwen2.5:3b"

    if not model_list:
        print("No models found.")
        return "qwen2.5:3b"
    
    for i, m in enumerate(model_list):
        print(f"{i+1}. {m}")
    
    choice = input(f"Select model (1-{len(model_list)}): ").strip()
    try:
        return model_list[int(choice)-1]
    except:
        print(f"Defaulting to {model_list[0]}")
        return model_list[0]

def main():
    target_model = select_model()
    ears = JarvisEars()
    mouth = JarvisMouth()
    
    print("\n[SYSTEM] JARVIS ONLINE.")
    print("[TIP] Press Ctrl+C to Stop Speech / Interrupt.")

    while True:
        try:
            user_text = ears.listen()
            if not user_text: continue

            print("Jarvis: ", end='', flush=True)
            
            try:
                stream = ollama.chat(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": "You are Jarvis. Conversational, plain text, concise."},
                        {"role": "user", "content": user_text}
                    ],
                    stream=True
                )

                buffer = ""
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    print(content, end='', flush=True)
                    buffer += content
                    
                    if re.search(r'[.!?]', buffer):
                        parts = re.split(r'(?<=[.!?])\s+', buffer)
                        for sentence in parts[:-1]:
                            mouth.speak(sentence)
                        buffer = parts[-1]
                
                if buffer:
                    mouth.speak(buffer)
                
                mouth.wait()
                print("\n")

            except KeyboardInterrupt:
                mouth.stop_talking() # Triggers the break in _play_chunked
                print("\n[INTERRUPTED] Stopping...")
                time.sleep(0.5) # Brief cool down

        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down.")
            break

if __name__ == "__main__":
    main()
