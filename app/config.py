import os
from dotenv import load_dotenv

load_dotenv()

# WebSocket Server Config
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8765))
WEBSOCKET_PATH = os.getenv("WEBSOCKET_PATH", "/voice")
EXPECTED_PROTOCOL_VERSION = "1"
SERVER_HELLO_TIMEOUT_S = 10 # Match client timeout
IDLE_TIMEOUT_S = int(os.getenv("IDLE_TIMEOUT_S", 300)) # Timeout for inactive connections

# Authentication
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL") # Example: URL to verify token

# AI Services (OpenAI Compatible Example)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Optional: For self-hosted/proxy
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_OUTPUT_FORMAT = "opus" # Must match spec
TTS_SAMPLE_RATE = 16000 # Must match spec and server hello
TTS_SOURCE_SAMPLE_RATE = 24000 # <--- Rate from OpenAI TTS
OPUS_FRAME_MS = 60 # <--- Match client's expected frame duration (e.g., 60ms)
OPUS_BITRATE = "auto" # Or set a specific bitrate like 32000

# Audio Processing
AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "audio_files")
SAVE_AUDIO_FILES = os.getenv("SAVE_AUDIO_FILES", "false").lower() == "true"
ASR_MIN_OPUS_PACKETS = int(os.getenv("ASR_MIN_OPUS_PACKETS", "10"))  # Minimum number of opus packets for ASR

# Voice Activity Detection
VAD_ENABLED = os.getenv("VAD_ENABLED", "false").lower() == "true"
VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "1000"))  # Stop after 1s silence
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))  # Default energy threshold

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# IoT (Placeholder)
IOT_SERVICE_URL = os.getenv("IOT_SERVICE_URL")

# Simple In-Memory Auth Token Store (Replace with real auth service call)
# WARNING: For demonstration only. Use a secure method in production.
VALID_TOKENS = set(os.getenv("VALID_BEARER_TOKENS", "").split(','))