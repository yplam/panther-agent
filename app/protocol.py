import json
from typing import Dict, Any, List, Optional
from app import config 
# --- Message Types ---
TYPE_HELLO = "hello"
TYPE_LISTEN = "listen"
TYPE_ABORT = "abort"
TYPE_IOT = "iot"
TYPE_STT = "stt"
TYPE_LLM = "llm"
TYPE_TTS = "tts"

# --- Listen States ---
LISTEN_STATE_START = "start"
LISTEN_STATE_STOP = "stop"
LISTEN_STATE_DETECT = "detect"

# --- Listen Modes ---
LISTEN_MODE_AUTO = "auto"
LISTEN_MODE_MANUAL = "manual"
LISTEN_MODE_REALTIME = "realtime"

# --- Abort Reasons ---
ABORT_REASON_WAKE_WORD = "wake_word_detected"

# --- TTS States ---
TTS_STATE_START = "start"
TTS_STATE_STOP = "stop"
TTS_STATE_SENTENCE_START = "sentence_start"
TTS_STATE_SENTENCE_END = "sentence_end"

# --- Audio Params ---
AUDIO_FORMAT_OPUS = "opus"
AUDIO_SAMPLE_RATE_16K = 16000
AUDIO_CHANNELS_MONO = 1

# --- Helper Functions ---

def create_server_hello(session_id: str, sample_rate: int = config.TTS_SAMPLE_RATE) -> str:
    """Creates the server hello JSON message."""
    message = {
        "type": TYPE_HELLO,
        "version": 1,
        "session_id": session_id,
        "transport": "websocket",
        "audio_params": {
            "format": AUDIO_FORMAT_OPUS,
            "sample_rate": sample_rate,
            "channels": AUDIO_CHANNELS_MONO,
            "frame_duration": 60
        }
    }
    return json.dumps(message)

def create_stt_message(text: str, session_id: Optional[str] = None) -> str:
    """Creates the STT result JSON message."""
    message = {"type": TYPE_STT, "text": text}
    if session_id:
        message["session_id"] = session_id
    return json.dumps(message)

def create_llm_emotion_message(emotion: str, session_id: Optional[str] = None, text: Optional[str] = None) -> str:
    """Creates the LLM emotion JSON message."""
    message = {"type": TYPE_LLM, "emotion": emotion}
    if session_id:
        message["session_id"] = session_id
    if text:
        message["text"] = text
    return json.dumps(message)

def create_tts_message(state: str, text: Optional[str] = None, sample_rate: Optional[int] = None, session_id: Optional[str] = None) -> str:
    """Creates the TTS control JSON message."""
    message: Dict[str, Any] = {"type": TYPE_TTS, "state": state}
    
    if text and (state == TTS_STATE_SENTENCE_START or state == TTS_STATE_SENTENCE_END):
        message["text"] = text
        
    if sample_rate and state == TTS_STATE_START:
        message["sample_rate"] = sample_rate
        
    if session_id:
        message["session_id"] = session_id
        
    return json.dumps(message)

def create_iot_command_message(commands: List[Dict[str, Any]], session_id: Optional[str] = None) -> str:
    """Creates the IoT command JSON message."""
    message = {"type": TYPE_IOT, "commands": commands}
    if session_id:
        message["session_id"] = session_id
    return json.dumps(message)

def create_listen_message(state: str, mode: str = LISTEN_MODE_MANUAL) -> str:
    """Creates a listen control message."""
    message = {"type": TYPE_LISTEN, "state": state}
    if state == LISTEN_STATE_START:
        message["mode"] = mode
    return json.dumps(message)

def parse_json(data: str) -> Optional[Dict[str, Any]]:
    """Safely parses a JSON string."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None