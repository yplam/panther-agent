import asyncio
import uuid
from typing import Dict, Any, Optional, List
import websockets
from enum import Enum, auto

from app.logger import logger
from app import config

class ClientState(Enum):
    CONNECTING = auto()      # Initial connection, before hello exchange
    HANDSHAKE_PENDING = auto() # Server sent hello, waiting for client tasks
    IDLE = auto()            # Connected, ready for interaction
    LISTENING = auto()       # Client sent listen_start, accumulating audio
    PROCESSING = auto()      # Client sent listen_stop, agent is running
    SPEAKING = auto()        # Server sent tts_start, sending audio
    CLOSING = auto()         # Connection is being closed

class ConnectedClient:
    """Represents the state and data for a single connected client."""
    def __init__(self, websocket: websockets.WebSocketServerProtocol, client_id: str, device_id: str):
        self.websocket = websocket
        self.client_id: str = client_id
        self.device_id: str = device_id
        self.state: ClientState = ClientState.CONNECTING
        self.session_id: str = "" # Generated on listen_start
        self.audio_buffer: List[bytes] = []  # List of opus audio packets
        self.listen_mode: Optional[str] = None
        self.last_activity_time: float = asyncio.get_event_loop().time()
        self.tts_task: Optional[asyncio.Task] = None # To allow cancellation
        self.conversation_history: List[Dict[str, str]] = [] # For LLM context
        
        # Voice activity detection state
        self.client_have_voice: bool = False
        self.client_voice_stop: bool = False
        self.client_no_voice_last_time: float = 0.0
        self.asr_server_receive: bool = True  # Whether ASR is ready to receive audio
        self.client_abort: bool = False  # Whether client aborted current processing
        self.close_after_chat: bool = False  # Whether to close connection after chat

    def update_activity(self):
        self.last_activity_time = asyncio.get_event_loop().time()

    def change_state(self, new_state: ClientState):
        if self.state != new_state:
            logger.info(f"Client {self.client_id}/{self.device_id}: State transition {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.update_activity() # Reset idle timer on state change

    def start_listening(self, mode: str):
        self.change_state(ClientState.LISTENING)
        self.listen_mode = mode
        self.session_id = str(uuid.uuid4()) # Generate unique ID for this interaction
        self.audio_buffer = []  # Clear previous buffer
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_no_voice_last_time = 0.0
        self.asr_server_receive = True
        self.client_abort = False
        logger.info(f"Client {self.client_id}: Starting listen session {self.session_id} (mode: {mode})")

    def stop_listening(self):
        if self.state == ClientState.LISTENING:
            self.change_state(ClientState.PROCESSING)
            self.client_voice_stop = True
            logger.info(f"Client {self.client_id}: Stopped listening for session {self.session_id}. Buffer size: {len(self.audio_buffer)} packets")

    def add_audio_packet(self, audio_packet: bytes) -> bool:
        """Add an audio packet to the buffer and return whether it should be processed."""
        if self.state != ClientState.LISTENING or not self.asr_server_receive:
            return False
            
        self.audio_buffer.append(audio_packet)
        self.update_activity()
        
        # Implement auto-stop based on silence detection if needed
        if self.listen_mode == "auto" and config.VAD_ENABLED:
            # This would call VAD detection code
            # For now, just return based on buffer length
            return len(self.audio_buffer) >= config.ASR_MIN_OPUS_PACKETS
            
        return self.client_voice_stop and len(self.audio_buffer) >= config.ASR_MIN_OPUS_PACKETS

    def reset_vad_states(self):
        """Reset voice activity detection related states"""
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_no_voice_last_time = 0.0

    def start_speaking(self, tts_task: asyncio.Task):
        self.change_state(ClientState.SPEAKING)
        self.tts_task = tts_task
        # Clear history related to the *previous* turn before starting to speak the new response
        # Or keep history based on session_id management strategy
        # self.conversation_history.clear() # Example: Clear after each turn

    async def stop_speaking(self, aborted: bool = False):
         current_tts_task = self.tts_task
         self.tts_task = None # Clear the task reference

         if current_tts_task and not current_tts_task.done():
             if aborted:
                 logger.warning(f"Client {self.client_id}: Aborting TTS task for session {self.session_id}")
                 current_tts_task.cancel()
                 try:
                     await current_tts_task # Wait for cancellation to propagate
                 except asyncio.CancelledError:
                     logger.info(f"Client {self.client_id}: TTS task successfully cancelled.")
                 except Exception as e:
                      logger.error(f"Client {self.client_id}: Error awaiting cancelled TTS task: {e}")
             else:
                 # If not aborted, the task should complete naturally
                 logger.info(f"Client {self.client_id}: TTS task finishing normally for session {self.session_id}")
                 # No need to await here if state transition handles next steps

         # Transition state AFTER handling the task
         if self.state == ClientState.SPEAKING:
             # Decide next state based on client logic (keep_listening_ flag in ESP32)
             # Since server doesn't know 'keep_listening_', default to IDLE.
             # The client will send 'listen start' again if needed.
             self.change_state(ClientState.IDLE)

         # Handle closing connection if needed
         if self.close_after_chat:
             logger.info(f"Client {self.client_id}: Closing connection after chat completion")
             if not self.websocket.closed:
                 await self.websocket.close(1000, "Chat completed")


class ClientManager:
    """Manages all connected clients."""
    def __init__(self):
        self._clients: Dict[websockets.WebSocketServerProtocol, ConnectedClient] = {}
        self._lock = asyncio.Lock()

    async def add_client(self, websocket: websockets.WebSocketServerProtocol, client_id: str, device_id: str) -> ConnectedClient:
        async with self._lock:
            client = ConnectedClient(websocket, client_id, device_id)
            self._clients[websocket] = client
            logger.info(f"Client connected: {client_id} ({device_id}) from {websocket.remote_address}")
            return client

    async def remove_client(self, websocket: websockets.WebSocketServerProtocol):
        async with self._lock:
            client = self._clients.pop(websocket, None)
            if client:
                client.change_state(ClientState.CLOSING) # Mark as closing
                logger.info(f"Client disconnected: {client.client_id} ({client.device_id})")
                # Cancel any ongoing TTS task for this client
                if client.tts_task and not client.tts_task.done():
                    logger.warning(f"Cancelling TTS task for disconnected client {client.client_id}")
                    client.tts_task.cancel()
                    # Don't await here, just cancel
            else:
                logger.warning(f"Attempted to remove non-existent client: {websocket.remote_address}")


    async def get_client(self, websocket: websockets.WebSocketServerProtocol) -> Optional[ConnectedClient]:
        async with self._lock:
            return self._clients.get(websocket)

    async def get_all_clients(self) -> List[ConnectedClient]:
         async with self._lock:
             return list(self._clients.values())

client_manager = ClientManager()