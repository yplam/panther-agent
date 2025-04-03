import asyncio
import websockets
import argparse
import json
import uuid
import time
import datetime
import numpy as np
import math
import soundfile as sf # Using soundfile for easier WAV I/O with numpy
import opuslib_next
import logging
from typing import Optional, Dict, Any, AsyncGenerator, List
import os

# --- Configuration & Constants ---
CLIENT_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
OPUS_FRAME_MS = 60 # Should match server/client Opus encoder/decoder settings
OPUS_FRAME_SIZE = (CLIENT_SAMPLE_RATE * OPUS_FRAME_MS) // 1000
# Use opuslib_next constants
OPUS_BITRATE = "auto" # Or e.g., 32000
# NOTE: The actual ESP32 firmware might use different Opus complexity settings (e.g., 3 or 5)
# depending on the board, while this script uses the default from opuslib_next.

# Logging setup
log_file = "client_log.txt"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)-8s %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ClientSim")

# State Tracking
# NOTE: This script simulates a specific scenario (sending one WAV file) and does not
# replicate the full complex state machine of the ESP32 firmware (e.g., Idle, Connecting,
# Listening with VAD/Wake Word, Speaking, Auto-reconnect logic, keep_listening).
is_receiving_tts = False
tts_packets = []  # Store opus packets instead of raw bytes
tts_decoder: Optional[opuslib_next.Decoder] = None
# Always use client sample rate, ignore server one
current_interaction = ""  # Track whether we're in wakeword or question phase
tts_stop_event = asyncio.Event()  # Event to signal when TTS is complete


# --- Audio Helper Functions ---

def read_and_resample_wav(filepath: str) -> Optional[np.ndarray]:
    """Reads a WAV file, converts to mono, resamples to CLIENT_SAMPLE_RATE."""
    try:
        data, samplerate = sf.read(filepath, dtype='int16')
        logger.info(f"Read '{filepath}': Sample rate={samplerate}, Channels={data.ndim}, Duration={len(data)/samplerate:.2f}s")
        # NOTE: This simulates reading a pre-recorded file. The actual ESP32 firmware
        # reads live audio from a microphone via an AudioCodec, potentially applies
        # Acoustic Echo Cancellation (AEC), Voice Activity Detection (VAD),
        # and Wake Word detection before encoding. This script bypasses those steps.

        # Ensure numpy array
        data = np.array(data, dtype=np.int16)

        # Convert to mono if stereo
        if data.ndim > 1 and data.shape[1] == 2:
            logger.info("Converting stereo WAV to mono by averaging channels.")
            data = data.mean(axis=1).astype(np.int16)
        elif data.ndim > 1:
            logger.warning(f"Unsupported number of channels ({data.shape[1]}). Taking first channel.")
            data = data[:, 0]

        # Resample if necessary
        if samplerate != CLIENT_SAMPLE_RATE:
            logger.info(f"Resampling from {samplerate}Hz to {CLIENT_SAMPLE_RATE}Hz...")
            # Use resampy (installed dependency) for resampling
            import resampy
            data = resampy.resample(data.astype(np.float32), sr_orig=samplerate, sr_new=CLIENT_SAMPLE_RATE)
            data = data.astype(np.int16)
            logger.info("Resampling complete.")

        # DEBUG: Save the processed audio to a temporary file
        debug_filename = f"debug_{os.path.basename(filepath)}"
        logger.info(f"Saving debug output to {debug_filename}")
        save_pcm_to_wav(data, CLIENT_SAMPLE_RATE, 1, debug_filename)

        return data

    except Exception as e:
        logger.error(f"Error reading or processing WAV file '{filepath}': {e}", exc_info=True)
        return None

async def encode_pcm_to_opus(pcm_data: np.ndarray) -> AsyncGenerator[bytes, None]:
    """Encodes 16kHz mono PCM data into Opus frames."""
    try:
        # Updated for opuslib_next
        encoder = opuslib_next.Encoder(CLIENT_SAMPLE_RATE, TARGET_CHANNELS, opuslib_next.APPLICATION_AUDIO)
        logger.info("Opus encoder initialized successfully.")
        # encoder.bitrate = OPUS_BITRATE
        # Use signal type optimized for voice
        encoder.signal = opuslib_next.SIGNAL_VOICE 
    except Exception as e:
        logger.error(f"Failed to initialize Opus encoder: {e}")
        return

    total_samples = len(pcm_data)
    current_pos = 0
    frame_count = 0

    while current_pos < total_samples:
        end_pos = current_pos + OPUS_FRAME_SIZE
        # Get frame data
        frame = pcm_data[current_pos:end_pos]

        # Pad the last frame if necessary
        if len(frame) < OPUS_FRAME_SIZE:
            padding_needed = OPUS_FRAME_SIZE - len(frame)
            padding = np.zeros(padding_needed, dtype=np.int16)
            frame = np.concatenate((frame, padding))
            logger.debug(f"Padding last frame with {padding_needed} zeros.")

        # Encode
        try:
            encoded_frame = encoder.encode(frame.tobytes(), OPUS_FRAME_SIZE)
            # Don't log every encoded frame to keep logs clean
            yield encoded_frame
            frame_count += 1
        except opuslib_next.OpusError as e:
            logger.error(f"Opus encoding error on frame {frame_count}: {e}")
            # Decide if to continue or stop
            break
        except Exception as e:
            logger.error(f"Unexpected error during encoding frame {frame_count}: {e}")
            break

        current_pos = end_pos
        # Yield control to the event loop periodically
        if frame_count % 10 == 0: # Adjust frequency as needed
             await asyncio.sleep(0.001)

    logger.info(f"Finished encoding {frame_count} Opus frames.")

def decode_opus_packets(opus_packets: List[bytes], sample_rate: int, channels: int) -> Optional[np.ndarray]:
    """Decodes a list of Opus packets into PCM data."""
    global tts_decoder
    
    if not tts_decoder:
        try:
            logger.info(f"Initializing Opus Decoder for TTS: {sample_rate}Hz, {channels}ch")
            tts_decoder = opuslib_next.Decoder(sample_rate, channels)
            # NOTE: ESP32 firmware resets decoder state explicitly between speaking states.
            # This script resets by creating a new decoder instance on TTS "start".
        except Exception as e:
            logger.error(f"Failed to initialize Opus decoder: {e}")
            return None
    
    try:
        # Decode each packet individually
        pcm_chunks = []
        frame_size = int(sample_rate * OPUS_FRAME_MS / 1000)  # Size in samples
        
        for i, packet in enumerate(opus_packets):
            if not packet:  # Skip empty packets
                continue
                
            try:
                # Decode expects the number of samples per channel for the frame
                decoded_pcm = tts_decoder.decode(packet, frame_size)
                pcm_chunks.append(decoded_pcm)
            except opuslib_next.OpusError as e:
                logger.error(f"Error decoding opus packet {i}: {e}")
                # Continue with next packet
        
        if not pcm_chunks:
            return None
            
        # Combine all PCM chunks
        combined_pcm = b''.join(pcm_chunks)
        pcm_data = np.frombuffer(combined_pcm, dtype=np.int16)
        logger.info(f"Decoded {len(opus_packets)} Opus packets -> {len(pcm_data)} PCM samples")
        return pcm_data
        
    except Exception as e:
        logger.error(f"Unexpected error during opus decoding: {e}")
        tts_decoder = None  # Reset decoder on error
        return None

def save_pcm_to_wav(pcm_data: np.ndarray, sample_rate: int, channels: int, filepath: str):
    """Saves PCM data to a WAV file."""
    try:
        sf.write(filepath, pcm_data, sample_rate, subtype='PCM_16')
        logger.info(f"Saved {len(pcm_data)/sample_rate:.2f}s of audio to '{filepath}' ({sample_rate}Hz, {channels}ch)")
    except Exception as e:
        logger.error(f"Error saving WAV file '{filepath}': {e}", exc_info=True)


# --- WebSocket Client Logic ---

async def send_audio_task(websocket, pcm_data):
    """Task to encode and send audio frames."""
    logger.info("Starting audio sending task...")
    async for opus_frame in encode_pcm_to_opus(pcm_data):
        try:
            await websocket.send(opus_frame)
            logger.info(f"[CLIENT->SERVER] BINARY: {len(opus_frame)} bytes, Decoded Opus: {OPUS_FRAME_SIZE*2} bytes")
            # Optional small delay to simulate real-time streaming if needed
            await asyncio.sleep(OPUS_FRAME_MS / 1000 * 0.9) # Simulate near real-time
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while sending audio.")
            break
        except Exception as e:
            logger.error(f"Error sending audio frame: {e}")
            break
    logger.info("Audio sending task finished.")

async def receive_messages(websocket):
    """Task to receive and handle messages from the server."""
    global is_receiving_tts, tts_packets, tts_decoder, current_interaction, tts_stop_event
    logger.info("Starting message receiving task...")
    
    try:
        async for message in websocket:
            if isinstance(message, str):
                # Handle Text Message
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    logger.info(f"[SERVER->CLIENT] TEXT: {message}")

                    if msg_type == "stt":
                        text = data.get("text", "<no text>")
                        print(f"\n--- ASR Result ---")
                        print(f"User: {text}")
                        print(f"--------------------\n")
                    elif msg_type == "llm":
                        emotion = data.get("emotion", "neutral")
                        text = data.get("text", "<no text>")
                        print(f"\n--- LLM Emotion ---")
                        print(f"Emotion: {emotion}")
                        if text:
                            print(f"Text: {text}")
                        print(f"--------------------\n")
                    elif msg_type == "tts":
                        state = data.get("state")
                        if state == "start":
                            logger.info("Server indicated TTS start.")
                            is_receiving_tts = True
                            tts_packets = []  # Clear previous packets
                            
                            # Get sample rate from TTS start message but ignore it
                            svr_sr = data.get("sample_rate")
                            if svr_sr and isinstance(svr_sr, int):
                                logger.info(f"Server TTS sample rate is {svr_sr}Hz, but using client rate {CLIENT_SAMPLE_RATE}Hz")
                                
                            # Reset decoder in case previous TTS had errors
                            tts_decoder = None
                        elif state == "stop":
                            logger.info("Server indicated TTS stop.")
                            is_receiving_tts = False
                            if tts_packets:
                                logger.info(f"Processing buffered TTS audio ({len(tts_packets)} packets)...")
                                pcm_data = decode_opus_packets(tts_packets, CLIENT_SAMPLE_RATE, TARGET_CHANNELS)
                                if pcm_data is not None and len(pcm_data) > 0:
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    if current_interaction == "wakeword":
                                        filename = f"response_wakeword.wav"
                                    else:  # question
                                        filename = f"response_question.wav"
                                    save_pcm_to_wav(pcm_data, CLIENT_SAMPLE_RATE, TARGET_CHANNELS, filename)
                                else:
                                    logger.warning("Failed to decode TTS audio or decoded audio was empty.")
                                tts_packets = []
                            else:
                                logger.info("No TTS audio data buffered.")
                                
                            # Signal that TTS is complete so we can proceed with next interaction
                            tts_stop_event.set()
                        elif state == "sentence_start":
                            text = data.get("text", "<no sentence text>")
                            print(f"\n--- TTS Sentence Start ---")
                            print(f"Assistant (speaking): {text}")
                            print(f"-------------------------\n")
                        elif state == "sentence_end":
                            text = data.get("text", "<no sentence text>")
                            print(f"\n--- TTS Sentence End ---")
                            print(f"Finished: {text}")
                            print(f"-----------------------\n")
                    elif msg_type == "iot":
                        commands = data.get("commands", [])
                        print("\n--- IoT Command ---")
                        print(f"{json.dumps(data, indent=2)}")
                        print("-------------------\n")
                    # Ignore 'hello' after handshake

                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON text message: {message[:200]}")
                except Exception as e:
                    logger.error(f"Error processing text message: {e}", exc_info=True)

            elif isinstance(message, bytes):
                # Handle Binary Message (Audio)
                if is_receiving_tts:
                    # Store each packet separately
                    tts_packets.append(message)
                    logger.info(f"[SERVER->CLIENT] BINARY: {len(message)} bytes, Decoded Opus: {OPUS_FRAME_SIZE*2} bytes")
                else:
                    logger.warning(f"[SERVER->CLIENT] Received unexpected binary data ({len(message)} bytes) when not expecting TTS.")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Connection closed normally by server.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed with error: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Error in receive loop: {e}", exc_info=True)
    finally:
        logger.info("Message receiving task finished.")
        # Ensure any remaining buffered TTS audio is processed on disconnect
        if is_receiving_tts and tts_packets:
            logger.warning(f"Connection closed during TTS, processing buffered audio...")
            pcm_data = decode_opus_packets(tts_packets, CLIENT_SAMPLE_RATE, TARGET_CHANNELS)
            if pcm_data is not None and len(pcm_data) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if current_interaction == "wakeword":
                    filename = f"response_wakeword_partial.wav"
                else:  # question
                    filename = f"response_question_partial.wav"
                save_pcm_to_wav(pcm_data, CLIENT_SAMPLE_RATE, TARGET_CHANNELS, filename)
            tts_packets = []
            is_receiving_tts = False


async def run_client(url: str, token: str, device_id: str, client_id: str):
    """Main function to run the WebSocket client simulation."""
    global current_interaction, tts_stop_event
    
    logger.info(f"Starting ESP32 client simulator, logging to {log_file}")
    
    # Reset the event at the start
    tts_stop_event = asyncio.Event()

    # 1. Prepare Audio Data for wakeword (audio_0.wav) and question (audio_1.wav)
    wakeword_wav = "audio_0.wav"
    question_wav = "audio_1.wav"
    
    wakeword_data = read_and_resample_wav(wakeword_wav)
    if wakeword_data is None:
        logger.error(f"Cannot proceed without valid wakeword audio data from {wakeword_wav}.")
        return
        
    question_data = read_and_resample_wav(question_wav)
    if question_data is None:
        logger.error(f"Cannot proceed without valid question audio data from {question_wav}.")
        return

    # 2. Prepare Headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Protocol-Version": "1",
        "Device-Id": device_id,
        "Client-Id": client_id,
    }
    logger.info(f"Connecting to {url} with headers:")
    for k, v in headers.items():
        # Mask sensitive parts of token in log
        log_v = v if k != "Authorization" else f"Bearer {v[:5]}..."
        logger.info(f"  {k}: {log_v}")

    # 3. Connect and Handle Communication
    try:
        async with websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=20, # Send pings to keep connection alive
            ping_timeout=20   # Timeout for pong response
        ) as websocket:
            logger.info("WebSocket connection established.")

            # --- Handshake ---
            # Send Client Hello
            client_hello = {
                "type": "hello",
                "version": 1,
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": CLIENT_SAMPLE_RATE,
                    "channels": TARGET_CHANNELS,
                    "frame_duration": OPUS_FRAME_MS
                }
            }
            await websocket.send(json.dumps(client_hello))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(client_hello)}")

            # Receive Server Hello (with timeout)
            try:
                server_hello_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                if not isinstance(server_hello_raw, str):
                     raise ValueError("Server 'hello' was not text.")
                server_hello = json.loads(server_hello_raw)
                logger.info(f"[SERVER->CLIENT] TEXT: {server_hello_raw}")

                if server_hello.get("type") != "hello" or server_hello.get("transport") != "websocket":
                    raise ValueError("Invalid server 'hello' message.")

                # Log server TTS sample rate but don't use it
                svr_sr = server_hello.get("audio_params", {}).get("sample_rate")
                if svr_sr and isinstance(svr_sr, int):
                    logger.info(f"Server TTS sample rate is {svr_sr}Hz, but using client rate {CLIENT_SAMPLE_RATE}Hz")
                    if svr_sr != CLIENT_SAMPLE_RATE:
                         logger.warning(f"Server TTS sample rate ({svr_sr}Hz) differs from client rate ({CLIENT_SAMPLE_RATE}Hz). Using client rate anyway.")
                else:
                     logger.warning(f"Server did not provide valid TTS sample rate, using client rate {CLIENT_SAMPLE_RATE}Hz")

                # Get session ID
                session_id = server_hello.get("session_id", "")
                if session_id:
                    logger.info(f"Server assigned session ID: {session_id}")
                else:
                    logger.warning("Server did not provide a session ID")

            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Server Hello.")
                return
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to process Server Hello: {e}")
                return

            logger.info("Handshake complete.")

            # --- Send device descriptors ---
            # Send Speaker descriptor
            speaker_descriptor = {
                "session_id": "",
                "type": "iot",
                "update": True,
                "descriptors": [{
                    "name": "Speaker",
                    "description": "扬声器",
                    "properties": {
                        "volume": {"description": "当前音量值", "type": "number"}
                    },
                    "methods": {
                        "SetVolume": {
                            "description": "设置音量",
                            "parameters": {
                                "volume": {"description": "0到100之间的整数", "type": "number"}
                            }
                        }
                    }
                }]
            }
            await websocket.send(json.dumps(speaker_descriptor))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(speaker_descriptor)}")
            
            # Send Screen descriptor
            screen_descriptor = {
                "session_id": "",
                "type": "iot",
                "update": True,
                "descriptors": [{
                    "name": "Screen",
                    "description": "这是一个屏幕，可设置主题和亮度",
                    "properties": {
                        "theme": {"description": "主题", "type": "string"},
                        "brightness": {"description": "当前亮度百分比", "type": "number"}
                    },
                    "methods": {
                        "SetTheme": {
                            "description": "设置屏幕主题",
                            "parameters": {
                                "theme_name": {"description": "主题模式, light 或 dark", "type": "string"}
                            }
                        },
                        "SetBrightness": {
                            "description": "设置亮度",
                            "parameters": {
                                "brightness": {"description": "0到100之间的整数", "type": "number"}
                            }
                        }
                    }
                }]
            }
            await websocket.send(json.dumps(screen_descriptor))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(screen_descriptor)}")
            
            # Send Chassis descriptor
            chassis_descriptor = {
                "session_id": "",
                "type": "iot",
                "update": True,
                "descriptors": [{
                    "name": "Chassis",
                    "description": "小机器人的底座：有履带可以移动；可以调整灯光效果",
                    "properties": {
                        "light_mode": {"description": "灯光效果编号", "type": "number"}
                    },
                    "methods": {
                        "GoForward": {"description": "向前走", "parameters": {}},
                        "GoBack": {"description": "向后退", "parameters": {}},
                        "TurnLeft": {"description": "向左转", "parameters": {}},
                        "TurnRight": {"description": "向右转", "parameters": {}},
                        "Dance": {"description": "跳舞", "parameters": {}},
                        "SwitchLightMode": {
                            "description": "打开灯",
                            "parameters": {
                                "lightmode": {"description": "1到6之间的整数", "type": "number"}
                            }
                        }
                    }
                }]
            }
            await websocket.send(json.dumps(chassis_descriptor))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(chassis_descriptor)}")

            # --- Start Interaction ---
            # Run receiver in the background
            receiver_task = asyncio.create_task(receive_messages(websocket))

            # Send wakeword audio data first
            logger.info(f"Sending wake word audio from {wakeword_wav}...")
            current_interaction = "wakeword"
            wakeword_sender = asyncio.create_task(send_audio_task(websocket, wakeword_data))
            await wakeword_sender

            # Send wake word detection notification
            detect_msg = {"session_id": "", "type": "listen", "state": "detect", "text": "你好小智"}
            await websocket.send(json.dumps(detect_msg))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(detect_msg)}")

            # Wait for TTS to complete (server sends "tts" with "state":"stop")
            logger.info("Waiting for wake word TTS response to complete...")
            await tts_stop_event.wait()
            time.sleep(0.1)
            logger.info("Wake word TTS response complete, proceeding with question")
            
            # Reset the event for next use
            tts_stop_event.clear()
            
            # Send listen start for user command
            listen_start_msg = {"session_id": "", "type": "listen", "state": "start", "mode": "auto"}
            await websocket.send(json.dumps(listen_start_msg))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(listen_start_msg)}")
            
            # Send device state update
            # device_states = {
            #     "session_id": "",
            #     "type": "iot",
            #     "update": True,
            #     "states": [
            #         {"name": "Speaker", "state": {"volume": 90}},
            #         {"name": "Screen", "state": {"theme": "light", "brightness": 75}},
            #         {"name": "Chassis", "state": {"light_mode": 1}}
            #     ]
            # }
            # await websocket.send(json.dumps(device_states))
            # logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(device_states)}")

            # Start sending question audio data
            logger.info(f"Sending question audio from {question_wav}...")
            current_interaction = "question"
            question_sender = asyncio.create_task(send_audio_task(websocket, question_data))
            await question_sender  # Wait for all audio to be sent

            # Send listen stop message to signal end of speech
            listen_stop_msg = {"session_id": "", "type": "listen", "state": "stop"}
            await websocket.send(json.dumps(listen_stop_msg))
            logger.info(f"[CLIENT->SERVER] TEXT: {json.dumps(listen_stop_msg)}")

            # Keep connection open for receiving responses
            logger.info("Audio sent. Waiting for server responses...")
            await asyncio.sleep(30.0)  # Wait for responses before disconnecting

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {url}")
    except websockets.exceptions.InvalidHandshake as e:
        logger.error(f"WebSocket handshake failed: {e.status_code} {getattr(e, 'reason', '')}")
        logger.error(f"Headers received: {getattr(e, 'headers', 'N/A')}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused by server at {url}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Client finished.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate ESP32 WebSocket Voice Client")
    parser.add_argument("-u", "--url", default="wss://api.tenclass.net/xiaozhi/v1/", help="WebSocket server URL")
    parser.add_argument("-t", "--token", default="test-token", help="Bearer token for Authorization header")
    parser.add_argument("-d", "--device-id", default="94:a9:90:28:d9:28", help="Device-Id header value (e.g., MAC address)")
    parser.add_argument("-c", "--client-id", default=str(uuid.uuid4()), help="Client-Id header value (UUID)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("-l", "--log-file", default="client_log.txt", help="Log file path")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Set websockets library logger level too if desired
        # logging.getLogger("websockets").setLevel(logging.DEBUG)
        
    # Update log file from command line if provided
    if args.log_file != "client_log.txt":
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Set up new handlers
        logging.basicConfig(
            level=logging.INFO if not args.verbose else logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging to {args.log_file}")

    try:
        asyncio.run(run_client(args.url, args.token, args.device_id, args.client_id))
    except KeyboardInterrupt:
        logger.info("Client interrupted by user.")