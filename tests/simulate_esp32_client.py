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

# --- Configuration & Constants ---
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
OPUS_FRAME_MS = 60 # Should match server/client Opus encoder/decoder settings
OPUS_FRAME_SIZE = (TARGET_SAMPLE_RATE * OPUS_FRAME_MS) // 1000
# Use opuslib_next constants
OPUS_BITRATE = "auto" # Or e.g., 32000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClientSim")

# State Tracking
is_receiving_tts = False
tts_audio_buffer = bytearray()
tts_decoder: Optional[opuslib_next.Decoder] = None
server_tts_sample_rate: int = TARGET_SAMPLE_RATE # Default, updated by server hello


# --- Audio Helper Functions ---

def read_and_resample_wav(filepath: str) -> Optional[np.ndarray]:
    """Reads a WAV file, converts to mono, resamples to TARGET_SAMPLE_RATE."""
    try:
        data, samplerate = sf.read(filepath, dtype='int16')
        logger.info(f"Read '{filepath}': Sample rate={samplerate}, Channels={data.ndim}, Duration={len(data)/samplerate:.2f}s")

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
        if samplerate != TARGET_SAMPLE_RATE:
            logger.info(f"Resampling from {samplerate}Hz to {TARGET_SAMPLE_RATE}Hz...")
            # Use resampy (installed dependency) for resampling
            import resampy
            data = resampy.resample(data.astype(np.float32), sr_orig=samplerate, sr_new=TARGET_SAMPLE_RATE)
            data = data.astype(np.int16)
            logger.info("Resampling complete.")

        return data

    except Exception as e:
        logger.error(f"Error reading or processing WAV file '{filepath}': {e}", exc_info=True)
        return None

async def encode_pcm_to_opus(pcm_data: np.ndarray) -> AsyncGenerator[bytes, None]:
    """Encodes 16kHz mono PCM data into Opus frames."""
    try:
        # Updated for opuslib_next
        encoder = opuslib_next.Encoder(TARGET_SAMPLE_RATE, TARGET_CHANNELS, opuslib_next.APPLICATION_AUDIO)
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
            # logger.debug(f"Encoded Opus frame {frame_count}: {len(encoded_frame)} bytes")
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
            # Optional small delay to simulate real-time streaming if needed
            # await asyncio.sleep(OPUS_FRAME_MS / 1000 * 0.9) # Simulate near real-time
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while sending audio.")
            break
        except Exception as e:
            logger.error(f"Error sending audio frame: {e}")
            break
    logger.info("Audio sending task finished.")

async def receive_messages(websocket):
    """Task to receive and handle messages from the server."""
    global is_receiving_tts, tts_audio_buffer, tts_decoder, server_tts_sample_rate
    logger.info("Starting message receiving task...")
    tts_packets = []  # Store opus packets instead of raw bytes
    
    try:
        async for message in websocket:
            if isinstance(message, str):
                # Handle Text Message
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    logger.info(f"Received TEXT: Type='{msg_type}', Data={json.dumps(data, indent=2)}") # Pretty print JSON

                    if msg_type == "stt":
                        text = data.get("text", "<no text>")
                        print(f"\n--- ASR Result ---")
                        print(f"User: {text}")
                        print(f"--------------------\n")
                    elif msg_type == "llm":
                        emotion = data.get("emotion", "neutral")
                        print(f"\n--- LLM Emotion ---")
                        print(f"Emotion: {emotion}")
                        print(f"--------------------\n")
                    elif msg_type == "tts":
                        state = data.get("state")
                        if state == "start":
                            logger.info("Server indicated TTS start.")
                            is_receiving_tts = True
                            tts_packets = []  # Clear previous packets
                            # Reset decoder in case previous TTS had errors
                            tts_decoder = None
                        elif state == "stop":
                            logger.info("Server indicated TTS stop.")
                            is_receiving_tts = False
                            if tts_packets:
                                logger.info(f"Processing buffered TTS audio ({len(tts_packets)} packets)...")
                                pcm_data = decode_opus_packets(tts_packets, server_tts_sample_rate, TARGET_CHANNELS)
                                if pcm_data is not None and len(pcm_data) > 0:
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    filename = f"received_tts_{timestamp}.wav"
                                    save_pcm_to_wav(pcm_data, server_tts_sample_rate, TARGET_CHANNELS, filename)
                                else:
                                    logger.warning("Failed to decode TTS audio or decoded audio was empty.")
                                tts_packets = []
                            else:
                                logger.info("No TTS audio data buffered.")
                        elif state == "sentence_start":
                             text = data.get("text", "<no sentence text>")
                             print(f"\n--- TTS Sentence ---")
                             print(f"Assistant (speaking): {text}")
                             print(f"----------------------\n")
                    elif msg_type == "iot":
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
                    # Store each packet separately instead of concatenating bytes
                    tts_packets.append(message)
                else:
                    logger.warning(f"Received unexpected binary data ({len(message)} bytes) when not expecting TTS.")

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
            logger.warning("Connection closed during TTS, processing buffered audio...")
            pcm_data = decode_opus_packets(tts_packets, server_tts_sample_rate, TARGET_CHANNELS)
            if pcm_data is not None and len(pcm_data) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"received_tts_partial_{timestamp}.wav"
                save_pcm_to_wav(pcm_data, server_tts_sample_rate, TARGET_CHANNELS, filename)
            tts_packets = []
            is_receiving_tts = False


async def run_client(url: str, token: str, device_id: str, client_id: str, wav_path: str):
    """Main function to run the WebSocket client simulation."""
    global server_tts_sample_rate

    # 1. Prepare Audio Data
    pcm_data = read_and_resample_wav(wav_path)
    if pcm_data is None:
        logger.error("Cannot proceed without valid audio data.")
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
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "channels": TARGET_CHANNELS,
                    "frame_duration": OPUS_FRAME_MS
                }
            }
            await websocket.send(json.dumps(client_hello))
            logger.info(f"Sent Client Hello: {json.dumps(client_hello)}")

            # Receive Server Hello (with timeout)
            try:
                server_hello_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                if not isinstance(server_hello_raw, str):
                     raise ValueError("Server 'hello' was not text.")
                server_hello = json.loads(server_hello_raw)
                logger.info(f"Received Server Hello: {json.dumps(server_hello)}")

                if server_hello.get("type") != "hello" or server_hello.get("transport") != "websocket":
                    raise ValueError("Invalid server 'hello' message.")

                # IMPORTANT: Get TTS sample rate from server
                svr_sr = server_hello.get("audio_params", {}).get("sample_rate")
                if svr_sr and isinstance(svr_sr, int):
                    server_tts_sample_rate = svr_sr
                    logger.info(f"Server TTS sample rate confirmed: {server_tts_sample_rate}Hz")
                    if server_tts_sample_rate != TARGET_SAMPLE_RATE:
                         logger.warning(f"Server TTS sample rate ({server_tts_sample_rate}Hz) differs from client input rate ({TARGET_SAMPLE_RATE}Hz). Decoder initialized for server rate.")
                else:
                     logger.warning(f"Server did not provide valid TTS sample rate, defaulting to {server_tts_sample_rate}Hz.")


            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Server Hello.")
                return
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to process Server Hello: {e}")
                return

            logger.info("Handshake complete.")

            # --- Start Interaction ---
            # Run receiver in the background
            receiver_task = asyncio.create_task(receive_messages(websocket))

            # Send listen start
            listen_start_msg = {"type": "listen", "state": "start", "mode": "manual"} # Or auto? Manual matches WAV file sending better.
            await websocket.send(json.dumps(listen_start_msg))
            logger.info(f"Sent Listen Start: {json.dumps(listen_start_msg)}")

            # Start sending audio
            sender_task = asyncio.create_task(send_audio_task(websocket, pcm_data))
            await sender_task # Wait for all audio to be sent

            # Send listen stop
            listen_stop_msg = {"type": "listen", "state": "stop"}
            await websocket.send(json.dumps(listen_stop_msg))
            logger.info(f"Sent Listen Stop: {json.dumps(listen_stop_msg)}")

            # Keep connection open for receiving responses
            logger.info("Audio sent. Waiting for server responses...")
            await receiver_task # Wait for receiver task to finish (e.g., on disconnect)

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
    parser.add_argument("wav_file", help="Path to the WAV file to send (will be resampled to 16kHz mono)")
    parser.add_argument("-u", "--url", default="ws://localhost:8765/voice", help="WebSocket server URL")
    parser.add_argument("-t", "--token", required=True, help="Bearer token for Authorization header")
    parser.add_argument("-d", "--device-id", default="SIMULATED_DEVICE_MAC", help="Device-Id header value (e.g., MAC address)")
    parser.add_argument("-c", "--client-id", default=str(uuid.uuid4()), help="Client-Id header value (UUID)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Set websockets library logger level too if desired
        # logging.getLogger("websockets").setLevel(logging.DEBUG)

    try:
        asyncio.run(run_client(args.url, args.token, args.device_id, args.client_id, args.wav_file))
    except KeyboardInterrupt:
        logger.info("Client interrupted by user.")