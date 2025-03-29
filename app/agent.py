import asyncio
import math
from typing import TypedDict, List, Dict, Any, Optional, Annotated

import numpy as np
import opuslib_next
import resampy
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For potential state saving
from opuslib_next import OpusError

from app import config, protocol, exceptions
from app.logger import logger
from app.services.asr import asr_service
from app.services.llm import llm_service
from app.services.tts import tts_service
from app.services.iot import iot_service # Using the in-memory version
from app.state_manager import ConnectedClient, ClientState
import websockets

# Define the state structure for the graph
class AgentState(TypedDict):
    client: ConnectedClient               # Reference to the client object
    input_audio: List[bytes]              # List of raw audio bytes (Opus packets)
    asr_text: Optional[str]               # Text from ASR
    llm_response_text: Optional[str]      # Text from LLM
    llm_emotion: Optional[str]            # Emotion from LLM
    iot_commands_to_send: Optional[List[Dict[str, Any]]] # Commands parsed by LLM for client
    # tts_audio_stream: AsyncGenerator[bytes, None] # TTS handled directly in node
    error_message: Optional[str]          # Error details
    # Add conversation history management if needed
    conversation_history: List[Dict[str, str]]


# --- Graph Nodes ---

async def run_asr(state: AgentState) -> AgentState:
    """Runs the ASR service on buffered input audio packets."""
    client = state["client"]
    logger.info(f"{client.client_id}: Running ASR for session {client.session_id}")
    try:
        audio_data = state["input_audio"]
        if not audio_data or len(audio_data) < 10:  # Ensure we have enough audio data
            raise ValueError("Insufficient audio data for ASR.")
            
        # Pass the list of opus packets directly to the updated ASR service
        transcription = await asr_service.transcribe(audio_data)
        state["asr_text"] = transcription
        state["error_message"] = None

        # Send STT result back to client
        stt_msg = protocol.create_stt_message(transcription if transcription else "<no speech detected>")
        logger.debug(f"{client.client_id}: Sending STT: {stt_msg}")
        await client.websocket.send(stt_msg)

        # Add user input to history
        if transcription:
            state["conversation_history"].append({"role": "user", "content": transcription})

    except exceptions.ServiceError as e:
        logger.error(f"{client.client_id}: ASR Service Error: {e}")
        state["error_message"] = f"Sorry, I couldn't understand that. ({e})"
        state["asr_text"] = None # Ensure no text proceeds
    except Exception as e:
        logger.error(f"{client.client_id}: Unexpected ASR Error: {e}", exc_info=True)
        state["error_message"] = "An unexpected error occurred during speech recognition."
        state["asr_text"] = None # Ensure no text proceeds
    return state

async def run_llm(state: AgentState) -> AgentState:
    """Runs the LLM service to generate response, emotion, and commands."""
    client = state["client"]
    logger.info(f"{client.client_id}: Running LLM for session {client.session_id}")
    try:
        asr_text = state.get("asr_text")
        if not asr_text:
            logger.warning(f"{client.client_id}: No ASR text available, skipping LLM.")
            # Maybe generate a default response?
            state["llm_response_text"] = "I didn't catch that, could you please repeat?"
            state["llm_emotion"] = "neutral"
            state["iot_commands_to_send"] = None
            state["error_message"] = None # Clear previous error if ASR failed silently
            return state

        # device_capabilities = await iot_service.get_device_capabilities(client.device_id)
        response_text, emotion, iot_commands = await llm_service.generate_response(
            prompt=asr_text,
            history=state["conversation_history"], # Pass history
            device_id=client.device_id,
            # device_capabilities=device_capabilities
        )
        state["llm_response_text"] = response_text
        state["llm_emotion"] = emotion
        state["iot_commands_to_send"] = iot_commands # Store commands needed by client
        state["error_message"] = None

        # Send LLM emotion
        if emotion:
            llm_msg = protocol.create_llm_emotion_message(emotion)
            logger.debug(f"{client.client_id}: Sending LLM emotion: {llm_msg}")
            await client.websocket.send(llm_msg)

        # Add assistant response to history (without commands/emotion part)
        if response_text:
             state["conversation_history"].append({"role": "assistant", "content": response_text})


    except exceptions.ServiceError as e:
        logger.error(f"{client.client_id}: LLM Service Error: {e}")
        state["error_message"] = f"Sorry, I'm having trouble thinking right now. ({e})"
    except Exception as e:
        logger.error(f"{client.client_id}: Unexpected LLM Error: {e}", exc_info=True)
        state["error_message"] = "An unexpected error occurred while processing your request."

    # Ensure LLM fields are cleared on error
    if state.get("error_message"):
        state["llm_response_text"] = None
        state["llm_emotion"] = None
        state["iot_commands_to_send"] = None

    return state

async def send_iot_commands(state: AgentState) -> AgentState:
    """Sends IoT commands back to the client via WebSocket."""
    client = state["client"]
    commands = state.get("iot_commands_to_send")
    if commands:
        logger.info(f"{client.client_id}: Sending IoT commands for session {client.session_id}: {commands}")
        try:
            iot_msg = protocol.create_iot_command_message(commands)
            logger.debug(f"{client.client_id}: Sending IoT: {iot_msg}")
            await client.websocket.send(iot_msg)
            # Optionally wait for client ACK or handle potential errors?
            # For now, assume fire-and-forget
            state["error_message"] = None # Clear error if successful
        except Exception as e:
            logger.error(f"{client.client_id}: Failed to send IoT commands: {e}", exc_info=True)
            state["error_message"] = "Failed to send device commands."
            # Decide if we should still proceed with TTS
    else:
         logger.debug(f"{client.client_id}: No IoT commands to send.")

    return state

async def _perform_tts_streaming(client: ConnectedClient, text_to_speak: str):
    """
    Helper coroutine to handle TTS streaming WITH resampling using opuslib_next.
    Receives 24kHz Opus -> Decodes -> Resamples to 16kHz -> Encodes to 16kHz Opus -> Streams.
    """
    opus_decoder = None
    opus_encoder = None
    try:
        # --- First validate Opus format with non-streaming call ---
        logger.info(f"{client.client_id}: Validating TTS Opus format with non-streaming call")
        try:
            opus_data = await tts_service.synthesize_non_streaming(
                text="Test audio validation.",
                sample_rate=config.TTS_SOURCE_SAMPLE_RATE
            )
            # Validation successful, opus_data will be bytes
            logger.info(f"{client.client_id}: Successfully validated Opus format")
        except Exception as e:
            logger.error(f"{client.client_id}: Failed to validate Opus format: {e}")
            raise exceptions.ServiceError("Failed to validate TTS Opus format")

        # --- Initialization ---
        source_sr = config.TTS_SOURCE_SAMPLE_RATE # e.g., 24000 from OpenAI
        target_sr = config.TTS_SAMPLE_RATE       # e.g., 16000 for client
        channels = 1
        frame_duration_ms = config.OPUS_FRAME_MS
        # Frame size in samples at SOURCE rate (for decoder)
        frame_size_source = math.ceil(source_sr * frame_duration_ms / 1000)
        # Frame size in samples at TARGET rate (for encoder)
        frame_size_target = math.ceil(target_sr * frame_duration_ms / 1000)

        logger.info(f"{client.client_id}: Initializing Resampling Pipeline (opuslib_next): {source_sr}Hz -> {target_sr}Hz")
        logger.info(f"{client.client_id}: Source Frame Size (samples): {frame_size_source}, Target Frame Size (samples): {frame_size_target}")

        # Decoder for incoming 24kHz Opus stream
        opus_decoder = opuslib_next.Decoder(source_sr, channels)
        # Encoder for outgoing 16kHz Opus stream
        opus_encoder = opuslib_next.Encoder(target_sr, channels, opuslib_next.APPLICATION_AUDIO)
        if config.OPUS_BITRATE != "auto":
            try:
                bitrate = int(config.OPUS_BITRATE)
                opus_encoder.bitrate = bitrate
            except (ValueError, TypeError) as e:
                logger.warning(f"{client.client_id}: Invalid OPUS_BITRATE value '{config.OPUS_BITRATE}', using default.")
        opus_encoder.signal = opuslib_next.SIGNAL_VOICE # Optimize for voice

        # Buffer for *resampled* 16k PCM data before encoding
        pcm_16k_buffer = np.array([], dtype=np.int16)

        # --- Start Streaming ---
        tts_start_msg = protocol.create_tts_message(protocol.TTS_STATE_START)
        logger.debug(f"{client.client_id}: Sending TTS Start: {tts_start_msg}")
        await client.websocket.send(tts_start_msg)

        chunk_count_in = 0
        chunk_count_out = 0
        stream_finished = False
        opus_buffer = bytearray()  # Buffer to accumulate Opus data
        min_opus_frame_size = 10  # Minimum size for a valid Opus frame
        max_opus_frame_size = 1275  # Maximum size of an Opus frame (per spec)
        accumulated_chunks = 0  # Count of accumulated chunks before processing

        # Get the async generator from the TTS service
        tts_generator = tts_service.synthesize(text_to_speak, sample_rate=source_sr) # Request source rate

        while not stream_finished:
            try:
                # Check for cancellation *before* potentially blocking on receiving data
                if client.tts_task is None or client.tts_task.cancelled():
                    raise asyncio.CancelledError()

                opus_chunk = await asyncio.wait_for(tts_generator.__anext__(), timeout=10.0)
                if opus_chunk:
                    opus_buffer.extend(opus_chunk)
                    chunk_count_in += 1
                    accumulated_chunks += 1
                    logger.debug(f"{client.client_id}: Received chunk {chunk_count_in}: {len(opus_chunk)} bytes, buffer size: {len(opus_buffer)}")

                # Try to process buffer when we have accumulated enough data or stream is finished
                if len(opus_buffer) >= frame_size_source * 2 or (stream_finished and opus_buffer):
                    logger.debug(f"{client.client_id}: Processing buffer of size {len(opus_buffer)} after {accumulated_chunks} chunks")
                    accumulated_chunks = 0  # Reset counter

                    while len(opus_buffer) >= min_opus_frame_size:
                        try:
                            # Try to decode the current buffer
                            decoded_pcm_24k_bytes = opus_decoder.decode(bytes(opus_buffer), frame_size_source)
                            decoded_pcm_24k = np.frombuffer(decoded_pcm_24k_bytes, dtype=np.int16)
                            
                            # If successful, clear the processed data
                            opus_buffer.clear()
                            
                            # Resample 24kHz to 16kHz
                            resampled_pcm_16k = resampy.resample(
                                decoded_pcm_24k,
                                sr_orig=source_sr,
                                sr_new=target_sr,
                                filter='kaiser_fast'
                            ).astype(np.int16)
                            
                            # Add to the 16k buffer
                            pcm_16k_buffer = np.concatenate((pcm_16k_buffer, resampled_pcm_16k))
                            
                        except OpusError as opus_err:
                            if "buffer too small" in str(opus_err).lower():
                                # Need more data, break inner loop and wait for more chunks
                                break
                            elif len(opus_buffer) > max_opus_frame_size:
                                # Buffer is too large, might be corrupted - remove some data
                                logger.warning(f"{client.client_id}: Buffer exceeded max size, discarding data")
                                opus_buffer = opus_buffer[max_opus_frame_size:]
                            else:
                                # Try removing one byte to resync
                                opus_buffer = opus_buffer[1:]
                            continue
                        
                        # Process accumulated 16kHz PCM data
                        while len(pcm_16k_buffer) >= frame_size_target:
                            if client.tts_task is None or client.tts_task.cancelled():
                                raise asyncio.CancelledError()

                            # Get a frame's worth of PCM data
                            pcm_frame = pcm_16k_buffer[:frame_size_target]
                            pcm_16k_buffer = pcm_16k_buffer[frame_size_target:]

                            try:
                                # Encode to 16kHz Opus
                                encoded_opus_16k = opus_encoder.encode(pcm_frame.tobytes(), frame_size_target)
                                await client.websocket.send(encoded_opus_16k)
                                chunk_count_out += 1
                                await asyncio.sleep(0.005)
                            except Exception as e:
                                logger.error(f"{client.client_id}: Error encoding/sending 16k Opus: {e}")
                                if isinstance(e, (websockets.exceptions.ConnectionClosed, asyncio.CancelledError)):
                                    raise
                                break

            except StopAsyncIteration:
                logger.info(f"{client.client_id}: TTS source stream finished.")
                stream_finished = True
            except asyncio.TimeoutError:
                logger.warning(f"{client.client_id}: Timeout waiting for next TTS chunk.")
                stream_finished = True
            except asyncio.CancelledError:
                logger.info(f"{client.client_id}: TTS streaming cancelled.")
                raise
            except Exception as e:
                logger.error(f"{client.client_id}: Error in TTS streaming loop: {e}", exc_info=True)
                stream_finished = True

        # --- Stream Finished - Handle Remaining PCM Buffer ---
        # NOTE: Incoming Opus buffer is no longer used/needed with per-chunk decoding
        logger.info(f"{client.client_id}: Processing remaining {len(pcm_16k_buffer)} samples in PCM buffer.")

        # Encode any remaining PCM data (padding if necessary)
        if len(pcm_16k_buffer) > 0:
            remaining_samples = len(pcm_16k_buffer)
            padding_needed = frame_size_target - (remaining_samples % frame_size_target)
            # Only pad if it's not already a multiple of the frame size
            if padding_needed > 0 and padding_needed != frame_size_target :
                 logger.debug(f"{client.client_id}: Padding final {remaining_samples} samples with {padding_needed} zeros.")
                 padding = np.zeros(padding_needed, dtype=np.int16)
                 pcm_16k_buffer = np.concatenate((pcm_16k_buffer, padding))
            elif padding_needed == frame_size_target: # Means it's already a multiple
                 pass # No padding needed
            else: # remaining_samples % frame_size_target == 0
                 pass # No padding needed

            logger.debug(f"{client.client_id}: Encoding final {len(pcm_16k_buffer)} PCM samples (incl. padding).")
            while len(pcm_16k_buffer) >= frame_size_target:
                if client.tts_task is None or client.tts_task.cancelled(): raise asyncio.CancelledError()
                pcm_frame = pcm_16k_buffer[:frame_size_target]
                pcm_16k_buffer = pcm_16k_buffer[frame_size_target:]
                try:
                    # Encode requires the number of samples per channel
                    encoded_opus_16k = opus_encoder.encode(pcm_frame.tobytes(), frame_size_target)
                    await client.websocket.send(encoded_opus_16k)
                    chunk_count_out += 1
                    await asyncio.sleep(0.005)
                except OpusError as enc_err:
                     logger.error(f"{client.client_id}: Opus encoding error on final buffer: {enc_err}", exc_info=True)
                     # Break loop on final encoding error?
                     break
                except Exception as e:
                     logger.error(f"{client.client_id}: Error sending final encoded chunk: {e}")
                     # Break loop if sending fails
                     if isinstance(e, (websockets.exceptions.ConnectionClosed, asyncio.CancelledError)): raise
                     break # Stop trying to send on other errors

        logger.info(f"{client.client_id}: Finished streaming. In chunks: {chunk_count_in}, Out chunks (16k Opus): {chunk_count_out}")

        # --- Send TTS Stop ---
        # Ensure stop is sent even if cancelled, unless connection is closed
        if not client.websocket.closed:
             tts_stop_msg = protocol.create_tts_message(protocol.TTS_STATE_STOP)
             logger.debug(f"{client.client_id}: Sending TTS Stop.")
             await client.websocket.send(tts_stop_msg)

        # --- Update Client State ---
        await client.stop_speaking(aborted=False)

    except websockets.exceptions.ConnectionClosed:
        logger.warning(f"{client.client_id}: Connection closed during TTS resampling pipeline.")
        await client.stop_speaking(aborted=True)
    except asyncio.CancelledError:
         logger.info(f"{client.client_id}: TTS resampling pipeline cancelled.")
         # Try to send stop if possible, then update state
         try:
             if not client.websocket.closed:
                 await client.websocket.send(protocol.create_tts_message(protocol.TTS_STATE_STOP))
         except Exception: pass # Ignore errors during cancellation cleanup
         await client.stop_speaking(aborted=True)
         # Do not re-raise CancelledError here if it's handled gracefully
         # Re-raise if the caller needs to know it was cancelled. Let's re-raise for now.
         raise
    except Exception as e:
         logger.error(f"{client.client_id}: Error within TTS resampling pipeline: {e}", exc_info=True)
         # Try to send stop message if possible
         try:
            if not client.websocket.closed:
                 await client.websocket.send(protocol.create_tts_message(protocol.TTS_STATE_STOP))
         except Exception: pass
         await client.stop_speaking(aborted=True) # Mark as aborted due to error
         # Re-raise exception for the main graph node or caller
         raise

    finally:
        # opuslib_next's Encoder/Decoder don't have explicit close methods,
        # rely on garbage collection.
        opus_decoder = None
        opus_encoder = None
        logger.debug(f"{client.client_id}: Exiting TTS streaming coroutine.")


# Main TTS streaming node
async def run_tts_and_stream(state: AgentState) -> AgentState:
    """Runs TTS and streams the audio back to the client (using resampling pipeline)."""
    client = state["client"]
    response_text = state.get("llm_response_text")
    error_message = state.get("error_message") # Get error from previous steps

    text_to_speak = error_message if error_message else response_text

    if not text_to_speak:
        logger.warning(f"{client.client_id}: No text available for TTS (LLM response or error), session {client.session_id}. Ending turn.")
        if client.state == ClientState.PROCESSING:
             client.change_state(ClientState.IDLE)
        return state # End the graph gracefully

    logger.info(f"{client.client_id}: Starting TTS (with resampling) for session {client.session_id}")

    # Create a task for the TTS streaming process
    tts_stream_task = asyncio.create_task(
        _perform_tts_streaming(client, text_to_speak), # Calls the modified function
        name=f"TTS_{client.client_id}_{client.session_id}"
    )
    client.start_speaking(tts_stream_task) # Store task reference

    try:
        await tts_stream_task # Await completion/cancellation/error
        logger.info(f"{client.client_id}: TTS resampling task completed for session {client.session_id}")
        state["error_message"] = None

    except asyncio.CancelledError:
         logger.warning(f"{client.client_id}: TTS resampling task was cancelled for session {client.session_id}")
         state["error_message"] = "TTS playback was interrupted."
    except exceptions.ServiceError as e: # Catch errors from the TTS service itself (e.g., API key)
        logger.error(f"{client.client_id}: TTS Service Error during setup/streaming: {e}")
        state["error_message"] = f"Sorry, I couldn't generate the audio response. ({e})"
        # State should have been handled within _perform_tts_streaming's error block
    except Exception as e: # Catch errors from resampling/encoding/sending
        logger.error(f"{client.client_id}: Unexpected TTS pipeline Error: {e}", exc_info=True)
        state["error_message"] = "An unexpected error occurred during audio playback processing."
        # State should have been handled within _perform_tts_streaming's error block

    # State transition IDLE handled within _perform_tts_streaming or stop_speaking
    return state

# --- Conditional Edges ---

def should_run_iot(state: AgentState) -> str:
    """Determines if IoT commands need to be sent."""
    client = state["client"]
    if state.get("error_message"):
        logger.warning(f"{client.client_id}: Error detected before IoT check, proceeding to TTS/End.")
        return "run_tts" # Skip IoT on error, maybe speak the error

    if state.get("iot_commands_to_send"):
        logger.info(f"{client.client_id}: Routing to send_iot_commands.")
        return "send_iot"
    else:
        logger.info(f"{client.client_id}: No IoT commands, routing directly to run_tts.")
        return "run_tts"

def check_asr_result(state: AgentState) -> str:
    """Routes based on ASR success or failure."""
    client = state["client"]
    if state.get("error_message") or not state.get("asr_text"):
         logger.warning(f"{client.client_id}: ASR failed or returned empty. Routing to TTS to speak error/default.")
         return "run_tts" # Go directly to TTS to speak the error or a default message
    else:
         logger.info(f"{client.client_id}: ASR successful. Routing to LLM.")
         return "run_llm"


# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("run_asr", run_asr)
workflow.add_node("run_llm", run_llm)
workflow.add_node("send_iot", send_iot_commands)
workflow.add_node("run_tts", run_tts_and_stream) # Combined TTS and streaming

# Set entry point
workflow.set_entry_point("run_asr")

# Add edges
workflow.add_conditional_edges(
    "run_asr",
    check_asr_result,
    {
        "run_llm": "run_llm",
        "run_tts": "run_tts" # If ASR fails, speak error/default
    }
)

workflow.add_conditional_edges(
    "run_llm",
    should_run_iot,
    {
        "send_iot": "send_iot",
        "run_tts": "run_tts"
    }
)
workflow.add_edge("send_iot", "run_tts") # After sending IoT, proceed to TTS
workflow.add_edge("run_tts", END) # End after TTS streaming finishes or fails

# Compile the graph
app = workflow.compile() # checkpointer=memory