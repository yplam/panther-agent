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

# Function to create listen messages (used in websocket_handler)
def create_listen_message(state: str, mode: str = "manual") -> str:
    """
    Create a JSON listen message to send to the client.
    
    Args:
        state: One of "start", "stop", or "detect"
        mode: One of "manual", "auto", or "realtime" (for "start" state only)
    
    Returns:
        JSON string ready to send to the client
    """
    return protocol.create_listen_message(state, mode)

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
        # --- Initialization ---
        source_sr = config.TTS_SOURCE_SAMPLE_RATE # e.g., 24000 from OpenAI
        target_sr = config.TTS_SAMPLE_RATE       # e.g., 16000 for client
        channels = 1
        frame_duration_ms = config.OPUS_FRAME_MS
        # Frame size in samples at SOURCE rate (for decoder)
        frame_size_source = math.ceil(source_sr * frame_duration_ms / 1000)
        # Frame size in samples at TARGET rate (for encoder)
        frame_size_target = math.ceil(target_sr * frame_duration_ms / 1000)

        logger.info(f"{client.client_id}: Initializing Resampling Pipeline: {source_sr}Hz -> {target_sr}Hz")
        logger.info(f"{client.client_id}: Source Frame Size: {frame_size_source}, Target Frame Size: {frame_size_target}")

        # Decoder for incoming 24kHz Opus stream
        opus_decoder = opuslib_next.Decoder(source_sr, channels)
        # Encoder for outgoing 16kHz Opus stream
        opus_encoder = opuslib_next.Encoder(target_sr, channels, opuslib_next.APPLICATION_AUDIO)
        if config.OPUS_BITRATE != "auto":
            try:
                bitrate = int(config.OPUS_BITRATE)
                opus_encoder.bitrate = bitrate
                logger.info(f"{client.client_id}: Set Opus encoder bitrate to {bitrate} bps")
            except ValueError:
                logger.warning(f"{client.client_id}: Invalid OPUS_BITRATE: {config.OPUS_BITRATE}, using auto")

        # --- Stream TTS and process frames ---
        try:
            # Announce TTS start
            tts_msg = protocol.create_tts_message(protocol.TTS_STATE_START)
            await client.websocket.send(tts_msg)
            
            # Send sentence start with text
            sentence_msg = protocol.create_tts_message(protocol.TTS_STATE_SENTENCE_START, text_to_speak)
            await client.websocket.send(sentence_msg)
            
            # Get streaming TTS
            async for chunk in tts_service.synthesize(text_to_speak, config.TTS_SOURCE_SAMPLE_RATE):
                if not client.websocket.open or client.client_abort:
                    logger.warning(f"{client.client_id}: Streaming interrupted - connection closed or aborted")
                    break
                
                try:
                    # Decode opus packet (TTS service output) at source rate
                    pcm_samples = opus_decoder.decode(chunk, frame_size_source)
                    
                    # Resample from source to target rate
                    resampled = resampy.resample(
                        np.frombuffer(pcm_samples, dtype=np.int16),
                        source_sr, target_sr
                    )
                    
                    # Convert back to bytes for encoding
                    resampled_bytes = resampled.astype(np.int16).tobytes()
                    
                    # Encode to target rate Opus
                    target_opus = opus_encoder.encode(resampled_bytes, frame_size_target)
                    
                    # Send to client
                    await client.websocket.send(target_opus)
                    
                except OpusError as e:
                    logger.error(f"{client.client_id}: Opus processing error: {e}")
                    continue
                except asyncio.CancelledError:
                    logger.info(f"{client.client_id}: TTS streaming cancelled")
                    break
                
                # Check for cancelation between packets
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            logger.info(f"{client.client_id}: TTS streaming cancelled during synthesis")
            raise
        except Exception as e:
            logger.error(f"{client.client_id}: Error during TTS streaming: {e}", exc_info=True)
            raise
        
        # Always send TTS stop message unless websocket is closed
        if client.websocket.open:
            try:
                tts_stop_msg = protocol.create_tts_message(protocol.TTS_STATE_STOP)
                await client.websocket.send(tts_stop_msg)
            except Exception as e:
                logger.error(f"{client.client_id}: Failed to send TTS stop message: {e}")
    
    except asyncio.CancelledError:
        logger.info(f"{client.client_id}: TTS Task cancelled")
        raise
    except Exception as e:
        logger.error(f"{client.client_id}: TTS failed: {e}", exc_info=True)
        if client.websocket.open:
            try:
                # Attempt to send TTS stop on error
                stop_msg = protocol.create_tts_message(protocol.TTS_STATE_STOP)
                await client.websocket.send(stop_msg)
            except:
                pass
    finally:
        # Clean up resources
        if opus_decoder:
            try:
                opus_decoder.destroy()
            except:
                pass
        if opus_encoder:
            try:
                opus_encoder.destroy()
            except:
                pass


# Main TTS streaming node
async def run_tts_and_stream(state: AgentState) -> AgentState:
    """Runs the TTS service to synthesize speech and streams it to the client."""
    client = state["client"]
    logger.info(f"{client.client_id}: Running TTS for session {client.session_id}")
    
    try:
        error_message = state.get("error_message")
        llm_response = state.get("llm_response_text")
        
        # Decide what to speak (error message or LLM response)
        text_to_speak = error_message if error_message else llm_response
        
        if not text_to_speak:
            logger.warning(f"{client.client_id}: No text to speak, skipping TTS.")
            client.change_state(ClientState.IDLE)
            return state
            
        # Create and run the TTS streaming task
        tts_task = asyncio.create_task(
            _perform_tts_streaming(client, text_to_speak),
            name=f"TTS_{client.client_id}_{client.session_id}"
        )
        
        # Register the task with the client for potential cancellation
        client.start_speaking(tts_task)
        
        # Wait for task to complete
        try:
            await tts_task
            # If we get here, the task completed successfully
            logger.info(f"{client.client_id}: TTS task completed successfully")
            if client.state == ClientState.SPEAKING:
                client.change_state(ClientState.IDLE)
        except asyncio.CancelledError:
            logger.info(f"{client.client_id}: TTS task was cancelled externally")
            # TTS task cancellation is handled by stop_speaking method
        except Exception as e:
            logger.error(f"{client.client_id}: Error in TTS task: {e}", exc_info=True)
            if client.state == ClientState.SPEAKING:
                client.change_state(ClientState.IDLE)
    
    except Exception as e:
        logger.error(f"{client.client_id}: Error in run_tts_and_stream: {e}", exc_info=True)
        if client.state != ClientState.IDLE:
            client.change_state(ClientState.IDLE)
    
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