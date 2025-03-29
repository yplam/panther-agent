from openai import AsyncOpenAI
import os
import io
import wave
import uuid

from app.logger import logger
from app.services.base import BaseTTSService
from app import config, exceptions
from typing import AsyncGenerator, Optional
import asyncio

from app.services.utils import openai_client


class OpenAITTSService(BaseTTSService):
    def __init__(self, client: AsyncOpenAI):
        if not client:
            raise ValueError("AsyncOpenAI client instance is required.")
        self.client = client
        self.output_dir = config.AUDIO_OUTPUT_DIR if hasattr(config, "AUDIO_OUTPUT_DIR") else "audio_files"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # OpenAI TTS doesn't inherently support sentence splitting for streaming hints
        # If needed, implement manual sentence splitting before calling TTS multiple times
        # or use a different TTS provider with better streaming support.

    def save_audio_to_file(self, audio_data: bytes, prefix: str = "tts") -> str:
        """Save audio data to file for debugging/logging."""
        file_name = f"{prefix}_{uuid.uuid4()}.opus"
        file_path = os.path.join(self.output_dir, file_name)
        
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Saved audio to {file_path}")
        return file_path

    async def synthesize_non_streaming(
        self,
        text: str,
        sample_rate: int = config.TTS_SAMPLE_RATE,
        output_format: str = config.TTS_OUTPUT_FORMAT
    ) -> bytes:
        """Non-streaming version to validate Opus format."""
        logger.info(f"Synthesizing TTS (non-streaming) for: '{text[:50]}...'")
        if output_format.lower() != "opus":
             logger.warning(f"Requested format '{output_format}' may not be optimal. Forcing Opus.")
             output_format = "opus"

        try:
            # Get binary response content
            response = await self.client.audio.speech.create(
                model=config.TTS_MODEL,
                voice=config.TTS_VOICE,
                input=text,
                response_format=output_format
            )
            
            # Read the content synchronously
            audio_data = response.read()
            logger.debug(f"Received {len(audio_data)} bytes of non-streaming audio data")
            
            # Save for debugging if configured
            if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES:
                self.save_audio_to_file(audio_data)
            
            return audio_data

        except Exception as e:
            logger.error(f"TTS service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"TTS failed: {e}")

    async def synthesize(
        self,
        text: str,
        sample_rate: int = config.TTS_SAMPLE_RATE,
        output_format: str = config.TTS_OUTPUT_FORMAT
    ) -> AsyncGenerator[bytes, None]:
        """Original streaming version."""
        logger.info(f"Synthesizing TTS for: '{text[:50]}...'")
        if output_format.lower() != "opus":
             logger.warning(f"Requested format '{output_format}' may not be optimal. Forcing Opus.")
             output_format = "opus" # Ensure Opus for the client

        try:
            # Note: OpenAI TTS API currently doesn't support specifying sample rate directly for Opus.
            # It defaults to 24kHz. The ESP32 client *should* handle this via its decoder,
            # but we *declared* 16kHz in server_hello. This is a mismatch!
            # Option 1: Tell client TTS is 24kHz (Update create_server_hello if client supports it)
            # Option 2: Use a different TTS service that allows 16kHz Opus.
            # Option 3: Resample 24kHz Opus to 16kHz on the server (adds complexity/latency - requires opuslib/ffmpeg).
            # Choosing Option 1 for simplicity, assuming client can handle 24kHz Opus if told.
            # --- IF CLIENT CAN ONLY HANDLE 16kHz ---
            # logger.warning("OpenAI TTS outputs 24kHz Opus. Resampling to 16kHz is NOT implemented.")
            # raise exceptions.ServiceError("Server TTS format mismatch. Cannot generate 16kHz Opus directly.")
            # --- IF CLIENT CAN HANDLE 24kHz ---
            logger.warning("OpenAI TTS outputs 24kHz Opus. Sending as is.")
            # Adjust server_hello if necessary!

            async with self.client.audio.speech.with_streaming_response.create(
                model=config.TTS_MODEL,
                voice=config.TTS_VOICE,
                input=text,
                response_format=output_format, # Use 'opus'
                # speed=1.0 # Optional parameter
            ) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(f"TTS API Error ({response.status_code}): {error_content.decode()}")
                    raise exceptions.ServiceError(f"TTS API failed with status {response.status_code}")

                # For debugging, accumulate chunks if configured
                debug_buffer = bytearray() if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES else None

                async for chunk in response.iter_bytes():
                    if chunk:
                        if debug_buffer is not None:
                            debug_buffer.extend(chunk)
                        yield chunk
                    # Add a small sleep to prevent hogging the event loop if chunks are very small/fast
                    await asyncio.sleep(0.001)

                # Save accumulated audio if debugging
                if debug_buffer:
                    self.save_audio_to_file(bytes(debug_buffer))

            logger.info("TTS synthesis stream finished.")

        except Exception as e:
            logger.error(f"TTS service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"TTS failed: {e}")

tts_service: BaseTTSService = OpenAITTSService(openai_client)