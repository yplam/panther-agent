import aiohttp
import io
import wave
import os
from typing import List, Optional, Tuple
import uuid

import numpy as np
import opuslib_next
from openai import AsyncOpenAI

from app.logger import logger
from app.services.base import BaseASRService
from app import config, exceptions
from app.services.utils import openai_client


class OpenAIASRService(BaseASRService):
    def __init__(self, client: AsyncOpenAI):
        if not client:
            raise ValueError("AsyncOpenAI client instance is required.")
        self.client = client
        self.output_dir = config.AUDIO_OUTPUT_DIR if hasattr(config, "AUDIO_OUTPUT_DIR") else "audio_files"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def decode_opus(opus_data: List[bytes]) -> bytes:
        """Decode opus packets to PCM data."""
        logger.debug(f"Decoding {len(opus_data)} opus packets")
        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, mono
        pcm_data = []

        for opus_packet in opus_data:
            if not opus_packet:  # Skip empty packets
                continue
            try:
                # 960 samples = 60ms at 16kHz
                pcm_frame = decoder.decode(opus_packet, 960)
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.error(f"Opus decoding error: {e}", exc_info=True)

        return b''.join(pcm_data)

    def save_audio_to_file(self, pcm_data: bytes, session_id: str) -> str:
        """Save PCM audio to WAV file for debugging/logging."""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(pcm_data)

        logger.info(f"Saved audio to {file_path}")
        return file_path

    async def transcribe(self, audio_data: bytes, content_type: str = "audio/opus") -> str:
        """
        Transcribe audio data.
        If audio_data is a list of opus packets, decode them first.
        If content_type is "audio/opus", treat as a single opus stream.
        """
        if isinstance(audio_data, list):
            # Handle list of opus packets from buffering
            pcm_data = self.decode_opus(audio_data)
            
            # Create WAV from PCM
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm_data)
            
            # Get WAV data
            wav_data = wav_buffer.getvalue()
            
            # Optionally save for debugging
            if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES:
                self.save_audio_to_file(pcm_data, "session")
            
            # Update for OpenAI API
            audio_data = wav_data
            content_type = "audio/wav"
        
        logger.info(f"Sending {len(audio_data)} bytes to ASR ({config.ASR_MODEL})...")
        try:
            # Create file-like object for OpenAI API
            file_ext = "wav" if content_type == "audio/wav" else "opus"
            audio_file = (f"audio.{file_ext}", io.BytesIO(audio_data), content_type)

            response = await self.client.audio.transcriptions.create(
                model=config.ASR_MODEL,
                file=audio_file,
                response_format="text"  # Get plain text directly
            )
            
            # The response object itself is the transcribed text when response_format="text"
            transcription = str(response).strip()
            logger.info(f"ASR Result: '{transcription}'")
            if not transcription:
                logger.warning("ASR returned empty transcription.")
                return ""
            return transcription
        except Exception as e:
            logger.error(f"ASR service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"ASR failed: {e}")


asr_service: BaseASRService = OpenAIASRService(openai_client)