# --- tts.py ---

import os
import io
import wave
import uuid
import tempfile
# import subprocess # Removed ffmpeg dependency
import asyncio
import time
import opuslib_next # Added opuslib_next
import numpy as np # Added for PCM manipulation and resampling
import resampy # Added for resampling
from typing import AsyncGenerator, Optional, List

from openai import AsyncOpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason
from azure.cognitiveservices.speech.audio import AudioOutputStream

from app.logger import logger
from app.services.base import BaseTTSService
from app import config, exceptions
from app.services.utils import openai_client, get_azure_speech_config


class OpenAITTSService(BaseTTSService):
    def __init__(self, client: AsyncOpenAI):
        if not client:
            raise ValueError("AsyncOpenAI client instance is required.")
        self.client = client
        self.output_dir = config.AUDIO_OUTPUT_DIR if hasattr(config, "AUDIO_OUTPUT_DIR") else "audio_files"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # --- Target Opus Configuration ---
        self.target_sample_rate = config.TTS_SAMPLE_RATE if hasattr(config, "TTS_SAMPLE_RATE") else 16000
        self.target_channels = 1 # Opus requires mono
        self.frame_duration_ms = 60 # Recommended for VoIP/Streaming (e.g., 20, 40, 60)
        self.frame_size_samples = int(self.target_sample_rate * self.frame_duration_ms / 1000) # Samples per frame
        self.sample_width = 2 # Bytes per sample (16-bit)

        logger.info(f"OpenAI TTS configured for {self.target_sample_rate}Hz, {self.target_channels}ch Opus output.")
        logger.info(f"Opus frame duration: {self.frame_duration_ms}ms ({self.frame_size_samples} samples)")


    def save_audio_to_file(self, audio_data: bytes, prefix: str = "tts", extension: str = "opus") -> str:
        """Save audio data to file for debugging/logging."""
        file_name = f"{prefix}_{uuid.uuid4()}.{extension}"
        file_path = os.path.join(self.output_dir, file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(audio_data)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
             logger.error(f"Failed to save audio file {file_path}: {e}")

        return file_path

    async def _wav_to_opus_packets(self, wav_data: bytes) -> List[bytes]:
        """Converts WAV data (bytes) to a list of Opus packets, resampling if necessary."""
        opus_packets = []
        try:
            with io.BytesIO(wav_data) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wf:
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    nframes = wf.getnframes()
                    pcm_data_bytes = wf.readframes(nframes)
                    logger.info(f"Read WAV: {nchannels}ch, {sampwidth * 8}bit, {framerate}Hz, {nframes} frames, {len(pcm_data_bytes)} bytes")

                    # --- Validation ---
                    if sampwidth != self.sample_width:
                        logger.error(f"Input WAV is not {self.sample_width * 8}-bit PCM ({sampwidth * 8}-bit detected).")
                        raise exceptions.ServiceError(f"Input audio must be {self.sample_width * 8}-bit PCM.")

                    # --- Preprocessing (Numpy) ---
                    # Convert PCM bytes to numpy array (int16)
                    pcm_data_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)

                    # --- Channel Conversion (if necessary) ---
                    if nchannels != self.target_channels:
                        if nchannels == 2 and self.target_channels == 1:
                             logger.warning(f"Input WAV is stereo, converting to mono.")
                             # Simple averaging for stereo to mono
                             pcm_data_int16 = pcm_data_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
                             logger.info("Converted stereo WAV to mono.")
                        else:
                            logger.error(f"Cannot automatically convert {nchannels} channels to {self.target_channels} channel(s).")
                            raise exceptions.ServiceError(f"Unsupported channel conversion: {nchannels} -> {self.target_channels}")
                    else:
                         logger.debug("Input WAV already has target channel count.")


                    # --- Resampling (if necessary) ---
                    if framerate != self.target_sample_rate:
                        logger.warning(f"Input WAV sample rate ({framerate}Hz) differs from target ({self.target_sample_rate}Hz). Resampling...")
                        # Convert int16 to float32 for resampy
                        pcm_data_float32 = pcm_data_int16.astype(np.float32) / 32768.0 # Normalize
                        # Resample
                        resampled_float32 = resampy.resample(pcm_data_float32, sr_orig=framerate, sr_new=self.target_sample_rate)
                        # Convert back to int16
                        pcm_data_int16 = (resampled_float32 * 32768.0).astype(np.int16)
                        logger.info(f"Resampled audio from {framerate}Hz to {self.target_sample_rate}Hz. New length: {len(pcm_data_int16)} samples.")
                    else:
                         logger.debug("Input WAV already has target sample rate.")


                    # Convert final processed numpy array back to bytes
                    processed_pcm_bytes = pcm_data_int16.tobytes()
                    logger.debug(f"Final PCM data length for encoding: {len(processed_pcm_bytes)} bytes")


            # --- Opus Encoding ---
            encoder = opuslib_next.Encoder(self.target_sample_rate, self.target_channels, opuslib_next.APPLICATION_AUDIO)
            frame_size_bytes = self.frame_size_samples * self.target_channels * self.sample_width # Bytes per frame

            logger.debug(f"Opus encoder initialized. Frame size: {frame_size_bytes} bytes ({self.frame_size_samples} samples).")

            for i in range(0, len(processed_pcm_bytes), frame_size_bytes):
                chunk = processed_pcm_bytes[i:i + frame_size_bytes]

                # Pad the last chunk if necessary
                if len(chunk) < frame_size_bytes:
                    padding_needed = frame_size_bytes - len(chunk)
                    chunk += b'\x00' * padding_needed # Zero padding
                    logger.debug(f"Padded last chunk with {padding_needed} bytes.")

                # Encode
                try:
                    # Ensure chunk is exactly the right size before encoding
                    if len(chunk) != frame_size_bytes:
                         logger.error(f"Chunk size mismatch before encoding: expected {frame_size_bytes}, got {len(chunk)}. Skipping frame.")
                         continue # Or handle differently

                    encoded_packet = encoder.encode(chunk, self.frame_size_samples)
                    opus_packets.append(encoded_packet)
                except opuslib_next.OpusError as e:
                    logger.error(f"Opus encoding error on chunk starting at index {i}: {e}")
                    # Decide handling: break, continue? Breaking is safer for stream integrity.
                    break
                except Exception as enc_e: # Catch other potential errors during encode
                    logger.error(f"Unexpected error during Opus chunk encoding: {enc_e}")
                    break

            logger.info(f"Encoded {len(opus_packets)} Opus packets from processed PCM.")
            return opus_packets

        except wave.Error as e:
            logger.error(f"Error reading WAV data: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Invalid WAV data received: {e}")
        except ImportError:
             logger.error("Missing dependency: 'resampy' or 'numpy' is not installed. Cannot perform resampling.")
             raise exceptions.ServiceError("Audio resampling library not found. Please install 'resampy' and 'numpy'.")
        except opuslib_next.OpusError as e:
            logger.error(f"Opus library error during initialization or encoding: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Opus encoding failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during WAV to Opus conversion: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Audio conversion failed: {e}")


    async def synthesize_non_streaming(
        self,
        text: str,
        sample_rate: int = None, # No longer used directly, uses self.target_sample_rate
        output_format: str = None # No longer used directly, always returns Opus
    ) -> bytes:
        """Non-streaming TTS. Requests WAV, resamples if needed, encodes to Opus using opuslib_next."""
        logger.info(f"Synthesizing TTS (non-streaming, WAV->Opus) for: '{text[:50]}...'")
        # Sample rate/output format args are ignored, using instance config

        try:
            # Request WAV format from OpenAI
            response = await self.client.audio.speech.create(
                model=config.TTS_MODEL,
                voice=config.TTS_VOICE,
                input=text,
                response_format="wav" # Request WAV
            )

            # Read the entire WAV content asynchronously
            wav_data = await response.aread()
            logger.debug(f"Received {len(wav_data)} bytes of non-streaming WAV data")

            # Convert WAV to Opus packets (handles resampling)
            opus_packets = await self._wav_to_opus_packets(wav_data)

            # Combine packets into single byte string
            opus_data_out = b"".join(opus_packets)

            # Save for debugging if configured
            if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES:
                 # Save original WAV for comparison
                self.save_audio_to_file(wav_data, prefix="tts_openai_wav", extension="wav")
                # Save final Opus
                self.save_audio_to_file(opus_data_out, prefix="tts_opus", extension="opus")

            logger.info(f"Non-streaming synthesis complete. Returning {len(opus_data_out)} bytes of Opus data.")
            return opus_data_out

        except Exception as e:
            # Catch exceptions from OpenAI client or our conversion
            logger.error(f"TTS service error (non-streaming): {e}", exc_info=True)
            if isinstance(e, exceptions.ServiceError): # Keep our specific errors
                 raise
            raise exceptions.ServiceError(f"TTS failed (non-streaming): {e}")

    async def synthesize(
        self,
        text: str,
        sample_rate: int = None, # No longer used directly
        output_format: str = None # No longer used directly
    ) -> AsyncGenerator[bytes, None]:
        """Streaming TTS synthesis. Requests WAV, resamples/converts, then streams Opus packets."""
        logger.info(f"Synthesizing TTS (streaming, WAV->Opus) for: '{text[:50]}...' at target {self.target_sample_rate}Hz")
        # Sample rate/output format args are ignored

        try:
            # Request WAV format from OpenAI
            response = await self.client.audio.speech.create(
                model=config.TTS_MODEL,
                voice=config.TTS_VOICE,
                input=text,
                response_format="wav" # Request WAV
            )

            # Read the entire WAV content asynchronously
            # For very long inputs, a chunked read + conversion might be better, but adds complexity.
            wav_data = await response.aread()
            logger.debug(f"Received {len(wav_data)} bytes of WAV data for streaming conversion.")

            # Save original WAV for debugging if configured
            if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES:
                self.save_audio_to_file(wav_data, prefix="tts_stream_openai_wav", extension="wav")

            # Convert WAV to Opus packets (handles resampling/mono)
            opus_packets = await self._wav_to_opus_packets(wav_data)

            # Stream the generated Opus packets with pacing
            packet_count = 0
            total_bytes_yielded = 0
            start_time = time.perf_counter()
            expected_next_packet_time = start_time # Initialize

            for packet in opus_packets:
                # Calculate when this packet *should* be sent
                expected_next_packet_time += (self.frame_duration_ms / 1000.0)
                current_time = time.perf_counter()
                wait_time = expected_next_packet_time - current_time

                # Wait if we are ahead of schedule
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                # else: # Optional: Log if falling behind
                #    logger.warning(f"Stream fell behind schedule by {-wait_time*1000:.2f} ms at packet {packet_count}")

                yield packet # Send the packet

                packet_count += 1
                total_bytes_yielded += len(packet)

                if packet_count % 50 == 0: # Log progress occasionally
                     logger.debug(f"Streamed {packet_count} Opus packets ({total_bytes_yielded} bytes)")

            logger.info(f"TTS synthesis stream completed. Streamed {packet_count} packets ({total_bytes_yielded} bytes)")

        except Exception as e:
            logger.error(f"TTS service error (streaming): {e}", exc_info=True)
            if isinstance(e, exceptions.ServiceError): # Keep our specific errors
                 raise
            raise exceptions.ServiceError(f"TTS streaming failed: {e}")


class AzureTTSService(BaseTTSService):
    def __init__(self, speech_config):
        self.speech_config = speech_config
        self.output_dir = config.AUDIO_OUTPUT_DIR if hasattr(config, "AUDIO_OUTPUT_DIR") else "audio_files"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # --- Target Opus Configuration ---
        self.target_sample_rate = config.TTS_SAMPLE_RATE if hasattr(config, "TTS_SAMPLE_RATE") else 16000
        self.target_channels = 1 # Opus requires mono
        self.frame_duration_ms = 60 # Recommended for VoIP/Streaming (e.g., 20, 40, 60)
        self.frame_size_samples = int(self.target_sample_rate * self.frame_duration_ms / 1000) # Samples per frame
        self.sample_width = 2 # Bytes per sample (16-bit)

        # Configure voices
        self.voice_name = config.AZURE_TTS_VOICE if hasattr(config, "AZURE_TTS_VOICE") else "en-US-JennyNeural"
        self.speech_config.speech_synthesis_voice_name = self.voice_name

        logger.info(f"Azure TTS configured with voice {self.voice_name}")
        logger.info(f"Azure TTS configured for {self.target_sample_rate}Hz, {self.target_channels}ch Opus output.")
        logger.info(f"Opus frame duration: {self.frame_duration_ms}ms ({self.frame_size_samples} samples)")

    def save_audio_to_file(self, audio_data: bytes, prefix: str = "tts", extension: str = "opus") -> str:
        """Save audio data to file for debugging/logging."""
        file_name = f"{prefix}_{uuid.uuid4()}.{extension}"
        file_path = os.path.join(self.output_dir, file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(audio_data)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
             logger.error(f"Failed to save audio file {file_path}: {e}")

        return file_path

    async def _wav_to_opus_packets(self, wav_data: bytes) -> List[bytes]:
        """Converts WAV data (bytes) to a list of Opus packets, resampling if necessary."""
        # This method is the same as in OpenAITTSService
        opus_packets = []
        try:
            with io.BytesIO(wav_data) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wf:
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    nframes = wf.getnframes()
                    pcm_data_bytes = wf.readframes(nframes)
                    logger.info(f"Read WAV: {nchannels}ch, {sampwidth * 8}bit, {framerate}Hz, {nframes} frames, {len(pcm_data_bytes)} bytes")

                    # --- Validation ---
                    if sampwidth != self.sample_width:
                        logger.error(f"Input WAV is not {self.sample_width * 8}-bit PCM ({sampwidth * 8}-bit detected).")
                        raise exceptions.ServiceError(f"Input audio must be {self.sample_width * 8}-bit PCM.")

                    # --- Preprocessing (Numpy) ---
                    # Convert PCM bytes to numpy array (int16)
                    pcm_data_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)

                    # --- Channel Conversion (if necessary) ---
                    if nchannels != self.target_channels:
                        if nchannels == 2 and self.target_channels == 1:
                             logger.warning(f"Input WAV is stereo, converting to mono.")
                             # Simple averaging for stereo to mono
                             pcm_data_int16 = pcm_data_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
                             logger.info("Converted stereo WAV to mono.")
                        else:
                            logger.error(f"Cannot automatically convert {nchannels} channels to {self.target_channels} channel(s).")
                            raise exceptions.ServiceError(f"Unsupported channel conversion: {nchannels} -> {self.target_channels}")
                    else:
                         logger.debug("Input WAV already has target channel count.")

                    # --- Resampling (if necessary) ---
                    if framerate != self.target_sample_rate:
                        logger.warning(f"Input WAV sample rate ({framerate}Hz) differs from target ({self.target_sample_rate}Hz). Resampling...")
                        # Convert int16 to float32 for resampy
                        pcm_data_float32 = pcm_data_int16.astype(np.float32) / 32768.0 # Normalize
                        # Resample
                        resampled_float32 = resampy.resample(pcm_data_float32, sr_orig=framerate, sr_new=self.target_sample_rate)
                        # Convert back to int16
                        pcm_data_int16 = (resampled_float32 * 32768.0).astype(np.int16)
                        logger.info(f"Resampled audio from {framerate}Hz to {self.target_sample_rate}Hz. New length: {len(pcm_data_int16)} samples.")
                    else:
                         logger.debug("Input WAV already has target sample rate.")

                    # Convert final processed numpy array back to bytes
                    processed_pcm_bytes = pcm_data_int16.tobytes()
                    logger.debug(f"Final PCM data length for encoding: {len(processed_pcm_bytes)} bytes")

            # --- Opus Encoding ---
            encoder = opuslib_next.Encoder(self.target_sample_rate, self.target_channels, opuslib_next.APPLICATION_AUDIO)
            frame_size_bytes = self.frame_size_samples * self.target_channels * self.sample_width # Bytes per frame

            logger.debug(f"Opus encoder initialized. Frame size: {frame_size_bytes} bytes ({self.frame_size_samples} samples).")

            for i in range(0, len(processed_pcm_bytes), frame_size_bytes):
                chunk = processed_pcm_bytes[i:i + frame_size_bytes]

                # Pad the last chunk if necessary
                if len(chunk) < frame_size_bytes:
                    padding_needed = frame_size_bytes - len(chunk)
                    chunk += b'\x00' * padding_needed # Zero padding
                    logger.debug(f"Padded last chunk with {padding_needed} bytes.")

                # Encode
                try:
                    # Ensure chunk is exactly the right size before encoding
                    if len(chunk) != frame_size_bytes:
                         logger.error(f"Chunk size mismatch before encoding: expected {frame_size_bytes}, got {len(chunk)}. Skipping frame.")
                         continue # Or handle differently

                    encoded_packet = encoder.encode(chunk, self.frame_size_samples)
                    opus_packets.append(encoded_packet)
                except opuslib_next.OpusError as e:
                    logger.error(f"Opus encoding error on chunk starting at index {i}: {e}")
                    # Decide handling: break, continue? Breaking is safer for stream integrity.
                    break
                except Exception as enc_e: # Catch other potential errors during encode
                    logger.error(f"Unexpected error during Opus chunk encoding: {enc_e}")
                    break

            logger.info(f"Encoded {len(opus_packets)} Opus packets from processed PCM.")
            return opus_packets

        except wave.Error as e:
            logger.error(f"Error reading WAV data: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Invalid WAV data received: {e}")
        except ImportError:
             logger.error("Missing dependency: 'resampy' or 'numpy' is not installed. Cannot perform resampling.")
             raise exceptions.ServiceError("Audio resampling library not found. Please install 'resampy' and 'numpy'.")
        except opuslib_next.OpusError as e:
            logger.error(f"Opus library error during initialization or encoding: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Opus encoding failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during WAV to Opus conversion: {e}", exc_info=True)
            raise exceptions.ServiceError(f"Audio conversion failed: {e}")

    async def synthesize_non_streaming(
        self,
        text: str,
        sample_rate: int = None, # No longer used directly, uses self.target_sample_rate
        output_format: str = None # No longer used directly, always returns Opus
    ) -> bytes:
        """Non-streaming TTS using Azure. Synthesizes to WAV, then converts to Opus."""
        logger.info(f"Synthesizing TTS with Azure (non-streaming) for: '{text[:50]}...'")
        
        try:
            # Create a temporary file to store the synthesized speech
            temp_file_path = os.path.join(self.output_dir, f"temp_tts_{uuid.uuid4()}.wav")
            
            # Create audio output config that writes to a file
            audio_output = AudioConfig(filename=temp_file_path)
            
            # Create the speech synthesizer
            synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_output)
            
            # Synthesize speech and wait for the result
            result = synthesizer.speak_text_async(text).get()
            
            # Check if synthesis was successful
            if result.reason != ResultReason.SynthesizingAudioCompleted:
                logger.error(f"Azure speech synthesis failed with reason: {result.reason}")
                raise exceptions.ServiceError(f"TTS failed: {result.reason}")
            
            # Read the generated WAV file
            with open(temp_file_path, 'rb') as f:
                wav_data = f.read()
            
            # Remove the temporary file
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
            
            logger.debug(f"Received {len(wav_data)} bytes of WAV data from Azure Speech Service")
            
            # Convert WAV to Opus packets
            opus_packets = await self._wav_to_opus_packets(wav_data)
            
            # Combine packets into single byte string
            opus_data_out = b"".join(opus_packets)
            
            # Save for debugging if configured
            if hasattr(config, "SAVE_AUDIO_FILES") and config.SAVE_AUDIO_FILES:
                # Save original WAV for comparison
                self.save_audio_to_file(wav_data, prefix="tts_azure_wav", extension="wav")
                # Save final Opus
                self.save_audio_to_file(opus_data_out, prefix="tts_opus", extension="opus")
            
            logger.info(f"Non-streaming synthesis complete. Returning {len(opus_data_out)} bytes of Opus data.")
            return opus_data_out
            
        except Exception as e:
            logger.error(f"Azure TTS service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"TTS failed: {e}")

    async def synthesize(
        self,
        text: str,
        sample_rate: int = None, # No longer used directly
        output_format: str = None # No longer used directly
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech using Azure Speech Services.
        For Azure implementation, we'll use the non-streaming approach and yield the complete audio.
        """
        logger.info(f"Synthesizing speech with Azure for: '{text[:50]}...'")
        
        try:
            # Get all audio data at once using non-streaming method
            opus_data = await self.synthesize_non_streaming(text)
            
            # Yield the complete opus data in a single chunk
            # This is a simplification - in a real implementation, you might want to
            # split this into smaller chunks
            yield opus_data
            
        except Exception as e:
            logger.error(f"Azure TTS service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"TTS failed: {e}")

# Initialize the appropriate TTS service based on configuration
if hasattr(config, "TTS_SERVICE") and config.TTS_SERVICE == "azure":
    speech_config = get_azure_speech_config()
    tts_service: BaseTTSService = AzureTTSService(speech_config)
else:
    tts_service: BaseTTSService = OpenAITTSService(openai_client)