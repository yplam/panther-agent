from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple, List, Dict, Any, Union, Optional
import asyncio

class BaseAuthService(ABC):
    @abstractmethod
    async def verify_token(self, token: str) -> bool:
        """Verifies the provided bearer token."""
        pass

class BaseASRService(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes, content_type: str = "audio/opus") -> str:
        """Transcribes audio data to text."""
        pass

class BaseLLMService(ABC):
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        # Add other relevant parameters like available tools, device state etc.
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]: # (response_text, emotion, iot_commands)
        """Generates text response, emotion, and potentially IoT commands."""
        pass

class BaseTTSService(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        sample_rate: int = 16000,
        output_format: str = "opus"
    ) -> AsyncGenerator[bytes, None]:
        """Synthesizes text to audio stream (Opus format)."""
        pass

class BaseIoTService(ABC):
    # Placeholder - Define methods based on actual IoT interactions needed
    @abstractmethod
    async def store_descriptors(self, device_id: str, descriptors: List[Dict[str, Any]]):
        pass

    @abstractmethod
    async def store_states(self, device_id: str, states: Dict[str, Any]):
        pass

    @abstractmethod
    async def execute_commands(self, device_id: str, commands: List[Dict[str, Any]]) -> bool:
        # This might actually be handled by sending the command back to the client
        pass

    @abstractmethod
    async def get_device_capabilities(self, device_id: str) -> Optional[List[Dict[str, Any]]]:
        pass # Needed for LLM to know what commands are possible