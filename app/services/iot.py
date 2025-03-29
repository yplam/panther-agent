from app.services.base import BaseIoTService
from app.logger import logger
from typing import List, Dict, Any, Optional
import asyncio

class InMemoryIoTService(BaseIoTService):
    """Simple in-memory storage for IoT device info. Not persistent."""
    def __init__(self):
        self._descriptors: Dict[str, List[Dict[str, Any]]] = {}
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def store_descriptors(self, device_id: str, descriptors: List[Dict[str, Any]]):
        async with self._lock:
            logger.info(f"Storing descriptors for device {device_id}: {descriptors}")
            # Simple append/replace logic, might need merging based on 'update' flag etc.
            if device_id not in self._descriptors:
                self._descriptors[device_id] = []
            # Assuming each message sends one descriptor item as per client code
            self._descriptors[device_id].extend(descriptors)


    async def store_states(self, device_id: str, states: Dict[str, Any]):
         async with self._lock:
            logger.info(f"Storing states for device {device_id}: {states}")
            if device_id not in self._states:
                self._states[device_id] = {}
            self._states[device_id].update(states) # Merge new states

    async def execute_commands(self, device_id: str, commands: List[Dict[str, Any]]) -> bool:
        # In this architecture, commands are sent back to the client via WebSocket.
        # This method might be used if the server *directly* controlled IoT devices.
        logger.warning(f"Simulating IoT command execution for {device_id}: {commands}. Actual execution happens via client.")
        # Here you might interact with an IoT platform API if needed.
        await asyncio.sleep(0.1) # Simulate async work
        return True # Simulate success

    async def get_device_capabilities(self, device_id: str) -> Optional[List[Dict[str, Any]]]:
        async with self._lock:
            logger.debug(f"Retrieving capabilities (descriptors) for device {device_id}")
            return self._descriptors.get(device_id)


iot_service: BaseIoTService = InMemoryIoTService()