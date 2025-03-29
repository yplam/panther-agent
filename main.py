import asyncio
import signal
import websockets
from functools import partial

from app import config
from app.logger import logger  # Change from 'from app import logger' to 'from app.logger import logger'
from app.websocket_handler import handle_connection, process_request

async def main():
    """Starts the WebSocket server."""
    stop = asyncio.Future()

    # Add signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    # Use partial to pass process_request to websockets.serve
    bound_process_request = partial(process_request)

    async with websockets.serve(
        handle_connection,
        host="0.0.0.0",
        port=config.WEBSOCKET_PORT,
        process_request=bound_process_request, # Pass the handshake validator
        ping_interval=20, # Keep connection alive
        ping_timeout=20
    ):
        logger.info(f"WebSocket server started on ws://0.0.0.0:{config.WEBSOCKET_PORT}{config.WEBSOCKET_PATH}")
        logger.info("AI Services Configured:")
        logger.info(f"  ASR Model: {config.ASR_MODEL}")
        logger.info(f"  LLM Model: {config.LLM_MODEL}")
        logger.info(f"  TTS Model: {config.TTS_MODEL} (Voice: {config.TTS_VOICE}, Sample Rate: {config.TTS_SAMPLE_RATE}Hz, Format: {config.TTS_OUTPUT_FORMAT})")
        if config.AUTH_SERVICE_URL:
             logger.info(f"  Auth Service: {config.AUTH_SERVICE_URL}")
        else:
             logger.warning("  Auth Service: Using In-Memory Tokens (INSECURE!)")

        await stop # Wait for shutdown signal

    logger.info("WebSocket server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested.")