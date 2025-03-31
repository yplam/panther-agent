#!/usr/bin/env python

import asyncio
import logging
import ssl
from typing import Optional, Dict, Any
import certifi
import opuslib_next

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
from websockets.legacy.client import WebSocketClientProtocol

# --- Configuration ---
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8765
LISTEN_PATH = "/xiaozhi/v1/"
TARGET_URI = "wss://api.tenclass.net/xiaozhi/v1/"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("websocket_proxy")

# --- Constants ---
EXPECTED_HEADERS = [
    'Authorization',
    'Protocol-Version',
    'Device-Id',
    'Client-Id'
]

# --- SSL Context using certifi ---
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_context.load_verify_locations(cafile=certifi.where()) # <-- Use certifi's CA bundle
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

async def forward_messages(source_ws: Any, target_ws: Any, direction: str):
    """
    Reads messages from source_ws and forwards them to target_ws.
    Handles both text and binary messages.
    Closes target_ws if source_ws closes or errors occur.
    """
    source_addr = source_ws.remote_address if hasattr(source_ws, 'remote_address') else 'N/A'
    target_desc = getattr(target_ws, 'host', 'Target') + ':' + str(getattr(target_ws, 'port', 'N/A'))
    decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
    try:
        async for message in source_ws:
            try:
                if isinstance(message, str):
                    logger.debug(f"[{direction}] Forwarding TEXT from {source_addr} to {target_desc}: {message[:100]}{'...' if len(message)>100 else ''}")
                    await target_ws.send(message)
                elif isinstance(message, bytes):
                    logger.debug(f"[{direction}] Forwarding BINARY from {source_addr} to {target_desc}: {len(message)} bytes")
                    try:
                        pcm_frame = decoder.decode(message, 960)  # 960 samples = 60ms
                        logger.debug(f"[{direction}] Decoded message: {len(pcm_frame)} bytes")
                    except Exception as e:
                        logger.error(f"[{direction}] Error decoding message: {e}")
                        pass
                    await target_ws.send(message)
                else:
                    logger.warning(f"[{direction}] Unknown message type from {source_addr}: {type(message)}")

            except ConnectionClosed:
                logger.info(f"[{direction}] Target connection {target_desc} closed while sending.")
                break
            except Exception as e:
                logger.error(f"[{direction}] Error sending message to {target_desc}: {e}")
                break
    except ConnectionClosedOK:
        logger.info(f"[{direction}] Source connection {source_addr} closed gracefully.")
    except ConnectionClosedError as e:
        logger.warning(f"[{direction}] Source connection {source_addr} closed with error: {e}")
    except Exception as e:
        logger.error(f"[{direction}] Error reading from source {source_addr}: {e}")
    finally:
        if target_ws and not target_ws.closed:
            logger.info(f"[{direction}] Closing target connection {target_desc} due to source issue.")
            await target_ws.close()


async def proxy_handler(client_ws: WebSocketServerProtocol, path: str):
    """
    Handles a new client connection, connects to the target, and forwards data.
    """
    client_addr = client_ws.remote_address
    logger.info(f"Client connected: {client_addr} on path '{path}'")

    if path != LISTEN_PATH:
        logger.warning(f"Client {client_addr} connected to wrong path '{path}'. Expected '{LISTEN_PATH}'. Closing.")
        await client_ws.close(code=1003, reason="Invalid path")
        return

    forward_headers: Dict[str, str] = {}
    missing_headers = []
    for header_name in EXPECTED_HEADERS:
        header_value = client_ws.request_headers.get(header_name)
        if header_value:
            forward_headers[header_name] = header_value
            logger.debug(f"Forwarding header {header_name}: {header_value}")
        else:
            logger.warning(f"Header '{header_name}' not found in client request from {client_addr}")
            missing_headers.append(header_name)

    user_agent_header = client_ws.request_headers.get('User-Agent')
    if user_agent_header:
        forward_headers['User-Agent'] = user_agent_header
        logger.debug(f"Forwarding User-Agent header: {user_agent_header}")
    else:
        logger.debug(f"User-Agent header not found in client request from {client_addr}. Not forwarding.")

    target_ws: Optional[WebSocketClientProtocol] = None
    try:
        logger.info(f"Connecting to target: {TARGET_URI} for client {client_addr}")
        target_ws = await websockets.connect(
            TARGET_URI,
            extra_headers=forward_headers,
            ssl=ssl_context, # Use the updated SSL context
            open_timeout=15,
            ping_interval=20,
            ping_timeout=20,
            user_agent_header=None
        )
        logger.info(f"Connected to target {TARGET_URI} successfully for client {client_addr}")

        task_client_to_target = asyncio.create_task(
            forward_messages(client_ws, target_ws, f"CLIENT->TARGET ({client_addr})")
        )
        task_target_to_client = asyncio.create_task(
            forward_messages(target_ws, client_ws, f"TARGET->CLIENT ({client_addr})")
        )

        done, pending = await asyncio.wait(
            [task_client_to_target, task_target_to_client],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            logger.debug(f"Cancelling pending task for {client_addr}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            try:
                await task
            except Exception as e:
                logger.error(f"Forwarding task for {client_addr} finished with error: {e}")


    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid TARGET_URI: {TARGET_URI}")
        await client_ws.close(code=1011, reason="Proxy configuration error")
    except ssl.SSLCertVerificationError as e: # Catch the specific error
         logger.error(f"SSL verification failed connecting to {TARGET_URI}: {e}")
         await client_ws.close(code=1011, reason="Proxy target SSL verification failed")
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"Failed to connect to target {TARGET_URI}: {e}")
        await client_ws.close(code=1011, reason="Proxy target connection failed")
    except ConnectionRefusedError:
         logger.error(f"Connection refused by target {TARGET_URI}")
         await client_ws.close(code=1011, reason="Proxy target refused connection")
    except OSError as e:
        logger.error(f"OS error connecting to target {TARGET_URI}: {e}")
        await client_ws.close(code=1011, reason="Proxy target connection OS error")
    except Exception as e:
        logger.exception(f"An unexpected error occurred for client {client_addr}: {e}")
        if not client_ws.closed:
            await client_ws.close(code=1011, reason="Proxy internal error")
    finally:
        if target_ws and not target_ws.closed:
            logger.info(f"Closing connection to target {TARGET_URI} for client {client_addr}")
            await target_ws.close()
        if not client_ws.closed:
             logger.info(f"Closing connection to client {client_addr}")
             await client_ws.close()
        logger.info(f"Client disconnected: {client_addr}")


async def main():
    # Need to capture the server object to close it gracefully
    stop = asyncio.Future() # Future to signal shutdown
    server = await websockets.serve(
        proxy_handler,
        LISTEN_HOST,
        LISTEN_PORT,
        # Optional: Add server-side SSL if ESP32 needs wss:// to proxy
        # ssl=server_ssl_context
    )
    logger.info(f"Starting WebSocket proxy server on {LISTEN_HOST}:{LISTEN_PORT}")
    logger.info(f"Listening for connections on path: {LISTEN_PATH}")
    logger.info(f"Forwarding connections to: {TARGET_URI}")

    await stop # Keep server running until stop is set

    # Graceful shutdown (optional but good practice)
    logger.info("Shutting down server...")
    server.close()
    await server.wait_closed()
    logger.info("Server shut down complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Proxy server interrupted by user.")
        # Optional: Here you could set the 'stop' future if you implemented
        # more complex signal handling for graceful shutdown on Ctrl+C.
        # For simple cases, asyncio.run handles KeyboardInterrupt okay.