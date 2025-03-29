import asyncio
import websockets
from httpx import Headers
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from typing import Optional
from app import config, protocol, exceptions
from app.logger import logger
from app.state_manager import client_manager, ClientState
from app.services.auth import auth_service
from app.services.iot import iot_service
from app.agent import app as agent_app, AgentState # Import the compiled graph and state type
import uuid

async def process_request(path: str, headers: Headers) -> Optional[tuple[int, Headers, bytes]]:
    """
    Handles the initial HTTP upgrade request (WebSocket handshake).
    Performs header validation and authentication *before* upgrading.
    """
    logger.debug(f"Incoming connection request for path: {path}")
    logger.debug(f"Headers: {headers}")

    # 1. Check Path
    if path != config.WEBSOCKET_PATH:
        logger.warning(f"Connection attempt to wrong path: {path}")
        return (404, Headers({"Content-Type": "text/plain"}), b"Not Found")

    # 2. Check Protocol Version Header
    protocol_version = headers.get("Protocol-Version")
    if protocol_version != config.EXPECTED_PROTOCOL_VERSION:
        logger.error(f"Invalid Protocol-Version: {protocol_version}. Expected: {config.EXPECTED_PROTOCOL_VERSION}")
        return (400, Headers({"Content-Type": "text/plain"}), b"Unsupported Protocol Version")

    # 3. Check Mandatory Headers
    auth_header = headers.get("Authorization")
    device_id = headers.get("Device-Id")
    client_id_header = headers.get("Client-Id") # Renamed to avoid conflict

    if not all([auth_header, device_id, client_id_header]):
        missing = [h for h, v in [("Authorization", auth_header), ("Device-Id", device_id), ("Client-Id", client_id_header)] if not v]
        logger.error(f"Missing mandatory headers: {', '.join(missing)}")
        return (400, Headers({"Content-Type": "text/plain"}), f"Missing headers: {', '.join(missing)}".encode())

    # 4. Validate Authorization Header Format
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"Invalid Authorization header format.")
        return (401, Headers({"Content-Type": "text/plain"}), b"Invalid Authorization header format")

    # 5. Perform Authentication
    token = auth_header.split(" ", 1)[1]
    try:
        is_valid = await auth_service.verify_token(token)
        if not is_valid:
            logger.error(f"Authentication failed for token prefix: {token[:10]}...")
            return (401, Headers({"Content-Type": "text/plain"}), b"Unauthorized")
    except Exception as e:
        logger.error(f"Authentication service error during handshake: {e}", exc_info=True)
        return (503, Headers({"Content-Type": "text/plain"}), b"Authentication service unavailable")

    # If all checks pass, allow the WebSocket upgrade by returning None
    logger.info(f"Handshake validation successful for Device-Id: {device_id}, Client-Id: {client_id_header}")
    # Store validated info for the handler (cannot directly pass, handler must re-read headers)
    return None


async def handle_connection(websocket: WebSocketServerProtocol, path: str):
    """Handles a single WebSocket client connection after handshake."""
    client = None
    headers = websocket.request_headers
    device_id = headers.get("Device-Id", "unknown_device")
    client_id = headers.get("Client-Id", str(uuid.uuid4())) # Use header value or generate fallback

    try:
        # 1. Add client to manager and set initial state
        client = await client_manager.add_client(websocket, client_id, device_id)
        client.change_state(ClientState.CONNECTING)

        # 2. Protocol Handshake ("hello" exchange)
        try:
            # Wait for client hello (with timeout)
            logger.info(f"{client.client_id}: Waiting for client 'hello'...")
            client_hello_raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=config.SERVER_HELLO_TIMEOUT_S
            )
            client.update_activity()

            if not isinstance(client_hello_raw, str):
                 raise exceptions.ProtocolError("Client 'hello' must be a text message.")

            client_hello = protocol.parse_json(client_hello_raw)
            if not client_hello or client_hello.get("type") != protocol.TYPE_HELLO:
                raise exceptions.ProtocolError(f"Invalid or missing client 'hello': {client_hello_raw[:100]}")

            logger.info(f"{client.client_id}: Received client 'hello': {client_hello}")
            # TODO: Validate client_hello contents further if needed (version, audio_params)

            # Send server hello
            server_hello = protocol.create_server_hello(config.TTS_SAMPLE_RATE) # Use configured TTS rate
            logger.info(f"{client.client_id}: Sending server 'hello': {server_hello}")
            await websocket.send(server_hello)
            client.change_state(ClientState.IDLE) # Handshake complete, ready
            logger.info(f"{client.client_id}: Handshake complete. Client is IDLE.")

        except asyncio.TimeoutError:
            logger.error(f"{client.client_id}: Timeout waiting for client 'hello'.")
            raise exceptions.ProtocolError("Client 'hello' timeout")
        except exceptions.ProtocolError as e:
            logger.error(f"{client.client_id}: Handshake protocol error: {e}")
            raise # Re-raise to close connection
        except ConnectionClosed:
            logger.warning(f"{client.client_id}: Connection closed during handshake.")
            # No need to raise, loop will exit. Cleanup happens in finally block.
            return # Exit handler cleanly

        # 3. Main Message Loop
        while websocket.open:
            try:
                # Use a timeout for receiving messages to detect idle clients
                message = await asyncio.wait_for(websocket.recv(), timeout=config.IDLE_TIMEOUT_S + 5) # Slightly more than client timeout
                client.update_activity()

                if isinstance(message, str):
                    # --- Handle Text Message (JSON) ---
                    logger.debug(f"{client.client_id}: Received TEXT message: {message[:200]}")
                    data = protocol.parse_json(message)
                    if not data:
                        logger.warning(f"{client.client_id}: Received invalid JSON: {message[:100]}")
                        # Decide action: ignore, warn, or close connection
                        continue

                    msg_type = data.get("type")

                    # Update conversation history if message has 'session_id' and matches current
                    # Note: Client code sends "" as session_id. Server generates it.
                    # This check might not be useful unless client starts sending it.
                    # msg_session_id = data.get("session_id")
                    # if msg_session_id and msg_session_id != client.session_id:
                    #      logger.warning(f"{client.client_id}: Received message with mismatched session_id. Ignoring? Current: {client.session_id}, Received: {msg_session_id}")
                    #      continue # Or handle appropriately


                    if msg_type == protocol.TYPE_LISTEN:
                        state = data.get("state")
                        mode = data.get("mode") # auto, manual, realtime
                        if state == protocol.LISTEN_STATE_START:
                            if client.state == ClientState.IDLE or client.state == ClientState.SPEAKING: # Can start listening from idle or after speaking
                                 # If speaking, abort current TTS first
                                 if client.state == ClientState.SPEAKING:
                                     logger.warning(f"{client.client_id}: Received listen:start while speaking. Aborting TTS.")
                                     await client.stop_speaking(aborted=True)
                                 client.start_listening(mode if mode else protocol.LISTEN_MODE_MANUAL)
                            else:
                                logger.warning(f"{client.client_id}: Received listen:start in unexpected state: {client.state.name}. Ignoring.")

                        elif state == protocol.LISTEN_STATE_STOP:
                            if client.state == ClientState.LISTENING:
                                client.stop_listening()
                                # Trigger the agent graph
                                if client.audio_buffer:
                                     logger.info(f"{client.client_id}: Triggering agent graph for session {client.session_id}")
                                     initial_graph_state = AgentState(
                                         client=client,
                                         input_audio=client.audio_buffer.copy(), # Pass immutable copy of packet list
                                         asr_text=None,
                                         llm_response_text=None,
                                         llm_emotion=None,
                                         iot_commands_to_send=None,
                                         error_message=None,
                                         conversation_history=client.conversation_history # Pass current history
                                     )
                                     # Run the graph asynchronously
                                     asyncio.create_task(
                                         agent_app.ainvoke(initial_graph_state),
                                         name=f"AgentGraph_{client.client_id}_{client.session_id}"
                                      )
                                else:
                                     logger.warning(f"{client.client_id}: Listen stop received but audio buffer is empty. Returning to IDLE.")
                                     client.change_state(ClientState.IDLE)

                            else:
                                logger.warning(f"{client.client_id}: Received listen:stop in unexpected state: {client.state.name}. Ignoring.")

                        elif state == protocol.LISTEN_STATE_DETECT:
                            wake_word = data.get("text", "<unknown>")
                            logger.info(f"{client.client_id}: Received wake word detected: '{wake_word}'")
                            # Action: Could potentially interrupt TTS like abort, or just log.
                            # Client code seems to handle this before sending audio often.

                        else:
                            logger.warning(f"{client.client_id}: Unknown listen state: {state}")

                    elif msg_type == protocol.TYPE_ABORT:
                        reason = data.get("reason")
                        logger.info(f"{client.client_id}: Received abort request. Reason: {reason}")
                        if client.state == ClientState.SPEAKING:
                            await client.stop_speaking(aborted=True)
                            # If reason is wake word, client might send listen:start next
                        else:
                            logger.warning(f"{client.client_id}: Received abort in non-speaking state: {client.state.name}. Ignoring.")

                    elif msg_type == protocol.TYPE_IOT:
                        # Handle device descriptor/state updates
                        if "descriptors" in data:
                            descriptors = data.get("descriptors", [])
                            logger.info(f"{client.client_id}: Received IoT descriptors: {len(descriptors)} items")
                            # Store descriptors using the IoT service
                            await iot_service.store_descriptors(client.device_id, descriptors)
                        elif "states" in data:
                             states = data.get("states", {})
                             logger.info(f"{client.client_id}: Received IoT states: {states}")
                             # Store states using the IoT service
                             await iot_service.store_states(client.device_id, states)
                        else:
                             logger.warning(f"{client.client_id}: Received unknown IoT message format: {data}")

                    elif msg_type == protocol.TYPE_HELLO:
                         logger.warning(f"{client.client_id}: Received unexpected 'hello' after handshake. Ignoring.")
                    else:
                        logger.warning(f"{client.client_id}: Received unknown message type: {msg_type}")

                elif isinstance(message, bytes):
                    # --- Handle Binary Message (Audio) ---
                    # logger.debug(f"{client.client_id}: Received BINARY message: {len(message)} bytes")
                    if client.state == ClientState.LISTENING:
                        # Add to buffer as an individual opus packet
                        should_process = client.add_audio_packet(message)
                        
                        # If in auto mode and packet processing indicates we should stop
                        if should_process:
                            # Stop listening and process audio
                            client.stop_listening()
                            
                            # Trigger the agent graph
                            logger.info(f"{client.client_id}: Triggering agent graph for session {client.session_id}")
                            initial_graph_state = AgentState(
                                client=client,
                                input_audio=client.audio_buffer.copy(),  # Pass copy of packet list
                                asr_text=None,
                                llm_response_text=None,
                                llm_emotion=None,
                                iot_commands_to_send=None,
                                error_message=None,
                                conversation_history=client.conversation_history # Pass current history
                            )
                            # Run the graph asynchronously
                            asyncio.create_task(
                                agent_app.ainvoke(initial_graph_state),
                                name=f"AgentGraph_{client.client_id}_{client.session_id}"
                            )
                    else:
                        logger.warning(f"{client.client_id}: Received audio data in non-listening state: {client.state.name}. Discarding.")
                else:
                    logger.warning(f"{client.client_id}: Received unexpected message format: {type(message)}")

            except asyncio.TimeoutError:
                logger.warning(f"{client.client_id}: Client idle timeout ({config.IDLE_TIMEOUT_S}s). Closing connection.")
                await websocket.close(code=1008, reason="Idle timeout") # 1008 = Policy Violation
                break # Exit loop after closing
            except ConnectionClosed:
                logger.info(f"{client.client_id}: Connection closed by client (or during send).")
                break # Exit loop
            except exceptions.ProtocolError as e:
                 logger.error(f"{client.client_id}: Protocol error: {e}. Closing connection.")
                 await websocket.close(code=1002, reason=f"Protocol error: {e}") # 1002 = Protocol Error
                 break
            except Exception as e:
                 logger.error(f"{client.client_id}: Unexpected error in message loop: {e}", exc_info=True)
                 # Attempt graceful close, otherwise finally block handles removal
                 await websocket.close(code=1011, reason="Internal server error") # 1011 = Internal Error
                 break

    except ConnectionClosedOK:
         logger.info(f"{client_id}/{device_id}: Connection closed normally.")
    except ConnectionClosedError as e:
         logger.warning(f"{client_id}/{device_id}: Connection closed with error: {e.code} {e.reason}")
    except exceptions.ProtocolError as e:
         logger.error(f"{client_id}/{device_id}: Initial handshake failed: {e}")
         # Ensure connection is closed if not already
         if not websocket.closed:
             await websocket.close(code=1002, reason=f"Handshake failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during connection handling for {client_id}/{device_id}: {e}", exc_info=True)
        if not websocket.closed:
             await websocket.close(code=1011, reason="Internal server error")
    finally:
        # --- Cleanup ---
        if client:
            await client_manager.remove_client(websocket)
        else:
             # If client object wasn't even created (e.g., handshake failed early)
             logger.info(f"Cleaning up failed connection attempt from {websocket.remote_address}")
        logger.info(f"Connection handler finished for {client_id}/{device_id}")