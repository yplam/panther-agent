# AI Voice Agent WebSocket Server

This server implements the backend for an ESP32-based AI voice assistant, handling WebSocket connections, audio processing (ASR, LLM, TTS), and state management using LangGraph.

## Features

*   WebSocket communication following the specified protocol.
*   Client authentication via Bearer token.
*   Header validation (`Protocol-Version`, `Device-Id`, `Client-Id`).
*   Handles Opus audio input from the client.
*   Orchestrates ASR -> LLM -> TTS flow using LangGraph.
*   Integrates with OpenAI-compatible APIs for AI services (configurable).
*   Supports Azure Cognitive Services as an alternative to OpenAI:
    *   Azure Speech Service for ASR and TTS
    *   Azure OpenAI Service for LLM
    *   Azure Text Analytics for emotion detection
*   Abstracted service layer for easy replacement of AI providers.
*   Handles `listen`, `abort`, `iot` messages from the client.
*   Sends `stt`, `llm`, `tts`, `iot` messages to the client.
*   Streams Opus audio output for TTS.
*   Basic in-memory IoT device state/descriptor handling (placeholder).
*   State management per client connection.
*   Graceful shutdown and logging.
*   Docker support.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ai-voice-agent-server
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root or set environment variables directly. See `.env.example` (you should create this) for required variables:

    ```dotenv
    # .env.example
    WEBSOCKET_PORT=8765
    WEBSOCKET_PATH=/voice
    IDLE_TIMEOUT_S=300

    # --- Authentication ---
    # Option 1: Simple In-Memory Tokens (FOR DEVELOPMENT/TESTING ONLY)
    VALID_BEARER_TOKENS=your_secret_token_1,another_secret_token
    # Option 2: Remote Auth Service URL (Recommended for Production)
    # AUTH_SERVICE_URL=https://your-auth-service.com/verify

    # --- Service Selection ---
    # Uncomment to use Azure services instead of OpenAI
    # ASR_SERVICE=azure
    # TTS_SERVICE=azure
    # LLM_SERVICE=azure

    # --- Language Settings ---
    DEFAULT_LANGUAGE=zh-CN
    TTS_LANGUAGE=zh-CN
    ASR_LANGUAGE=zh-CN

    # --- OpenAI Services ---
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # OPENAI_BASE_URL= # Optional: For local models or proxies
    ASR_MODEL=whisper-1
    LLM_MODEL=gpt-4o # Or gpt-3.5-turbo, etc.
    TTS_MODEL=tts-1
    TTS_VOICE=nova # Good for Chinese, other options: echo, fable, onyx, alloy, shimmer

    # --- Azure Services ---
    # Azure OpenAI Service (LLM)
    # AZURE_OPENAI_KEY=your_azure_openai_key
    # AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
    # AZURE_OPENAI_API_VERSION=2023-05-15
    # AZURE_DEPLOYMENT_NAME=your-deployment-name

    # Azure Speech Service (ASR and TTS)
    # AZURE_SPEECH_KEY=your_azure_speech_key
    # AZURE_SPEECH_REGION=eastus
    # AZURE_SPEECH_RECOGNITION_LANGUAGE=zh-CN
    # AZURE_SPEECH_SYNTHESIS_LANGUAGE=zh-CN
    # AZURE_TTS_VOICE=zh-CN-XiaoxiaoNeural

    # Azure Text Analytics (for emotion detection with Azure LLM)
    # AZURE_TEXT_ANALYTICS_KEY=your_azure_text_analytics_key
    # AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/

    # --- Logging ---
    LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR
    ```
    **Important:** For production, use a secure method for managing tokens (like `AUTH_SERVICE_URL`) instead of `VALID_BEARER_TOKENS`.

## Running the Server

*   **Directly:**
    ```bash
    source venv/bin/activate
    python main.py
    ```

*   **With Docker:**
    1.  Build the image:
        ```bash
        docker build -t ai-voice-agent-server .
        ```
    2.  Run the container, passing environment variables:
        ```bash
        docker run -p 8765:8765 \
          -e WEBSOCKET_PORT=8765 \
          -e OPENAI_API_KEY="your_key_here" \
          -e VALID_BEARER_TOKENS="your_token" \
          -e LOG_LEVEL="INFO" \
          --name voice-agent-server \
          ai-voice-agent-server
        ```
        (Add other necessary `-e` flags based on your `.env` file).

## Protocol Notes & Mismatches

*   **TTS Sample Rate:** The documentation strongly recommends 16kHz TTS output. However, the example OpenAI TTS service generates **24kHz** Opus. This implementation sends 24kHz Opus and declares `24000` in the server "hello". **Ensure your ESP32 client's Opus decoder can handle 24kHz.** If not, you MUST either:
    *   Use a different TTS provider that outputs 16kHz Opus.
    *   Implement server-side resampling from 24kHz to 16kHz (adds complexity and latency).
*   **Session ID:** The server generates a unique `session_id` for each listen->speak interaction. The client code often sends `session_id: ""`. The server currently ignores the incoming `session_id`.
*   **IoT Commands:** The LLM tool definition and parsing logic (`app/services/llm.py`) needs to be adapted based on the *exact* command structure expected by the ESP32 client's `ThingManager`. The current example (`deviceId`, `action`, `value`) is a placeholder.
*   **Error Handling:** Error messages from AI services are wrapped and potentially spoken back to the user via TTS.

## TODO / Potential Improvements

*   Implement robust remote authentication (`RemoteAuthService`).
*   Refine LLM system prompt and tool definitions for better responses and IoT control.
*   Implement persistent storage for IoT descriptors/states (e.g., database).
*   Add server-side VAD if needed (e.g., for `auto` mode without client VAD).
*   Implement server-side resampling for TTS if 16kHz is strictly required and the provider doesn't support it.
*   Add more comprehensive unit and integration tests.
*   Implement checkpointing for LangGraph state if needed for recovery.
*   Handle `listen:detect` messages more actively if required.
*   Refine conversation history management within the AgentState.

## Using Azure Cognitive Services

The server supports using Azure Cognitive Services as an alternative to OpenAI:

1. **Azure Speech Service for ASR and TTS**:
   - Requires an Azure Speech Service resource
   - Configure with AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
   - Set ASR_SERVICE=azure and/or TTS_SERVICE=azure to enable

2. **Azure OpenAI Service for LLM**:
   - Requires an Azure OpenAI Service resource with a deployed model
   - Configure with AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_DEPLOYMENT_NAME
   - Set LLM_SERVICE=azure to enable

3. **Azure Text Analytics for Emotion Detection**:
   - Optional, used with Azure LLM to determine emotion from responses
   - Configure with AZURE_TEXT_ANALYTICS_KEY and AZURE_TEXT_ANALYTICS_ENDPOINT

The services can be used independently - for example, you can use Azure Speech for ASR/TTS while still using OpenAI for the LLM component, or any other combination.