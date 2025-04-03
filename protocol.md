**AI Agent ESP32 WebSocket 通讯协议文档**

**版本: 1.0**

**最后更新日期: 2025-04-01**

---

## 1. 概述

本文档定义了基于 ESP32 的 AI Agent 客户端（以下简称“客户端”或“设备”）与后端 AI 服务（以下简称“服务器”）之间通过 WebSocket 进行通信的协议。该协议支持双向的文本和音频流传输，以实现语音识别 (ASR)、语音合成 (TTS)、大语言模型 (LLM) 交互以及基本的 IoT 设备状态同步和控制。

---

## 2. 基本信息

-   **协议版本**: 1 (在 Header 和 `hello` 消息中指定)
-   **传输方式**: WebSocket (全双工通信)
-   **数据格式**:
    -   控制/元数据消息: JSON (UTF-8 编码)
    -   音频数据: 二进制帧
-   **音频编码**: OPUS
-   **客户端音频参数 (上行)**:
    -   采样率: 16000 Hz
    -   通道数: 1 (单声道)
    -   帧长: 60 ms (每帧包含 60ms 的音频数据)
-   **服务器音频参数 (下行)**:
    -   采样率: 由服务器在 `hello` 响应中指定 (日志中显示为 24000 Hz)
    -   通道数: 1 (单声道)
    -   帧长: 60 ms (与客户端匹配)
    *注意：客户端采样率固定，服务端需适配客户端采样率*

---

## 3. 连接建立与握手

### 3.1 WebSocket 连接请求

客户端发起 WebSocket 连接时，必须在 HTTP(S) 请求头中包含以下请求头：

```http
Authorization: Bearer <access_token>       # 访问令牌，用于身份验证
Protocol-Version: 1                         # 本协议的版本号
Device-Id: <设备MAC地址>                   # 设备的唯一硬件标识符 (例如: 94:a9:90:28:d9:28)
Client-Id: <设备UUID>                      # 设备的唯一软件/实例标识符 (例如: 9a35728c-637b-4dc3-80dc-8c705cca80fd)
```

### 3.2 握手 (Hello Exchange)

WebSocket 连接成功建立后，双方需要进行一次 `hello` 消息交换以确认协议参数和建立会话上下文。

1.  **客户端 -> 服务器**: 发送 `hello` 消息，声明自身能力。

    ```json
    {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60
        }
    }
    ```

2.  **服务器 -> 客户端**: 响应 `hello` 消息，确认连接并提供服务器参数和会话 ID。

    ```json
    {
        "type": "hello",
        "version": 1,                  # 确认协议版本
        "transport": "websocket",
        "audio_params": {
            "format": "opus",          # 确认音频格式
            "sample_rate": 16000,      # 服务器期望/发送的音频采样率
            "channels": 1,
            "frame_duration": 60
        },
        "session_id": "706bfb75"       # 服务器分配的会话 ID，用于后续消息关联
    }
    ```

**注意**: 客户端在收到此 `session_id` 后，应在后续与该会话相关的请求中携带此 ID。

---

## 4. 消息格式 (JSON)

所有非音频数据均使用 WebSocket 的文本帧以 JSON 格式传输。

### 4.1 通用字段

大多数 JSON 消息包含以下字段：

-   `type` (string, **必需**): 消息类型，决定消息的用途和结构。
-   `session_id` (string, **推荐**): 由服务器在 `hello` 响应中分配的会话标识符。
    -   服务器发送的所有与特定交互会话相关的消息（`stt`, `tts`, `llm`）都应包含此 ID。
    -   客户端发送的消息理论上也应包含此 ID 以关联上下文，但服务端需兼容 ID 为空的情形。

### 4.2 客户端 -> 服务器 消息

#### 4.2.1 `hello` (已在 3.2 节描述)

#### 4.2.2 `listen` (语音识别控制)

-   **开始监听**: 指示客户端开始捕获音频并发送至服务器。

    ```json
    {
        "session_id": "<会话ID>",  // 建议使用服务器分配的 ID
        "type": "listen",
        "state": "start",
        "mode": "<监听模式>"      // "auto", "manual", "realtime"
    }
    ```
    *监听模式说明:*
        *   `"auto"`: 服务器根据语音活动（VAD）或语义理解自动停止监听。
        *   `"manual"`: 客户端需要显式发送 `stop` 消息来停止监听。
        *   `"realtime"`: 持续监听并实时传输，通常用于流式识别。

-   **停止监听**: 指示客户端停止捕获和发送音频。

    ```json
    {
        "session_id": "<会话ID>",  // 建议使用服务器分配的 ID
        "type": "listen",
        "state": "stop"
    }
    ```

-   **唤醒词检测**: 客户端本地检测到唤醒词后通知服务器。

    ```json
    {
        "session_id": "<会话ID>",  // 建议使用服务器分配的 ID
        "type": "listen",
        "state": "detect",
        "text": "<检测到的唤醒词>" // 例如: "你好小智"
    }
    ```
    *注意：在发送 `detect` 消息之前，客户端可能会直接发送唤醒词对应的音频数据*

#### 4.2.3 `abort` (中止操作)

用于客户端主动中止当前的 TTS 播放或其他正在进行的操作。

```json
{
    "session_id": "<会话ID>", // 建议使用服务器分配的 ID
    "type": "abort",
    "reason": "<中止原因>"     // 可选，例如 "wake_word_detected", "user_interruption"
}
```

#### 4.2.4 `iot` (设备能力与状态)

-   **上报设备描述 (Descriptors)**: 客户端启动后或描述更新时，向服务器声明其可控制的组件及其属性和方法。

    ```json
    {
        "session_id": "",
        "type": "iot",
        "update": true,          // 表示这是一个更新操作
        "descriptors": [         // 包含一个或多个设备描述的数组，每次可发一个或者多个
            {
                "name": "Speaker", // 组件名称
                "description": "扬声器", // 组件描述
                "properties": {      // 可读/可写的属性
                    "volume": {"description": "当前音量值", "type": "number"}
                },
                "methods": {         // 可调用的方法
                    "SetVolume": {
                        "description": "设置音量",
                        "parameters": {
                            "volume": {"description": "0到100之间的整数", "type": "number"}
                        }
                    }
                }
            }
            // ... 其他组件的描述 (日志显示分多条消息发送)
            // {"name": "Screen", ...}
            // {"name": "Chassis", ...}
        ]
    }
    ```
    *注意：客户端在 `hello` 交换后立即发送一个或者多个 `iot` descriptor 消息。*

-   **上报设备状态 (States)**: 客户端向服务器报告其当前状态。

    ```json
    {
        "session_id": "",        // 日志显示为空字符串
        "type": "iot",
        "update": true,          // 日志显示此字段
        "states": [              // 包含一个或多个组件当前状态的数组
            {
                "name": "Speaker", // 组件名称
                "state": {         // 当前状态键值对
                    "volume": 90
                }
            },
            {
                "name": "Screen",
                "state": {
                    "theme": "light",
                    "brightness": 75
                }
            },
            {
                "name": "Chassis",
                "state": {
                    "light_mode": 1
                }
            }
        ]
    }
    ```
    *注意：客户端在一次交互过程中（收到 TTS stop 后，开始下一次 listen 之前）上报了设备状态。*

### 4.3 服务器 -> 客户端 消息

#### 4.3.1 `hello` (已在 3.2 节描述)

#### 4.3.2 `stt` (语音识别结果)

服务器将 ASR 识别出的文本发送给客户端。

```json
{
    "type": "stt",
    "text": "<识别出的用户语音文本>", // 例如: "小智", "没事，拜拜。"
    "session_id": "<会话ID>"       // 关联的会话 ID
}
```

#### 4.3.3 `tts` (语音合成控制与状态)

-   **开始合成/播放**: 通知客户端即将开始接收并播放 TTS 音频流。

    ```json
    {
        "type": "tts",
        "state": "start",
        "sample_rate": 16000,     //  TTS 音频的采样率
        "session_id": "<会话ID>"
    }
    ```
    *此消息后，服务器会开始发送二进制 OPUS 音频帧。*

-   **句子开始**: 标记一个新的句子开始播放，并提供该句子的文本内容。

    ```json
    {
        "type": "tts",
        "state": "sentence_start",
        "text": "<当前开始播放的句子文本>", // 例如: "嗨，我在呢，有什么好玩的事吗？"
        "session_id": "<会话ID>"
    }
    ```

-   **句子结束**:  标记一个句子的音频数据已发送完毕。

    ```json
    {
        "type": "tts",
        "state": "sentence_end",
        "text": "<刚刚播放完毕的句子文本>", // 例如: "嗨，我在呢，有什么好玩的事吗？"
        "session_id": "<会话ID>"
    }
    ```

-   **停止合成/播放**: 通知客户端 TTS 音频流已结束。

    ```json
    {
        "type": "tts",
        "state": "stop",
        "session_id": "<会话ID>"
    }
    ```

#### 4.3.4 `llm` (大语言模型相关)

服务器发送 LLM 的处理结果，可能包括情感状态或其他非语音信息。

```json
{
    "type": "llm",
    "emotion": "<情感类型>", // 例如: "happy"
    "text": "<可选文本表示>", // 例如: "😊" (Emoji)
    "session_id": "<会话ID>"
}
```

#### 4.3.5 `iot` (设备控制命令)

服务器向客户端下发控制指令。

```json
{
    "type": "iot",
    "commands": [             // 一个或多个命令组成的数组
        {
            "name": "<组件名称>", // 例如: "Speaker"
            "method": "<方法名称>", // 例如: "SetVolume"
            "parameters": {      // 方法所需的参数
                "<参数名>": "<参数值>" // 例如: "volume": 80
            }
        }
        // ... 可能包含对其他组件的命令
    ],
    "session_id": "<会话ID>"   // 可选，如果命令与特定会话相关
}
```

---

## 5. 二进制数据传输 (音频)

-   音频数据通过 WebSocket 的**二进制帧 (Binary Frame)** 传输。
-   **客户端 -> 服务器**: 发送 OPUS 编码的麦克风音频数据。通常在发送 `{"type":"listen", "state":"start"}` 之后开始，直到发送 `{"type":"listen", "state":"stop"}` 或服务器指示停止。日志显示在唤醒 (`detect`) 消息前和用户指令期间都有音频发送。
-   **服务器 -> 客户端**: 发送 OPUS 编码的 TTS 合成音频数据。通常在收到 `{"type":"tts", "state":"start"}` 之后开始，直到收到 `{"type":"tts", "state":"stop"}`。
-   **编码参数**:
    -   客户端上行: 16000Hz, 1 channel, 60ms frames.
    -   服务器下行: 16000Hz , 1 channel, 60ms frames.
-   **数据量**: 日志显示 OPUS 压缩效率很高，每帧 60ms (16kHz/16bit 为 1920 bytes PCM) 被压缩到几十到二百字节不等。

---

## 6. 会话与上下文管理 (Session & Context)

1.  **会话创建**: 服务器在响应客户端的第一个 `hello` 消息时，通过 `session_id` 字段创建一个唯一的会话标识符。这个 `session_id` 定义了本次连接的初始上下文。
2.  **上下文关联**:
    -   服务器在后续发送的与该会话相关的消息 (`stt`, `tts`, `llm`) 中，会携带相同的 `session_id`，以便客户端知道这些消息属于哪个交互流程。
    -   客户端**理论上**也应在其发送的与特定会话相关的消息（如 `listen`, `abort`）中使用服务器提供的 `session_id`。然而，抓包日志显示，客户端在某些情况下（如 `listen start/detect`, `iot` 消息）发送的是 `session_id:""`。这可能意味着：
        -   客户端实现未完全遵循预期，或者
        -   某些客户端发起的动作（如上报状态/描述，或开始新的监听）被设计为不强依赖于之前的 `session_id`，服务器能够通过连接本身来维持上下文。
    -   服务器需要具备一定的鲁棒性来处理客户端 `session_id` 可能为空的情况，可能通过 WebSocket 连接实例本身来维持状态。
3.  **IoT 消息的上下文**: 日志显示 `iot` 消息（无论是 `descriptors` 还是 `states`）由客户端发送时，`session_id` 为空。这表明 IoT 的能力声明和状态上报可能被视为独立于具体的对话会话，或者是在连接级别进行管理的。

---

## 7. 典型交互流程示例

1.  **连接与握手**:
    -   客户端发起 WebSocket 连接，携带 Headers (`Authorization`, `Protocol-Version`, `Device-Id`, `Client-Id`)。
    -   服务器接受连接。
    -   `C -> S`: `{"type":"hello", ...}`
    -   `S -> C`: `{"type":"hello", ..., "session_id": "706bfb75"}`

2.  **设备能力上报**:
    -   `C -> S`: `{"type":"iot", "update":true, "descriptors": [{"name":"Speaker", ...}]}`
    -   `C -> S`: `{"type":"iot", "update":true, "descriptors": [{"name":"Screen", ...}]}`
    -   `C -> S`: `{"type":"iot", "update":true, "descriptors": [{"name":"Chassis", ...}]}`

3.  **唤醒与首次交互**:
    -   `C -> S`: (发送唤醒词音频帧)
    -   `C -> S`: `{"session_id":"","type":"listen","state":"detect","text":"你好小智"}` (检测到唤醒词)
    -   `S -> C`: `{"type":"tts","state":"start","sample_rate":16000,"session_id":"706bfb75"}` (服务器准备响应)
    -   `S -> C`: `{"type":"stt","text":"小智","session_id":"706bfb75"}` (服务器确认听到的唤醒词部分)
    -   `S -> C`: `{"type":"llm","text":"😊","emotion":"happy","session_id":"706bfb75"}` (LLM 生成的情感/状态)
    -   `S -> C`: `{"type":"tts","state":"sentence_start","text":"嗨，我在呢，有什么好玩的事吗？","session_id":"706bfb75"}` (开始播报第一句)
    -   `S -> C`: (发送大量二进制 OPUS 音频帧 - TTS 语音)
    -   `S -> C`: `{"type":"tts","state":"sentence_end","text":"嗨，我在呢，有什么好玩的事吗？","session_id":"706bfb75"}` (第一句结束)
    -   `S -> C`: `{"type":"tts","state":"stop","session_id":"706bfb75"}` (TTS 播报结束)

4.  **用户指令与后续交互**:
    -   `C -> S`: `{"session_id":"","type":"listen","state":"start","mode":"auto"}` (客户端开始监听用户指令)
    -   `C -> S`: `{"session_id":"","type":"iot","update":true,"states": [...]}` (客户端上报当前状态)
    -   `C -> S`: (发送二进制 OPUS 音频帧 - 用户说 "没事，拜拜。")
    -   `S -> C`: `{"type":"tts","state":"start","sample_rate":16000,"session_id":"706bfb75"}` (服务器准备响应或确认)
    -   `S -> C`: `{"type":"stt","text":"没事，拜拜。","session_id":"706bfb75"}` (识别出用户指令)
    -   `S -> C`: `{"type":"tts","state":"stop","session_id":"706bfb75"}` (服务器可能没有语音回复，直接停止 TTS 状态)

5.  **连接关闭**:
    -   客户端主动关闭连接 (日志显示客户端发送了 Close Code 1000，但服务器未收到 Close Frame，表明可能关闭不规范)。
    -   服务器检测到连接关闭。

---

## 8. 错误处理

-   **连接错误**: 如果 WebSocket 连接无法建立（例如，网络问题、认证失败），客户端应处理连接失败事件，并根据策略进行重试。
-   **协议错误**: 如果收到无法解析的 JSON 消息或格式不正确的二进制数据，接收方应记录错误，并可选择忽略该消息或关闭连接。
-   **运行时错误**: 服务器可能在处理过程中遇到错误（如 ASR/TTS/LLM 服务失败）。服务器应通过 WebSocket 发送错误消息（协议中未明确定义错误消息格式，建议补充）或直接关闭连接并携带相应的关闭代码。
-   **连接中断**: 任何一方检测到 WebSocket 连接异常断开时，应清理相关资源。客户端通常需要实现重连机制。
-   **日志警告**: `[CLIENT->SERVER] Source connection (...) closed with error: sent 1000 (OK); no close frame received` 表明客户端关闭流程可能不完全符合 WebSocket 标准，虽然发送了关闭意图 (code 1000)，但握手未完成。服务器侧应能容忍这种情况。

---

## 9. 待明确/改进点

-   **客户端 `session_id` 使用**: 需要明确客户端在哪些消息中必须使用服务器分配的 `session_id`，以及如何处理空 `session_id` 的情况。当前日志显示行为与预期不完全一致。
-   **错误消息格式**: 建议定义标准的错误消息格式，例如 `{"type": "error", "code": <错误码>, "message": "<错误信息>", "session_id": "<相关会话ID>"}`，以便客户端能更好地理解和处理服务端错误。
-   **`iot` 命令**: 需要实际的服务器下发 `iot` 命令的日志或示例来确认其具体格式和使用场景。
-   **`listen` 模式**: `manual` 和 `realtime` 模式的具体交互流程需要更详细的描述或示例。
-   **连接关闭规范**: 建议客户端遵循标准的 WebSocket 关闭握手流程。

