from openai import AsyncOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from app.logger import logger
from app.services.base import BaseLLMService
from app import config, exceptions
from typing import Tuple, Optional, List, Dict, Any, AsyncGenerator
import json

from app.services.utils import openai_client, get_azure_openai_client


class OpenAILLMService(BaseLLMService):
    def __init__(self, client: AsyncOpenAI):
        if not client:
            raise ValueError("AsyncOpenAI client instance is required.")
        self.client = client

        # TODO: Define tools based on IoT capabilities if needed
        self.tools = [
             {
                "type": "function",
                "function": {
                    "name": "control_iot_device",
                    "description": "Controls an IoT device like a light or switch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commands": {
                                "type": "array",
                                "description": "A list of commands to execute.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                         # Define properties based on client's expected command format
                                         # e.g., "deviceId", "component", "capability", "command", "args"
                                        "deviceId": {"type": "string", "description": "Target device ID (e.g., MAC address)"},
                                        "action": {"type": "string", "description": "Action to perform (e.g., 'turnOn', 'setBrightness')"},
                                        "value": {"type": "string", "description": "Value for the action (optional)"}
                                    },
                                     "required": ["deviceId", "action"]
                                }
                            }
                        },
                        "required": ["commands"],
                    },
                },
            }
            # Add more tools here if needed
        ]


    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        device_id: Optional[str] = None, # Pass device ID for context
        # device_capabilities: Optional[List[Dict[str, Any]]] = None # Pass capabilities
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]: # (response_text, emotion, iot_commands)

        logger.info(f"Sending prompt to LLM ({config.LLM_MODEL}): '{prompt}'")
        messages = []
        # Basic System Prompt - Enhance this significantly!
        messages.append({"role": "system", "content": "You are a helpful voice assistant. Respond concisely. Infer emotion (neutral, happy, sad, thinking, worried) and respond in JSON format: {\"response\": \"Your text response\", \"emotion\": \"detected_emotion\"}. Only use the provided tools if necessary."})

        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            completion = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.7,
                # IMPORTANT: Use JSON mode if your model supports it reliably
                # response_format={"type": "json_object"}, # Use if model supports enforced JSON output
                tools=self.tools if device_id else None, # Only provide tools if relevant
                tool_choice="auto" if device_id else None,
            )

            response_message = completion.choices[0].message
            response_text = "Sorry, I couldn't process that."
            emotion = "neutral"
            iot_commands = None

            # Check for tool calls
            if response_message.tool_calls:
                logger.info(f"LLM requested tool calls: {response_message.tool_calls}")
                # For now, we only handle the first tool call if it's iot control
                tool_call = response_message.tool_calls[0]
                if tool_call.function.name == "control_iot_device":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        iot_commands = arguments.get("commands")
                        if iot_commands and device_id:
                             # Add device_id if not already present in each command by the LLM
                            for cmd in iot_commands:
                                if "deviceId" not in cmd:
                                    cmd["deviceId"] = device_id
                            logger.info(f"Extracted IoT commands: {iot_commands}")
                            # We need a follow-up response after the tool call
                            # Let's ask the LLM to generate a confirmation/response text
                            messages.append(response_message) # Add assistant's tool request
                            messages.append({ # Add tool execution result (simulate success for now)
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": json.dumps({"success": True, "message": "Commands sent to device."}) # Simulate success
                            })
                            follow_up_completion = await self.client.chat.completions.create(
                                model=config.LLM_MODEL,
                                messages=messages,
                                temperature=0.7,
                                # response_format={"type": "json_object"}, # Use if model supports enforced JSON output
                            )
                            follow_up_message = follow_up_completion.choices[0].message.content
                            # Attempt to parse the follow-up response JSON
                            try:
                                content_json = json.loads(follow_up_message)
                                response_text = content_json.get("response", "Okay, done.")
                                emotion = content_json.get("emotion", "neutral")
                            except json.JSONDecodeError:
                                logger.warning(f"LLM follow-up response was not valid JSON: {follow_up_message}")
                                response_text = follow_up_message # Use raw text if not JSON
                                emotion = "neutral" # Default emotion
                        else:
                             logger.warning("IoT command requested but no commands found or device_id missing.")
                             # Fallback to generating a normal response without tool call
                             response_text = "I can't control devices right now."
                             emotion = "sad"

                    except json.JSONDecodeError as json_err:
                        logger.error(f"Failed to parse LLM tool arguments: {json_err}")
                        response_text = "There was an issue processing the device command."
                        emotion = "sad"
                    except Exception as e:
                         logger.error(f"Error processing tool call: {e}")
                         response_text = "Something went wrong with the device control."
                         emotion = "sad"
                else:
                    logger.warning(f"Unhandled tool call: {tool_call.function.name}")
                    # Fallback if an unexpected tool is called
                    response_text = "I received an unexpected request."
                    emotion = "worried"

            # No tool calls, process normal response
            elif response_message.content:
                raw_content = response_message.content
                # Attempt to parse the response JSON directly
                try:
                    content_json = json.loads(raw_content)
                    response_text = content_json.get("response", "Sorry, I had trouble formulating a response.")
                    emotion = content_json.get("emotion", "neutral")
                except json.JSONDecodeError:
                    logger.warning(f"LLM response was not valid JSON: {raw_content}")
                    response_text = raw_content # Use raw text if not JSON
                    emotion = "neutral" # Default emotion

            logger.info(f"LLM Response: '{response_text}', Emotion: {emotion}, IoT: {iot_commands}")
            return response_text, emotion, iot_commands

        except Exception as e:
            logger.error(f"LLM service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"LLM failed: {e}")


class AzureLLMService(BaseLLMService):
    def __init__(self, client: AsyncOpenAI, text_analytics_client=None):
        if not client:
            raise ValueError("Azure OpenAI client instance is required.")
        self.client = client
        self.text_analytics_client = text_analytics_client
        
        # Similar tools as OpenAILLMService
        self.tools = [
             {
                "type": "function",
                "function": {
                    "name": "control_iot_device",
                    "description": "Controls an IoT device like a light or switch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commands": {
                                "type": "array",
                                "description": "A list of commands to execute.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "deviceId": {"type": "string", "description": "Target device ID (e.g., MAC address)"},
                                        "action": {"type": "string", "description": "Action to perform (e.g., 'turnOn', 'setBrightness')"},
                                        "value": {"type": "string", "description": "Value for the action (optional)"}
                                    },
                                     "required": ["deviceId", "action"]
                                }
                            }
                        },
                        "required": ["commands"],
                    },
                },
            }
        ]

    async def _analyze_emotion(self, text: str) -> str:
        """Use Azure Text Analytics to determine the emotion in the text."""
        if not self.text_analytics_client:
            return "neutral"  # Default if no text analytics client
        
        try:
            documents = [text]
            response = self.text_analytics_client.analyze_sentiment(documents=documents)
            
            result = [doc for doc in response if not doc.is_error]
            if result and result[0].sentiment:
                # Map Azure sentiment to our emotions
                sentiment = result[0].sentiment
                if sentiment == "positive":
                    return "happy"
                elif sentiment == "negative":
                    return "sad"
                elif sentiment == "neutral":
                    return "neutral"
                elif sentiment == "mixed":
                    return "worried"
            
            return "neutral"
        except Exception as e:
            logger.warning(f"Error analyzing emotion: {e}")
            return "neutral"

    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        device_id: Optional[str] = None, # Pass device ID for context
        # device_capabilities: Optional[List[Dict[str, Any]]] = None # Pass capabilities
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]: # (response_text, emotion, iot_commands)

        logger.info(f"Sending prompt to Azure OpenAI ({config.AZURE_DEPLOYMENT_NAME}): '{prompt}'")
        messages = []
        # Basic System Prompt - Enhance this significantly!
        messages.append({"role": "system", "content": "You are a helpful voice assistant. Respond concisely."})

        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            completion = await self.client.chat.completions.create(
                model=config.AZURE_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.7,
                tools=self.tools if device_id else None,
                tool_choice="auto" if device_id else None,
            )

            response_message = completion.choices[0].message
            response_text = "Sorry, I couldn't process that."
            emotion = "neutral"
            iot_commands = None

            # Check for tool calls
            if response_message.tool_calls:
                logger.info(f"Azure LLM requested tool calls: {response_message.tool_calls}")
                # For now, we only handle the first tool call if it's iot control
                tool_call = response_message.tool_calls[0]
                if tool_call.function.name == "control_iot_device":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        iot_commands = arguments.get("commands")
                        if iot_commands and device_id:
                             # Add device_id if not already present in each command by the LLM
                            for cmd in iot_commands:
                                if "deviceId" not in cmd:
                                    cmd["deviceId"] = device_id
                            logger.info(f"Extracted IoT commands: {iot_commands}")
                            # We need a follow-up response after the tool call
                            # Let's ask the LLM to generate a confirmation/response text
                            messages.append(response_message) # Add assistant's tool request
                            messages.append({ # Add tool execution result (simulate success for now)
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": json.dumps({"success": True, "message": "Commands sent to device."}) # Simulate success
                            })
                            follow_up_completion = await self.client.chat.completions.create(
                                model=config.AZURE_DEPLOYMENT_NAME,
                                messages=messages,
                                temperature=0.7,
                            )
                            response_text = follow_up_completion.choices[0].message.content
                            # For Azure, we'll use Text Analytics to determine emotion
                            emotion = await self._analyze_emotion(response_text)
                        else:
                             logger.warning("IoT command requested but no commands found or device_id missing.")
                             # Fallback to generating a normal response without tool call
                             response_text = "I can't control devices right now."
                             emotion = "sad"

                    except json.JSONDecodeError as json_err:
                        logger.error(f"Failed to parse Azure LLM tool arguments: {json_err}")
                        response_text = "There was an issue processing the device command."
                        emotion = "sad"
                    except Exception as e:
                         logger.error(f"Error processing tool call: {e}")
                         response_text = "Something went wrong with the device control."
                         emotion = "sad"
                else:
                    logger.warning(f"Unhandled tool call: {tool_call.function.name}")
                    # Fallback if an unexpected tool is called
                    response_text = "I received an unexpected request."
                    emotion = "worried"

            # No tool calls, process normal response
            elif response_message.content:
                response_text = response_message.content
                # For Azure, determine emotion using Text Analytics
                emotion = await self._analyze_emotion(response_text)

            logger.info(f"Azure LLM Response: '{response_text}', Emotion: {emotion}, IoT: {iot_commands}")
            return response_text, emotion, iot_commands

        except Exception as e:
            logger.error(f"Azure LLM service error: {e}", exc_info=True)
            raise exceptions.ServiceError(f"LLM failed: {e}")


# Initialize the appropriate LLM service based on configuration
if hasattr(config, "LLM_SERVICE") and config.LLM_SERVICE == "azure":
    # Initialize Azure Text Analytics client (optional)
    text_analytics_client = None
    if hasattr(config, "AZURE_TEXT_ANALYTICS_KEY") and hasattr(config, "AZURE_TEXT_ANALYTICS_ENDPOINT"):
        try:
            text_analytics_client = TextAnalyticsClient(
                endpoint=config.AZURE_TEXT_ANALYTICS_ENDPOINT,
                credential=AzureKeyCredential(config.AZURE_TEXT_ANALYTICS_KEY)
            )
            logger.info("Azure Text Analytics client initialized for emotion detection")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure Text Analytics client: {e}")
    
    # Initialize the Azure OpenAI client
    azure_openai_client = get_azure_openai_client()
    llm_service: BaseLLMService = AzureLLMService(azure_openai_client, text_analytics_client)
else:
    llm_service: BaseLLMService = OpenAILLMService(openai_client)