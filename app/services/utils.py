import os

import httpx
from openai import AsyncOpenAI
from azure.cognitiveservices.speech import SpeechConfig

from app.logger import logger

from app import config


def create_openai_client() -> AsyncOpenAI:
    """Creates and configures the AsyncOpenAI client."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured.")

    http_client = None
    proxy_url = getattr(config, 'HTTP_PROXY', None) or os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')

    if proxy_url:
        logger.info(f"Using HTTP proxy for OpenAI requests: {proxy_url}")
        # Consider adding timeouts: httpx.Timeout(30.0, connect=10.0)
        http_client = httpx.AsyncClient(proxy=proxy_url)
    else:
        logger.info("No HTTP proxy configured for OpenAI requests.")

    try:
        client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL, # Optional
            http_client=http_client
        )
        logger.info("AsyncOpenAI client created successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        # Clean up httpx client if created, though tricky in async context here
        # Rely on higher-level shutdown or garbage collection
        raise ValueError(f"Failed to initialize OpenAI client: {e}")


def get_azure_openai_client() -> AsyncOpenAI:
    """Creates and configures the Azure OpenAI client."""
    if not hasattr(config, "AZURE_OPENAI_KEY") or not config.AZURE_OPENAI_KEY:
        raise ValueError("AZURE_OPENAI_KEY not configured.")
    
    if not hasattr(config, "AZURE_OPENAI_ENDPOINT") or not config.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not configured.")
    
    if not hasattr(config, "AZURE_OPENAI_API_VERSION") or not config.AZURE_OPENAI_API_VERSION:
        logger.warning("AZURE_OPENAI_API_VERSION not specified, using default '2023-05-15'")
        api_version = "2023-05-15"
    else:
        api_version = config.AZURE_OPENAI_API_VERSION

    http_client = None
    proxy_url = getattr(config, 'HTTP_PROXY', None) or os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')

    if proxy_url:
        logger.info(f"Using HTTP proxy for Azure OpenAI requests: {proxy_url}")
        http_client = httpx.AsyncClient(proxy=proxy_url)
    else:
        logger.info("No HTTP proxy configured for Azure OpenAI requests.")

    try:
        # Azure OpenAI requires a different setup
        client = AsyncOpenAI(
            api_key=config.AZURE_OPENAI_KEY,  
            base_url=f"{config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{config.AZURE_DEPLOYMENT_NAME}",
            api_version=api_version,
            http_client=http_client
        )
        logger.info(f"Azure OpenAI client created successfully for deployment {config.AZURE_DEPLOYMENT_NAME}.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize Azure OpenAI client: {e}")


def get_azure_speech_config() -> SpeechConfig:
    """Creates and configures the Azure Speech config."""
    if not hasattr(config, "AZURE_SPEECH_KEY") or not config.AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY not configured.")
    
    if not hasattr(config, "AZURE_SPEECH_REGION") or not config.AZURE_SPEECH_REGION:
        raise ValueError("AZURE_SPEECH_REGION not configured.")

    try:
        speech_config = SpeechConfig(subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION)
        
        # Set language properties using the updated settings
        speech_config.speech_recognition_language = getattr(config, "AZURE_SPEECH_RECOGNITION_LANGUAGE", config.DEFAULT_LANGUAGE)
        speech_config.speech_synthesis_language = getattr(config, "AZURE_SPEECH_SYNTHESIS_LANGUAGE", config.DEFAULT_LANGUAGE)
        
        logger.info(f"Azure Speech config created for region {config.AZURE_SPEECH_REGION} with language {speech_config.speech_recognition_language}")
        return speech_config
    except Exception as e:
        logger.error(f"Failed to initialize Azure Speech config: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize Azure Speech config: {e}")


openai_client = create_openai_client()