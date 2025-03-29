import os

import httpx
from openai import AsyncOpenAI

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

openai_client = create_openai_client()