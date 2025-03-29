import aiohttp

from app.logger import logger
from app.services.base import BaseAuthService
from app import config

class SimpleAuthService(BaseAuthService):
    """Simple token validation (replace with real service call)."""
    async def verify_token(self, token: str) -> bool:
        logger.warning("Using simple in-memory token validation. NOT FOR PRODUCTION.")
        is_valid = token in config.VALID_TOKENS
        if not is_valid:
            logger.warning(f"Invalid token received: {token[:10]}...") # Log prefix only
        return is_valid

class RemoteAuthService(BaseAuthService):
    """Example of calling a remote auth service."""
    async def verify_token(self, token: str) -> bool:
        if not config.AUTH_SERVICE_URL:
            logger.error("AUTH_SERVICE_URL is not configured.")
            return False
        try:
            async with aiohttp.ClientSession() as session:
                # Adjust headers/payload as needed for your auth service
                headers = {"Authorization": f"Bearer {token}"}
                async with session.post(config.AUTH_SERVICE_URL, headers=headers) as response:
                    logger.debug(f"Auth service response status: {response.status}")
                    return response.status == 200 # Or check response body
        except aiohttp.ClientError as e:
            logger.error(f"Error connecting to auth service: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            return False

# Choose the implementation (use RemoteAuthService in a real scenario)
auth_service: BaseAuthService = SimpleAuthService()
# auth_service: BaseAuthService = RemoteAuthService()