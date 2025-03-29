class AuthenticationError(Exception):
    """Custom exception for authentication failures."""
    pass

class ProtocolError(Exception):
    """Custom exception for WebSocket protocol violations."""
    pass

class ServiceError(Exception):
    """Custom exception for external AI/IoT service failures."""
    pass