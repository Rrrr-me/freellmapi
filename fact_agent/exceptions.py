"""Custom exceptions for Fact Agent."""


class FactAgentError(Exception):
    """Base exception for Fact Agent errors."""
    pass


class ConfigurationError(FactAgentError):
    """Configuration-related errors."""
    pass


class MemoryError(FactAgentError):
    """Memory system errors."""
    pass


class MemoryExtractionError(MemoryError):
    """Fact extraction failures."""
    pass


class MemoryRetrievalError(MemoryError):
    """Memory retrieval failures."""
    pass


class LLMAPIError(FactAgentError):
    """LLM API communication errors."""
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RenderError(FactAgentError):
    """Cloud rendering errors."""
    pass


class RenderAPIError(RenderError):
    """JANCTION API errors."""
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(FactAgentError):
    """Authentication/authorization errors."""
    pass