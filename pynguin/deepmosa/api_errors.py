class APIError(Exception):
    """Base exception for all OpenAI API errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class APIUnknownError(APIError):
    def __init__(self):
        super().__init__("Failed to send message to GPT")

class APILengthError(APIError):
    def __init__(self):
        super().__init__("Conversation was too long for context window")

class APIRefusalError(APIError):
    def __init__(self, message = None):
        super().__init__(message or "OpenAI API safety system fault")

class APIContentFilterError(APIError):
    def __init__(self):
        super().__init__("Model's output included restricted content")
