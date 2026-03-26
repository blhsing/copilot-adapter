"""Abstract base classes for API format adapters."""

from abc import ABC, abstractmethod


class StreamConverter(ABC):
    """Stateful converter that transforms OpenAI SSE lines to another format."""

    @abstractmethod
    def feed(self, line: str) -> str:
        """Process one SSE line from the OpenAI stream. Return formatted output or empty string."""

    @abstractmethod
    def format_error(self, error_line: str) -> str:
        """Format an error line for this stream format."""


class FormatAdapter(ABC):
    """Translates between a specific API format and OpenAI's chat/completions format."""

    @abstractmethod
    def convert_chat_request(self, body: dict) -> dict:
        """Convert an incoming request body to OpenAI chat/completions format."""

    @abstractmethod
    def convert_chat_response(self, openai_resp: dict, original_body: dict) -> dict:
        """Convert an OpenAI chat/completions response back to this format."""

    @abstractmethod
    def create_stream_converter(self, original_body: dict) -> StreamConverter:
        """Create a stateful stream converter for this format."""

    @abstractmethod
    def is_streaming(self, body: dict) -> bool:
        """Return True if the request body indicates streaming."""

    @abstractmethod
    def convert_models_response(self, openai_resp: dict) -> dict:
        """Convert an OpenAI model list response to this format."""

    def infer_initiator(self, body: dict) -> str:
        """Infer 'user' or 'agent' from the request body.

        Returns 'agent' if the conversation appears to be an agentic
        follow-up (e.g. the last message is a tool result), 'user' otherwise.
        """
        return "user"
