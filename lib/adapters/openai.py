"""OpenAI format adapter (passthrough)."""

from .base import FormatAdapter, StreamConverter


class _OpenAIStreamConverter(StreamConverter):
    def feed(self, line: str) -> str:
        if line.strip():
            return f"{line}\n"
        if line == "":
            return "\n"
        return ""

    def format_error(self, error_line: str) -> str:
        return f"data: {error_line}\n\n"


class OpenAIAdapter(FormatAdapter):
    """Passthrough adapter — the Copilot API already speaks OpenAI format."""

    def convert_chat_request(self, body: dict) -> dict:
        return body

    def convert_chat_response(self, openai_resp: dict, original_body: dict) -> dict:
        return openai_resp

    def create_stream_converter(self, original_body: dict) -> StreamConverter:
        return _OpenAIStreamConverter()

    def is_streaming(self, body: dict) -> bool:
        return bool(body.get("stream"))

    def convert_models_response(self, openai_resp: dict) -> dict:
        return openai_resp
