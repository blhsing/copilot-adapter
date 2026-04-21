"""OpenAI format adapter (passthrough)."""

import logging

from .base import FormatAdapter, StreamConverter

logger = logging.getLogger(__name__)


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

    def infer_initiator(self, body: dict) -> str:
        messages = body.get("messages", [])
        if not messages:
            return "user"

        # Only the last message is a reliable signal. A role=tool message
        # means the client is auto-continuing after a tool call. Prior
        # tool activity in history does NOT imply the current turn is
        # agent-initiated — chat UIs re-send full history every turn.
        if messages[-1].get("role") == "tool":
            logger.debug("infer_initiator: last message role=tool -> agent")
            return "agent"

        logger.debug(
            "infer_initiator: last message role=%s -> user (%d messages)",
            messages[-1].get("role"),
            len(messages),
        )
        return "user"
