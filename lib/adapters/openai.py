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

        if messages[-1].get("role") == "tool":
            logger.debug("infer_initiator: last message role=tool -> agent")
            return "agent"

        # Prior tool activity means this is an agentic follow-up.
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                logger.debug(
                    "infer_initiator: prior tool message in history, "
                    "last role=%s -> agent",
                    messages[-1].get("role"),
                )
                return "agent"
            if role == "assistant" and msg.get("tool_calls"):
                logger.debug(
                    "infer_initiator: prior tool_calls in history, "
                    "last role=%s -> agent",
                    messages[-1].get("role"),
                )
                return "agent"

        logger.debug(
            "infer_initiator: no tool activity detected, %d messages, "
            "last role=%s -> user",
            len(messages),
            messages[-1].get("role"),
        )
        return "user"
