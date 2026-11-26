"""HTTP client with retry logic for Fact Agent (stdlib + requests)."""

import json
import logging
import time
from typing import Any, Generator
from urllib import request, error

from fact_agent.config import get_settings
from fact_agent.exceptions import LLMAPIError, RenderAPIError, AuthenticationError
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)


class HTTPClient:
    """HTTP client with automatic retries and error handling."""

    def __init__(self, timeout: int = 60, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with retries."""
        req = request.Request(url, data=data, headers=headers or {}, method=method)

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                    return json.loads(body) if body else {}
            except error.HTTPError as e:
                body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 401:
                    raise AuthenticationError(f"Authentication failed: {body}") from e
                if e.code >= 500 and attempt < self.max_retries - 1:
                    last_exception = LLMAPIError(
                        f"HTTP {e.code}: {e.reason}",
                        status_code=e.code,
                        response_body=body,
                    )
                    wait_time = (2 ** attempt) + 0.1 * attempt
                    logger.warning("Server error, retrying", extra={"attempt": attempt + 1, "wait": wait_time})
                    time.sleep(wait_time)
                    continue
                raise LLMAPIError(
                    f"HTTP {e.code}: {e.reason}",
                    status_code=e.code,
                    response_body=body,
                ) from e
            except error.URLError as e:
                last_exception = LLMAPIError(f"Network error: {e.reason}")
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + 0.1 * attempt
                    logger.warning("Network error, retrying", extra={"attempt": attempt + 1, "wait": wait_time})
                    time.sleep(wait_time)
                    continue
                raise last_exception from e
            except TimeoutError:
                last_exception = LLMAPIError("Request timed out")
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + 0.1 * attempt
                    logger.warning("Timeout, retrying", extra={"attempt": attempt + 1, "wait": wait_time})
                    time.sleep(wait_time)
                    continue
                raise last_exception from None
            except json.JSONDecodeError as e:
                raise LLMAPIError(f"Invalid JSON response: {e}") from e

        # Should not reach here, but just in case
        raise last_exception or LLMAPIError("Request failed after retries")

    def post_json(self, url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
        """POST JSON payload."""
        h = {"Content-Type": "application/json", **(headers or {})}
        data = json.dumps(payload).encode("utf-8")
        return self._make_request(url, "POST", h, data)

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> dict:
        """GET JSON response."""
        return self._make_request(url, "GET", headers)


class LLMClient(HTTPClient):
    """LLM API client."""

    def __init__(self, api_url: str, api_key: str, model: str = "auto", timeout: int = 60):
        super().__init__(timeout=timeout)
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat completion request and return assistant message content."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.debug("LLM request", extra={"messages": len(messages), "model": self.model})
        response = self.post_json(self.api_url, payload, self._auth_headers())

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("Unexpected LLM response format", extra={"response": response})
            raise LLMAPIError("Invalid response format from LLM") from e

    def chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_tool_calls: int = 5,
    ) -> str:
        """Send chat completion with tool support and handle tool calls."""
        from fact_agent.tools import execute_tool, get_tool_registry

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("LLM request with tools", extra={"messages": len(messages), "tools": len(tools) if tools else 0})

        tool_calls_made = 0
        current_messages = messages.copy()

        while tool_calls_made < max_tool_calls:
            response = self.post_json(self.api_url, payload, self._auth_headers())

            try:
                choice = response["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError) as e:
                logger.error("Unexpected LLM response format", extra={"response": response})
                raise LLMAPIError("Invalid response format from LLM") from e

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                # No tool calls, return content
                content = message.get("content", "")
                return content

            # Add assistant message with tool calls to conversation
            current_messages.append(message)

            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                logger.info("Executing tool", extra={"tool": function_name, "tool_args": function_args})

                # Execute tool
                result = execute_tool(function_name, **function_args)

                # Add tool result to conversation
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result.content,
                })

                tool_calls_made += 1

            # Update payload for next iteration
            payload["messages"] = current_messages

        # Max tool calls reached, return last content
        return current_messages[-1].get("content", "") if current_messages else ""

    def chat_completion_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Stream chat completion tokens as they arrive.

        Yields:
            Content chunks (tokens) as strings.
        """
        import urllib.request

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("Streaming LLM request", extra={"messages": len(messages)})

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                # Read streaming response line by line
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: "
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            logger.error("Streaming HTTP error", extra={"status": e.code, "body": body})
            raise
        except urllib.error.URLError as e:
            logger.error("Streaming network error", exc_info=e)
            raise
        except Exception as e:
            logger.error("Streaming error", exc_info=e)
            raise


class RenderClient(HTTPClient):
    """JANCTION render API client."""

    def __init__(self, api_base: str, api_key: str, timeout: int = 30):
        super().__init__(timeout=timeout)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _request(self, endpoint: str, method: str = "GET", payload: dict | None = None) -> dict:
        url = f"{self.api_base}{endpoint}"
        headers = self._auth_headers()
        if payload:
            headers["Content-Type"] = "application/json"
        if method == "POST":
            return self.post_json(url, payload, headers)
        return self.get_json(url, headers)

    def check_account(self) -> dict:
        """Get account info and limits."""
        return self._request("/v1/me")

    def estimate_preview(
        self,
        frame_start: int = 1,
        frame_end: int = 1,
        width: int = 640,
        height: int = 360,
        samples: int = 16,
    ) -> dict:
        """Request a preview render cost estimate."""
        payload = {
            "kind": "preview",
            "frame_start": frame_start,
            "frame_end": frame_end,
            "width": width,
            "height": height,
            "samples": samples,
        }
        return self._request("/v1/estimate", "POST", payload)