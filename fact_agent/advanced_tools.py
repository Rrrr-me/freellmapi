"""Advanced agent tools: code execution, API calls, structured extraction."""

import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time_ms: float


@dataclass
class APICallResult:
    """Result of API call."""
    success: bool
    status_code: int
    response_data: Any
    error: Optional[str] = None


class PythonSandbox:
    """Safe Python code execution in isolated subprocess."""

    def __init__(
        self,
        timeout: int = 30,
        memory_limit_mb: int = 256,
        allowed_imports: Optional[list[str]] = None,
        blocked_imports: Optional[list[str]] = None,
    ):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.allowed_imports = allowed_imports or []
        self.blocked_imports = blocked_imports or [
            "os", "sys", "subprocess", "shutil", "pathlib",
            "importlib", "pkgutil", "runpy", "ctypes", "cffi",
            "socket", "http", "urllib", "requests", "ftplib",
            "telnetlib", "poplib", "imaplib", "smtplib",
        ]

    def _validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """Check code for blocked imports."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.blocked_imports:
                        return False, f"Blocked import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(
                    node.module.startswith(b) or node.module == b
                    for b in self.blocked_imports
                ):
                    return False, f"Blocked import: {node.module}"

        return True, None

    def execute(self, code: str, capture_output: bool = True) -> ExecutionResult:
        """Execute Python code in sandboxed subprocess."""
        # Validate
        valid, error = self._validate_code(code)
        if not valid:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=error or "Validation failed",
                return_code=-1,
                execution_time_ms=0,
            )

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            start = time.time()
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=capture_output,
                text=True,
                timeout=self.timeout,
            )
            elapsed = (time.time() - start) * 1000

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                return_code=result.returncode,
                execution_time_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout}s",
                return_code=-1,
                execution_time_ms=self.timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution error: {e}",
                return_code=-1,
                execution_time_ms=0,
            )
        finally:
            # Clean up
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


class APIClient:
    """HTTP API client for external service calls."""

    def __init__(
        self,
        default_timeout: int = 30,
        max_retries: int = 3,
        default_headers: Optional[dict[str, str]] = None,
    ):
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {
            "User-Agent": "FactAgent/1.0",
            "Accept": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        data: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> APICallResult:
        """Make HTTP request with retries."""
        req_headers = self.default_headers.copy()
        if headers:
            req_headers.update(headers)

        req_timeout = timeout or self.default_timeout

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=req_headers,
                    params=params,
                    json=json_data,
                    data=data,
                    timeout=req_timeout,
                )

                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = response.text

                return APICallResult(
                    success=200 <= response.status_code < 300,
                    status_code=response.status_code,
                    response_data=response_data,
                    error=None if response.ok else f"HTTP {response.status_code}",
                )

            except requests.Timeout:
                last_error = "Request timed out"
            except requests.ConnectionError as e:
                last_error = f"Connection error: {e}"
            except Exception as e:
                last_error = f"Request error: {e}"

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        return APICallResult(
            success=False,
            status_code=0,
            response_data=None,
            error=last_error,
        )

    def get(self, url: str, **kwargs) -> APICallResult:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> APICallResult:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> APICallResult:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> APICallResult:
        return self.request("DELETE", url, **kwargs)


class StructuredExtractor:
    """Extract structured data from text using LLM."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(
        self,
        text: str,
        schema: dict,
        instructions: str = "",
    ) -> dict:
        """Extract structured data matching schema."""
        import json

        prompt = f"""Extract structured data from the following text according to the JSON schema.

Schema:
{json.dumps(schema, indent=2)}

{instructions}

Text:
{text}

Return ONLY valid JSON matching the schema. No extra text."""

        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a precise data extraction engine. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            # Parse JSON from response
            start = response.find("{")
            end = response.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON object in response")

            return json.loads(response[start:end + 1])

        except Exception as e:
            logger.error("Structured extraction failed", exc_info=e)
            raise


# Tool wrappers for agent integration
_sandbox = PythonSandbox()
_api_client = APIClient()


def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in sandbox.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        Dict with success, stdout, stderr, return_code, execution_time_ms
    """
    global _sandbox
    if timeout != _sandbox.timeout:
        _sandbox = PythonSandbox(timeout=timeout)

    result = _sandbox.execute(code)
    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
        "execution_time_ms": result.execution_time_ms,
    }


def api_request(
    method: str,
    url: str,
    headers: dict = None,
    params: dict = None,
    json_data: dict = None,
    timeout: int = 30,
) -> dict:
    """Make HTTP API request.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        url: Request URL
        headers: Optional headers
        params: Query parameters
        json_data: JSON body
        timeout: Request timeout

    Returns:
        Dict with success, status_code, response_data, error
    """
    result = _api_client.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        json_data=json_data,
        timeout=timeout,
    )
    return {
        "success": result.success,
        "status_code": result.status_code,
        "response_data": result.response_data,
        "error": result.error,
    }


def extract_structured(text: str, schema: dict, instructions: str = "") -> dict:
    """Extract structured data from text (requires LLM client).

    Note: This needs an LLM client to be passed. Use the agent's LLM client.

    Args:
        text: Text to extract from
        schema: JSON schema for extraction
        instructions: Additional extraction instructions

    Returns:
        Extracted data matching schema
    """
    raise NotImplementedError("Requires LLM client - use agent's extractor")


# Tool specifications for LLM function calling
PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code in a sandboxed environment. Returns stdout, stderr, and execution time. Use for calculations, data processing, testing code snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    },
}

API_REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "api_request",
        "description": "Make HTTP API requests to external services. Supports GET, POST, PUT, DELETE with JSON payloads.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "description": "HTTP method",
                },
                "url": {
                    "type": "string",
                    "description": "Request URL",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers",
                    "additionalProperties": {"type": "string"},
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters",
                    "additionalProperties": {"type": "string"},
                },
                "json_data": {
                    "type": "object",
                    "description": "JSON request body",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["method", "url"],
        },
    },
}

STRUCTURED_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_structured",
        "description": "Extract structured data from unstructured text using a JSON schema. Requires LLM - call via agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to extract from",
                },
                "schema": {
                    "type": "object",
                    "description": "JSON schema for the extracted data",
                },
                "instructions": {
                    "type": "string",
                    "description": "Additional extraction instructions",
                },
            },
            "required": ["text", "schema"],
        },
    },
}

ALL_ADVANCED_TOOLS = [
    PYTHON_EXEC_TOOL,
    API_REQUEST_TOOL,
    STRUCTURED_EXTRACT_TOOL,
]