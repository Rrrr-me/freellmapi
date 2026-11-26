"""Agent tools: web search, calculator, file operations, etc."""

import json
import logging
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from fact_agent.config import get_settings
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    content: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Tool(ABC):
    """Base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (for function calling)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description (for LLM)."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def to_function_spec(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class WebSearchTool(Tool):
    """Web search using Bing HTML (no API key required)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information, facts, or verification. Returns top results with snippets."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """Execute web search via Bing HTML."""
        try:
            url = "https://www.bing.com/search"
            params = {"q": query, "setlang": "en-US"}
            query_string = urllib.parse.urlencode(params)

            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            req = urllib.request.Request(f"{url}?{query_string}", headers=headers, method="GET")

            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8")

            # Parse results from HTML
            results = self._parse_bing(html, max_results)

            if not results:
                return ToolResult(
                    success=True,
                    content="No results found for query.",
                    metadata={"query": query, "results": []},
                )

            # Format results
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. **{r['title']}**\n"
                    f"   URL: {r['url']}\n"
                    f"   Snippet: {r['snippet']}"
                )

            content = f"Web search results for '{query}':\n\n" + "\n\n".join(formatted)

            return ToolResult(
                success=True,
                content=content,
                metadata={"query": query, "results": results},
            )

        except Exception as e:
            logger.error("Web search failed", exc_info=e)
            return ToolResult(
                success=False,
                content=f"Search failed: {e}",
                metadata={"query": query, "error": str(e)},
            )

    def _parse_bing(self, html: str, max_results: int) -> list[dict]:
        """Parse Bing HTML results."""
        results = []

        # Find all b_algo result items
        algo_blocks = re.findall(
            r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>',
            html,
            re.DOTALL
        )

        for block in algo_blocks[:max_results]:
            # Extract title and URL from h2 > a
            title_match = re.search(
                r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL
            )

            # Extract snippet from b_caption > p
            snippet_match = re.search(
                r'<div[^>]*class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>',
                block,
                re.DOTALL
            )

            if title_match:
                url = title_match.group(1)
                title = title_match.group(2)
                snippet = snippet_match.group(1) if snippet_match else ""

                # Unescape HTML entities in URL first
                import html
                url = html.unescape(url)

                # Clean HTML tags from title/snippet
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()

                # Extract real URL from Bing tracking URL
                real_url = self._extract_bing_url(url)
                url = real_url or url

                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet[:300],
                    })

        return results

    def _extract_bing_url(self, tracking_url: str) -> str | None:
        """Extract real URL from Bing tracking redirect."""
        # Bing tracking URLs have format: https://www.bing.com/ck/a?!&&p=...&u=a1aHR0cHM6Ly9...
        # The 'u' parameter contains base64 encoded real URL
        import base64
        import urllib.parse

        try:
            parsed = urllib.parse.urlparse(tracking_url)
            params = urllib.parse.parse_qs(parsed.query)

            if 'u' in params:
                encoded = params['u'][0]
                # URL-safe base64 decode - handle the a1 prefix which is Bing's encoding marker
                if encoded.startswith('a1'):
                    encoded = encoded[2:]  # Remove a1 prefix
                encoded = encoded.replace('-', '+').replace('_', '/')
                # Add padding if needed
                padding = 4 - len(encoded) % 4
                if padding != 4:
                    encoded += '=' * padding
                decoded = base64.urlsafe_b64decode(encoded).decode('utf-8')
                return decoded
        except Exception:
            pass

        return None


class CalculatorTool(Tool):
    """Safe calculator for mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions safely. Supports basic arithmetic, functions (sqrt, sin, cos, log, etc.), and constants (pi, e)."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 * pi * r', 'sqrt(144) + 5')",
                },
            },
            "required": ["expression"],
        }

    # Safe functions and constants
    SAFE_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum,
        "pi": 3.141592653589793,
        "e": 2.718281828459045,
        "tau": 6.283185307179586,
    }

    # Add math functions
    import math
    SAFE_NAMES.update({
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "degrees": math.degrees,
        "radians": math.radians,
        "factorial": math.factorial,
    })

    def execute(self, expression: str) -> ToolResult:
        """Safely evaluate mathematical expression."""
        try:
            # Validate expression - only allow safe characters
            if not re.match(r"^[0-9a-zA-Z_+\-*/()., ]+$", expression):
                return ToolResult(
                    success=False,
                    content="Invalid expression: only numbers, operators, and safe functions allowed",
                    metadata={"expression": expression},
                )

            # Evaluate with safe namespace
            result = eval(expression, {"__builtins__": {}}, self.SAFE_NAMES)

            return ToolResult(
                success=True,
                content=f"{expression} = {result}",
                metadata={"expression": expression, "result": result},
            )

        except ZeroDivisionError:
            return ToolResult(
                success=False,
                content="Error: Division by zero",
                metadata={"expression": expression},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Calculation error: {e}",
                metadata={"expression": expression, "error": str(e)},
            )


class FileOpsTool(Tool):
    """File operations within workspace."""

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def description(self) -> str:
        return "Read, write, list, or delete files in the workspace. Paths are relative to the project directory."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete", "exists"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (relative to workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)",
                },
            },
            "required": ["operation", "path"],
        }

    def __init__(self, workspace: Path = None):
        self.workspace = workspace or Path.cwd()

    def execute(self, operation: str, path: str, content: str = None) -> ToolResult:
        """Execute file operation."""
        try:
            # Resolve path within workspace
            target = (self.workspace / path).resolve()

            # Security: ensure path is within workspace
            try:
                target.relative_to(self.workspace.resolve())
            except ValueError:
                return ToolResult(
                    success=False,
                    content="Error: Path must be within workspace",
                    metadata={"path": path},
                )

            if operation == "read":
                if not target.is_file():
                    return ToolResult(
                        success=False,
                        content=f"File not found: {path}",
                        metadata={"path": path},
                    )
                text = target.read_text(encoding="utf-8")
                return ToolResult(
                    success=True,
                    content=text,
                    metadata={"path": path, "size": len(text)},
                )

            elif operation == "write":
                if content is None:
                    return ToolResult(
                        success=False,
                        content="Error: content required for write",
                        metadata={"path": path},
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return ToolResult(
                    success=True,
                    content=f"Written {len(content)} chars to {path}",
                    metadata={"path": path, "size": len(content)},
                )

            elif operation == "list":
                if not target.exists():
                    return ToolResult(
                        success=False,
                        content=f"Path not found: {path}",
                        metadata={"path": path},
                    )
                if target.is_file():
                    items = [target.name]
                else:
                    items = sorted([f.name for f in target.iterdir()])
                return ToolResult(
                    success=True,
                    content="\n".join(items) if items else "(empty)",
                    metadata={"path": path, "count": len(items)},
                )

            elif operation == "delete":
                if not target.exists():
                    return ToolResult(
                        success=False,
                        content=f"Path not found: {path}",
                        metadata={"path": path},
                    )
                if target.is_file():
                    target.unlink()
                else:
                    import shutil
                    shutil.rmtree(target)
                return ToolResult(
                    success=True,
                    content=f"Deleted {path}",
                    metadata={"path": path},
                )

            elif operation == "exists":
                return ToolResult(
                    success=True,
                    content=f"Exists: {target.exists()}",
                    metadata={"path": path, "exists": target.exists()},
                )

            else:
                return ToolResult(
                    success=False,
                    content=f"Unknown operation: {operation}",
                    metadata={"operation": operation},
                )

        except Exception as e:
            logger.error("File operation failed", exc_info=e)
            return ToolResult(
                success=False,
                content=f"File operation failed: {e}",
                metadata={"operation": operation, "path": path, "error": str(e)},
            )


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self, workspace: Path = None):
        self.tools: dict[str, Tool] = {}
        self.workspace = workspace or Path.cwd()
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in tools."""
        self.register(WebSearchTool())
        self.register(CalculatorTool())
        self.register(FileOpsTool(self.workspace))

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.debug("Registered tool", extra={"name": tool.name})

    def unregister(self, name: str):
        """Unregister a tool."""
        self.tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def get_function_specs(self) -> list[dict]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_function_spec() for tool in self.tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                content=f"Unknown tool: {name}",
                metadata={"tool": name},
            )
        return tool.execute(**kwargs)


# Global registry instance
_default_registry: ToolRegistry | None = None


def get_tool_registry(workspace: Path = None) -> ToolRegistry:
    """Get or create global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(workspace)
        # Register advanced tools
        _default_registry.register(CodeExecutionTool())
        _default_registry.register(APIRequestTool())
    return _default_registry


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool using global registry."""
    return get_tool_registry().execute(name, **kwargs)


class CodeExecutionTool(Tool):
    """Python code execution in sandbox."""

    @property
    def name(self) -> str:
        return "execute_python"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed environment. Returns stdout, stderr, and execution time. Use for calculations, data processing, testing code snippets."

    @property
    def parameters(self) -> dict:
        return {
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
        }

    def execute(self, code: str, timeout: int = 30) -> ToolResult:
        from fact_agent.advanced_tools import execute_python
        result = execute_python(code, timeout=timeout)
        if result["success"]:
            content = f"Exit code: {result['return_code']}\n"
            if result["stdout"]:
                content += f"STDOUT:\n{result['stdout']}\n"
            if result["stderr"]:
                content += f"STDERR:\n{result['stderr']}\n"
            content += f"Execution time: {result['execution_time_ms']:.1f}ms"
            return ToolResult(success=True, content=content, metadata=result)
        else:
            return ToolResult(
                success=False,
                content=f"Execution failed (exit {result['return_code']}):\n{result['stderr']}",
                metadata=result,
            )


class APIRequestTool(Tool):
    """HTTP API requests."""

    @property
    def name(self) -> str:
        return "api_request"

    @property
    def description(self) -> str:
        return "Make HTTP API requests to external services. Supports GET, POST, PUT, DELETE with JSON payloads."

    @property
    def parameters(self) -> dict:
        return {
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
        }

    def execute(
        self,
        method: str,
        url: str,
        headers: dict = None,
        params: dict = None,
        json_data: dict = None,
        timeout: int = 30,
    ) -> ToolResult:
        from fact_agent.advanced_tools import api_request
        result = api_request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json_data=json_data,
            timeout=timeout,
        )
        if result["success"]:
            import json
            content = f"Status: {result['status_code']}\n"
            content += f"Response:\n{json.dumps(result['response_data'], indent=2)}"
            return ToolResult(success=True, content=content, metadata=result)
        else:
            return ToolResult(
                success=False,
                content=f"Request failed (status {result['status_code']}): {result['error']}",
                metadata=result,
            )