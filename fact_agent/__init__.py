"""Fact Agent - A fact-based AI agent with persistent memory and cloud rendering."""

__version__ = "0.8.0"
__author__ = "Fact Agent Developer"
__license__ = "MIT"

# Main exports
from .config import get_settings, Settings
from .http_client import LLMClient, RenderClient
from .memory_extractor import extract_memories, store_memories, extract_and_store
from .memory_retriever import retrieve_memories, keyword_retrieval
from .vector_memory import VectorMemory, retrieve_vector_memories
from .memory_consolidator import MemoryConsolidator, MemoryJanitor, TTLPolicy, ConsolidationResult, get_categories, categorize_fact
from .blender_scenes import (
    SceneConfig,
    SceneType,
    CameraType,
    RenderEngine,
    BlenderSceneGenerator,
    MysteryForestScene,
    AbstractGeometryScene,
    ProductShowcaseScene,
    DataVisualizationScene,
    get_scene_generator,
    list_available_scenes,
    generate_scene_script,
    SCENE_REGISTRY,
)
from .cloud_renderer import CloudRenderer, RenderConfig, RenderJob, JobStatus
from .render_pipeline import RenderPipeline
from .tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    WebSearchTool,
    CalculatorTool,
    FileOpsTool,
    get_tool_registry,
    execute_tool,
)

__all__ = [
    # Config
    "get_settings",
    "Settings",
    # HTTP
    "LLMClient",
    "RenderClient",
    # Memory
    "extract_memories",
    "store_memories",
    "extract_and_store",
    "retrieve_memories",
    "keyword_retrieval",
    "VectorMemory",
    "retrieve_vector_memories",
    # Memory Consolidation
    "MemoryConsolidator",
    "MemoryJanitor",
    "TTLPolicy",
    "ConsolidationResult",
    "get_categories",
    "categorize_fact",
    # Blender Scenes
    "SceneConfig",
    "SceneType",
    "CameraType",
    "RenderEngine",
    "BlenderSceneGenerator",
    "MysteryForestScene",
    "AbstractGeometryScene",
    "ProductShowcaseScene",
    "DataVisualizationScene",
    "get_scene_generator",
    "list_available_scenes",
    "generate_scene_script",
    "SCENE_REGISTRY",
    # Rendering
    "CloudRenderer",
    "RenderConfig",
    "RenderJob",
    "JobStatus",
    "RenderPipeline",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "WebSearchTool",
    "CalculatorTool",
    "FileOpsTool",
    "get_tool_registry",
    "execute_tool",
]