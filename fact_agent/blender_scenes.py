"""Blender Scene Library for Fact Agent - Parameterized scenes and composition."""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)


class SceneType(Enum):
    """Predefined scene types."""
    MYSTERY_FOREST = "mystery_forest"
    ABSTRACT_GEOMETRY = "abstract_geometry"
    PRODUCT_SHOWCASE = "product_showcase"
    DATA_VISUALIZATION = "data_visualization"
    SHIP_OF_THESEUS = "ship_of_theseus"
    MINIMAL_STUDIO = "minimal_studio"
    CYBERPUNK_ALLEY = "cyberpunk_alley"
    NATURE_TIMELAPSE = "nature_timelapse"
    TITLE_SEQUENCE = "title_sequence"


class CameraType(Enum):
    """Camera types."""
    CINEMATIC = "cinematic"
    ORTHOGRAPHIC = "orthographic"
    PANORAMIC = "panoramic"
    DRONE = "drone"
    HANDHELD = "handheld"


class RenderEngine(Enum):
    """Render engines."""
    CYCLES = "CYCLES"
    EEVEE = "EEVEE"
    WORKBENCH = "WORKBENCH"


@dataclass
class SceneConfig:
    """Configuration for a scene."""
    # Resolution
    resolution_x: int = 1080
    resolution_y: int = 1920
    resolution_percentage: int = 100

    # Frame range
    frame_start: int = 1
    frame_end: int = 900
    fps: int = 30

    # Render settings
    engine: RenderEngine = RenderEngine.CYCLES
    samples: int = 128
    use_denoising: bool = True
    device: str = "GPU"  # GPU, CPU

    # Camera
    camera_type: CameraType = CameraType.CINEMATIC
    camera_location: tuple = (0, -13, 4.5)
    camera_rotation: tuple = (0, 0, 0)
    lens: float = 35

    # Lighting
    use_hdri: bool = False
    hdri_path: Optional[str] = None
    key_light_energy: float = 900
    fill_light_energy: float = 300

    # Output
    output_format: str = "PNG"
    output_quality: int = 90
    file_prefix: str = "render"

    # Custom parameters (scene-specific)
    custom_params: dict = field(default_factory=dict)


@dataclass
class SceneTemplate:
    """A reusable scene template."""
    name: str
    scene_type: SceneType
    description: str
    default_config: SceneConfig
    generate_func: str  # Name of generation function

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scene_type": self.scene_type.value,
            "description": self.description,
            "default_config": self.default_config.__dict__,
            "generate_func": self.generate_func,
        }


class BlenderSceneGenerator:
    """Generates Blender Python scripts for scenes."""

    def __init__(self, config: SceneConfig = None):
        self.config = config or SceneConfig()
        self.script_lines = []

    def _add(self, line: str = "", indent: int = 0):
        """Add a line to the script."""
        self.script_lines.append("    " * indent + line)

    def _add_import(self):
        """Add standard imports."""
        self._add("import bpy")
        self._add("import math")
        self._add("from mathutils import Vector, Euler")
        self._add("import random")
        self._add("")

    def _add_scene_setup(self):
        """Add scene initialization."""
        self._add("# ============================================================")
        self._add("# SCENE SETUP")
        self._add("# ============================================================")
        self._add("")

        # Clear scene
        self._add("# Clear existing scene")
        self._add("bpy.ops.object.select_all(action='SELECT')")
        self._add("bpy.ops.object.delete(use_global=False)")
        self._add("")

        # Scene settings
        self._add("scene = bpy.context.scene")
        self._add(f"scene.frame_start = {self.config.frame_start}")
        self._add(f"scene.frame_end = {self.config.frame_end}")
        self._add(f"scene.render.fps = {self.config.fps}")
        self._add("")

        # Render settings
        self._add("# Render settings")
        self._add(f"scene.render.engine = '{self.config.engine.value}'")
        self._add(f"scene.cycles.samples = {self.config.samples}")
        if self.config.use_denoising:
            self._add("scene.cycles.use_denoising = True")
        self._add(f"scene.cycles.device = '{self.config.device}'")
        self._add("")

        # Resolution
        self._add(f"scene.render.resolution_x = {self.config.resolution_x}")
        self._add(f"scene.render.resolution_y = {self.config.resolution_y}")
        self._add(f"scene.render.resolution_percentage = {self.config.resolution_percentage}")
        self._add("")

        # Output format
        self._add(f"scene.render.image_settings.file_format = '{self.config.output_format}'")
        if self.config.output_format == "JPEG":
            self._add(f"scene.render.image_settings.quality = {self.config.output_quality}")
        self._add("")

    def _add_materials(self):
        """Add standard materials."""
        self._add("# ============================================================")
        self._add("# MATERIALS")
        self._add("# ============================================================")
        self._add("")

        self._add("def make_material(name, color, roughness=0.5, metallic=0, emission=0, emission_color=None):")
        self._add("    mat = bpy.data.materials.new(name)")
        self._add("    mat.use_nodes = True")
        self._add("    nodes = mat.node_tree.nodes")
        self._add("    nodes.clear()")
        self._add("")
        self._add("    output = nodes.new('ShaderNodeOutputMaterial')")
        self._add("    principled = nodes.new('ShaderNodeBsdfPrincipled')")
        self._add("    principled.inputs['Base Color'].default_value = (*color, 1)")
        self._add("    principled.inputs['Roughness'].default_value = roughness")
        self._add("    principled.inputs['Metallic'].default_value = metallic")
        self._add("")
        self._add("    if emission > 0:")
        self._add("        principled.inputs['Emission Color'].default_value = (*(emission_color or color), 1)")
        self._add("        principled.inputs['Emission Strength'].default_value = emission")
        self._add("")
        self._add("    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])")
        self._add("    return mat")
        self._add("")

    def _add_world(self):
        """Add world environment."""
        self._add("# ============================================================")
        self._add("# WORLD ENVIRONMENT")
        self._add("# ============================================================")
        self._add("")

        if self.config.use_hdri and self.config.hdri_path:
            self._add("# HDRI Environment")
            self._add("world = bpy.data.worlds.new('HDRI_World')")
            self._add("scene.world = world")
            self._add("world.use_nodes = True")
            self._add("nodes = world.node_tree.nodes")
            self._add("nodes.clear()")
            self._add("")
            self._add("output = nodes.new('ShaderNodeOutputWorld')")
            self._add("bg = nodes.new('ShaderNodeBackground')")
            self._add("env = nodes.new('ShaderNodeTexEnvironment')")
            self._add(f"env.image = bpy.data.images.load('{self.config.hdri_path}')")
            self._add("mapping = nodes.new('ShaderNodeMapping')")
            self._add("texcoord = nodes.new('ShaderNodeTexCoord')")
            self._add("world.node_tree.links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])")
            self._add("world.node_tree.links.new(mapping.outputs['Vector'], env.inputs['Vector'])")
            self._add("world.node_tree.links.new(env.outputs['Color'], bg.inputs['Color'])")
            self._add("world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])")
        else:
            self._add("# Solid color background")
            self._add("world = bpy.data.worlds.new('Background')")
            self._add("scene.world = world")
            self._add("world.use_nodes = True")
            self._add("bg = world.node_tree.nodes.get('Background')")
            self._add("bg.inputs['Color'].default_value = (0.01, 0.01, 0.02, 1)")
            self._add("bg.inputs['Strength'].default_value = 0.1")
        self._add("")

    def _add_camera(self):
        """Add camera setup."""
        self._add("# ============================================================")
        self._add("# CAMERA")
        self._add("# ============================================================")
        self._add("")

        loc = self.config.camera_location
        rot = self.config.camera_rotation
        self._add(f"bpy.ops.object.camera_add(location={loc})")
        self._add("camera = bpy.context.object")
        self._add(f"camera.name = 'Main_Camera'")
        self._add(f"camera.data.lens = {self.config.lens}")
        self._add("")

        if rot != (0, 0, 0):
            self._add(f"camera.rotation_euler = {rot}")
        else:
            self._add("# Point camera at origin")
            self._add("target = Vector((0, 0, 0))")
            self._add("camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()")

        self._add("scene.camera = camera")
        self._add("")

    def _add_lighting(self):
        """Add standard lighting."""
        self._add("# ============================================================")
        self._add("# LIGHTING")
        self._add("# ============================================================")
        self._add("")

        # Key light
        self._add(f"bpy.ops.object.light_add(type='AREA', location=(0, -3, 7))")
        self._add("key_light = bpy.context.object")
        self._add("key_light.name = 'Key_Light'")
        self._add(f"key_light.data.energy = {self.config.key_light_energy}")
        self._add("key_light.data.shape = 'DISK'")
        self._add("key_light.data.size = 5")
        self._add("")

        # Fill light
        self._add(f"bpy.ops.object.light_add(type='AREA', location=(5, 3, 4))")
        self._add("fill_light = bpy.context.object")
        self._add("fill_light.name = 'Fill_Light'")
        self._add(f"fill_light.data.energy = {self.config.fill_light_energy}")
        self._add("fill_light.data.shape = 'RECTANGLE'")
        self._add("fill_light.data.size = 4")
        self._add("")

        # Rim light
        self._add("bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))")
        self._add("rim_light = bpy.context.object")
        self._add("rim_light.name = 'Rim_Light'")
        self._add("rim_light.data.energy = 200")
        self._add("rim_light.data.shape = 'RECTANGLE'")
        self._add("rim_light.data.size = 3")
        self._add("")

    def generate_script(self) -> str:
        """Generate the complete Blender script."""
        self.script_lines = []

        self._add("# Auto-generated Blender scene script")
        self._add(f"# Scene: {self.config.custom_params.get('scene_name', 'Custom')}")
        self._add(f"# Generated by Fact Agent Blender Scene Library")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Scene-specific generation (override in subclass)
        self._add("# ============================================================")
        self._add("# SCENE CONTENT")
        self._add("# ============================================================")
        self._add("")
        self._add("def create_scene_content():")
        self._add("    '''Override this in subclass'''")
        self._add("    pass")
        self._add("")
        self._add("create_scene_content()")
        self._add("")

        # Animation helpers
        self._add("# ============================================================")
        self._add("# ANIMATION")
        self._add("# ============================================================")
        self._add("")
        self._add("def animate_camera():")
        self._add("    '''Basic camera animation - override for custom motion'''")
        self._add("    camera = bpy.data.objects['Main_Camera']")
        self._add("    camera.keyframe_insert(data_path='location', frame=1)")
        self._add("    # Add custom animation here")
        self._add("")
        self._add("animate_camera()")
        self._add("")

        self._add("# Final setup")
        self._add("scene.frame_set(1)")
        self._add('print("SCENE GENERATED SUCCESSFULLY")')

        return "\n".join(self.script_lines)


class MysteryForestScene(BlenderSceneGenerator):
    """Mystery forest scene (original first_scene)."""

    def generate_script(self) -> str:
        self.script_lines = []

        self._add("# Auto-generated Mystery Forest Scene")
        self._add("# Based on original first_scene.py")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Scene content
        self._add("# ============================================================")
        self._add("# MYSTERY FOREST CONTENT")
        self._add("# ============================================================")
        self._add("")

        # Materials
        self._add("ground_mat = make_material('Dark forest ground', (0.018, 0.035, 0.045))")
        self._add("tree_mat = make_material('Shadow trees', (0.012, 0.025, 0.035))")
        self._add("orb_mat = make_material('Mysterious blue glow', (0.05, 0.55, 1.0), emission=5)")
        self._add("ring_mat = make_material('Ancient silver rings', (0.25, 0.55, 0.75), emission=1)")
        self._add("")

        # Ground
        self._add("# Ground plane")
        self._add("bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))")
        self._add("ground = bpy.context.object")
        self._add("ground.name = 'Forest floor'")
        self._add("ground.data.materials.append(ground_mat)")
        self._add("")

        # Trees
        self._add("# Tree rows")
        self._add("for side in (-1, 1):")
        self._add("    for i in range(9):")
        self._add("        y = -2 + i * 2.5")
        self._add("        x = side * (3.5 + (i % 3) * 0.7)")
        self._add("        ")
        self._add("        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.35, radius2=0.12, depth=5, location=(x, y, 2.5))")
        self._add("        trunk = bpy.context.object")
        self._add("        trunk.name = 'Dark tree trunk'")
        self._add("        trunk.data.materials.append(tree_mat)")
        self._add("        ")
        self._add("        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=1.25, radius2=0, depth=3, location=(x, y, 5.5))")
        self._add("        leaves = bpy.context.object")
        self._add("        leaves.name = 'Forest canopy'")
        self._add("        leaves.data.materials.append(tree_mat)")
        self._add("")

        # Orb
        self._add("# Glowing orb")
        self._add("bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.75, location=(0, 0, 2.5))")
        self._add("orb = bpy.context.object")
        self._add("orb.name = 'Mysterious floating orb'")
        self._add("orb.data.materials.append(orb_mat)")
        self._add("")

        # Rings
        self._add("# Floating rings")
        self._add("for radius in (1.05, 1.35):")
        self._add("    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.025, major_segments=48, minor_segments=8, location=(0, 0, 2.5))")
        self._add("    ring = bpy.context.object")
        self._add("    ring.name = 'Floating energy ring'")
        self._add("    ring.data.materials.append(ring_mat)")
        self._add("    ring.rotation_euler[0] = math.radians(65)")
        self._add("")

        # Animation
        self._add("# Orb rotation animation")
        self._add("orb.rotation_euler[2] = 0")
        self._add("orb.keyframe_insert(data_path='rotation_euler', frame=1)")
        self._add("orb.rotation_euler[2] = math.tau")
        self._add(f"orb.keyframe_insert(data_path='rotation_euler', frame={self.config.frame_end})")
        self._add("")

        # Camera animation
        self._add("# Camera dolly")
        self._add("camera = bpy.data.objects['Main_Camera']")
        self._add("camera.keyframe_insert(data_path='location', frame=1)")
        self._add("camera.location.y = -10")
        self._add(f"camera.keyframe_insert(data_path='location', frame={self.config.frame_end})")
        self._add("")

        self._add("scene.frame_set(1)")
        self._add('print("MYSTERY FOREST SCENE CREATED")')

        return "\n".join(self.script_lines)


class AbstractGeometryScene(BlenderSceneGenerator):
    """Abstract geometric animation."""

    def generate_script(self) -> str:
        self.script_lines = []

        self._add("# Auto-generated Abstract Geometry Scene")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Materials
        self._add("# Colorful materials")
        self._add("colors = [")
        self._add("    (1.0, 0.2, 0.2),  # Red")
        self._add("    (0.2, 1.0, 0.2),  # Green")
        self._add("    (0.2, 0.2, 1.0),  # Blue")
        self._add("    (1.0, 1.0, 0.2),  # Yellow")
        self._add("    (1.0, 0.2, 1.0),  # Magenta")
        self._add("    (0.2, 1.0, 1.0),  # Cyan")
        self._add("]")
        self._add("materials = [make_material(f'Color_{i}', c, roughness=0.3, metallic=0.2) for i, c in enumerate(colors)]")
        self._add("")

        # Objects
        self._add("# Create animated geometric shapes")
        self._add("shapes = []")
        self._add("for i in range(12):")
        self._add("    angle = i * (math.tau / 12)")
        self._add("    radius = 3.0")
        self._add("    x = radius * math.cos(angle)")
        self._add("    y = radius * math.sin(angle)")
        self._add("    z = 2.0")
        self._add("    ")
        self._add("    # Alternate shape types")
        self._add("    if i % 3 == 0:")
        self._add("        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z))")
        self._add("    elif i % 3 == 1:")
        self._add("        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.7, location=(x, y, z))")
        self._add("    else:")
        self._add("        bpy.ops.mesh.primitive_torus_add(major_radius=0.7, minor_radius=0.2, location=(x, y, z))")
        self._add("    ")
        self._add("    obj = bpy.context.object")
        self._add("    obj.name = f'GeoShape_{i}'")
        self._add("    obj.data.materials.append(materials[i % len(materials)])")
        self._add("    shapes.append(obj)")
        self._add("")

        # Animation
        self._add("# Rotation animation")
        self._add("for i, obj in enumerate(shapes):")
        self._add("    speed = 1.0 + (i % 5) * 0.3")
        self._add("    obj.rotation_euler = (0, 0, 0)")
        self._add("    obj.keyframe_insert(data_path='rotation_euler', frame=1)")
        self._add("    obj.rotation_euler = (math.tau * speed, math.tau * 0.5, math.tau * 0.3)")
        self._add(f"    obj.keyframe_insert(data_path='rotation_euler', frame={self.config.frame_end})")
        self._add("    ")
        self._add("    # Scale pulse")
        self._add("    obj.scale = (1, 1, 1)")
        self._add("    obj.keyframe_insert(data_path='scale', frame=1)")
        self._add("    obj.scale = (1.5, 1.5, 1.5)")
        self._add(f"    obj.keyframe_insert(data_path='scale', frame={self.config.frame_end // 2})")
        self._add("    obj.scale = (1, 1, 1)")
        self._add(f"    obj.keyframe_insert(data_path='scale', frame={self.config.frame_end})")
        self._add("")

        # Camera animation
        self._add("# Camera orbits")
        self._add("camera = bpy.data.objects['Main_Camera']")
        self._add("camera.keyframe_insert(data_path='location', frame=1)")
        self._add(f"camera.location = (0, -15, 5)")
        self._add(f"camera.keyframe_insert(data_path='location', frame={self.config.frame_end})")
        self._add("")

        self._add("scene.frame_set(1)")
        self._add('print("ABSTRACT GEOMETRY SCENE CREATED")')

        return "\n".join(self.script_lines)


class ProductShowcaseScene(BlenderSceneGenerator):
    """Clean product showcase scene."""

    def generate_script(self) -> str:
        self.script_lines = []

        self._add("# Auto-generated Product Showcase Scene")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Studio setup
        self._add("# Studio backdrop")
        self._add("bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))")
        self._add("floor = bpy.context.object")
        self._add("floor.name = 'Studio_Floor'")
        self._add("floor_mat = make_material('Studio_Gray', (0.15, 0.15, 0.15), roughness=0.8)")
        self._add("floor.data.materials.append(floor_mat)")
        self._add("")

        # Backdrop curve
        self._add("bpy.ops.mesh.primitive_plane_add(size=20, location=(0, -9, 5))")
        self._add("backdrop = bpy.context.object")
        self._add("backdrop.name = 'Studio_Backdrop'")
        self._add("backdrop.rotation_euler.x = -math.radians(90)")
        self._add("backdrop.data.materials.append(floor_mat)")
        self._add("")

        # Product placeholder
        self._add("# Product (placeholder cube - replace with your model)")
        self._add("bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))")
        self._add("product = bpy.context.object")
        self._add("product.name = 'Product'")
        self._add("product_mat = make_material('Product_White', (0.95, 0.95, 0.95), roughness=0.1, metallic=0.0)")
        self._add("product.data.materials.append(product_mat)")
        self._add("")

        # Accent lights
        self._add("# Rim lights for product")
        self._add("for pos, name in [((3, -2, 3), 'Rim_Left'), ((-3, -2, 3), 'Rim_Right'), ((0, 3, 4), 'Top')]:")
        self._add("    bpy.ops.object.light_add(type='AREA', location=pos)")
        self._add("    light = bpy.context.object")
        self._add("    light.name = name")
        self._add("    light.data.energy = 500")
        self._add("    light.data.shape = 'RECTANGLE'")
        self._add("    light.data.size = 2")
        self._add("")

        # Turntable animation
        self._add("# Turntable animation")
        self._add("product.rotation_euler = (0, 0, 0)")
        self._add("product.keyframe_insert(data_path='rotation_euler', frame=1)")
        self._add("product.rotation_euler = (0, 0, math.tau)")
        self._add(f"product.keyframe_insert(data_path='rotation_euler', frame={self.config.frame_end})")
        self._add("")

        # Camera
        self._add("# Camera slight dolly")
        self._add("camera = bpy.data.objects['Main_Camera']")
        self._add("camera.keyframe_insert(data_path='location', frame=1)")
        self._add("camera.location = (0, -8, 2.5)")
        self._add(f"camera.keyframe_insert(data_path='location', frame={self.config.frame_end})")
        self._add("")

        self._add("scene.frame_set(1)")
        self._add('print("PRODUCT SHOWCASE SCENE CREATED")')

        return "\n".join(self.script_lines)


class DataVisualizationScene(BlenderSceneGenerator):
    """Data visualization scene (bar chart, etc.)."""

    def generate_script(self) -> str:
        self.script_lines = []

        self._add("# Auto-generated Data Visualization Scene")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Sample data
        self._add("# Sample data (replace with real data)")
        self._add("data = [")
        self._add("    ('Jan', 45), ('Feb', 52), ('Mar', 48), ('Apr', 61),")
        self._add("    ('May', 55), ('Jun', 67), ('Jul', 72), ('Aug', 69),")
        self._add("    ('Sep', 75), ('Oct', 82), ('Nov', 78), ('Dec', 88)")
        self._add("]")
        self._add("max_val = max(v for _, v in data)")
        self._add("")

        # Materials
        self._add("bar_mat = make_material('Bar_Blue', (0.2, 0.5, 1.0), roughness=0.2, metallic=0.1)")
        self._add("label_mat = make_material('Label_White', (0.9, 0.9, 0.9), emission=2)")
        self._add("axis_mat = make_material('Axis_Gray', (0.4, 0.4, 0.4), roughness=0.8)")
        self._add("")

        # Create bars
        self._add("# Create bar chart")
        self._add("bar_width = 0.6")
        self._add("spacing = 1.0")
        self._add("start_x = -len(data) * (bar_width + spacing) / 2")
        self._add("")
        self._add("for i, (label, value) in enumerate(data):")
        self._add("    x = start_x + i * (bar_width + spacing)")
        self._add("    height = (value / max_val) * 5.0  # Max height 5 units")
        self._add("    ")
        self._add("    # Bar")
        self._add("    bpy.ops.mesh.primitive_cube_add(size=1, location=(x + bar_width/2, 0, height/2))")
        self._add("    bar = bpy.context.object")
        self._add("    bar.name = f'Bar_{label}'")
        self._add("    bar.scale = (bar_width, 0.8, height)")
        self._add("    bar.data.materials.append(bar_mat)")
        self._add("    ")
        self._add("    # Value label (text)")
        self._add("    bpy.ops.object.text_add(location=(x + bar_width/2, -1.2, height + 0.3))")
        self._add("    txt = bpy.context.object")
        self._add("    txt.name = f'Label_{label}'")
        self._add("    txt.data.body = str(value)")
        self._add("    txt.data.size = 0.4")
        self._add("    txt.data.align_x = 'CENTER'")
        self._add("    txt.rotation_euler.x = -math.radians(90)")
        self._add("    txt.data.materials.append(label_mat)")
        self._add("    ")
        self._add("    # Month label")
        self._add("    bpy.ops.object.text_add(location=(x + bar_width/2, -1.2, -0.5))")
        self._add("    mtxt = bpy.context.object")
        self._add("    mtxt.name = f'Month_{label}'")
        self._add("    mtxt.data.body = label")
        self._add("    mtxt.data.size = 0.35")
        self._add("    mtxt.data.align_x = 'CENTER'")
        self._add("    mtxt.rotation_euler.x = -math.radians(90)")
        self._add("    mtxt.data.materials.append(label_mat)")
        self._add("")

        # Axes
        self._add("# Axes")
        self._add("bpy.ops.mesh.primitive_cube_add(size=1, location=(start_x + len(data)*(bar_width+spacing)/2, 0.4, 0))")
        self._add("x_axis = bpy.context.object")
        self._add("x_axis.name = 'X_Axis'")
        self._add("x_axis.scale = (len(data) * (bar_width + spacing) / 2, 0.05, 0.05)")
        self._add("x_axis.data.materials.append(axis_mat)")
        self._add("")
        self._add("bpy.ops.mesh.primitive_cube_add(size=1, location=(start_x - 0.5, 0, max_val * 5.0 / max_val / 2))")
        self._add("y_axis = bpy.context.object")
        self._add("y_axis.name = 'Y_Axis'")
        self._add("y_axis.scale = (0.05, 0.05, 5.5)")
        self._add("y_axis.data.materials.append(axis_mat)")
        self._add("")

        # Animation - bars grow
        self._add("# Bar growth animation")
        self._add("for i, (label, value) in enumerate(data):")
        self._add("    bar = bpy.data.objects[f'Bar_{label}']")
        self._add("    target_height = (value / max_val) * 5.0")
        self._add("    bar.scale.z = 0.01")
        self._add("    bar.keyframe_insert(data_path='scale', frame=1)")
        self._add("    bar.scale.z = target_height")
        self._add(f"    delay = i * 5  # Staggered")
        self._add(f"    bar.keyframe_insert(data_path='scale', frame=30 + delay)")
        self._add("")

        # Camera
        self._add("# Camera")
        self._add("camera = bpy.data.objects['Main_Camera']")
        self._add("camera.location = (0, -12, 6)")
        self._add("target = Vector((0, 0, 2))")
        self._add("camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()")
        self._add("camera.data.lens = 35")
        self._add("")

        self._add("scene.frame_set(1)")
        self._add('print("DATA VISUALIZATION SCENE CREATED")')

        return "\n".join(self.script_lines)


class ShipOfTheseusScene(BlenderSceneGenerator):
    """Ship of Theseus paradox visualization."""

    def generate_script(self) -> str:
        self.script_lines = []

        self._add("# Auto-generated Ship of Theseus Paradox Scene")
        self._add("")

        self._add_import()
        self._add_scene_setup()
        self._add_materials()
        self._add_world()
        self._add_camera()
        self._add_lighting()

        # Ship of Theseus specific content
        self._add("# ============================================================")
        self._add("# SHIP OF THESEUS PARADOX")
        self._add("# ============================================================")
        self._add("")

        # Materials
        self._add("ship_wood = make_material('Ship Wood', (0.55, 0.35, 0.2))")
        self._add("new_wood = make_material('New Wood', (0.4, 0.25, 0.15))")
        self._add("highlight = make_material('Glow', (1.0, 0.8, 0.2), emission=3)")
        self._add("dark = make_material('Dark', (0.1, 0.05, 0.02))")
        self._add("text_mat = make_material('Text', (1.0, 0.9, 0.5), emission=5)")
        self._add("water_mat = make_material('Water', (0.0, 0.2, 0.4), roughness=0.1, emission=0.5)")
        self._add("")

        # Water plane
        self._add("# Water plane")
        self._add("bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -1))")
        self._add("water = bpy.context.object")
        self._add("water.name = 'Ocean'")
        self._add("water.data.materials.append(water_mat)")
        self._add("")

        # Ship creation function
        self._add("def create_ship(name, x_offset, wood_mat, rotation_y=0):")
        self._add("    \"\"\"Create a complete ship at x_offset.\"\"\"")
        self._add("    ")
        self._add("    # Hull")
        self._add("    bpy.ops.mesh.primitive_cylinder_add(radius=2.5, depth=10, location=(x_offset, 0, 2))")
        self._add("    hull = bpy.context.object")
        self._add("    hull.name = f\"{name}_Hull\"")
        self._add("    hull.data.materials.append(wood_mat)")
        self._add("    ")
        self._add("    # Bow (front point)")
        self._add("    bpy.ops.mesh.primitive_cone_add(radius1=1.5, radius2=0, depth=3, ")
        self._add("                                     location=(x_offset + 5.5, 0, 2))")
        self._add("    bow = bpy.context.object")
        self._add("    bow.name = f\"{name}_Bow\"")
        self._add("    bow.rotation_euler = (0, 0, math.pi/2)")
        self._add("    bow.data.materials.append(wood_mat)")
        self._add("    ")
        self._add("    # Stern (back)")
        self._add("    bpy.ops.mesh.primitive_cube_add(size=2, location=(x_offset - 5.5, 0, 2))")
        self._add("    stern = bpy.context.object")
        self._add("    stern.name = f\"{name}_Stern\"")
        self._add("    stern.scale = (1.5, 2, 3)")
        self._add("    stern.data.materials.append(wood_mat)")
        self._add("    ")
        self._add("    # Mast")
        self._add("    bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=15, location=(x_offset, 0, 10))")
        self._add("    mast = bpy.context.object")
        self._add("    mast.name = f\"{name}_Mast\"")
        self._add("    mast.data.materials.append(dark)")
        self._add("    ")
        self._add("    # Sail")
        self._add("    bpy.ops.mesh.primitive_plane_add(size=8, location=(x_offset + 4, 0, 10))")
        self._add("    sail = bpy.context.object")
        self._add("    sail.name = f\"{name}_Sail\"")
        self._add("    sail.rotation_euler = (0, math.pi/2, 0)")
        self._add("    sail.data.materials.append(make_material(\"Sail\", (0.95, 0.95, 0.95)))")
        self._add("    ")
        self._add("    # Planks (deck details)")
        self._add("    for i in range(-4, 5):")
        self._add("        bpy.ops.mesh.primitive_cube_add(size=1, ")
        self._add("                                         location=(x_offset + i * 1.2, 0, 2.5))")
        self._add("        plank = bpy.context.object")
        self._add("        plank.name = f\"{name}_Plank\"")
        self._add("        plank.scale = (1, 0.2, 0.1)")
        self._add("        plank.data.materials.append(make_material(\"Plank\", (0.45, 0.3, 0.18)))")
        self._add("    ")
        self._add("    # Rotate whole ship")
        self._add("    hull.rotation_euler = (0, 0, rotation_y)")
        self._add("    ")
        self._add("    return hull")
        self._add("")

        # Create the TWO ships
        self._add("# Create the TWO ships")
        self._add("original_ship = create_ship(\"Original\", -12, ship_wood, math.pi/6)")
        self._add("new_ship = create_ship(\"New\", 12, new_wood, -math.pi/6)")
        self._add("")

        # Text labels
        self._add("def create_text(text, location, size=1.5, material=text_mat):")
        self._add("    bpy.ops.object.text_add(location=location)")
        self._add("    txt = bpy.context.object")
        self._add("    txt.data.body = text")
        self._add("    txt.data.size = size")
        self._add("    txt.data.extrude = 0.1")
        self._add("    txt.data.align_x = 'CENTER'")
        self._add("    txt.data.materials.append(material)")
        self._add("    return txt")
        self._add("")

        # Title
        self._add("title = create_text(\"SHIP OF THESEUS PARADOX\", (0, -20, 12), 1.8, text_mat)")
        self._add("")

        # Questions
        self._add("q1 = create_text(\"If you replace EVERY part of a ship,\\none plank at a time...\", (-12, 0, 8), 1.2, text_mat)")
        self._add("q2 = create_text(\"Is it still the SAME ship?\", (12, 0, 8), 1.2, text_mat)")
        self._add("")

        # Arrows pointing to each ship
        self._add("arrow_orig = create_text(\"← Original Ship\", (-12, 0, 6), 0.9, highlight)")
        self._add("arrow_new = create_text(\"New Ship →\", (12, 0, 6), 0.9, new_wood)")
        self._add("")

        # Deep question
        self._add("deep = create_text(\"And if you REBUILD the original\\nfrom the discarded planks...\", (0, 0, 4), 1.1, highlight)")
        self._add("deep2 = create_text(\"Which ONE is the REAL Ship of Theseus?\", (0, 0, 2), 1.3, highlight)")
        self._add("")

        # Camera
        self._add("# Camera")
        self._add("bpy.ops.object.camera_add(location=(0, -40, 15))")
        self._add("cam = bpy.context.object")
        self._add("cam.name = \"MainCam\"")
        self._add("cam.data.lens = 35")
        self._add("target = bpy.mathutils.Vector((0, 0, 3))")
        self._add("cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()")
        self._add("scene.camera = cam")
        self._add("")

        # Lighting
        self._add("# Lighting")
        self._add("bpy.ops.object.light_add(type='SUN', location=(10, -10, 30))")
        self._add("sun = bpy.context.object")
        self._add("sun.data.energy = 5")
        self._add("sun.data.angle = 0.5")
        self._add("")
        self._add("bpy.ops.object.light_add(type='AREA', location=(-10, 10, 20))")
        self._add("fill = bpy.context.object")
        self._add("fill.data.energy = 200")
        self._add("fill.data.shape = 'RECTANGLE'")
        self._add("fill.data.size = 10")
        self._add("")
        self._add("bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))")
        self._add("glow = bpy.context.object")
        self._add("glow.data.energy = 50")
        self._add("glow.data.color = (1.0, 0.6, 0.2)")
        self._add("")

        # World
        self._add("# World")
        self._add("world = bpy.data.worlds.new(\"World\")")
        self._add("scene.world = world")
        self._add("world.use_nodes = True")
        self._add("bg = world.node_tree.nodes.get(\"Background\")")
        self._add("bg.inputs['Color'].default_value = (0.0, 0.05, 0.1, 1)")
        self._add("bg.inputs['Strength'].default_value = 0.1")
        self._add("")

        # Animation
        self._add("def animate_ship(obj, direction=1):")
        self._add("    \"\"\"Animate rotation and subtle floating.\"\"\"")
        self._add("    for frame in range(scene.frame_start, scene.frame_end + 1):")
        self._add("        scene.frame_set(frame)")
        self._add("        # Rotation")
        self._add("        obj.rotation_euler.z = direction * frame * 0.02")
        self._add("        # Subtle bobbing")
        self._add("        obj.location.z = 2 + math.sin(frame * 0.1) * 0.1")
        self._add("        obj.keyframe_insert(data_path=\"location\", frame=frame)")
        self._add("        obj.keyframe_insert(data_path=\"rotation_euler\", frame=frame)")
        self._add("")

        # Animate both ships in opposite directions
        self._add("animate_ship(original_ship, direction=1)")
        self._add("animate_ship(new_ship, direction=-1)")
        self._add("")

        # Camera slow orbit
        self._add("cam = bpy.data.objects['Camera']")
        self._add("for frame in range(scene.frame_start, scene.frame_end + 1):")
        self._add("    scene.frame_set(frame)")
        self._add("    angle = frame * 0.01")
        self._add("    cam.location.x = 35 * math.cos(angle)")
        self._add("    cam.location.y = -35 * math.sin(angle)")
        self._add("    cam.keyframe_insert(data_path=\"location\", frame=frame)")
        self._add("")

        # Text fade in/out
        self._add("for obj, start_fade, end_fade in [")
        self._add("    (title, 1, 20),")
        self._add("    (q1, 20, 60),")
        self._add("    (q2, 20, 60),")
        self._add("    (deep, 60, 100),")
        self._add("    (deep2, 60, 100),")
        self._add("    (arrow_orig, 10, 50),")
        self._add("    (arrow_new, 10, 50),")
        self._add("]:")
        self._add("    obj.hide_render = True")
        self._add("    obj.keyframe_insert(data_path=\"hide_render\", frame=start_fade - 5)")
        self._add("    obj.hide_render = False")
        self._add("    obj.keyframe_insert(data_path=\"hide_render\", frame=start_fade)")
        self._add("    obj.hide_render = True")
        self._add("    obj.keyframe_insert(data_path=\"hide_render\", frame=end_fade)")
        self._add("")

        # Final frame
        self._add("scene.frame_set(1)")
        self._add('print("SHIP OF THESEUS PARADOX SCENE CREATED")')

        return "\n".join(self.script_lines)


# Scene registry
SCENE_REGISTRY = {
    SceneType.MYSTERY_FOREST: MysteryForestScene,
    SceneType.ABSTRACT_GEOMETRY: AbstractGeometryScene,
    SceneType.PRODUCT_SHOWCASE: ProductShowcaseScene,
    SceneType.DATA_VISUALIZATION: DataVisualizationScene,
    SceneType.SHIP_OF_THESEUS: ShipOfTheseusScene,
    # SceneType.MINIMAL_STUDIO: MinimalStudioScene,  # TODO
    # SceneType.CYBERPUNK_ALLEY: CyberpunkAlleyScene,  # TODO
    # SceneType.NATURE_TIMELAPSE: NatureTimelapseScene,  # TODO
    # SceneType.TITLE_SEQUENCE: TitleSequenceScene,  # TODO
}


def get_scene_generator(scene_type: SceneType, config: SceneConfig = None) -> BlenderSceneGenerator:
    """Get a scene generator instance."""
    generator_class = SCENE_REGISTRY.get(scene_type)
    if not generator_class:
        raise ValueError(f"Unknown scene type: {scene_type}")
    return generator_class(config)


def list_available_scenes() -> list[dict]:
    """List all available scene types."""
    return [
        {
            "type": st.value,
            "name": st.name.replace("_", " ").title(),
            "available": st in SCENE_REGISTRY,
        }
        for st in SceneType
    ]


def generate_scene_script(
    scene_type: SceneType,
    config: SceneConfig = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a Blender scene script."""
    generator = get_scene_generator(scene_type, config)
    script = generator.generate_script()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script)
        logger.info("Scene script saved", extra={"path": str(output_path)})

    return script


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Blender Scene Generator")
    parser.add_argument("--list", action="store_true", help="List available scenes")
    parser.add_argument("--scene", type=str, help="Scene type to generate")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--frames", type=int, default=900, help="Frame count")
    parser.add_argument("--resolution", type=str, default="1080x1920", help="Resolution WxH")

    args = parser.parse_args()

    if args.list:
        scenes = list_available_scenes()
        print("Available scenes:")
        for s in scenes:
            status = "✓" if s["available"] else "✗ (TODO)"
            print(f"  {status} {s['type']}: {s['name']}")
        sys.exit(0)

    if not args.scene:
        parser.error("--scene required (or --list)")

    try:
        scene_type = SceneType(args.scene)
    except ValueError:
        print(f"Unknown scene: {args.scene}")
        sys.exit(1)

    # Parse resolution
    w, h = map(int, args.resolution.split("x"))

    config = SceneConfig(
        frame_end=args.frames,
        resolution_x=w,
        resolution_y=h,
    )

    output_path = Path(args.output) if args.output else None

    script = generate_scene_script(scene_type, config, output_path)

    if not output_path:
        print(script)