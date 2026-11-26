from fact_agent.blender_scenes import RenderEngine
#!/usr/bin/env python3
"""
Fully Automated Blender Render Pipeline
========================================

One-command video generation:
  python scripts/auto_render.py mystery_forest --frames 900 --resolution 1920x1080

Features:
- Auto-generates scene script from template
- Finds Blender executable (auto-detects or uses $BLENDER_PATH)
- Runs render with progress tracking
- Optionally creates MP4 video from frames
- Cloud fallback (JANCTION) if no local Blender

Usage:
  python scripts/auto_render.py <scene_type> [options]

Scene types: mystery_forest, abstract_geometry, product_showcase, data_visualization

Options:
  --frames N          Number of frames (default: 900)
  --resolution WxH    Resolution (default: 1920x1080)
  --fps N             Frames per second (default: 30)
  --engine CYCLES|EEVEE|WORKBENCH (default: CYCLES)
  --samples N         Render samples (default: 128)
  --device GPU|CPU    (default: GPU)
  --format PNG|JPEG|EXR (default: PNG)
  --output-dir DIR    Output directory (default: ./renders/<scene>_<timestamp>)
  --make-video        Create MP4 from frames (requires ffmpeg)
  --fps-video N       Video FPS (default: 30)
  --cloud             Use JANCTION cloud rendering (requires API key)
  --keep-script       Keep generated .py script
  --blender-path PATH Path to Blender executable
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add fact_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fact_agent.blender_scenes import (
    SceneConfig,
    SceneType,
    generate_scene_script,
    list_available_scenes,
)
from fact_agent.render_pipeline import RenderPipeline, RenderConfig, JobStatus
from fact_agent.config import get_settings


def find_blender() -> str | None:
    """Find Blender executable."""
    # Check explicit path first
    if os.environ.get("BLENDER_PATH"):
        path = Path(os.environ["BLENDER_PATH"])
        if path.exists():
            return str(path)

    # Common locations
    candidates = [
        "blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
    ]

    for c in candidates:
        if shutil.which(c) or Path(c).exists():
            return c

    return None


def run_blender(script_path: Path, output_dir: Path, blender_path: str) -> bool:
    """Run Blender to render the scene."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change to output directory so Blender writes frames there
    cmd = [
        blender_path,
        "--background",
        "--python", str(script_path),
        "--",
        "--output", str(output_dir),
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {output_dir}")

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Print output in real-time
        for line in process.stdout:
            print(f"  [Blender] {line.rstrip()}")

        process.wait()

        if process.returncode == 0:
            print("✅ Blender render completed successfully")
            return True
        else:
            print(f"❌ Blender failed with exit code {process.returncode}")
            return False

    except FileNotFoundError:
        print(f"❌ Blender not found at: {blender_path}")
        return False
    except Exception as e:
        print(f"❌ Error running Blender: {e}")
        return False


def make_video(frames_dir: Path, output_path: Path, fps: int = 30) -> bool:
    """Create MP4 video from rendered frames using ffmpeg."""
    frame_pattern = frames_dir / "%04d.png"
    if not list(frames_dir.glob("*.png")):
        # Try other extensions
        frame_pattern = frames_dir / "%04d.jpg"
        if not list(frames_dir.glob("*.jpg")):
            frame_pattern = frames_dir / "%04d.exr"
            if not list(frames_dir.glob("*.exr")):
                print("❌ No frame images found to create video")
                return False

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-framerate", str(fps),
        "-i", str(frame_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    print(f"Creating video: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Video created: {output_path}")
            return True
        else:
            print(f"❌ ffmpeg failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ ffmpeg not found. Install ffmpeg to create videos.")
        return False
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False


def render_cloud(scene_script: Path, config: SceneConfig, output_dir: Path) -> bool:
    """Fallback: render via JANCTION cloud."""
    print("☁️  Attempting cloud render via JANCTION...")

    try:
        settings = get_settings()
        pipeline = RenderPipeline(settings)

        render_config = RenderConfig(
            frame_start=config.frame_start,
            frame_end=config.frame_end,
            width=config.resolution_x,
            height=config.resolution_y,
            samples=config.samples,
            engine=config.engine.value,
            device=config.device,
            output_format=config.output_format,
        )

        print("Submitting job to JANCTION...")
        job = pipeline.submit_job(config=render_config)
        print(f"Job submitted: {job.id}")

        def progress(job):
            print(f"  [{job.status.value}] {job.progress:.1%} ({job.frames_completed}/{job.frames_total})")

        job = pipeline.wait_for_job(
            job.id,
            poll_interval=15,
            timeout=3600,
            progress_callback=progress,
        )

        if job.status != JobStatus.COMPLETED:
            print(f"❌ Cloud render failed: {job.error_message}")
            return False

        print("Downloading results...")
        files = pipeline.download_result(job, output_dir)
        print(f"✅ Downloaded {len(files)} files")
        return True

    except Exception as e:
        print(f"❌ Cloud render failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fully Automated Blender Render Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local render
  python scripts/auto_render.py mystery_forest --frames 900

  # High-res product shot
  python scripts/auto_render.py product_showcase --frames 300 --resolution 1920x1080 --make-video

  # Cloud render (requires JANCTION API key)
  python scripts/auto_render.py data_visualization --frames 200 --cloud

  # Quick test
  python scripts/auto_render.py abstract_geometry --frames 30 --resolution 640x360
        """
    )

    parser.add_argument("scene_type", help="Scene type to render")
    parser.add_argument("--frames", type=int, default=900, help="Number of frames")
    parser.add_argument("--resolution", type=str, default="1920x1080", help="Resolution WxH")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--engine", choices=["CYCLES", "EEVEE", "WORKBENCH"], default="CYCLES")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--format", choices=["PNG", "JPEG", "EXR"], default="PNG")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--make-video", action="store_true", help="Create MP4 from frames")
    parser.add_argument("--fps-video", type=int, default=30, help="Video FPS")
    parser.add_argument("--cloud", action="store_true", help="Use JANCTION cloud rendering")
    parser.add_argument("--keep-script", action="store_true", help="Keep generated .py script")
    parser.add_argument("--blender-path", type=str, help="Path to Blender executable")
    parser.add_argument("--list-scenes", action="store_true", help="List available scenes")

    args = parser.parse_args()

    if args.list_scenes:
        scenes = list_available_scenes()
        print("Available scenes:")
        for s in scenes:
            status = "✓" if s["available"] else "✗ (TODO)"
            print(f"  {status} {s['type']}: {s['name']}")
        return

    # Validate scene type
    try:
        scene_type = SceneType(args.scene_type)
    except ValueError:
        print(f"Unknown scene type: {args.scene_type}")
        print("Available:")
        for s in list_available_scenes():
            print(f"  {s['type']}")
        sys.exit(1)

    # Check if scene is implemented
    available = any(s["type"] == scene_type.value and s["available"] for s in list_available_scenes())
    if not available:
        print(f"Scene '{scene_type.value}' not implemented yet")
        sys.exit(1)

    # Parse resolution
    try:
        w, h = map(int, args.resolution.split("x"))
    except ValueError:
        print("Invalid resolution. Use WxH (e.g., 1920x1080)")
        sys.exit(1)

    # Find Blender (if not cloud)
    blender_path = args.blender_path or find_blender()
    if not args.cloud and not blender_path:
        print("❌ Blender not found. Options:")
        print("  1. Install Blender and add to PATH")
        print("  2. Set BLENDER_PATH environment variable")
        print("  3. Use --blender-path /path/to/blender")
        print("  4. Use --cloud for JANCTION cloud rendering")
        sys.exit(1)

    # Build config
    config = SceneConfig(
        frame_start=1,
        frame_end=args.frames,
        resolution_x=w,
        resolution_y=h,
        fps=args.fps,
        engine=RenderEngine(args.engine),
        samples=args.samples,
        device=args.device,
        output_format=args.format,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./renders/{scene_type.value}_{timestamp}")

    print(f"🎬 Auto-Render Pipeline")
    print(f"   Scene: {scene_type.value}")
    print(f"   Frames: {args.frames}")
    print(f"   Resolution: {w}x{h}")
    print(f"   Engine: {args.engine}")
    print(f"   Samples: {args.samples}")
    print(f"   Output: {output_dir}")

    # Generate scene script
    print("📝 Generating scene script...")
    script_content = generate_scene_script(scene_type, config)

    script_path = output_dir / "scene.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)

    if args.keep_script:
        print(f"📄 Script saved to: {script_path}")

    # Render
    start_time = time.time()
    success = False

    if args.cloud:
        print("⚠️  Cloud rendering requires a .blend file, not a Python script.")
        print("   Cloud rendering via JANCTION requires a pre-made .blend file.")
        print("   Use local Blender rendering instead (remove --cloud flag).")
        sys.exit(1)
    else:
        success = run_blender(script_path, output_dir, blender_path)

    if not success:
        print("❌ Render failed")
        sys.exit(1)

    # Create video if requested
    if args.make_video and success:
        video_path = output_dir / f"{scene_type.value}_{timestamp}.mp4"
        print("🎞️  Creating video...")
        make_video(output_dir, video_path, args.fps_video)

    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f}s")
    print(f"   Output: {output_dir}")

    if args.make_video:
        print(f"   Video: {output_dir / f'{scene_type.value}_{timestamp}.mp4'}")


if __name__ == "__main__":
    main()