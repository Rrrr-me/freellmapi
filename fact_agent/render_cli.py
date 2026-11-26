#!/usr/bin/env python3
"""CLI for JANCTION cloud rendering."""

import sys
import argparse
import json
from pathlib import Path
from .cloud_renderer import CloudRenderer
from .render_pipeline import RenderPipeline, RenderConfig, JobStatus
from .logging_setup import setup_logging
from .config import get_settings
from .blender_scenes import (
    SceneConfig,
    SceneType,
    list_available_scenes,
    generate_scene_script,
)


settings = get_settings()
setup_logging(settings)


def cmd_account(args):
    """Check account status and limits."""
    renderer = CloudRenderer()
    try:
        result = renderer.check_account()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Account Status:")
            print(f"  Authenticated: {result['authenticated']}")
            if result['job_limit_yen']:
                print(f"  Job Limit: ¥{result['job_limit_yen']:,}")
            if result['daily_limit_yen']:
                print(f"  Daily Limit: ¥{result['daily_limit_yen']:,}")
            if result['limits_applied'] is not None:
                print(f"  Limits Applied: {result['limits_applied']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_estimate(args):
    """Get a preview render cost estimate."""
    renderer = CloudRenderer()
    try:
        result = renderer.estimate_preview(
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            width=args.width,
            height=args.height,
            samples=args.samples,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Preview Estimate:")
            print(f"  Frames: {args.frame_start}-{args.frame_end}")
            print(f"  Resolution: {args.width}×{args.height}")
            print(f"  Samples: {args.samples}")
            print(f"  Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_submit(args):
    """Submit a render job."""
    pipeline = RenderPipeline()
    config = RenderConfig(
        blend_file=Path(args.blend_file),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        width=args.width,
        height=args.height,
        samples=args.samples,
        engine=args.engine,
        device=args.device,
        output_format=args.format,
        priority=args.priority,
    )
    try:
        job = pipeline.submit_job(config)
        print(f"Job submitted: {job.id}")
        print(f"Status: {job.status.value}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Check job status."""
    pipeline = RenderPipeline()
    try:
        job = pipeline.get_job_status(args.job_id)
        if args.json:
            print(json.dumps(job.raw, indent=2))
        else:
            print(f"Job: {job.id}")
            print(f"Status: {job.status.value}")
            print(f"Progress: {job.progress:.1%}")
            print(f"Frames: {job.frames_completed}/{job.frames_total}")
            if job.cost_yen:
                print(f"Cost: ¥{job.cost_yen:,}")
            if job.error_message:
                print(f"Error: {job.error_message}")
            if job.output_url:
                print(f"Output: {job.output_url}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_wait(args):
    """Wait for job completion."""
    pipeline = RenderPipeline()
    try:
        def progress(job):
            print(f"  [{job.status.value}] {job.progress:.1%} ({job.frames_completed}/{job.frames_total})")

        job = pipeline.wait_for_job(
            args.job_id,
            poll_interval=args.interval,
            timeout=args.timeout,
            progress_callback=progress,
        )

        print(f"\nFinal status: {job.status.value}")
        if job.status == JobStatus.COMPLETED:
            print(f"Cost: ¥{job.cost_yen:,}")
            if job.output_url:
                print(f"Output: {job.output_url}")
        elif job.error_message:
            print(f"Error: {job.error_message}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_download(args):
    """Download job results."""
    pipeline = RenderPipeline()
    try:
        job = pipeline.get_job_status(args.job_id)
        if job.status != JobStatus.COMPLETED:
            print(f"Job not completed (status: {job.status.value})", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir)
        files = pipeline.download_result(job, output_dir)
        print(f"Downloaded {len(files)} files to {output_dir}")
        for f in files:
            print(f"  {f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_render(args):
    """Complete pipeline: scene script -> cloud render -> download."""
    pipeline = RenderPipeline()

    scene_script = Path(args.scene_script)
    output_dir = Path(args.output_dir)

    config = RenderConfig(
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        width=args.width,
        height=args.height,
        samples=args.samples,
        engine=args.engine,
        device=args.device,
        output_format=args.format,
        priority=args.priority,
    )

    try:
        def progress(job):
            print(f"  [{job.status.value}] {job.progress:.1%} ({job.frames_completed}/{job.frames_total})")

        from fact_agent.render_pipeline import render_scene_direct
        files = render_scene_direct(scene_script, config, output_dir, pipeline)
        print(f"Downloaded {len(files)} files to {output_dir}")
        for f in files:
            print(f"  {f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_generate_scene(args):
    """Generate a Blender scene script from a template."""
    try:
        scene_type = SceneType(args.scene_type)
    except ValueError:
        print(f"Unknown scene type: {args.scene_type}")
        print("Available types:")
        for s in list_available_scenes():
            status = "✓" if s["available"] else "✗"
            print(f"  {status} {s['type']}")
        sys.exit(1)

    # Check if scene is available
    available = any(s["type"] == scene_type.value and s["available"] for s in list_available_scenes())
    if not available:
        print(f"Scene type '{scene_type.value}' is not implemented yet")
        sys.exit(1)

    # Parse resolution
    try:
        w, h = map(int, args.resolution.split("x"))
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 1920x1080)")
        sys.exit(1)

    from .blender_scenes import RenderEngine as RE

    config = SceneConfig(
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        resolution_x=w,
        resolution_y=h,
        fps=args.fps,
        engine=RE(args.engine),
        samples=args.samples,
        device=args.device,
        output_format=args.format,
    )

    try:
        script = generate_scene_script(scene_type, config)

        output_path = Path(args.output) if args.output else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script)
            print(f"Scene script saved to: {output_path}")
        else:
            print(script)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_scenes(args):
    """List available scene templates."""
    scenes = list_available_scenes()
    print("Available scene templates:")
    for s in scenes:
        status = "✓" if s["available"] else "✗ (TODO)"
        print(f"  {status} {s['type']}: {s['name']}")


def main():
    parser = argparse.ArgumentParser(description="Fact Agent Render CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # account
    acc = subparsers.add_parser("account", help="Check account status")
    acc.add_argument("--json", action="store_true", help="Output as JSON")

    # estimate
    est = subparsers.add_parser("estimate", help="Get preview render estimate")
    est.add_argument("--frame-start", type=int, default=1, help="Start frame")
    est.add_argument("--frame-end", type=int, default=1, help="End frame")
    est.add_argument("--width", type=int, default=640, help="Width in pixels")
    est.add_argument("--height", type=int, default=360, help="Height in pixels")
    est.add_argument("--samples", type=int, default=16, help="Sample count")
    est.add_argument("--json", action="store_true", help="Output as JSON")

    # submit
    sub = subparsers.add_parser("submit", help="Submit render job")
    sub.add_argument("blend_file", help="Path to .blend file")
    sub.add_argument("--frame-start", type=int, default=1)
    sub.add_argument("--frame-end", type=int, default=1)
    sub.add_argument("--width", type=int, default=1920)
    sub.add_argument("--height", type=int, default=1080)
    sub.add_argument("--samples", type=int, default=128)
    sub.add_argument("--engine", choices=["CYCLES", "EEVEE", "WORKBENCH"], default="CYCLES")
    sub.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    sub.add_argument("--format", choices=["PNG", "JPEG", "EXR", "MP4"], default="PNG")
    sub.add_argument("--priority", choices=["low", "normal", "high"], default="normal")

    # status
    st = subparsers.add_parser("status", help="Check job status")
    st.add_argument("job_id", help="Job ID")
    st.add_argument("--json", action="store_true", help="Output as JSON")

    # wait
    wt = subparsers.add_parser("wait", help="Wait for job completion")
    wt.add_argument("job_id", help="Job ID")
    wt.add_argument("--interval", type=float, default=10.0, help="Poll interval (seconds)")
    wt.add_argument("--timeout", type=float, default=3600.0, help="Timeout (seconds)")

    # download
    dl = subparsers.add_parser("download", help="Download job results")
    dl.add_argument("job_id", help="Job ID")
    dl.add_argument("output_dir", help="Output directory")

    # render (full pipeline with .blend file)
    rd = subparsers.add_parser("render", help="Full pipeline: scene script (.py) -> cloud -> download")
    rd.add_argument("scene_script", help="Path to Blender scene script (.py)")
    rd.add_argument("output_dir", help="Output directory")
    rd.add_argument("--frame-start", type=int, default=1)
    rd.add_argument("--frame-end", type=int, default=1)
    rd.add_argument("--width", type=int, default=1920)
    rd.add_argument("--height", type=int, default=1080)
    rd.add_argument("--samples", type=int, default=128)
    rd.add_argument("--engine", choices=["CYCLES", "EEVEE", "WORKBENCH"], default="CYCLES")
    rd.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    rd.add_argument("--format", choices=["PNG", "JPEG", "EXR", "MP4"], default="PNG")
    rd.add_argument("--priority", choices=["low", "normal", "high"], default="normal")

    # generate-scene (create .py scene script from template)
    gs = subparsers.add_parser("generate-scene", help="Generate Blender scene script from template")
    gs.add_argument("scene_type", help="Scene type (mystery_forest, abstract_geometry, etc.)")
    gs.add_argument("--output", "-o", help="Output file path (default: stdout)")
    gs.add_argument("--frame-start", type=int, default=1, help="Start frame")
    gs.add_argument("--frame-end", type=int, default=900, help="End frame")
    gs.add_argument("--resolution", type=str, default="1080x1920", help="Resolution WxH")
    gs.add_argument("--fps", type=int, default=30, help="Frames per second")
    gs.add_argument("--engine", choices=["CYCLES", "EEVEE", "WORKBENCH"], default="CYCLES")
    gs.add_argument("--samples", type=int, default=128, help="Render samples")
    gs.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    gs.add_argument("--format", choices=["PNG", "JPEG", "EXR", "MP4"], default="PNG")

    # list-scenes
    ls = subparsers.add_parser("list-scenes", help="List available scene templates")

    args = parser.parse_args()

    commands = {
        "account": cmd_account,
        "estimate": cmd_estimate,
        "submit": cmd_submit,
        "status": cmd_status,
        "wait": cmd_wait,
        "download": cmd_download,
        "render": cmd_render,
        "generate-scene": cmd_generate_scene,
        "list-scenes": cmd_list_scenes,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()