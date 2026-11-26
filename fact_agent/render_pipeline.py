"""Complete cloud rendering pipeline for Fact Agent."""

import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fact_agent.config import get_settings, load_janction_key
from fact_agent.http_client import RenderClient
from fact_agent.exceptions import RenderAPIError, AuthenticationError
from fact_agent.logging_setup import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Render job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenderJob:
    """Render job information."""
    id: str
    status: JobStatus
    progress: float = 0.0
    frames_completed: int = 0
    frames_total: int = 0
    cost_yen: float = 0.0
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class RenderConfig:
    """Configuration for a render job."""
    blend_file: Path
    frame_start: int = 1
    frame_end: int = 1
    width: int = 1920
    height: int = 1080
    samples: int = 128
    engine: str = "CYCLES"  # CYCLES, EEVEE, WORKBENCH
    device: str = "GPU"  # GPU, CPU
    output_format: str = "PNG"  # PNG, JPEG, EXR, MP4
    output_quality: int = 90  # For JPEG
    use_denoising: bool = True
    priority: str = "normal"  # low, normal, high


class RenderPipeline:
    """Complete cloud rendering pipeline."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: Optional[RenderClient] = None
        self._api_key: Optional[str] = None

    @property
    def client(self) -> RenderClient:
        if self._client is None:
            self._api_key = load_janction_key(self.settings)
            self._client = RenderClient(
                api_base=self.settings.janction_api_base,
                api_key=self._api_key,
                timeout=self.settings.janction_timeout,
            )
        return self._client

    def _request(self, endpoint: str, method: str = "GET", payload: dict | None = None, files: dict | None = None) -> dict:
        """Make API request with error handling."""
        url = f"{self.settings.janction_api_base}{endpoint}"
        headers = {"X-API-Key": self._api_key or load_janction_key(self.settings)}

        if payload and not files:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        else:
            data = None

        import urllib.request

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.settings.janction_timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 401:
                raise AuthenticationError(f"Authentication failed: {body}") from e
            raise RenderAPIError(
                f"HTTP {e.code}: {e.reason}",
                status_code=e.code,
                response_body=body,
            ) from e
        except urllib.error.URLError as e:
            raise RenderAPIError(f"Network error: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise RenderAPIError(f"Invalid JSON response: {e}") from e

    def submit_job(self, config: RenderConfig) -> RenderJob:
        """Submit a render job with .blend file."""
        if not config.blend_file.is_file():
            raise RenderAPIError(f"Blend file not found: {config.blend_file}")

        # Prepare multipart form data for file upload
        import urllib.request
        import uuid

        boundary = uuid.uuid4().hex
        headers = {
            "X-API-Key": self._api_key or load_janction_key(self.settings),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        # Build multipart body
        body_parts = []

        # Add config fields
        config_data = {
            "frame_start": config.frame_start,
            "frame_end": config.frame_end,
            "width": config.width,
            "height": config.height,
            "samples": config.samples,
            "engine": config.engine,
            "device": config.device,
            "output_format": config.output_format,
            "output_quality": config.output_quality,
            "use_denoising": config.use_denoising,
            "priority": config.priority,
        }

        for key, value in config_data.items():
            body_parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            )

        # Add blend file
        file_content = config.blend_file.read_bytes()
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="blend_file"; filename="{config.blend_file.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        )
        body_parts.append(file_content.decode("latin-1"))  # Binary safe
        body_parts.append(f"\r\n--{boundary}--\r\n")

        body = "".join(body_parts).encode("latin-1")

        url = f"{self.settings.janction_api_base}/v1/jobs"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 401:
                raise AuthenticationError(f"Authentication failed: {body}") from e
            raise RenderAPIError(
                f"Job submission failed: HTTP {e.code}",
                status_code=e.code,
                response_body=body,
            ) from e

        return self._parse_job(result)

    def _parse_job(self, data: dict) -> RenderJob:
        """Parse job from API response."""
        return RenderJob(
            id=str(data.get("id", data.get("job_id", ""))),
            status=JobStatus(data.get("status", "pending").lower()),
            progress=float(data.get("progress", 0)),
            frames_completed=int(data.get("frames_completed", 0)),
            frames_total=int(data.get("frames_total", 1)),
            cost_yen=float(data.get("cost_yen", data.get("cost", 0))),
            output_url=data.get("output_url", data.get("download_url")),
            error_message=data.get("error", data.get("error_message")),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            raw=data,
        )

    def get_job_status(self, job_id: str) -> RenderJob:
        """Get current job status."""
        result = self._request(f"/v1/jobs/{job_id}")
        return self._parse_job(result)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
        progress_callback: Optional[callable] = None,
    ) -> RenderJob:
        """Poll job until completion."""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise RenderAPIError(f"Job {job_id} timed out after {timeout}s")

            job = self.get_job_status(job_id)

            if progress_callback:
                progress_callback(job)

            logger.info(
                "Job status",
                extra={
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "frames": f"{job.frames_completed}/{job.frames_total}",
                },
            )

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job

            time.sleep(poll_interval)

    def download_result(self, job: RenderJob, output_dir: Path) -> list[Path]:
        """Download render output files."""
        if not job.output_url:
            raise RenderAPIError("No output URL available for completed job")

        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        import urllib.request

        # Handle both single file and archive
        if job.output_url.endswith((".zip", ".tar.gz", ".tgz")):
            # Download archive and extract
            import zipfile
            import tarfile

            archive_path = output_dir / f"render_{job.id}.zip"
            urllib.request.urlretrieve(job.output_url, archive_path)

            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(output_dir)
                    downloaded = [output_dir / n for n in zf.namelist()]
            else:
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(output_dir)
                    downloaded = [output_dir / n for n in tf.getnames()]

            archive_path.unlink()  # Remove archive
        else:
            # Single file download
            filename = job.output_url.split("/")[-1] or f"render_{job.id}.png"
            output_path = output_dir / filename
            urllib.request.urlretrieve(job.output_url, output_path)
            downloaded = [output_path]

        logger.info("Downloaded render results", extra={"files": len(downloaded), "dir": str(output_dir)})
        return downloaded

    def render_sync(
        self,
        config: RenderConfig,
        output_dir: Path,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
        progress_callback: Optional[callable] = None,
    ) -> list[Path]:
        """Submit job, wait for completion, and download results."""
        logger.info("Submitting render job", extra={"config": config.__dict__})

        job = self.submit_job(config)
        logger.info("Job submitted", extra={"job_id": job.id})

        job = self.wait_for_job(job.id, poll_interval, timeout, progress_callback)

        if job.status == JobStatus.FAILED:
            raise RenderAPIError(f"Render failed: {job.error_message}")

        if job.status == JobStatus.CANCELLED:
            raise RenderAPIError("Render was cancelled")

        return self.download_result(job, output_dir)


def create_blend_from_scene(scene_script: Path, output_path: Path) -> bool:
    """Run Blender script to generate .blend file.

    This runs Blender in background mode with the scene script.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["blender", "--background", "--python", str(scene_script), "--", "--output", str(output_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error("Blender script failed", extra={"stderr": result.stderr})
            return False

        return output_path.is_file()

    except FileNotFoundError:
        logger.error("Blender not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Blender script timed out")
        return False
    except Exception as e:
        logger.error("Blender execution error", exc_info=e)
        return False


def render_scene_direct(
    scene_script: Path,
    config: RenderConfig,
    output_dir: Path,
    pipeline: Optional[RenderPipeline] = None,
) -> list[Path]:
    """Complete pipeline: generate .blend from scene script, render in cloud, download."""
    pipeline = pipeline or RenderPipeline()

    # Generate .blend file from scene script
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".blend", delete=False) as tmp:
        blend_path = Path(tmp.name)

    try:
        logger.info("Generating .blend from scene script", extra={"script": str(scene_script)})

        if not create_blend_from_scene(scene_script, blend_path):
            raise RenderAPIError(f"Failed to generate .blend from {scene_script}")

        config.blend_file = blend_path
        return pipeline.render_sync(config, output_dir)

    finally:
        # Clean up temp .blend
        if blend_path.exists():
            blend_path.unlink()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python render_pipeline.py <command> [args]")
        print("Commands: account, estimate, submit, status, download, render")
        sys.exit(1)

    cmd = sys.argv[1]
    pipeline = RenderPipeline()

    try:
        if cmd == "account":
            from fact_agent.cloud_renderer import check_account
            print(json.dumps(check_account(), indent=2))

        elif cmd == "estimate":
            from fact_agent.cloud_renderer import estimate_free_preview
            print(json.dumps(estimate_free_preview(), indent=2))

        elif cmd == "submit":
            if len(sys.argv) < 3:
                print("Usage: python render_pipeline.py submit <blend_file>")
                sys.exit(1)
            blend_file = Path(sys.argv[2])
            config = RenderConfig(blend_file=blend_file)
            job = pipeline.submit_job(config)
            print(f"Job submitted: {job.id}")

        elif cmd == "status":
            if len(sys.argv) < 3:
                print("Usage: python render_pipeline.py status <job_id>")
                sys.exit(1)
            job = pipeline.get_job_status(sys.argv[2])
            print(f"Status: {job.status.value}, Progress: {job.progress:.1%}")

        elif cmd == "render":
            if len(sys.argv) < 4:
                print("Usage: python render_pipeline.py render <scene_script> <output_dir>")
                sys.exit(1)
            scene_script = Path(sys.argv[2])
            output_dir = Path(sys.argv[3])

            config = RenderConfig(
                frame_start=1,
                frame_end=900,
                width=1080,
                height=1920,
                samples=128,
            )

            def progress(job):
                print(f"  [{job.status.value}] {job.progress:.1%} ({job.frames_completed}/{job.frames_total})")

            files = render_scene_direct(scene_script, config, output_dir, pipeline)
            print(f"Downloaded {len(files)} files to {output_dir}")
            for f in files:
                print(f"  {f}")

        else:
            print(f"Unknown command: {cmd}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)