"""JANCTION Cloud Rendering Client for Fact Agent."""

import json
import logging
import pathlib
import stat
from typing import Any

from fact_agent.config import get_settings, load_janction_key
from fact_agent.http_client import RenderClient
from fact_agent.exceptions import RenderAPIError, AuthenticationError
from fact_agent.logging_setup import get_logger
from fact_agent.render_pipeline import RenderPipeline, RenderConfig, RenderJob, JobStatus

logger = get_logger(__name__)


class CloudRenderer:
    """High-level cloud rendering interface."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: RenderClient | None = None
        self._pipeline: RenderPipeline | None = None

    @property
    def client(self) -> RenderClient:
        if self._client is None:
            api_key = load_janction_key(self.settings)
            self._client = RenderClient(
                api_base=self.settings.janction_api_base,
                api_key=api_key,
                timeout=self.settings.janction_timeout,
            )
        return self._client

    @property
    def pipeline(self) -> RenderPipeline:
        if self._pipeline is None:
            self._pipeline = RenderPipeline(self.settings)
        return self._pipeline

    def check_account(self) -> dict[str, Any]:
        """Get account status and limits."""
        try:
            account = self.client.check_account()
            limits = account.get("limits", {})

            return {
                "authenticated": True,
                "job_limit_yen": limits.get("job_yen"),
                "daily_limit_yen": limits.get("day_yen"),
                "limits_applied": limits.get("applies"),
                "raw": account,
            }
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("Account check failed", exc_info=e)
            raise RenderAPIError(f"Account check failed: {e}") from e

    def estimate_preview(
        self,
        frame_start: int = 1,
        frame_end: int = 1,
        width: int = 640,
        height: int = 360,
        samples: int = 16,
    ) -> dict[str, Any]:
        """Request a preview render cost estimate."""
        try:
            return self.client.estimate_preview(
                frame_start=frame_start,
                frame_end=frame_end,
                width=width,
                height=height,
                samples=samples,
            )
        except Exception as e:
            logger.error("Preview estimate failed", exc_info=e)
            raise RenderAPIError(f"Estimate failed: {e}") from e

    def submit_render_job(self, blend_file: pathlib.Path, **params) -> RenderJob:
        """Submit a Blender file for rendering."""
        config = RenderConfig(blend_file=blend_file, **params)
        return self.pipeline.submit_job(config)

    def get_job_status(self, job_id: str) -> RenderJob:
        """Get job status."""
        return self.pipeline.get_job_status(job_id)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 3600.0,
        progress_callback=None,
    ) -> RenderJob:
        """Wait for job completion."""
        return self.pipeline.wait_for_job(job_id, poll_interval, timeout, progress_callback)

    def download_result(self, job: RenderJob, output_dir: pathlib.Path) -> list[pathlib.Path]:
        """Download render output files."""
        return self.pipeline.download_result(job, output_dir)

    def render_scene(
        self,
        scene_script: pathlib.Path,
        output_dir: pathlib.Path,
        config: RenderConfig = None,
        progress_callback=None,
    ) -> list[pathlib.Path]:
        """Complete pipeline: generate .blend from scene script, render in cloud, download."""
        from fact_agent.render_pipeline import render_scene_direct
        return render_scene_direct(scene_script, config or RenderConfig(), output_dir, self.pipeline)


def check_account() -> dict[str, Any]:
    """Convenience function for backward compatibility."""
    renderer = CloudRenderer()
    return renderer.check_account()


def estimate_free_preview(
    frame_start: int = 1,
    frame_end: int = 1,
    width: int = 640,
    height: int = 360,
    samples: int = 16,
) -> dict[str, Any]:
    """Convenience function for backward compatibility."""
    renderer = CloudRenderer()
    return renderer.estimate_preview(
        frame_start=frame_start,
        frame_end=frame_end,
        width=width,
        height=height,
        samples=samples,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = check_account()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Account check failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)