from pathlib import Path

from PIL import Image

from .renderer import Renderer

MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB


class EpisodeToolkit:
    def __init__(
        self,
        sample_dir: str | Path,
        output_dir: str | Path,
        renderer: Renderer,
        preview_budget: int = 10,
    ):
        self.sample_dir = Path(sample_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.renderer = renderer
        self.preview_budget = preview_budget
        self.preview_count = 0
        self._cached_assets: list[str] | None = None

    # --- helpers ---

    def _resolve_output_path(self, path: str) -> Path:
        """Resolve a path relative to /output/ and ensure it stays inside."""
        resolved = (self.output_dir / path).resolve()
        if not resolved.is_relative_to(self.output_dir):
            raise ValueError(f"Path '{path}' escapes the output directory")
        return resolved

    # --- tools ---

    def write_file(self, path: str, content: str) -> str:
        try:
            resolved = self._resolve_output_path(path)
        except ValueError as e:
            return f"Error: {e}"
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > MAX_FILE_SIZE_BYTES:
            return f"Error: file size {len(content_bytes)} bytes exceeds limit ({MAX_FILE_SIZE_BYTES} bytes)"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content_bytes)
        return f"Wrote {resolved.relative_to(self.output_dir)}"

    def read_file(self, path: str) -> str:
        try:
            resolved = self._resolve_output_path(path)
        except ValueError as e:
            return f"Error: {e}"
        if not resolved.exists():
            return f"Error: file '{path}' does not exist"
        size = resolved.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            return f"Error: file size {size} bytes exceeds limit ({MAX_FILE_SIZE_BYTES} bytes)"
        return resolved.read_text()

    async def preview(self) -> Image.Image | str:
        if self.preview_count >= self.preview_budget:
            return f"Error: preview budget exceeded ({self.preview_budget} calls)"
        index = self.output_dir / "index.html"
        if not index.exists():
            return "Error: /output/index.html does not exist"
        self.preview_count += 1
        return await self.renderer.render(index)

    def list_assets(self) -> list[str]:
        if self._cached_assets is None:
            assets_dir = self.sample_dir / "assets"
            if not assets_dir.exists():
                self._cached_assets = []
            else:
                self._cached_assets = sorted(p.name for p in assets_dir.iterdir() if p.is_file())
        return self._cached_assets

    def view_target(self) -> Image.Image | str:
        target = self.sample_dir / "screenshot.png"
        if not target.exists():
            return "Error: target screenshot not found"
        img = Image.open(target)
        img.load()
        return img
