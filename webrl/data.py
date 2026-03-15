import asyncio
import json
import logging
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

WIDTH = 1280
HEIGHT = 800

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".avif"}
EXCLUDED_EXTENSIONS = {".js", ".css", ".html", ".htm", ".woff", ".woff2", ".ttf", ".otf", ".eot"}


@dataclass
class Sample:
    id: str
    screenshot_path: Path
    assets_dir: Path
    source_url: str | None
    difficulty: str  # "easy", "medium", "hard"


def _stable_filename(url: str) -> str:
    """Generate a stable filename from a URL, preserving the extension."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    name = Path(parsed.path).stem or "image"
    # Sanitize name
    name = re.sub(r"[^\w\-.]", "_", name)[:50]
    return f"{name}_{url_hash}{ext}"


def _is_image_url(url: str) -> bool:
    """Check if a URL likely points to an image based on extension."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext in EXCLUDED_EXTENSIONS:
        return False
    if ext in IMAGE_EXTENSIONS:
        return True
    return False


async def capture_sample(
    url: str,
    output_dir: Path,
    difficulty: str = "medium",
) -> Sample:
    """Capture a screenshot and extract image assets from a URL.

    Args:
        url: The URL to capture.
        output_dir: Directory to save the sample to.
        difficulty: Difficulty tier ("easy", "medium", "hard").

    Returns:
        A Sample object pointing to the saved data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    screenshot_path = output_dir / "screenshot.png"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        context = await browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
        )
        page = await context.new_page()

        # Intercept responses to capture image URLs by Content-Type,
        # catching images served without standard file extensions.
        content_type_image_urls: set[str] = set()

        async def _on_response(response):
            ct = response.headers.get("content-type", "")
            if ct.startswith("image/") and response.url and not response.url.startswith("data:"):
                content_type_image_urls.add(response.url)

        page.on("response", _on_response)

        try:
            await page.goto(url, wait_until="networkidle", timeout=15_000)
        except Exception:
            await page.goto(url, wait_until="domcontentloaded", timeout=10_000)

        # Take screenshot
        await page.screenshot(
            path=str(screenshot_path),
            type="png",
            clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT},
        )

        # Extract image URLs
        image_urls: set[str] = set()

        # <img> src attributes
        img_srcs = await page.eval_on_selector_all(
            "img[src]",
            "els => els.map(e => e.src)",
        )
        image_urls.update(img_srcs)

        # CSS background-image URLs (computed styles)
        bg_urls = await page.evaluate("""
            () => {
                const urls = [];
                for (const el of document.querySelectorAll('*')) {
                    const bg = getComputedStyle(el).backgroundImage;
                    if (bg && bg !== 'none') {
                        const matches = bg.matchAll(/url\\(["']?([^"')]+)["']?\\)/g);
                        for (const m of matches) urls.push(m[1]);
                    }
                }
                return urls;
            }
        """)
        image_urls.update(bg_urls)

        # Favicon and icon references
        icon_hrefs = await page.eval_on_selector_all(
            'link[rel*="icon"][href]',
            "els => els.map(e => e.href)",
        )
        image_urls.update(icon_hrefs)

        await context.close()
        await browser.close()

    # Resolve relative URLs and filter to images
    resolved_urls: set[str] = set()
    for img_url in image_urls:
        if not img_url or img_url.startswith("data:"):
            continue
        absolute = urljoin(url, img_url)
        if _is_image_url(absolute):
            resolved_urls.add(absolute)
    # Include images detected by Content-Type during navigation (catches
    # CDN images without standard file extensions).
    resolved_urls.update(content_type_image_urls)

    # Download assets
    downloaded = await _download_assets(resolved_urls, assets_dir)

    sample_id = output_dir.name

    # Save metadata
    metadata = {
        "id": sample_id,
        "source_url": url,
        "difficulty": difficulty,
        "assets": downloaded,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return Sample(
        id=sample_id,
        screenshot_path=screenshot_path,
        assets_dir=assets_dir,
        source_url=url,
        difficulty=difficulty,
    )


async def _download_assets(urls: set[str], assets_dir: Path) -> list[str]:
    """Download image assets to the assets directory. Returns list of saved filenames."""
    import httpx

    saved: list[str] = []
    sem = asyncio.Semaphore(10)

    async def download_one(url: str) -> str | None:
        filename = _stable_filename(url)
        dest = assets_dir / filename
        async with sem:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    return None
                dest.write_bytes(resp.content)
                return filename
            except Exception as exc:
                logger.warning("Failed to download %s: %s", url, exc)
                return None

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        results = await asyncio.gather(*[download_one(url) for url in urls])
        saved = [f for f in results if f is not None]
    return saved


def load_sample(sample_dir: Path) -> Sample:
    """Load an existing sample from disk.

    Args:
        sample_dir: Path to the sample directory.

    Returns:
        A Sample object.

    Raises:
        FileNotFoundError: If screenshot.png or assets/ directory is missing.
    """
    sample_dir = Path(sample_dir)
    screenshot_path = sample_dir / "screenshot.png"
    assets_dir = sample_dir / "assets"

    if not screenshot_path.exists():
        raise FileNotFoundError(f"Missing screenshot: {screenshot_path}")

    # Load metadata if available
    metadata_path = sample_dir / "metadata.json"
    source_url = None
    difficulty = "medium"
    sample_id = sample_dir.name

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        source_url = metadata.get("source_url")
        difficulty = metadata.get("difficulty", "medium")
        sample_id = metadata.get("id", sample_id)

    return Sample(
        id=sample_id,
        screenshot_path=screenshot_path,
        assets_dir=assets_dir,
        source_url=source_url,
        difficulty=difficulty,
    )


def load_dataset(dataset_dir: Path) -> list[Sample]:
    """Load all samples from a dataset directory.

    Each subdirectory containing a screenshot.png is treated as a sample.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        List of Sample objects, sorted by id.
    """
    dataset_dir = Path(dataset_dir)
    samples = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and (child / "screenshot.png").exists():
            samples.append(load_sample(child))
    return samples
