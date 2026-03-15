import asyncio
import logging
from pathlib import Path
from playwright.async_api import async_playwright, Route
from PIL import Image
import io

logger = logging.getLogger(__name__)

WIDTH = 1280
HEIGHT = 800
RENDER_TIMEOUT_SECONDS = 60


class Renderer:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            args=[
                "--disable-gpu",
                "--disable-lcd-text",
                "--font-render-hinting=none",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            locale="en-US",
            timezone_id="UTC",
        )

        # Block all non-local network requests at the context level
        async def _block_non_local(route: Route) -> None:
            url = route.request.url
            if url.startswith("file://"):
                await route.continue_()
            else:
                logger.debug("Blocked external request: %s", url)
                await route.abort()

        await self._context.route("**/*", _block_non_local)
        return self

    async def __aexit__(self, *exc):
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def render(self, html_path: str | Path) -> Image.Image:
        return await asyncio.wait_for(
            self._render_impl(html_path),
            timeout=RENDER_TIMEOUT_SECONDS,
        )

    async def _render_impl(self, html_path: str | Path) -> Image.Image:
        html_path = Path(html_path).resolve()
        page = await self._context.new_page()
        try:
            await page.goto(
                html_path.as_uri(),
                wait_until="networkidle",
                timeout=30000,
            )

            # Disable animations/transitions for deterministic rendering (after navigation)
            await page.add_style_tag(content="""\
*, *::before, *::after {
    animation-duration: 0s !important;
    animation-delay: 0s !important;
    transition-duration: 0s !important;
    transition-delay: 0s !important;
    scroll-behavior: auto !important;
}
""")

            png_bytes = await page.screenshot(
                type="png",
                clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT},
            )

            return Image.open(io.BytesIO(png_bytes))
        finally:
            await page.close()
