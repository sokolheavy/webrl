from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .anti_cheat import run_anti_cheat
from .renderer import Renderer
from .similarity import ssim_score, lpips_score

# Weighting for the final score formula.
# LPIPS (learned perceptual similarity) is weighted higher because it correlates
# better with human perception of visual similarity than SSIM.
SSIM_WEIGHT = 0.3
LPIPS_WEIGHT = 0.7

# Score when anti-cheat checks fail. Zero removes any reward signal for
# cheating, ensuring honest low-quality attempts always beat cheats in RL.
ANTI_CHEAT_FAILURE_SCORE = 0.0


@dataclass
class JudgeResult:
    score: float  # 0.0 - 1.0
    ssim: float  # raw SSIM score
    lpips: float  # raw LPIPS distance
    anti_cheat_passed: bool
    anti_cheat_failures: list[str] = field(default_factory=list)
    rendered_screenshot: Image.Image | None = None


async def judge(
    target_screenshot: Path | Image.Image,
    output_html: Path,
    renderer: Renderer | None = None,
) -> JudgeResult:
    html = output_html.read_text()
    passed, failures = run_anti_cheat(html)

    if renderer is not None:
        rendered = await renderer.render(output_html)
    else:
        async with Renderer() as r:
            rendered = await r.render(output_html)

    if isinstance(target_screenshot, Image.Image):
        target = target_screenshot
    else:
        target = Image.open(target_screenshot)
        target.load()

    ssim = ssim_score(target, rendered)
    dist = lpips_score(target, rendered)
    ssim_normalized = max(0.0, min(1.0, ssim))
    lpips_normalized = max(0.0, min(1.0, dist))
    score = SSIM_WEIGHT * ssim_normalized + LPIPS_WEIGHT * (1.0 - lpips_normalized)

    if not passed:
        score = ANTI_CHEAT_FAILURE_SCORE

    return JudgeResult(
        score=score,
        ssim=ssim,
        lpips=dist,
        anti_cheat_passed=passed,
        anti_cheat_failures=failures,
        rendered_screenshot=rendered,
    )
