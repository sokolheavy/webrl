import random
import shutil
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import Sample, load_dataset
from .judge import JudgeResult, judge
from .prompt import build_prompt
from .renderer import Renderer
from .tools import EpisodeToolkit


@dataclass
class EpisodeState:
    prompt: str
    target_screenshot: Image.Image
    asset_list: list[str]


@dataclass
class StepResult:
    output: str | Image.Image
    done: bool
    steps_remaining: int


class Environment:
    def __init__(
        self,
        dataset_dir: Path,
        max_steps: int = 20,
        preview_budget: int = 10,
        _preloaded_samples: list[Sample] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_steps = max_steps
        self.preview_budget = preview_budget
        self.samples = _preloaded_samples if _preloaded_samples is not None else load_dataset(self.dataset_dir)
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_dir}")

        self._sample: Sample | None = None
        self._toolkit: EpisodeToolkit | None = None
        self._renderer: Renderer | None = None
        self._episode_root: Path | None = None
        self._output_dir: Path | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._step_count = 0
        self._target_image: Image.Image | None = None

    @property
    def step_count(self) -> int:
        return self._step_count

    async def setup_episode(self, sample: Sample | None = None) -> EpisodeState:
        """Set up a new episode, optionally with a specific sample."""
        # Clean up previous episode resources
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

        # Clean up previous episode directory
        if self._episode_root is not None and self._episode_root.exists():
            shutil.rmtree(self._episode_root, ignore_errors=True)
            self._episode_root = None
            self._output_dir = None

        self._sample = sample if sample is not None else random.choice(self.samples)
        self._step_count = 0

        # Initialize exit stack first, then register temp dir for cleanup
        self._exit_stack = AsyncExitStack()

        # Create a per-episode root dir with isolated output and assets symlink
        self._episode_root = Path(tempfile.mkdtemp(prefix="webrl_episode_"))
        # Register cleanup so temp dir is removed even if Renderer init fails.
        # Capture path by value to avoid issues if _episode_root is reassigned.
        episode_root = self._episode_root
        self._exit_stack.callback(
            lambda: shutil.rmtree(episode_root, ignore_errors=True)
            if episode_root.exists()
            else None
        )

        self._output_dir = self._episode_root / "output"
        self._output_dir.mkdir()

        # Symlink assets into episode root so ../assets/ relative paths work
        assets_src = self._sample.screenshot_path.parent / "assets"
        if assets_src.exists():
            assets_link = self._episode_root / "assets"
            assets_link.symlink_to(assets_src.resolve())

        # Initialize renderer via AsyncExitStack
        self._renderer = await self._exit_stack.enter_async_context(Renderer())

        self._toolkit = EpisodeToolkit(
            sample_dir=self._sample.screenshot_path.parent,
            output_dir=self._output_dir,
            renderer=self._renderer,
            preview_budget=self.preview_budget,
        )

        asset_list = self._toolkit.list_assets()
        with Image.open(self._sample.screenshot_path) as img:
            img.load()
            self._target_image = img.copy()
        target_screenshot = self._target_image

        prompt = build_prompt(asset_list, preview_budget=self.preview_budget)

        return EpisodeState(
            prompt=prompt,
            target_screenshot=target_screenshot,
            asset_list=asset_list,
        )

    _INFORMATIONAL_TOOLS = frozenset({"view_target", "list_assets"})

    async def step(self, tool_name: str, tool_input: dict) -> StepResult:
        """Dispatch a tool call and return the result."""
        if self._toolkit is None:
            raise RuntimeError("Call setup_episode() before step()")

        if tool_name not in self._INFORMATIONAL_TOOLS:
            self._step_count += 1
        steps_remaining = self.max_steps - self._step_count
        done = steps_remaining <= 0

        output: str | Image.Image
        match tool_name:
            case "write_file":
                output = self._toolkit.write_file(
                    path=tool_input["path"],
                    content=tool_input["content"],
                )
            case "read_file":
                output = self._toolkit.read_file(path=tool_input["path"])
            case "preview":
                output = await self._toolkit.preview()
            case "list_assets":
                output = ", ".join(self._toolkit.list_assets()) or "No assets available"
            case "view_target":
                output = self._toolkit.view_target()
            case _:
                output = f"Error: unknown tool '{tool_name}'"

        return StepResult(
            output=output,
            done=done,
            steps_remaining=steps_remaining,
        )

    async def score(self) -> JudgeResult:
        """Score the current episode output against the target."""
        if self._sample is None or self._output_dir is None:
            raise RuntimeError("Call setup_episode() before score()")

        output_html = self._output_dir / "index.html"
        if not output_html.exists():
            return JudgeResult(
                score=0.0,
                ssim=0.0,
                lpips=1.0,
                anti_cheat_passed=False,
                anti_cheat_failures=["No index.html found in output directory"],
            )

        return await judge(
            target_screenshot=self._target_image or self._sample.screenshot_path,
            output_html=output_html,
            renderer=self._renderer,
        )

    async def cleanup(self):
        """Release renderer resources and clean up temp directory."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._renderer = None
            self._episode_root = None
            self._output_dir = None
