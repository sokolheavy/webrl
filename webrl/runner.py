import asyncio
import base64
import io
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import Sample, load_dataset
from .environment import Environment, StepResult
from .judge import JudgeResult

logger = logging.getLogger(__name__)


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _build_tool_result(result: StepResult) -> dict:
    """Convert a StepResult into an MCP tool result."""
    content = []
    if isinstance(result.output, Image.Image):
        content.append(
            {
                "type": "image",
                "data": _image_to_base64(result.output),
                "mimeType": "image/png",
            }
        )
    else:
        content.append({"type": "text", "text": result.output})

    if result.done:
        content.append(
            {
                "type": "text",
                "text": "No more steps remaining. Please finish now.",
            }
        )
    else:
        content.append(
            {
                "type": "text",
                "text": f"({result.steps_remaining} steps remaining)",
            }
        )

    return {"content": content}


def _save_episode_artifacts(
    save_dir: Path,
    sample: Sample,
    output_dir: Path | None,
    judge_result: JudgeResult,
    steps_used: int,
    turns: int,
    error: str | None,
) -> None:
    """Save episode artifacts for debugging."""
    episode_dir = save_dir / sample.id
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Target screenshot
    shutil.copy2(sample.screenshot_path, episode_dir / "target.png")

    # Output HTML (and any other files the agent wrote)
    if output_dir is not None and output_dir.exists():
        for path in output_dir.rglob("*"):
            if path.is_file():
                dest = episode_dir / path.relative_to(output_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)

    # Rendered screenshot from the judge
    if judge_result.rendered_screenshot is not None:
        judge_result.rendered_screenshot.save(episode_dir / "rendered.png")

    # Score metadata
    meta = {
        "sample_id": sample.id,
        "score": judge_result.score,
        "ssim": judge_result.ssim,
        "lpips": judge_result.lpips,
        "anti_cheat_passed": judge_result.anti_cheat_passed,
        "anti_cheat_failures": judge_result.anti_cheat_failures,
        "steps_used": steps_used,
        "conversation_turns": turns,
        "error": error,
    }
    (episode_dir / "score.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved episode artifacts to %s", episode_dir)


def _create_episode_tools(env: Environment) -> list:
    """Create MCP tools that dispatch through the environment."""
    from claude_agent_sdk import tool

    async def _dispatch(name: str, args: dict) -> dict:
        result = await env.step(name, args)
        logger.info(
            "tool=%s steps_remaining=%d done=%s",
            name,
            result.steps_remaining,
            result.done,
        )
        return _build_tool_result(result)

    @tool(
        "write_file",
        "Write or overwrite a file in the /output/ directory. "
        "Use this to create your index.html and any supporting files.",
        {"path": str, "content": str},
    )
    async def write_file(args: dict) -> dict:
        logger.info("write_file path=%s (%d bytes)", args["path"], len(args["content"]))
        return await _dispatch("write_file", args)

    @tool(
        "read_file",
        "Read a file you have previously written in /output/.",
        {"path": str},
    )
    async def read_file(args: dict) -> dict:
        logger.info("read_file path=%s", args["path"])
        return await _dispatch("read_file", args)

    @tool(
        "preview",
        "Render /output/index.html in a headless browser and return a screenshot. "
        "Use this to compare your progress against the target.",
        {},
    )
    async def preview(args: dict) -> dict:
        logger.info("preview")
        return await _dispatch("preview", args)

    @tool(
        "list_assets",
        "List the image assets available in /assets/. "
        "Reference these in your HTML with relative paths like '../assets/logo.png'.",
        {},
    )
    async def list_assets(args: dict) -> dict:
        logger.info("list_assets")
        return await _dispatch("list_assets", args)

    @tool(
        "view_target",
        "View the target screenshot again to re-examine what you need to reproduce.",
        {},
    )
    async def view_target(args: dict) -> dict:
        logger.info("view_target")
        return await _dispatch("view_target", args)

    return [write_file, read_file, preview, list_assets, view_target]


@dataclass
class EpisodeResult:
    sample_id: str
    judge_result: JudgeResult
    steps_used: int
    conversation_turns: int
    elapsed_seconds: float
    error: str | None = None


async def run_episode(
    env: Environment,
    sample: Sample,
    *,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int | None = None,
    save_dir: Path | None = None,
) -> EpisodeResult:
    """Run a single episode: Claude drives the environment via tool use."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolUseBlock,
        create_sdk_mcp_server,
        query,
    )

    start = time.monotonic()
    turns = 0
    error: str | None = None

    try:
        logger.info("Setting up episode for sample '%s'", sample.id)
        state = await env.setup_episode(sample)
        logger.info(
            "Episode ready: %d assets, max_steps=%d",
            len(state.asset_list),
            env.max_steps,
        )

        tools = _create_episode_tools(env)
        server = create_sdk_mcp_server("episode", "1.0.0", tools)

        tool_names = [f"mcp__episode__{t.name}" for t in tools]

        if max_turns is None:
            max_turns = env.max_steps + 5

        options = ClaudeAgentOptions(
            model=model,
            system_prompt=state.prompt,
            mcp_servers={"episode": server},
            tools=[],
            allowed_tools=tool_names,
            disallowed_tools=[
                "Write",
                "Edit",
                "Read",
                "Bash",
                "Glob",
                "Grep",
                "WebFetch",
                "WebSearch",
                "NotebookEdit",
                "Agent",
                "Skill",
            ],
            permission_mode="bypassPermissions",
            max_turns=max_turns,
            stderr=lambda line: logger.debug("cli: %s", line.rstrip()),
            extra_args={"debug-to-stderr": None},
        )

        logger.info("Starting agent loop (model=%s, max_turns=%d)", model, max_turns)
        async for message in query(
            prompt=(
                "View the target screenshot and recreate it as closely as possible. "
                "Start by examining the target, then write your HTML."
            ),
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                turns += 1
                tool_names_used = []
                text_snippet = None
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_names_used.append(block.name)
                    elif (
                        isinstance(block, TextBlock) and block.text and not text_snippet
                    ):
                        text_snippet = (
                            block.text
                            if len(block.text) <= 200
                            else block.text[:200] + "..."
                        )

                parts = [f"turn {turns}"]
                if text_snippet:
                    parts.append(text_snippet)
                if tool_names_used:
                    parts.append(f"tools: [{', '.join(tool_names_used)}]")
                logger.info(" | ".join(parts))

            elif isinstance(message, ResultMessage):
                logger.info(
                    "result: %s (cost=$%.4f, api_time=%.1fs)",
                    message.subtype,
                    message.total_cost_usd or 0,
                    (message.duration_api_ms or 0) / 1000,
                )
            elif isinstance(message, SystemMessage):
                logger.debug("system: %s %s", message.subtype, message.data)
            else:
                logger.debug("message: %s", type(message).__name__)

        logger.info("Agent finished after %d turns, scoring...", turns)
        judge_result = await env.score()
        logger.info(
            "Score: %.4f (ssim=%.4f, lpips=%.4f, anti_cheat=%s)",
            judge_result.score,
            judge_result.ssim,
            judge_result.lpips,
            "passed" if judge_result.anti_cheat_passed else "FAILED",
        )

    except Exception as exc:
        error = str(exc)
        logger.error("Episode failed: %s", error)
        judge_result = JudgeResult(
            score=0.0,
            ssim=0.0,
            lpips=1.0,
            anti_cheat_passed=False,
            anti_cheat_failures=[f"Runner error: {error}"],
        )
    finally:
        steps_used = env.step_count

        if save_dir is not None:
            _save_episode_artifacts(
                save_dir=save_dir,
                sample=sample,
                output_dir=env._output_dir,
                judge_result=judge_result,
                steps_used=steps_used,
                turns=turns,
                error=error,
            )

        await env.cleanup()

    elapsed = time.monotonic() - start
    return EpisodeResult(
        sample_id=sample.id,
        judge_result=judge_result,
        steps_used=steps_used,
        conversation_turns=turns,
        elapsed_seconds=elapsed,
        error=error,
    )


async def run_batch(
    dataset_dir: Path,
    *,
    model: str = "claude-sonnet-4-20250514",
    max_steps: int = 20,
    preview_budget: int = 10,
    max_turns: int | None = None,
    sample_ids: list[str] | None = None,
    save_dir: Path | None = None,
    concurrency: int = 4,
) -> list[EpisodeResult]:
    """Run episodes concurrently across all samples in a dataset."""
    dataset = load_dataset(dataset_dir)
    if not dataset:
        raise ValueError(f"No samples found in {dataset_dir}")

    samples = dataset
    if sample_ids:
        id_set = set(sample_ids)
        samples = [s for s in samples if s.id in id_set]
        if not samples:
            raise ValueError(f"No matching samples for IDs: {sample_ids}")

    sem = asyncio.Semaphore(concurrency)
    completed_count = 0

    async def run_one(i: int, sample: Sample) -> EpisodeResult:
        nonlocal completed_count
        async with sem:
            print(f"[{i}/{len(samples)}] Running sample '{sample.id}'...")
            env = Environment(
                dataset_dir=dataset_dir,
                max_steps=max_steps,
                preview_budget=preview_budget,
                _preloaded_samples=dataset,
            )
            result = await run_episode(
                env, sample, model=model, max_turns=max_turns,
                save_dir=save_dir,
            )
            completed_count += 1
            status = "ERROR" if result.error else f"score={result.judge_result.score:.4f}"
            print(f"  [{completed_count}/{len(samples)}] {sample.id}: {status} ({result.steps_used} steps, {result.elapsed_seconds:.1f}s)")
            return result

    results = await asyncio.gather(
        *[run_one(i, s) for i, s in enumerate(samples, 1)]
    )
    return list(results)
