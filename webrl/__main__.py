import argparse
import asyncio
import logging
import sys
from pathlib import Path


def cmd_capture(args: argparse.Namespace) -> None:
    """Capture a screenshot and assets from a URL."""
    from .data import capture_sample

    sample = asyncio.run(
        capture_sample(args.url, Path(args.output), difficulty=args.difficulty)
    )
    print(f"Saved sample '{sample.id}' to {args.output}")
    print(f"  Screenshot: {sample.screenshot_path}")
    print(f"  Assets:     {sample.assets_dir}")


def cmd_score(args: argparse.Namespace) -> None:
    """Score an output HTML against a target screenshot."""
    from .judge import judge

    target = Path(args.target)
    html = Path(args.html)
    if not target.exists():
        print(f"Error: target screenshot not found: {target}", file=sys.stderr)
        sys.exit(1)
    if not html.exists():
        print(f"Error: HTML file not found: {html}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(judge(target, html))

    print(f"Score:       {result.score:.4f}")
    print(f"SSIM:        {result.ssim:.4f}")
    print(f"LPIPS:       {result.lpips:.4f}")
    print(f"Anti-cheat:  {'PASSED' if result.anti_cheat_passed else 'FAILED'}")
    if result.anti_cheat_failures:
        for failure in result.anti_cheat_failures:
            print(f"  - {failure}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single episode against Claude."""
    from .environment import Environment
    from .runner import run_episode

    dataset = Path(args.dataset)
    env = Environment(
        dataset_dir=dataset,
        max_steps=args.max_steps,
        preview_budget=args.preview_budget,
    )

    # Find the requested sample
    if args.sample:
        matches = [s for s in env.samples if s.id == args.sample]
        if not matches:
            print(f"Error: sample '{args.sample}' not found", file=sys.stderr)
            sys.exit(1)
        sample = matches[0]
    else:
        import random
        sample = random.choice(env.samples)

    save_dir = Path(args.save_dir) if args.save_dir else None
    result = asyncio.run(
        run_episode(
            env,
            sample,
            model=args.model,
            max_turns=args.max_turns,
            save_dir=save_dir,
        )
    )

    if result.error:
        print(f"\nError:       {result.error}")
    print(f"\nSample:      {result.sample_id}")
    print(f"Score:       {result.judge_result.score:.4f}")
    print(f"SSIM:        {result.judge_result.ssim:.4f}")
    print(f"LPIPS:       {result.judge_result.lpips:.4f}")
    print(f"Anti-cheat:  {'PASSED' if result.judge_result.anti_cheat_passed else 'FAILED'}")
    if result.judge_result.anti_cheat_failures:
        for failure in result.judge_result.anti_cheat_failures:
            print(f"  - {failure}")
    print(f"Steps:       {result.steps_used}")
    print(f"Turns:       {result.conversation_turns}")
    print(f"Time:        {result.elapsed_seconds:.1f}s")
    if save_dir:
        print(f"Artifacts:   {save_dir / result.sample_id}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Run all samples in a dataset sequentially."""
    import json
    from .runner import run_batch

    save_dir = Path(args.save_dir) if args.save_dir else None
    results = asyncio.run(
        run_batch(
            Path(args.dataset),
            model=args.model,
            max_steps=args.max_steps,
            preview_budget=args.preview_budget,
            max_turns=args.max_turns,
            sample_ids=args.samples,
            save_dir=save_dir,
            concurrency=args.concurrency,
        )
    )

    # Summary
    scores = [r.judge_result.score for r in results if r.error is None]
    print(f"\n{'=' * 40}")
    print(f"Completed {len(results)} episodes")
    if scores:
        print(f"Mean score:  {sum(scores) / len(scores):.4f}")
        print(f"Min score:   {min(scores):.4f}")
        print(f"Max score:   {max(scores):.4f}")
    errors = [r for r in results if r.error]
    if errors:
        print(f"Errors:      {len(errors)}")

    # Optionally write JSON
    if args.output:
        out = []
        for r in results:
            out.append({
                "sample_id": r.sample_id,
                "score": r.judge_result.score,
                "ssim": r.judge_result.ssim,
                "lpips": r.judge_result.lpips,
                "anti_cheat_passed": r.judge_result.anti_cheat_passed,
                "anti_cheat_failures": r.judge_result.anti_cheat_failures,
                "steps_used": r.steps_used,
                "conversation_turns": r.conversation_turns,
                "elapsed_seconds": r.elapsed_seconds,
                "error": r.error,
            })
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results written to {args.output}")


def cmd_episode(args: argparse.Namespace) -> None:
    """Set up an episode and print its details."""
    from .environment import Environment

    env = Environment(
        dataset_dir=Path(args.dataset),
        max_steps=args.max_steps,
        preview_budget=args.preview_budget,
    )

    async def run() -> None:
        state = await env.setup_episode()
        print("Episode ready\n")
        print("Prompt:")
        print(state.prompt)
        print(f"\nAssets: {', '.join(state.asset_list) or '(none)'}")
        print(f"\nTarget screenshot size: {state.target_screenshot.size}")
        await env.cleanup()

    asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="webrl",
        description="RL environment for screenshot-to-website recreation",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Enable verbose logging (-v info, -vv debug)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # capture
    p_capture = subparsers.add_parser("capture", help="Capture a sample from a URL")
    p_capture.add_argument("url", help="URL to capture")
    p_capture.add_argument("output", help="Output directory for the sample")
    p_capture.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty tier (default: medium)",
    )

    # score
    p_score = subparsers.add_parser(
        "score", help="Score an output HTML against a target"
    )
    p_score.add_argument("target", help="Path to target screenshot (PNG)")
    p_score.add_argument("html", help="Path to output index.html")
    p_score.add_argument("--assets", help="(deprecated, ignored)")

    # run
    p_run = subparsers.add_parser("run", help="Run a single episode against Claude")
    p_run.add_argument("dataset", help="Path to dataset directory")
    p_run.add_argument("--sample", help="Sample ID to run (default: random)")
    p_run.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    p_run.add_argument("--max-steps", type=int, default=20)
    p_run.add_argument("--preview-budget", type=int, default=10)
    p_run.add_argument("--max-turns", type=int, default=None, help="Max conversation turns (default: max-steps + 5)")
    p_run.add_argument("--save-dir", help="Directory to save episode artifacts (target, output, rendered screenshot, scores)")

    # batch
    p_batch = subparsers.add_parser("batch", help="Run all samples in a dataset")
    p_batch.add_argument("dataset", help="Path to dataset directory")
    p_batch.add_argument("--samples", nargs="*", help="Specific sample IDs to run")
    p_batch.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    p_batch.add_argument("--max-steps", type=int, default=20)
    p_batch.add_argument("--preview-budget", type=int, default=10)
    p_batch.add_argument("--max-turns", type=int, default=None, help="Max conversation turns (default: max-steps + 5)")
    p_batch.add_argument("--output", help="Output JSON file for results")
    p_batch.add_argument("--concurrency", type=int, default=4, help="Max concurrent episodes (default: 4)")
    p_batch.add_argument("--save-dir", help="Directory to save episode artifacts for debugging")

    # episode
    p_episode = subparsers.add_parser(
        "episode", help="Set up an episode and print its details"
    )
    p_episode.add_argument("dataset", help="Path to dataset directory")
    p_episode.add_argument("--max-steps", type=int, default=20)
    p_episode.add_argument("--preview-budget", type=int, default=10)

    args = parser.parse_args()

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    match args.command:
        case "capture":
            cmd_capture(args)
        case "score":
            cmd_score(args)
        case "run":
            cmd_run(args)
        case "batch":
            cmd_batch(args)
        case "episode":
            cmd_episode(args)


if __name__ == "__main__":
    main()
