def build_prompt(asset_list: list[str], preview_budget: int = 10) -> str:
    assets_section = "\n".join(f"  - {name}" for name in asset_list) if asset_list else "  (none)"

    return f"""\
You are a web developer. You will be shown a screenshot of a website. Your task \
is to recreate that website as a single HTML file that visually matches the \
screenshot as closely as possible.

## Output

Write your HTML to `/output/index.html` using the `write_file` tool.

## Assets

The original site's image assets are available in `/assets/`. Reference them \
using relative paths from your output file (e.g. `../assets/logo.png`).

Available assets:
{assets_section}

## Constraints

1. **No external resources.** Do not load anything from CDNs, external URLs, or \
external fonts. Everything must be inline or reference local assets only.
2. **No base64 screenshot embedding.** Do not embed the target screenshot or any \
large base64-encoded image as a shortcut. Build the page with real HTML and CSS.
3. **Use semantic HTML.** The page must be built with a variety of standard HTML \
elements (headings, paragraphs, sections, divs, lists, etc.). Do not render the \
page as a single `<canvas>`, `<svg>`, or `<img>` element.

## Viewport

The page will be rendered and screenshotted at **1280x800** pixels. Design your \
layout for this exact viewport size.

## Available Tools

- **write_file(path, content)** — Write or overwrite a file in `/output/`. Use \
this to create your `index.html` and any supporting files.
- **read_file(path)** — Read a file you previously wrote in `/output/`.
- **preview()** — Render `/output/index.html` in a headless browser and return a \
screenshot. Use this to compare your progress against the target. You have a \
budget of exactly {preview_budget} preview calls, so use them wisely.
- **list_assets()** — List the image assets available in `/assets/`.
- **view_target()** — View the target screenshot again to re-examine details.

## Strategy

Take an iterative approach:
1. Study the target screenshot carefully. Note the layout, colors, typography, \
spacing, and images used.
2. Write an initial version of `index.html` that captures the overall structure \
and layout.
3. Use `preview()` to see how your page looks and compare it to the target.
4. Refine your HTML and CSS to fix differences — adjust colors, spacing, font \
sizes, alignment, and other visual details.
5. Repeat the preview-and-refine cycle until the result closely matches the target.

Focus on getting the layout structure right first, then fine-tune visual details."""
