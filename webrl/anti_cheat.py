import base64
import re

from bs4 import BeautifulSoup, Tag

BASE64_DATA_PATTERN = re.compile(r"data:[^;]*;base64,([A-Za-z0-9+/=\s]+)")
MAX_BASE64_BYTES = 10 * 1024  # 10 KB
MIN_ELEMENT_COUNT = 10
MIN_UNIQUE_TAGS = 5
MAX_DOM_DEPTH = 50
MIN_DOM_DEPTH = 3

CheckResult = tuple[bool, str]


def _decoded_base64_size(b64_data: str) -> int:
    cleaned = re.sub(r"\s", "", b64_data)
    try:
        return len(base64.b64decode(cleaned))
    except Exception:
        return 0


def check_no_large_base64_images(html: str) -> CheckResult:
    """Reject if any base64-encoded image exceeds the size threshold.

    Scans the entire raw HTML to catch all embedding vectors (src, srcset,
    poster, object data, style attributes, style blocks, etc.).
    """
    for match in BASE64_DATA_PATTERN.finditer(html):
        size = _decoded_base64_size(match.group(1))
        if size > MAX_BASE64_BYTES:
            return (
                False,
                f"Base64 image found with decoded size {size} bytes (limit {MAX_BASE64_BYTES})",
            )

    return True, "No large base64 images found"


_FULL_WIDTH_RE = re.compile(
    r"(?:min-)?width\s*:\s*(?:calc\s*\(\s*)?100(?:\.0+)?\s*(?:vw|%)(?:\s*\))?",
    re.IGNORECASE,
)
_FULL_HEIGHT_RE = re.compile(
    r"(?:min-)?height\s*:\s*(?:calc\s*\(\s*)?100(?:\.0+)?\s*(?:vh|%)(?:\s*\))?",
    re.IGNORECASE,
)


def _collect_css_for_tag(tag: Tag, soup: BeautifulSoup) -> str:
    """Collect inline style + any <style> block rules targeting this tag by id/class."""
    parts = [tag.get("style", "")]
    tag_id = tag.get("id", "")
    tag_classes = tag.get("class", [])
    # Use word-boundary matching to avoid false positives
    # (e.g. "div" should not match "divider").
    tag_pattern = re.compile(r"(?<![a-zA-Z0-9_-])" + re.escape(tag.name) + r"(?![a-zA-Z0-9_-])")
    for style_tag in soup.find_all("style"):
        if style_tag.string:
            css_text = style_tag.string
            if (
                tag_pattern.search(css_text)
                or (tag_id and re.search(r"#" + re.escape(tag_id) + r"(?![a-zA-Z0-9_-])", css_text))
                or any(re.search(r"\." + re.escape(c) + r"(?![a-zA-Z0-9_-])", css_text) for c in tag_classes)
            ):
                parts.append(css_text)
    return " ".join(parts)


def _is_viewport_covering(tag: Tag, soup: BeautifulSoup | None = None) -> bool:
    """Check if an element has styles that cover the full viewport."""
    if soup is not None:
        css = _collect_css_for_tag(tag, soup)
    else:
        css = tag.get("style", "")
    if not css:
        return False
    return bool(_FULL_WIDTH_RE.search(css)) and bool(_FULL_HEIGHT_RE.search(css))


_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def _is_visible(tag: Tag) -> bool:
    """Check if a tag is likely visible (not display:none)."""
    style = tag.get("style", "")
    return not bool(_HIDDEN_RE.search(style)) if style else True


def check_no_single_element_tricks(soup: BeautifulSoup) -> CheckResult:
    """Reject if body resolves to a single canvas/svg/img covering the viewport."""
    body = soup.find("body")
    if not body:
        return True, "No body element found"

    # Walk down through single-child wrappers, ignoring hidden siblings
    node = body
    while True:
        children = [c for c in node.children if isinstance(c, Tag) and _is_visible(c)]
        if len(children) != 1:
            break
        node = children[0]

    if node is body:
        return True, "Body has multiple child elements"

    if node.name in ("canvas", "svg", "img"):
        if _is_viewport_covering(node, soup):
            return (
                False,
                f"Single <{node.name}> element covers the full viewport",
            )
        # Even without explicit viewport styles, a lone canvas/svg/img is suspicious
        return (
            False,
            f"Body effectively contains only a single <{node.name}> element",
        )

    return True, "No single-element viewport tricks detected"


def _visible_elements(body: Tag) -> list[Tag]:
    """Return all visible elements in body (filtering out display:none subtrees).

    Uses a top-down traversal that prunes hidden subtrees for O(n) performance.
    """
    visible: list[Tag] = []
    stack = list(reversed([c for c in body.children if isinstance(c, Tag)]))
    while stack:
        tag = stack.pop()
        if not _is_visible(tag):
            continue  # prune entire hidden subtree
        visible.append(tag)
        # Push children in reverse so we process them in document order
        for child in reversed([c for c in tag.children if isinstance(c, Tag)]):
            stack.append(child)
    return visible


def check_minimum_element_count(soup: BeautifulSoup) -> CheckResult:
    """Reject if fewer than MIN_ELEMENT_COUNT visible elements in body."""
    body = soup.find("body")
    if not body:
        return False, "No body element found"

    count = len(_visible_elements(body))
    if count < MIN_ELEMENT_COUNT:
        return (
            False,
            f"Only {count} visible DOM elements in body (minimum {MIN_ELEMENT_COUNT})",
        )

    return True, f"{count} visible DOM elements found"


def check_element_diversity(soup: BeautifulSoup) -> CheckResult:
    """Reject if fewer than MIN_UNIQUE_TAGS unique tag names among visible elements."""
    body = soup.find("body")
    if not body:
        return False, "No body element found"

    unique_tags = {tag.name for tag in _visible_elements(body)}
    if len(unique_tags) < MIN_UNIQUE_TAGS:
        return (
            False,
            f"Only {len(unique_tags)} unique visible tags ({', '.join(sorted(unique_tags))}); minimum {MIN_UNIQUE_TAGS}",
        )

    return True, f"{len(unique_tags)} unique visible tags found"


def _compute_depth(tag: Tag) -> int:
    """Compute max depth of the DOM tree rooted at tag (iterative)."""
    max_depth = 0
    stack = [(tag, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            max_depth = depth
        for child in node.children:
            if isinstance(child, Tag):
                stack.append((child, depth + 1))
    return max_depth


def check_no_degenerate_nesting(soup: BeautifulSoup) -> CheckResult:
    """Reject if DOM is too deep or completely flat."""
    body = soup.find("body")
    if not body:
        return False, "No body element found"

    depth = _compute_depth(body)
    if depth > MAX_DOM_DEPTH:
        return False, f"DOM depth is {depth} (maximum {MAX_DOM_DEPTH})"
    if depth < MIN_DOM_DEPTH:
        return False, f"DOM depth is {depth} (minimum {MIN_DOM_DEPTH})"

    return True, f"DOM depth is {depth}"


def check_no_viewport_background_image(soup: BeautifulSoup) -> CheckResult:
    """Reject if a viewport-covering element uses a data: background-image."""
    bg_pattern = re.compile(r"background-image\s*:\s*url\(\s*data:", re.IGNORECASE)

    for tag in soup.find_all(True):
        style = tag.get("style", "")
        if style and bg_pattern.search(style) and _is_viewport_covering(tag, soup):
            return (
                False,
                f"<{tag.name}> covers viewport with a data: background-image",
            )

    # For <style> blocks, only flag if combined with low element count
    # (a legitimate page with many elements may use small data-URI icons in CSS)
    body = soup.find("body")
    element_count = len(body.find_all(True)) if body else 0
    for style_tag in soup.find_all("style"):
        if style_tag.string and bg_pattern.search(style_tag.string):
            if element_count < MIN_ELEMENT_COUNT:
                return (
                    False,
                    "A <style> block contains a data: background-image with low element count (potential viewport cover)",
                )

    return True, "No viewport-covering background-image detected"


_SOUP_CHECKS = [
    check_no_single_element_tricks,
    check_minimum_element_count,
    check_element_diversity,
    check_no_degenerate_nesting,
    check_no_viewport_background_image,
]

ALL_CHECKS = [check_no_large_base64_images, *_SOUP_CHECKS]


def run_anti_cheat(html: str) -> tuple[bool, list[str]]:
    """Run all anti-cheat checks on the given HTML string.

    Returns (all_passed, failures) where failures is a list of reason strings.
    """
    soup = BeautifulSoup(html, "html.parser")
    failures: list[str] = []

    # Base64 check runs on raw HTML to catch all embedding vectors
    passed, reason = check_no_large_base64_images(html)
    if not passed:
        failures.append(reason)

    for check in _SOUP_CHECKS:
        passed, reason = check(soup)
        if not passed:
            failures.append(reason)

    return len(failures) == 0, failures
