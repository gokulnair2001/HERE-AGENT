#!/usr/bin/env python3
"""Convert an HTML file to a clean Markdown file by removing noise."""

import argparse
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    sys.exit("Missing dependency: pip install beautifulsoup4")

try:
    from markdownify import markdownify as md
except ImportError:
    sys.exit("Missing dependency: pip install markdownify")


NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside", "iframe",
    "noscript", "svg", "form", "button", "input", "select", "textarea",
    "link", "meta", "object", "embed", "applet",
]

NOISE_CLASSES = [
    "sidebar", "menu", "nav", "navbar", "footer", "header", "ads", "ad",
    "advertisement", "banner", "popup", "modal", "cookie", "social",
    "share", "comment", "comments", "related", "recommended", "widget",
    "breadcrumb", "pagination",
]

NOISE_IDS = [
    "sidebar", "menu", "nav", "navbar", "footer", "header", "ads", "ad",
    "banner", "popup", "modal", "cookie", "comments", "related", "widget",
]


def remove_noise(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove non-content elements from the parsed HTML."""
    # Remove noise tags entirely
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove elements with noisy class names
    for cls in NOISE_CLASSES:
        for tag in soup.find_all(class_=lambda c: c and cls in " ".join(c).lower()):
            tag.decompose()

    # Remove elements with noisy IDs
    for id_pattern in NOISE_IDS:
        for tag in soup.find_all(id=lambda i: i and id_pattern in i.lower()):
            tag.decompose()

    # Remove hidden elements
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()

    return soup


def extract_content(soup: BeautifulSoup) -> str:
    """Try to find the main content area, fall back to body or full soup."""
    for selector in ["main", "article", '[role="main"]', "#content", ".content", "#main", ".main"]:
        content = soup.select_one(selector)
        if content:
            return str(content)

    body = soup.find("body")
    return str(body) if body else str(soup)


def html_to_markdown(html: str) -> str:
    """Convert raw HTML string to clean Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    soup = remove_noise(soup)
    content_html = extract_content(soup)

    markdown = md(
        content_html,
        heading_style="ATX",
        bullets="-",
        strip=["img"],
        newline_style="backslash",
    )

    # Collapse excessive blank lines
    lines = markdown.splitlines()
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip() + "\n"


def convert_file(input_path: Path, output_path: Path) -> None:
    """Convert a single HTML file to Markdown."""
    html = input_path.read_text(encoding="utf-8", errors="replace")
    markdown = html_to_markdown(html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Converted: {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HTML file(s) to clean Markdown.")
    parser.add_argument("input", help="Path to an HTML file or a folder containing HTML files")
    parser.add_argument("-o", "--output", help="Output .md file (single file mode) or output directory (folder mode)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file mode
        output_path = Path(args.output) if args.output else input_path.with_suffix(".md")
        convert_file(input_path, output_path)

    elif input_path.is_dir():
        # Folder mode — recursively find all .html files
        html_files = sorted(input_path.rglob("*.html"))
        if not html_files:
            sys.exit(f"No .html files found in: {input_path}")

        print(f"Found {len(html_files)} HTML file(s) in {input_path}\n")

        for html_file in html_files:
            if args.output:
                # Dump all .md files into a single output directory
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / html_file.with_suffix(".md").name
            else:
                # Place .md next to the original .html
                output_path = html_file.with_suffix(".md")
            convert_file(html_file, output_path)

        print(f"\nDone. Converted {len(html_files)} file(s).")

    else:
        sys.exit(f"Path not found: {input_path}")


if __name__ == "__main__":
    main()
