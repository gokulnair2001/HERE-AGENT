#!/usr/bin/env python3
"""
RAG Agent — chat with your docs.

Single-file CLI that converts HTML to Markdown (if needed), indexes docs,
and starts an interactive agent with tool calling.

Usage:
  python rag_agent.py /path/to/html-or-md-docs
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

REQUIRED_PACKAGES = [
    "langchain-community", "langchain-text-splitters", "chromadb",
    "sentence-transformers", "groq", "anthropic", "rich", "python-dotenv",
    "beautifulsoup4", "markdownify",
]

HSA_HOME = Path.home() / ".hsa"
HSA_VENV = HSA_HOME / "venv"
HSA_CONFIG_FILE = HSA_HOME / "config.json"


def _load_config() -> dict:
    """Load persisted config from ~/.hsa/config.json."""
    if HSA_CONFIG_FILE.exists():
        try:
            return json.loads(HSA_CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_config(data: dict) -> None:
    """Merge data into ~/.hsa/config.json."""
    HSA_HOME.mkdir(parents=True, exist_ok=True)
    existing = _load_config()
    existing.update(data)
    HSA_CONFIG_FILE.write_text(json.dumps(existing, indent=2))


def _is_auth_error(exc: Exception) -> bool:
    """Return True if the exception looks like an invalid/expired API key."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("401", "403", "authentication", "invalid api key", "api_key", "unauthorized", "invalid x-api-key"))


def _in_hsa_venv() -> bool:
    """Check if we're running inside the hsa managed venv."""
    return str(HSA_VENV) in sys.executable


def _bootstrap_venv():
    """Create the hsa venv, install deps, and re-exec this script inside it."""
    HSA_HOME.mkdir(parents=True, exist_ok=True)

    if not HSA_VENV.exists():
        print("Creating virtual environment (one-time)...")
        subprocess.check_call([sys.executable, "-m", "venv", str(HSA_VENV)])

    venv_python = str(HSA_VENV / "bin" / "python")

    print("Installing dependencies (one-time)...")
    subprocess.check_call(
        [venv_python, "-m", "pip", "install", "-q"] + REQUIRED_PACKAGES,
        stdout=subprocess.DEVNULL,
    )

    # Re-exec this script using the venv python
    os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])


# --- Bootstrap: handle --uninstall early (no deps needed) ---
if "--uninstall" in sys.argv:
    CLI_NAME = "hsa"
    for _bd in [Path.home() / ".local" / "bin", Path("/usr/local/bin")]:
        _wp = _bd / CLI_NAME
        if _wp.exists():
            _wp.unlink()
            print(f"Removed: {_wp}")
    if HSA_HOME.exists():
        shutil.rmtree(HSA_HOME)
        print(f"Removed: {HSA_HOME}")
    sys.exit(0)

# --- Bootstrap: ensure we're in the venv with deps installed ---
if not _in_hsa_venv():
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        _bootstrap_venv()  # re-execs into venv, never returns
else:
    # In venv but deps might be missing (first run after venv creation)
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
        )

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md_convert

load_dotenv()

# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside", "iframe",
    "noscript", "svg", "form", "button", "input", "select", "textarea",
    "link", "meta", "object", "embed", "applet",
]

_NOISE_CLASSES = [
    "sidebar", "menu", "nav", "navbar", "footer", "header", "ads", "ad",
    "advertisement", "banner", "popup", "modal", "cookie", "social",
    "share", "comment", "comments", "related", "recommended", "widget",
    "breadcrumb", "pagination",
]

_NOISE_IDS = [
    "sidebar", "menu", "nav", "navbar", "footer", "header", "ads", "ad",
    "banner", "popup", "modal", "cookie", "comments", "related", "widget",
]


def _remove_noise(soup: BeautifulSoup) -> BeautifulSoup:
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for cls in _NOISE_CLASSES:
        for tag in soup.find_all(class_=lambda c: c and cls in " ".join(c).lower()):
            tag.decompose()
    for id_pattern in _NOISE_IDS:
        for tag in soup.find_all(id=lambda i: i and id_pattern in i.lower()):
            tag.decompose()
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()
    return soup


def _extract_content(soup: BeautifulSoup) -> str:
    for selector in ["main", "article", '[role="main"]', "#content", ".content", "#main", ".main"]:
        content = soup.select_one(selector)
        if content:
            return str(content)
    body = soup.find("body")
    return str(body) if body else str(soup)


def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    soup = _remove_noise(soup)
    content_html = _extract_content(soup)
    markdown = md_convert(
        content_html, heading_style="ATX", bullets="-",
        strip=["img"], newline_style="backslash",
    )
    lines = markdown.splitlines()
    cleaned, blank_count = [], 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def convert_html_to_md(html_dir: str, md_dir: str) -> str:
    """Convert all .html files in html_dir to .md files in md_dir. Returns md_dir path."""
    html_path = Path(html_dir)
    out_path = Path(md_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    html_files = sorted(html_path.rglob("*.html"))
    if not html_files:
        sys.exit(f"No .html files found in: {html_path}")

    console.print(f"[info]Converting {len(html_files)} HTML file(s) to Markdown...[/info]")
    for html_file in html_files:
        html = html_file.read_text(encoding="utf-8", errors="replace")
        markdown = _html_to_markdown(html)
        out_file = out_path / html_file.with_suffix(".md").name
        out_file.write_text(markdown, encoding="utf-8")

    console.print(f"[success]Converted {len(html_files)} files → {out_path}[/success]")
    return str(out_path)

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "success": "bold green",
    "source": "dim italic",
    "user_prompt": "bold bright_green",
})
console = Console(theme=custom_theme)

DEFAULT_VECTORDB_DIR = "./vectordb"
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_TOP_K = 8

AVAILABLE_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ],
    "claude": [
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-haiku-4-5-20251001",
    ],
}

# ---------------------------------------------------------------------------
# Platform configuration — auto-detected or user-specified
# ---------------------------------------------------------------------------

SDK_NAME = "HERE SDK"

PLATFORM_CONFIGS = {
    "ios": {
        "platform_label": "iOS",
        "sdk_name": f"{SDK_NAME} for iOS (Navigate Edition)",
        "language": "Swift",
        "language_keywords": ["swift", "objective-c", "ios", "uikit", "xcode"],
        "code_fence": "swift",
    },
    "android": {
        "platform_label": "Android",
        "sdk_name": f"{SDK_NAME} for Android (Navigate Edition)",
        "language": "Kotlin",
        "language_keywords": ["kotlin", "java", "android", "gradle"],
        "code_fence": "kotlin",
    },
    "flutter": {
        "platform_label": "Flutter",
        "sdk_name": f"{SDK_NAME} for Flutter (Navigate Edition)",
        "language": "Dart",
        "language_keywords": ["dart", "flutter", "widget"],
        "code_fence": "dart",
    },
    "generic": {
        "platform_label": "Generic",
        "sdk_name": SDK_NAME,
        "language": "the SDK's native language",
        "language_keywords": [],
        "code_fence": "",
    },
}


def detect_platform(docs_path: str) -> str:
    """Auto-detect the platform from the docs directory name.

    Looks for patterns like 'heresdk-ios-*', 'heresdk-android-*', etc.
    Falls back to 'generic' if nothing matches.
    """
    name = Path(docs_path).name.lower()
    # Walk up one level too (html files may be nested)
    parent_name = Path(docs_path).parent.name.lower()
    for candidate in (name, parent_name):
        for platform in ("ios", "android", "flutter"):
            if platform in candidate:
                return platform
    # Scan filenames for language hints
    sample_files = list(Path(docs_path).glob("*.md"))[:5]
    for f in sample_files:
        content = f.read_text(encoding="utf-8", errors="replace")[:2000].lower()
        if "uikit" in content or "swift" in content or "objective-c" in content:
            return "ios"
        if "kotlin" in content or "android" in content:
            return "android"
        if "dart" in content or "flutter" in content:
            return "flutter"
    return "generic"


def get_platform_config(platform: str) -> dict:
    """Return the config dict for the given platform key."""
    return PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS["generic"])


def _pick_provider() -> str:
    """Ask Groq or Claude — skip if already saved in config."""
    cfg = _load_config()
    saved = cfg.get("provider")
    if saved in ("groq", "claude"):
        console.print(f"[info]Provider: [bold]{saved}[/bold]  [dim](saved — delete ~/.hsa/config.json to reset)[/dim][/info]")
        return saved
    console.print("[bold]Select LLM provider:[/bold]")
    console.print("  [cyan]1[/cyan]. Groq    [dim](free & fast — console.groq.com)[/dim]")
    console.print("  [cyan]2[/cyan]. Claude  [dim](Anthropic — console.anthropic.com)[/dim]")
    choice = Prompt.ask("Enter 1 or 2", default="1").strip()
    provider = "claude" if choice == "2" else "groq"
    _save_config({"provider": provider})
    return provider


def _pick_model(provider: str) -> str:
    """Interactive model picker — shows models for the active provider."""
    models = AVAILABLE_MODELS[provider]
    default = DEFAULT_GROQ_MODEL if provider == "groq" else DEFAULT_CLAUDE_MODEL
    console.print(f"[bold]Select a {provider.capitalize()} model:[/bold]")
    for i, model in enumerate(models, 1):
        marker = " [dim](default)[/dim]" if model == default else ""
        console.print(f"  [cyan]{i}[/cyan]. {model}{marker}")
    idx_default = str(models.index(default) + 1) if default in models else "1"
    choice = Prompt.ask("Enter number or model name", default=idx_default).strip()
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        selected = models[int(choice) - 1]
    elif choice in models:
        selected = choice
    else:
        matches = [m for m in models if choice.lower() in m.lower()]
        selected = matches[0] if matches else default
    console.print(f"Using model: [cyan]{selected}[/cyan]\n")
    return selected

SYSTEM_PROMPT = """\
You are a precise technical assistant for the {sdk_name}.

━━━ STRICT GROUNDING RULES ━━━
1. CONTEXT IS YOUR ONLY SOURCE. Do NOT use any prior knowledge about the HERE SDK, its APIs,
   or any external documentation. Every fact in your answer must trace back to a chunk below.
2. BEFORE writing your answer, silently scan the context and identify which chunk(s) directly
   support each part of your response. If no chunk supports a claim, omit that claim entirely.
3. PARTIAL CONTEXT: If the context partially answers the question, answer what is confirmed,
   then explicitly state: "⚠ The context does not cover [specific missing part] — try rephrasing
   or check if those docs are indexed."
4. NO ANSWER: If the context contains nothing relevant, respond exactly with:
   "I don't have enough information in the indexed documentation to answer this.
    The relevant docs may not be indexed — try rephrasing, or run with --fresh to re-ingest."
5. EXACT NAMES ONLY: Copy class names, method names, parameter names, and return types
   character-for-character from the context. Never guess or normalize capitalization.
6. NEVER invent method signatures, parameter types, return types, or enum values.
   If a signature is only partially visible in the context, say so explicitly.
7. SOURCE ATTRIBUTION: Every class or method you mention must be followed by its source
   in parentheses, e.g. MapView (MapView.md).

━━━ RESPONSE FORMAT ━━━
{skill_instructions}

━━━ CONTEXT ━━━
{context}

━━━ QUESTION ━━━
{question}

━━━ ANSWER ━━━
(Ground every statement in the context above. Cite sources inline.)"""

# ---------------------------------------------------------------------------
# Skills — auto-detected from the user's question
# ---------------------------------------------------------------------------

SKILLS = {
    "code": {
        "keywords": ["code", "example", "snippet", "implement", "how to", "show me", "write", "sample", "usage"],
        "instructions": (
            "SKILL: Code Generation\n"
            "- Write a self-contained {language} example that directly answers the question.\n"
            "- Use ONLY class names, method names, and signatures that appear verbatim in the context.\n"
            "- If the context shows a method signature, reproduce it exactly — do not add or remove parameters.\n"
            "- Include all necessary imports shown in the context.\n"
            "- Add concise inline comments on non-obvious lines.\n"
            "- Wrap code in ```{code_fence} fenced blocks.\n"
            "- If a required step (e.g. initialisation, permission) is mentioned in the context but the\n"
            "  full code is not shown, add a TODO comment instead of inventing code.\n"
            "- Do NOT produce a code example if the context lacks enough detail — explain what is missing instead."
        ),
    },
    "compare": {
        "keywords": ["compare", "difference", "vs", "versus", "differ", "which one", "better", "alternative", "instead of"],
        "instructions": (
            "SKILL: API Comparison\n"
            "- Present a Markdown table: Feature | {first} | {second} (derive names from the question).\n"
            "- Populate cells ONLY with information found in the context; write '—' where context is silent.\n"
            "- Highlight differences in purpose, key methods, parameters, and return types.\n"
            "- If the context only covers one of the two items, say so clearly before the table.\n"
            "- End with a short 'When to use each' section grounded in the context."
        ),
    },
    "troubleshoot": {
        "keywords": ["error", "crash", "fail", "issue", "bug", "problem", "not working", "exception", "nil", "null", "debug", "fix", "wrong"],
        "instructions": (
            "SKILL: Troubleshooting\n"
            "- List probable causes as numbered items, ordered by likelihood based on the context.\n"
            "- For each cause: (a) describe the root cause, (b) provide a fix with {language} code if the\n"
            "  context supports it, (c) cite the source chunk.\n"
            "- Highlight any prerequisites, initialization order requirements, or permissions mentioned\n"
            "  in the context that are commonly missed.\n"
            "- If the context mentions specific error codes or exception types, quote them exactly.\n"
            "- If the context is insufficient to diagnose the issue, state what additional information\n"
            "  (logs, SDK version, platform) would be needed."
        ),
    },
    "tutorial": {
        "keywords": ["tutorial", "guide", "walkthrough", "step by step", "steps", "setup", "get started", "integrate", "configure", "build"],
        "instructions": (
            "SKILL: Tutorial / Step-by-Step Guide\n"
            "- Start with a '## Prerequisites' section listing imports, permissions, and setup steps\n"
            "  found in the context.\n"
            "- Number each step. Give each step a short `### Step N: Title` heading.\n"
            "- Include {language} code snippets for each step where the context provides enough detail.\n"
            "- If a step is referenced in the context but lacks detail, add a placeholder note rather\n"
            "  than inventing content.\n"
            "- End with a '## What's Next' section if the context mentions follow-up topics."
        ),
    },
}

DEFAULT_SKILL_INSTRUCTIONS = (
    "Structure your answer with:\n"
    "1. A one-sentence direct answer to the question.\n"
    "2. Supporting detail using Markdown headings, bullet points, and inline code where relevant.\n"
    "3. A 'Sources' line at the end listing every .md file you drew from."
)


def detect_skill(question: str, platform_cfg: dict = None) -> str:
    """Detect which skill to apply based on keywords in the question."""
    q = question.lower()
    best_skill = None
    best_count = 0
    extra_code_kw = (platform_cfg or {}).get("language_keywords", [])
    for skill_name, skill in SKILLS.items():
        keywords = skill["keywords"]
        if skill_name == "code":
            keywords = keywords + extra_code_kw
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count = count
            best_skill = skill_name
    return best_skill


QUERY_EXPANSION_PROMPT = """\
You are a search query optimizer for {sdk_name} API documentation.

Your job is to generate 3 DIVERSE search queries from the user's question so that a vector
search retrieves the most relevant API chunks. Make the queries meaningfully different from
each other — vary the angle, not just the wording.

Query strategies to consider (pick the most useful 3):
  1. CLASS-LEVEL  — name the likely SDK class(es) involved (e.g. "MapView initialisation")
  2. METHOD-LEVEL — name the likely method or property (e.g. "loadScene mapScheme")
  3. CONCEPT-LEVEL — use the underlying concept in plain English (e.g. "change map style at runtime")
  4. ERROR/SYMPTOM — if the question mentions a problem, phrase it as an error symptom
  5. RELATED CLASS — a helper/delegate/listener class that often pairs with the main class

User question: {question}

Return EXACTLY 3 search queries, one per line. No numbering, no explanations, no blank lines."""


def expand_query(question: str, api_key: str, provider: str, model: str, platform_cfg: dict = None) -> list[str]:
    """Use the LLM to generate better search queries from a natural language question."""
    cfg = platform_cfg or get_platform_config("generic")
    messages = [{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(sdk_name=cfg["sdk_name"], question=question)}]
    content = call_llm(messages, provider, model, api_key, temperature=0.0, max_tokens=200)
    return [l.strip() for l in content.strip().split("\n") if l.strip()]


def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def ingest(md_dir: str, vectordb_dir: str, embed_model: str, fresh: bool = False) -> None:
    """Load .md files, chunk them, embed, and store in ChromaDB."""
    if fresh and Path(vectordb_dir).exists():
        import shutil as _shutil
        _shutil.rmtree(vectordb_dir)
        console.print(f"[warning]Deleted existing vector DB at {vectordb_dir}[/warning]")

    md_path = Path(md_dir)
    if not md_path.is_dir():
        sys.exit(f"Directory not found: {md_path}")

    md_files = sorted(md_path.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files found in: {md_path}")

    console.print(f"[info]Loading {len(md_files)} .md file(s) from {md_path}...[/info]")

    # Load all .md files
    loader = DirectoryLoader(
        str(md_path), glob="*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    console.print(f"[info]Loaded {len(docs)} document(s)[/info]")

    # Split by markdown headers first, then by size
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )

    # Further split large chunks by character count
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in docs:
        source_file = Path(doc.metadata.get("source", "unknown"))
        class_name = source_file.stem  # e.g., "MapView" from "MapView.md"

        # Split by headers
        header_chunks = header_splitter.split_text(doc.page_content)
        for chunk in header_chunks:
            # Carry source metadata
            chunk.metadata["source"] = doc.metadata.get("source", "unknown")
            chunk.metadata["class_name"] = class_name
        # Further split any large chunks
        final_chunks = text_splitter.split_documents(header_chunks)

        # Prepend class name to each chunk for better embedding alignment
        for chunk in final_chunks:
            chunk.page_content = f"[Class: {class_name}] {chunk.page_content}"
            chunk.metadata["class_name"] = class_name

        all_chunks.extend(final_chunks)

    console.print(f"[info]Created {len(all_chunks)} chunk(s)[/info]")

    # Embed and store
    console.print(f"[info]Embedding with model '{embed_model}' (first run downloads the model)...[/info]")
    embeddings = get_embeddings(embed_model)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=vectordb_dir,
    )

    console.print(f"[success]Stored {len(all_chunks)} vectors in {vectordb_dir}[/success]")
    console.print('[success]Ingestion complete.[/success] Run: [bold]python rag_agent.py chat[/bold]')


def get_retriever(vectordb_dir: str, embed_model: str, top_k: int):
    """Load ChromaDB and return the vectorstore and top_k."""
    embeddings = get_embeddings(embed_model)
    vectorstore = Chroma(
        persist_directory=vectordb_dir,
        embedding_function=embeddings,
    )
    return vectorstore, top_k


def retrieve_with_expansion(vectorstore, top_k: int, question: str, api_key: str, provider: str, model: str, platform_cfg: dict = None):
    """Retrieve docs using the original query + LLM-expanded queries, then deduplicate."""
    all_queries = [question]
    try:
        expanded = expand_query(question, api_key, provider, model, platform_cfg)
        all_queries.extend(expanded)
    except Exception:
        pass  # Fall back to original query only

    seen_contents = set()
    unique_docs = []
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)

    # Limit total context to avoid token overflow
    return unique_docs[:top_k + 4]


def rerank_docs(docs, question: str):
    """Lightweight keyword reranking — boost docs that share words with the question."""
    q_words = set(question.lower().split())

    def score(doc):
        content_words = set(doc.page_content.lower().split())
        class_name = doc.metadata.get("class_name", "").lower()
        # Exact class name match in question is a strong signal
        class_bonus = 5 if class_name and class_name in question.lower() else 0
        return len(q_words & content_words) + class_bonus

    return sorted(docs, key=score, reverse=True)


def ask_llm(question: str, context_docs, provider: str, model: str, api_key: str, platform_cfg: dict = None):
    """Send question + retrieved context to the active LLM and return the answer."""
    cfg = platform_cfg or get_platform_config("generic")
    context_parts = []
    for doc in context_docs:
        source = Path(doc.metadata.get("source", "unknown")).stem
        context_parts.append(f"[{source}]\n{doc.page_content}")
    context = "\n---\n".join(context_parts)

    # Detect and apply skill
    skill_name = detect_skill(question, cfg)
    if skill_name:
        skill_instructions = SKILLS[skill_name]["instructions"].format(**cfg)
    else:
        skill_instructions = DEFAULT_SKILL_INSTRUCTIONS

    prompt = SYSTEM_PROMPT.format(
        sdk_name=cfg["sdk_name"],
        context=context,
        question=question,
        skill_instructions=skill_instructions,
    )

    content = call_llm([{"role": "user", "content": prompt}], provider, model, api_key, temperature=0.1)
    return content, skill_name


SKILL_LABELS = {
    "code": "\U0001f4bb Code Generation",
    "compare": "\u2696\ufe0f  API Comparison",
    "troubleshoot": "\U0001f527 Troubleshooting",
    "tutorial": "\U0001f4d6 Tutorial",
}


def print_answer(answer: str, docs, skill_name: str = None) -> None:
    """Render the answer as formatted markdown with source citations."""
    title = "[bold bright_white]Answer[/bold bright_white]"
    if skill_name and skill_name in SKILL_LABELS:
        title += f"  [dim]({SKILL_LABELS[skill_name]})[/dim]"

    console.print()
    console.print(Panel(
        Markdown(answer),
        title=title,
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    if docs:
        seen = set()
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                sources.append(Path(source).stem)
        source_text = "  ".join(f"[source]• {s}[/source]" for s in sources)
        console.print(f"  [dim]Sources:[/dim] {source_text}")
    console.print()


def query(question: str, vectordb_dir: str, embed_model: str, provider: str, model: str, top_k: int, api_key: str, platform_cfg: dict = None) -> None:
    """Ask a single question."""
    if not Path(vectordb_dir).is_dir():
        sys.exit(f"Vector DB not found at '{vectordb_dir}'. Run 'ingest' first.")

    vectorstore, top_k = get_retriever(vectordb_dir, embed_model, top_k)
    docs = retrieve_with_expansion(vectorstore, top_k, question, api_key, provider, model, platform_cfg)
    docs = rerank_docs(docs, question)
    try:
        answer, skill_name = ask_llm(question, docs, provider, model, api_key, platform_cfg)
    except Exception as e:
        if _is_auth_error(e):
            api_key = _refresh_api_key(provider)
            answer, skill_name = ask_llm(question, docs, provider, model, api_key, platform_cfg)
        else:
            raise
    print_answer(answer, docs, skill_name)


def chat(vectordb_dir: str, embed_model: str, provider: str, top_k: int, api_key: str, platform_cfg: dict = None) -> None:
    """Interactive chat loop."""
    cfg = platform_cfg or get_platform_config("generic")
    if not Path(vectordb_dir).is_dir():
        sys.exit(f"Vector DB not found at '{vectordb_dir}'. Run 'ingest' first.")

    console.print("[info]Loading knowledge base...[/info]")
    vectorstore, top_k = get_retriever(vectordb_dir, embed_model, top_k)
    console.print()

    model = _pick_model(provider)

    console.print(Panel(
        f"[bold]{cfg['sdk_name']} Documentation Agent[/bold]\n\n"
        f"Platform: [cyan]{cfg['language']}[/cyan] | Model: [cyan]{model}[/cyan] via [cyan]{provider.capitalize()}[/cyan]\n\n"
        f"[bold]Skills[/bold] (auto-detected from your question):\n"
        f"  \U0001f4bb [bold]Code Generation[/bold]  — \"Show me how to...\", \"example of...\"\n"
        f"  \u2696\ufe0f  [bold]API Comparison[/bold]   — \"Compare X vs Y\", \"difference between...\"\n"
        f"  \U0001f527 [bold]Troubleshooting[/bold]  — \"Error with...\", \"not working...\"\n"
        f"  \U0001f4d6 [bold]Tutorial[/bold]         — \"Step by step...\", \"how to set up...\"\n\n"
        f"Type your question and press Enter.\n"
        f"Commands: [bold]quit[/bold] / [bold]exit[/bold] / [bold]q[/bold] to leave.",
        border_style="bright_green",
        padding=(1, 2),
    ))
    console.print()

    while True:
        try:
            question = Prompt.ask("[user_prompt]You[/user_prompt]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[warning]Bye![/warning]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[warning]Bye![/warning]")
            break

        answer = skill_name = None
        with console.status("[info]Thinking...[/info]", spinner="dots"):
            docs = retrieve_with_expansion(vectorstore, top_k, question, api_key, provider, model, cfg)
            docs = rerank_docs(docs, question)
            try:
                answer, skill_name = ask_llm(question, docs, provider, model, api_key, cfg)
            except Exception as e:
                if not _is_auth_error(e):
                    raise
                # answer stays None → handled below, outside the spinner

        if answer is None:
            # Key was rejected — prompt outside spinner so the terminal works cleanly
            api_key = _refresh_api_key(provider)
            answer, skill_name = ask_llm(question, docs, provider, model, api_key, cfg)

        print_answer(answer, docs, skill_name)




def _vectordb_dir_for(md_dir: str) -> str:
    """Derive a vectordb path from the docs directory so each doc set gets its own DB."""
    return os.path.join(os.path.dirname(os.path.abspath(md_dir)), ".vectordb_" + Path(md_dir).name)


_PROVIDER_META = {
    "groq":   {"env": "GROQ_API_KEY",      "cfg_key": "groq_api_key",      "url": "https://console.groq.com/keys"},
    "claude": {"env": "ANTHROPIC_API_KEY",  "cfg_key": "anthropic_api_key", "url": "https://console.anthropic.com/keys"},
}


def _ensure_api_key(provider: str) -> str:
    """Return the API key for the given provider.

    Priority: env var → saved config → prompt user (then save).
    """
    meta = _PROVIDER_META[provider]
    # 1. Environment variable
    key = os.environ.get(meta["env"])
    if key:
        return key
    # 2. Saved config
    key = _load_config().get(meta["cfg_key"])
    if key:
        os.environ[meta["env"]] = key   # make available to child libs
        return key
    # 3. Prompt once and persist
    console.print(f"[warning]No API key found for {provider.capitalize()}.[/warning]")
    console.print(f"Get one at: [bold]{meta['url']}[/bold]\n")
    key = Prompt.ask(f"Paste your {provider.capitalize()} API key").strip()
    if not key:
        sys.exit("No API key provided.")
    _save_config({meta["cfg_key"]: key})
    os.environ[meta["env"]] = key
    return key


def _refresh_api_key(provider: str) -> str:
    """Called when a key is rejected — prompt for a replacement and save it."""
    meta = _PROVIDER_META[provider]
    console.print(f"\n[warning]⚠  Your {provider.capitalize()} API key is invalid or has expired.[/warning]")
    console.print(f"Get a new one at: [bold]{meta['url']}[/bold]\n")
    key = Prompt.ask(f"Paste a new {provider.capitalize()} API key (or press Enter to quit)").strip()
    if not key:
        sys.exit("Aborted.")
    _save_config({meta["cfg_key"]: key})
    os.environ[meta["env"]] = key
    return key


def call_llm(messages: list, provider: str, model: str, api_key: str, **kwargs) -> str:
    """Unified LLM call — routes to Groq or Anthropic based on provider."""
    if provider == "groq":
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content
    elif provider == "claude":
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 2048),
            system=system_msg,
            messages=user_msgs,
        )
        return resp.content[0].text
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _auto_ingest_if_needed(md_dir: str, vectordb_dir: str, embed_model: str) -> None:
    """Ingest docs into the vector DB if it doesn't already exist."""
    if Path(vectordb_dir).is_dir():
        return
    console.print(f"[info]First run — indexing docs from {md_dir}...[/info]")
    ingest(md_dir, vectordb_dir, embed_model, fresh=False)
    console.print()


CLI_NAME = "hsa"


def _install_cli():
    """Install this script as a CLI command on PATH."""
    script_path = os.path.abspath(__file__)

    # Ensure venv + deps exist
    if not HSA_VENV.exists():
        _bootstrap_venv()  # re-execs into venv

    # Always use ~/.local/bin (no sudo needed)
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    wrapper_path = bin_dir / CLI_NAME
    venv_python = HSA_VENV / "bin" / "python"

    # Copy the script to ~/.hsa/ so it's self-contained
    installed_script = HSA_HOME / "rag_agent.py"
    shutil.copy2(script_path, installed_script)

    # Write a wrapper shell script that uses the venv python
    wrapper_content = f"""#!/bin/sh
exec "{venv_python}" "{installed_script}" "$@"
"""
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)

    # Check if bin_dir is on PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    on_path = str(bin_dir) in path_dirs

    print(f"Installed: {wrapper_path}")
    print(f"  venv:   {HSA_VENV}")
    print(f"  script: {installed_script}")
    if not on_path:
        shell = os.environ.get("SHELL", "")
        rc_file = "~/.zshrc" if "zsh" in shell else "~/.bashrc"
        print(f"\nAdd {bin_dir} to your PATH:")
        print(f'  echo \'export PATH="{bin_dir}:$PATH"\' >> {rc_file} && source {rc_file}')
    else:
        print(f"\nReady! Try: {CLI_NAME} /path/to/docs")


def _uninstall_cli():
    """Remove the CLI wrapper and the venv."""
    removed = False
    for bin_dir in [Path.home() / ".local" / "bin", Path("/usr/local/bin")]:
        link_path = bin_dir / CLI_NAME
        if link_path.exists():
            link_path.unlink()
            print(f"Removed: {link_path}")
            removed = True
    if HSA_HOME.exists():
        shutil.rmtree(HSA_HOME)
        print(f"Removed: {HSA_HOME}")
        removed = True
    if not removed:
        print(f"{CLI_NAME} is not installed.")


def main():
    # ------------------------------------------------------------------
    # Quick-start: detect if first arg is a path (not a subcommand or flag)
    # Must happen before argparse since subparsers would eat it.
    # ------------------------------------------------------------------
    SUBCOMMANDS = {"ingest", "query", "chat"}
    first_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if first_arg and not first_arg.startswith("-") and first_arg not in SUBCOMMANDS and os.path.isdir(first_arg):
        # Quick-start mode: hsa /path/to/docs [--fresh] [--provider groq|claude] etc.
        quick_parser = argparse.ArgumentParser(description="RAG Agent — chat with your docs")
        quick_parser.add_argument("docs_path", help="Path to directory of .html or .md files.")
        quick_parser.add_argument("--vectordb", default=None)
        quick_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
        quick_parser.add_argument("--provider", default=None, choices=["groq", "claude"],
                                  help="LLM provider (saved after first choice if omitted)")
        quick_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        quick_parser.add_argument("--fresh", action="store_true")
        quick_parser.add_argument("--platform", default=None, choices=list(PLATFORM_CONFIGS.keys()),
                                  help="SDK platform (auto-detected from docs if omitted)")
        args = quick_parser.parse_args()

        docs_path = Path(args.docs_path)

        # Auto-detect: HTML or Markdown?
        html_files = list(docs_path.rglob("*.html"))
        md_files = list(docs_path.glob("*.md"))

        if html_files:
            # HTML dir may contain stray .md files — always convert HTML
            md_dir = str(docs_path.parent / (docs_path.name + "_md"))
            convert_html_to_md(str(docs_path), md_dir)
        elif md_files:
            md_dir = str(docs_path)
        else:
            sys.exit(f"No .html or .md files found in: {docs_path}")

        vectordb_dir = args.vectordb or _vectordb_dir_for(md_dir)

        # Provider → key (provider flag overrides saved config for this run)
        if args.provider:
            provider = args.provider
        else:
            provider = _pick_provider()
        api_key = _ensure_api_key(provider)

        if args.fresh and Path(vectordb_dir).exists():
            shutil.rmtree(vectordb_dir)

        _auto_ingest_if_needed(md_dir, vectordb_dir, args.embed_model)

        platform = args.platform or detect_platform(md_dir)
        platform_cfg = get_platform_config(platform)
        console.print(f"[info]Detected platform: {platform} ({platform_cfg['sdk_name']})[/info]")

        chat(vectordb_dir, args.embed_model, provider, args.top_k, api_key, platform_cfg)
        return

    # ------------------------------------------------------------------
    # Subcommand / flag mode
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="RAG Agent — chat with your docs",
        usage="%(prog)s [docs_path] [options]\n       %(prog)s {ingest,query,chat,agent} ...",
    )
    parser.add_argument("--install", action="store_true", help=f"Install as '{CLI_NAME}' CLI command")
    parser.add_argument("--uninstall", action="store_true", help=f"Remove '{CLI_NAME}' CLI command")
    parser.add_argument("--vectordb", default=None, help="Path to ChromaDB directory (auto-derived if omitted)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--provider", default=None, choices=["groq", "claude"],
                        help="LLM provider: groq or claude (saved after first choice if omitted)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--fresh", action="store_true", help="Re-ingest docs even if vector DB exists")
    parser.add_argument("--platform", default=None, choices=list(PLATFORM_CONFIGS.keys()),
                        help="SDK platform: ios, android, flutter, generic (auto-detected if omitted)")

    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest .md files into vector DB")
    ingest_parser.add_argument("md_dir", nargs="?", default="docs/md", help="Directory containing .md files (default: docs/md)")
    ingest_parser.add_argument("--fresh", action="store_true", dest="ingest_fresh", help="Delete existing vector DB before re-ingesting")

    # query
    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", help="The question to ask")

    # chat
    subparsers.add_parser("chat", help="Interactive chat mode (single-shot retrieval)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------
    if args.install:
        _install_cli()
        return
    if args.uninstall:
        _uninstall_cli()
        return

    # ------------------------------------------------------------------
    # Subcommand mode
    # ------------------------------------------------------------------
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        vectordb_dir = args.vectordb or DEFAULT_VECTORDB_DIR
        ingest(args.md_dir, vectordb_dir, args.embed_model, args.ingest_fresh)
    else:
        provider = args.provider or _pick_provider()
        api_key = _ensure_api_key(provider)
        vectordb_dir = args.vectordb or DEFAULT_VECTORDB_DIR
        # Resolve platform config
        platform = args.platform or detect_platform("docs/md")
        platform_cfg = get_platform_config(platform)
        console.print(f"[info]Platform: {platform} ({platform_cfg['sdk_name']})[/info]")

        if args.command == "query":
            default_model = DEFAULT_GROQ_MODEL if provider == "groq" else DEFAULT_CLAUDE_MODEL
            query(args.question, vectordb_dir, args.embed_model, provider, default_model, args.top_k, api_key, platform_cfg)
        elif args.command == "chat":
            chat(vectordb_dir, args.embed_model, provider, args.top_k, api_key, platform_cfg)


if __name__ == "__main__":
    main()
