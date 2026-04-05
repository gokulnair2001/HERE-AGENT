#!/usr/bin/env python3
"""
RAG Agent — chat with your docs.

Single-file CLI that converts HTML to Markdown (if needed), indexes docs,
and starts an interactive agent with tool calling.

Usage:
  python rag_agent.py /path/to/html-or-md-docs
"""

import argparse
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
    "sentence-transformers", "groq", "rich", "python-dotenv",
    "beautifulsoup4", "markdownify",
]

HSA_HOME = Path.home() / ".hsa"
HSA_VENV = HSA_HOME / "venv"


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
DEFAULT_TOP_K = 8

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

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


def _pick_model(default: str = DEFAULT_GROQ_MODEL) -> str:
    """Interactive model picker shown at startup."""
    console.print("[bold]Select a model:[/bold]")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        marker = " [dim](default)[/dim]" if model == default else ""
        console.print(f"  [cyan]{i}[/cyan]. {model}{marker}")
    choice = Prompt.ask(
        "Enter number or model name",
        default=str(AVAILABLE_MODELS.index(default) + 1) if default in AVAILABLE_MODELS else "1",
    ).strip()
    # Accept a number
    if choice.isdigit() and 1 <= int(choice) <= len(AVAILABLE_MODELS):
        selected = AVAILABLE_MODELS[int(choice) - 1]
    # Accept a name (exact or substring)
    elif choice in AVAILABLE_MODELS:
        selected = choice
    else:
        matches = [m for m in AVAILABLE_MODELS if choice.lower() in m.lower()]
        selected = matches[0] if matches else default
    console.print(f"Using model: [cyan]{selected}[/cyan]\n")
    return selected

SYSTEM_PROMPT = """\
You are a technical assistant for the {sdk_name}.

STRICT RULES:
1. Answer ONLY based on the provided context. Do NOT use prior knowledge about the HERE SDK.
2. If the context does not contain the answer, respond with: "I don't have enough information in the documentation to answer this. The relevant docs may not be indexed, or try rephrasing your question."
3. Always cite the exact class, method, or property names as they appear in the context.
4. Do NOT invent method signatures, parameters, or return types. If a signature is partially visible, say so.
5. When referencing a class, include [SourceFile] attribution.

{skill_instructions}

Context (each block is prefixed with [SourceFile]):
{context}

Question: {question}

Answer (cite sources, refuse if context is insufficient):"""

# ---------------------------------------------------------------------------
# Skills — auto-detected from the user's question
# ---------------------------------------------------------------------------

SKILLS = {
    "code": {
        "keywords": ["code", "example", "snippet", "implement", "how to", "show me", "write", "sample", "usage"],
        "instructions": (
            "SKILL: Code Generation\n"
            "- Provide a working {language} code example that directly answers the question.\n"
            "- Include necessary imports.\n"
            "- Add brief inline comments explaining key lines.\n"
            "- If the context shows method signatures, use them exactly.\n"
            "- Wrap code in ```{code_fence} fenced blocks."
        ),
    },
    "compare": {
        "keywords": ["compare", "difference", "vs", "versus", "differ", "which one", "better", "alternative", "instead of"],
        "instructions": (
            "SKILL: API Comparison\n"
            "- Present a clear side-by-side comparison using a Markdown table.\n"
            "- Columns: Feature | Class/Method A | Class/Method B\n"
            "- Highlight key differences in purpose, parameters, and return types.\n"
            "- End with a short recommendation on when to use each."
        ),
    },
    "troubleshoot": {
        "keywords": ["error", "crash", "fail", "issue", "bug", "problem", "not working", "exception", "nil", "null", "debug", "fix", "wrong"],
        "instructions": (
            "SKILL: Troubleshooting\n"
            "- Start with the most likely cause of the issue.\n"
            "- List possible causes as numbered steps, ordered by likelihood.\n"
            "- For each cause, provide a concrete fix with code if applicable.\n"
            "- Mention any known gotchas or prerequisites (e.g., permissions, initialization order).\n"
            "- If the context is insufficient, list what information would be needed to diagnose further."
        ),
    },
    "tutorial": {
        "keywords": ["tutorial", "guide", "walkthrough", "step by step", "steps", "setup", "get started", "integrate", "configure", "build"],
        "instructions": (
            "SKILL: Tutorial / Step-by-Step Guide\n"
            "- Break the answer into numbered steps.\n"
            "- Each step should have a short heading and explanation.\n"
            "- Include {language} code snippets where relevant.\n"
            "- Note any prerequisites at the top (imports, permissions, setup).\n"
            "- End with a 'What's Next' suggestion if applicable."
        ),
    },
}

DEFAULT_SKILL_INSTRUCTIONS = (
    "Provide a clear, well-structured answer. "
    "Use Markdown formatting: headings, bullet points, code blocks where appropriate."
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

Given a user question, generate 2-3 short search queries that would find the relevant API classes, methods, or properties. Think about:
- What class names are likely relevant (e.g., "show a map" → MapView, MapScene)
- What method or property names might exist
- Related classes the user might need

User question: {question}

Return ONLY the search queries, one per line. No explanations."""


def expand_query(question: str, api_key: str, groq_model: str, platform_cfg: dict = None) -> list[str]:
    """Use the LLM to generate better search queries from a natural language question."""
    cfg = platform_cfg or get_platform_config("generic")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(sdk_name=cfg["sdk_name"], question=question)}],
        temperature=0.0,
        max_tokens=150,
    )
    lines = [l.strip() for l in response.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines


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


def retrieve_with_expansion(vectorstore, top_k: int, question: str, api_key: str, groq_model: str, platform_cfg: dict = None):
    """Retrieve docs using the original query + LLM-expanded queries, then deduplicate."""
    all_queries = [question]
    try:
        expanded = expand_query(question, api_key, groq_model, platform_cfg)
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


def ask_groq(question: str, context_docs, groq_model: str, api_key: str, platform_cfg: dict = None):
    """Send question + retrieved context to Groq and return the answer."""
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

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content, skill_name


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


def query(question: str, vectordb_dir: str, embed_model: str, groq_model: str, top_k: int, api_key: str, platform_cfg: dict = None) -> None:
    """Ask a single question."""
    if not Path(vectordb_dir).is_dir():
        sys.exit(f"Vector DB not found at '{vectordb_dir}'. Run 'ingest' first.")

    vectorstore, top_k = get_retriever(vectordb_dir, embed_model, top_k)
    docs = retrieve_with_expansion(vectorstore, top_k, question, api_key, groq_model, platform_cfg)
    docs = rerank_docs(docs, question)
    answer, skill_name = ask_groq(question, docs, groq_model, api_key, platform_cfg)
    print_answer(answer, docs, skill_name)


def chat(vectordb_dir: str, embed_model: str, groq_model: str, top_k: int, api_key: str, platform_cfg: dict = None) -> None:
    """Interactive chat loop."""
    cfg = platform_cfg or get_platform_config("generic")
    if not Path(vectordb_dir).is_dir():
        sys.exit(f"Vector DB not found at '{vectordb_dir}'. Run 'ingest' first.")

    console.print("[info]Loading knowledge base...[/info]")
    vectorstore, top_k = get_retriever(vectordb_dir, embed_model, top_k)
    console.print()

    groq_model = _pick_model(groq_model)

    console.print(Panel(
        f"[bold]{cfg['sdk_name']} Documentation Agent[/bold]\n\n"
        f"Platform: [cyan]{cfg['language']}[/cyan] | Model: [cyan]{groq_model}[/cyan] via Groq\n\n"
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

        with console.status("[info]Thinking...[/info]", spinner="dots"):
            docs = retrieve_with_expansion(vectorstore, top_k, question, api_key, groq_model, cfg)
            docs = rerank_docs(docs, question)
            answer, skill_name = ask_groq(question, docs, groq_model, api_key, cfg)

        print_answer(answer, docs, skill_name)




def _vectordb_dir_for(md_dir: str) -> str:
    """Derive a vectordb path from the docs directory so each doc set gets its own DB."""
    return os.path.join(os.path.dirname(os.path.abspath(md_dir)), ".vectordb_" + Path(md_dir).name)


def _ensure_api_key() -> str:
    """Get the Groq API key from env or prompt the user."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print("[warning]GROQ_API_KEY not found in environment.[/warning]")
        console.print("Get a free key at: [bold]https://console.groq.com/keys[/bold]\n")
        api_key = Prompt.ask("Paste your Groq API key").strip()
        if not api_key:
            sys.exit("No API key provided.")
        os.environ["GROQ_API_KEY"] = api_key
    return api_key


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
        # Quick-start mode: hsa /path/to/docs [--fresh] [--groq-model X] etc.
        quick_parser = argparse.ArgumentParser(description="RAG Agent — chat with your docs")
        quick_parser.add_argument("docs_path", help="Path to directory of .html or .md files.")
        quick_parser.add_argument("--vectordb", default=None)
        quick_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
        quick_parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL)
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
        api_key = _ensure_api_key()

        if args.fresh and Path(vectordb_dir).exists():
            shutil.rmtree(vectordb_dir)

        _auto_ingest_if_needed(md_dir, vectordb_dir, args.embed_model)

        platform = args.platform or detect_platform(md_dir)
        platform_cfg = get_platform_config(platform)
        console.print(f"[info]Detected platform: {platform} ({platform_cfg['sdk_name']})[/info]")

        chat(vectordb_dir, args.embed_model, args.groq_model, args.top_k, api_key, platform_cfg)
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
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL, help="Groq LLM model name")
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
        api_key = _ensure_api_key()
        vectordb_dir = args.vectordb or DEFAULT_VECTORDB_DIR
        # Resolve platform config
        platform = args.platform or detect_platform("docs/md")
        platform_cfg = get_platform_config(platform)
        console.print(f"[info]Platform: {platform} ({platform_cfg['sdk_name']})[/info]")

        if args.command == "query":
            query(args.question, vectordb_dir, args.embed_model, args.groq_model, args.top_k, api_key, platform_cfg)
        elif args.command == "chat":
            chat(vectordb_dir, args.embed_model, args.groq_model, args.top_k, api_key, platform_cfg)


if __name__ == "__main__":
    main()
