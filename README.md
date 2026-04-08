# HSA — HERE SDK Agent

A single-file CLI agent that lets you chat with HERE SDK documentation (HTML or Markdown) for **any platform** — iOS, Android, Flutter, or generic. One command does everything: creates a venv, installs deps, converts HTML, indexes docs, and starts an interactive agent.

## Install

```bash
curl -O https://raw.githubusercontent.com/gokulnair2001/HERE-AGENT/main/rag_agent.py
python3 rag_agent.py --install
```

This creates a self-contained venv at `~/.hsa/` and puts `hsa` on your PATH (`~/.local/bin`). No sudo needed.

> If `hsa` isn't found after install, add `~/.local/bin` to your PATH:
> ```bash
> echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
> ```

## Quick Start

```bash
hsa /path/to/your/docs
```

Point it at a folder of `.html` or `.md` files. On first run it:
1. Converts HTML to Markdown (if the folder contains `.html` files)
2. **Auto-detects the platform** (iOS/Android/Flutter) from the docs directory name or file contents
3. Prompts you to choose a provider — **Claude Code** (default, no extra key) or **Groq** (free API key)
4. Indexes your docs into a local vector DB
5. Starts an interactive chat with query expansion, reranking, and skill detection

Subsequent runs skip steps 1–4 and go straight to chat.

## LLM Providers

HSA supports two providers. Your choice is saved to `~/.hsa/config.json` — you're only asked once.

| Provider | How it works | Key required |
|---|---|---|
| **Claude Code** *(default)* | Remote-controls the `claude` CLI already running on your machine | None — reuses existing Claude Code auth |
| **Groq** | Calls the Groq cloud API | Free key at [console.groq.com/keys](https://console.groq.com/keys) |

```bash
# Override provider for a single run
hsa /path/to/docs --provider claude-code
hsa /path/to/docs --provider groq

# Reset saved provider/key
rm ~/.hsa/config.json
```

## Multi-Platform Support

The agent auto-detects the platform from your docs folder name (e.g. `heresdk-ios-*` → iOS, `heresdk-android-*` → Android). You can also specify it explicitly:

```bash
# Auto-detect (default)
hsa /path/to/heresdk-ios-docs

# Explicit platform
hsa /path/to/docs --platform ios
hsa /path/to/docs --platform android
hsa /path/to/docs --platform flutter
hsa /path/to/docs --platform generic
```

| Platform  | Language | Code Blocks |
|-----------|----------|-------------|
| `ios`     | Swift    | `` ```swift ``  |
| `android` | Kotlin   | `` ```kotlin `` |
| `flutter` | Dart     | `` ```dart ``   |
| `generic` | Auto     | Plain       |

The platform setting controls:
- System prompts (e.g. "HERE SDK for Android")
- Code generation language (Swift / Kotlin / Dart)
- Skill keyword detection (platform-specific terms like "uikit", "gradle", "widget")
- Query expansion prompts

## Usage

```bash
# HTML docs — auto-converted to Markdown
hsa /path/to/html/docs

# Markdown docs — used directly
hsa /path/to/md/docs

# Force re-index after docs change
hsa /path/to/docs --fresh
```

### Subcommands (advanced)

```bash
# Ingest docs into vector DB manually
hsa ingest docs/md --fresh

# Single question
hsa query "How do I use MapView?"

# Interactive chat
hsa chat --platform android
```

## How It Works

```
You type a question
       ↓
LLM rephrases it into 3 diverse search queries   (query expansion)
       ↓
Each query searches the vector DB independently  (semantic search)
       ↓
Results are deduplicated and capped at ~12 chunks
       ↓
Chunks are sorted by keyword overlap             (reranking)
       ↓
Question type is detected                        (skill detection)
       ↓
All chunks + question sent to LLM as context     (grounded answer)
       ↓
LLM answers strictly from the context, with source citations
```

### Skills (auto-detected from your question)

| Skill | Triggered by | Output format |
|---|---|---|
| 💻 **Code Generation** | "show me how to", "example of" | Self-contained code snippet |
| ⚖️ **API Comparison** | "compare", "vs", "difference between" | Markdown table |
| 🔧 **Troubleshooting** | "error", "not working", "crash" | Numbered causes + fixes |
| 📖 **Tutorial** | "step by step", "how to set up" | Numbered steps with prereqs |

## MCP Server — Use with Claude Code

You can connect HSA to Claude Code as an **MCP server** so Claude Code can search your HERE SDK docs directly while you're working on any project. The normal CLI (`hsa chat`, `hsa query`) continues to work as before.

### Step 1: Ingest your docs (if not done already)

```bash
hsa /path/to/your/docs
```

The vector DB is stored automatically at `~/.hsa/vectordbs/<name>`.

### Step 2: Connect to Claude Code (one-time)

```bash
claude mcp add --scope user here-sdk-docs -- hsa mcp-serve
```

That's it. `--scope user` makes it available in all your projects. The MCP server auto-discovers the vector DB from `~/.hsa/vectordbs/`.

### Step 3: Use it

Open Claude Code in your app project. Claude Code now has two extra tools:

| Tool | What it does |
|---|---|
| `search_here_docs(query)` | Searches the indexed docs and returns matching chunks with source attribution |
| `list_indexed_classes()` | Lists all API classes/files in the index |

When you ask Claude Code a question about the HERE SDK, it will automatically call these tools to fetch the relevant docs and use them in its response — while also being able to edit your code directly.

### Example

```
You (in Claude Code, working on your iOS app):
> "Add MapView with a custom map style to my ViewController"

Claude Code:
  1. Calls search_here_docs("MapView custom map style initialization")
  2. Gets back relevant doc chunks about MapView, MapScheme, loadScene
  3. Writes the actual Swift code in your ViewController.swift
```

## Uninstall

```bash
hsa --uninstall
```

Removes `~/.hsa/` (venv + script) and `~/.local/bin/hsa`.

## Requirements

- Python 3.10+
- **Claude Code** installed (for the default provider) — [claude.ai/code](https://claude.ai/code)
- Or a **Groq API key** (free tier works) — [console.groq.com/keys](https://console.groq.com/keys)
