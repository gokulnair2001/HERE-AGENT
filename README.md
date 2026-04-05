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
3. Prompts for your Groq API key (free at https://console.groq.com/keys)
4. Indexes your docs into a local vector DB
5. Starts an interactive chat with query expansion, reranking, and skill detection

Subsequent runs skip steps 1–4 and go straight to chat.

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

# Set API key via env to skip the prompt
export GROQ_API_KEY=gsk_your_key_here
hsa /path/to/docs
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

The chat uses a multi-stage retrieval pipeline:
1. **Query expansion** — LLM generates 2-3 refined search queries from your question
2. **Semantic search** — retrieves relevant doc chunks from the vector DB
3. **Keyword reranking** — boosts chunks that share terms with your question
4. **Skill detection** — auto-detects the best response format (code, comparison, troubleshooting, tutorial)
5. **Grounded answer** — LLM answers strictly from retrieved context with source citations

## Uninstall

```bash
hsa --uninstall
```

Removes `~/.hsa/` (venv + script) and `~/.local/bin/hsa`.

## Requirements

- Python 3.10+
- Groq API key (free tier works)
