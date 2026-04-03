# HSA — HERE SDK Agent

A single-file CLI agent that lets you chat with any documentation (HTML or Markdown) using Groq + ChromaDB. One command does everything: creates a venv, installs deps, converts HTML, indexes docs, and starts an interactive agent.

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
2. Prompts for your Groq API key (free at https://console.groq.com/keys)
3. Indexes your docs into a local vector DB
4. Starts an interactive agent with search, read, and list tools

Subsequent runs skip steps 1–3 and go straight to chat.

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

# Simple chat (single-shot retrieval, no tool calling)
hsa chat

# Agent mode (tool calling, multi-step reasoning)
hsa agent --md-dir docs/md
```

## How It Works

The agent has 3 tools it can use autonomously:
- **search_docs** — semantic search over the vector DB
- **read_class_doc** — reads a full `.md` file for a specific class
- **list_classes** — lists all documented classes

It decides which tools to call (up to 8 steps), gathers context, then answers grounded in the docs.

## Uninstall

```bash
hsa --uninstall
```

Removes `~/.hsa/` (venv + script) and `~/.local/bin/hsa`.

## Requirements

- Python 3.10+
- Groq API key (free tier works)
