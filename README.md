# HERE Agent

A single-file CLI agent that lets you chat with any documentation (HTML or Markdown) using Groq + ChromaDB. One command does everything: installs deps, converts HTML, indexes docs, starts an interactive agent.

## Quick Start

```bash
curl -O https://raw.githubusercontent.com/gokulnair2001/HERE-AGENT/main/rag_agent.py
python3 rag_agent.py /path/to/your/docs
```

Point it at a folder of `.html` or `.md` files. On first run it:
1. Installs missing Python packages
2. Converts HTML to Markdown (if the folder contains `.html` files)
3. Prompts for your Groq API key (free at https://console.groq.com/keys)
4. Indexes your docs into a local vector DB
5. Starts an interactive agent with search, read, and list tools

Subsequent runs skip steps 1–4 and go straight to chat.

## Usage

```bash
# HTML docs — auto-converted to Markdown
python3 rag_agent.py /path/to/html/docs

# Markdown docs — used directly
python3 rag_agent.py /path/to/md/docs

# Force re-index after docs change
python3 rag_agent.py /path/to/docs --fresh

# Set API key via env to skip the prompt
export GROQ_API_KEY=gsk_your_key_here
python3 rag_agent.py /path/to/docs
```

### Subcommands (advanced)

```bash
# Ingest docs into vector DB manually
python3 rag_agent.py ingest docs/md --fresh

# Single question
python3 rag_agent.py query "How do I use MapView?"

# Simple chat (single-shot retrieval, no tool calling)
python3 rag_agent.py chat

# Agent mode (tool calling, multi-step reasoning)
python3 rag_agent.py agent --md-dir docs/md
```

## How It Works

The agent has 3 tools it can use autonomously:
- **search_docs** — semantic search over the vector DB
- **read_class_doc** — reads a full `.md` file for a specific class
- **list_classes** — lists all documented classes

It decides which tools to call (up to 8 steps), gathers context, then answers grounded in the docs.

## Requirements

- Python 3.10+
- Groq API key (free tier works)
