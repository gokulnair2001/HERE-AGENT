#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  HERE SDK RAG Agent — Setup"
echo "========================================"
echo

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv .venv
else
    echo "→ Virtual environment already exists."
fi

source .venv/bin/activate

# 2. Install dependencies
echo "→ Installing dependencies..."
pip install -q beautifulsoup4 markdownify langchain-community langchain-text-splitters \
    chromadb sentence-transformers groq rich python-dotenv

# 3. Check for .env
if [ ! -f ".env" ]; then
    echo
    echo "⚠  No .env file found."
    echo "   Create one with your Groq API key:"
    echo "   echo 'GROQ_API_KEY=gsk_your_key_here' > .env"
    echo "   Get a free key at https://console.groq.com/keys"
    echo
fi

# 4. Convert HTML → Markdown
HTML_DIR="docs/html"
MD_DIR="docs/md"

html_count=$(find "$HTML_DIR" -name "*.html" 2>/dev/null | wc -l | tr -d ' ')

if [ "$html_count" -eq 0 ]; then
    echo
    echo "⚠  No HTML files found in $HTML_DIR"
    echo "   Drop your HTML documentation files there and re-run this script."
    exit 1
fi

echo "→ Converting $html_count HTML file(s) to Markdown..."
python html_to_md.py "$HTML_DIR" -o "$MD_DIR"

# 5. Ingest into vector DB
echo "→ Building vector database (this may take a minute on first run)..."
python rag_agent.py ingest "$MD_DIR" --fresh

echo
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo
echo "  Start chatting:"
echo "    source .venv/bin/activate"
echo "    python rag_agent.py chat"
echo
echo "  Or ask a single question:"
echo "    python rag_agent.py query \"How do I use MapView?\""
echo
