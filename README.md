# HERE SDK RAG Agent

A local RAG agent that answers questions about the HERE SDK using your documentation.

## Quick Start (for team members)

1. Drop your HTML documentation files into `docs/html/`
2. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
   Get a free key at https://console.groq.com/keys
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Start chatting:
   ```bash
   source .venv/bin/activate
   python rag_agent.py chat
   ```

That's it. The setup script handles everything: venv, dependencies, HTML→Markdown conversion, and vector DB creation.

## Folder Structure

```
HERE Aent KB/
├── docs/
│   ├── html/          ← Drop HTML files here
│   └── md/            ← Auto-generated Markdown
├── vectordb/          ← Auto-generated vector store
├── html_to_md.py      ← HTML → Markdown converter
├── rag_agent.py       ← RAG pipeline (ingest, query, chat)
├── setup.sh           ← One-command setup
├── .env               ← Groq API key (not committed)
└── README.md
```

## Commands

```bash
# Interactive chat
python rag_agent.py chat

# Single question
python rag_agent.py query "How do I use MapView?"

# Re-ingest after updating docs (deletes old vector DB)
python rag_agent.py ingest --fresh

# Convert HTML manually
python html_to_md.py docs/html -o docs/md
```

## Updating Documentation

When docs change:
1. Replace HTML files in `docs/html/`
2. Re-run `./setup.sh` (or `python rag_agent.py ingest --fresh`)
