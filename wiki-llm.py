"""wiki-llm CLI entry point — avoids .exe shim issues on Windows.

Usage (all commands):
    uv run wiki-llm.py setup
    uv run wiki-llm.py run-all --config config/my_wiki.py
    uv run wiki-llm.py generate --config config/my_wiki.py
    uv run wiki-llm.py chat
"""
from src.cli import app

app()
