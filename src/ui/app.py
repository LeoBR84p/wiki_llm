"""NiceGUI chat interface for RAG queries over the generated wiki.

Provides a single-page web UI that accepts natural-language questions,
retrieves relevant wiki pages via the ChatEngine BM25 index, and streams
the LLM answer back to the user.  Conversation history is kept per-session.
"""

from __future__ import annotations

import asyncio
import os

from ..llm.factory import create_client
from ..models.config import WikiConfig
from .chat_engine import ChatEngine


def start_ui(cfg: WikiConfig, *, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the NiceGUI HTTP server with the wiki chat interface.

    Builds the BM25 index from the wiki directory, then registers a NiceGUI
    page at ``/`` and calls ``ui.run``.  This function blocks until the server
    is stopped.  Host and port can be overridden via the WIKI_UI_HOST and
    WIKI_UI_PORT environment variables so that Docker / Kubernetes deployments
    do not need to modify the config file.

    Args:
        cfg: Active WikiConfig; used to create the LLM client, build the
            ChatEngine index, and set the page title.
        host: Network interface to bind to.  Defaults to "0.0.0.0" so that
            the server is reachable from outside the container.
        port: TCP port for the NiceGUI HTTP server.

    Raises:
        RuntimeError: If nicegui is not installed.
    """
    try:
        from nicegui import app as ng_app, ui  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("nicegui not installed. Run: pip install nicegui") from exc

    llm = create_client(cfg.llm)
    engine = ChatEngine(cfg)
    engine.build_index()

    storage_secret = os.environ.get("WIKI_UI_STORAGE_SECRET", "wiki-llm-secret")

    @ui.page("/")
    async def index_page() -> None:
        ui.dark_mode()
        with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md"):
            ui.label(cfg.wiki_name).classes("text-h5 text-bold")
            ui.separator()

            messages_container = ui.column().classes("w-full gap-2")

            async def send_message() -> None:
                question = input_box.value.strip()
                if not question:
                    return
                input_box.value = ""
                with messages_container:
                    with ui.card().classes("w-full bg-blue-1"):
                        ui.label(f"You: {question}").classes("text-body1")
                    thinking = ui.label("...thinking...").classes("text-grey")

                try:
                    answer = await engine.ask(question, llm)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Error: {exc}"

                thinking.delete()
                with messages_container:
                    with ui.card().classes("w-full bg-green-1"):
                        ui.markdown(answer).classes("text-body1")

            with ui.row().classes("w-full"):
                input_box = ui.input(placeholder="Ask a question about the wiki...").classes("flex-1")
                input_box.on("keydown.enter", send_message)
                ui.button("Send", on_click=send_message).props("color=primary")

            with ui.row():
                ui.button("Clear history", on_click=lambda: (engine.clear_history(), messages_container.clear()))

    effective_host = os.environ.get("WIKI_UI_HOST", host)
    effective_port = int(os.environ.get("WIKI_UI_PORT", port))

    ui.run(
        host=effective_host,
        port=effective_port,
        title=cfg.wiki_name,
        storage_secret=storage_secret,
        reload=False,
    )
