"""Stage 7 — LangGraph repair agent.

Adapted from EXAMPLE/wiki_ng/agente_lint.py.

Flow:
  setup_state → dispatch_items (Send fan-out) → repair_item x N → END

Repair types:
  - broken_link: creates a stub page for a wikilink whose target does not exist
  - orphan: asks the LLM which existing pages should link to the orphan, then adds those links
"""

from __future__ import annotations

import logging
import operator
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from jinja2 import Template

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import WikiConfig
from ..models.evaluation import RepairState

logger = logging.getLogger(__name__)

_CHARS_INVALID = frozenset('\\/:*?"<>|')
_SYSTEM_PAGES = {"index.md", "log.md", "lint_report.md"}

_file_locks: dict[str, threading.Lock] = {}
_file_locks_meta = threading.Lock()
_creating_pages: set[str] = set()
_creating_pages_lock = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    """Return (or create) a per-file threading.Lock for the given path.

    Uses a global dict keyed by the resolved absolute path string so that
    concurrent async tasks repairing different pages never contend on the
    same lock.

    Args:
        path: The file path that needs exclusive write access.

    Returns:
        A threading.Lock unique to that file path.
    """
    key = str(path.resolve())
    with _file_locks_meta:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def _safe_slug(name: str) -> str:
    """Convert a link target name to a lowercase, filesystem-safe slug.

    Args:
        name: Raw link target string (e.g. from a broken wikilink).

    Returns:
        A slug with invalid chars replaced by hyphens and consecutive hyphens collapsed.
    """
    s = "".join(c if c not in _CHARS_INVALID else "-" for c in name.lower().strip())
    return re.sub(r"-{2,}", "-", s).strip("-") or "pagina"


def _write_atomic(path: Path, content: str, skip_if_exists: bool = False) -> bool:
    """Write content to path atomically via a temporary file and rename.

    Args:
        path: Destination file path.
        content: UTF-8 text to write.
        skip_if_exists: When True, returns False immediately if the file already exists.

    Returns:
        True if the file was written, False if skipped due to skip_if_exists.
    """
    if skip_if_exists and path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f"._tmp_{uuid.uuid4().hex[:8]}" + path.suffix)
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _find_page(wiki_dir: Path, stem: str) -> Path | None:
    """Search wiki_dir recursively for a Markdown file with the given stem.

    Args:
        wiki_dir: Root wiki directory to search.
        stem: The filename stem (without extension) to find.

    Returns:
        The first matching Path, or None if not found.
    """
    matches = list(wiki_dir.rglob(f"{stem}.md"))
    return matches[0] if matches else None


def _add_link_to_page(wiki_dir: Path, page_id: str, link_id: str) -> bool:
    """Append [[link_id]] to the end of the page identified by page_id (thread-safe).

    If the link already exists in the page, does nothing and returns False.
    Uses a per-file lock from _file_lock() to prevent concurrent write races.

    Args:
        wiki_dir: Root wiki directory used to locate the page.
        page_id: Stem of the page that should receive the new link.
        link_id: Stem of the page to link to.

    Returns:
        True if the link was added, False if already present or page not found.
    """
    path = _find_page(wiki_dir, page_id)
    if path is None:
        return False
    with _file_lock(path):
        text = path.read_text(encoding="utf-8")
        if f"[[{link_id}]]" in text:
            return False
        text = text.rstrip() + f"\n\n- [[{link_id}]]\n"
        path.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class RepairGlobalState(TypedDict):
    """Global state passed through the LangGraph repair workflow.

    Attributes:
        wiki_dir: Absolute path of the wiki root directory as a string.
        orphans: Page stems with no inbound wikilinks.
        broken_links: List of {origem, destino} dicts from static_analysis.
        repaired: Accumulated list of successfully repaired item names (fan-in).
        errors: Accumulated list of error strings from failed repairs (fan-in).
    """
    wiki_dir: str
    orphans: list[str]
    broken_links: list[dict]  # [{origem, destino}]
    repaired: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


class RepairItemState(TypedDict):
    """Per-item state dispatched to each repair_item node via Send fan-out.

    Attributes:
        wiki_dir: Absolute path of the wiki root directory as a string.
        repair_type: Either ``"broken_link"`` or ``"orphan"``.
        target: The page stem being repaired.
        sources: For broken_link repairs, the list of pages that reference the target.
    """
    wiki_dir: str
    repair_type: str  # "broken_link" | "orphan"
    target: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _build_item_states(state: RepairGlobalState) -> list[dict[str, Any]]:
    """Convert global orphans and broken_links into a flat list of RepairItemState dicts.

    Groups broken links by destination so each unique broken target becomes a
    single repair item with all its source pages listed.  Orphan entries each
    become one item with an empty sources list.

    Args:
        state: The current RepairGlobalState from the graph.

    Returns:
        A list of dicts conforming to RepairItemState, ready for Send fan-out.
    """
    items: list[dict[str, Any]] = []

    for orphan in state["orphans"]:
        items.append({
            "wiki_dir": state["wiki_dir"],
            "repair_type": "orphan",
            "target": orphan,
            "sources": [],
        })

    # Group broken links by destination
    by_dest: dict[str, list[str]] = {}
    for bl in state["broken_links"]:
        by_dest.setdefault(bl["destino"], []).append(bl["origem"])

    for dest, srcs in by_dest.items():
        items.append({
            "wiki_dir": state["wiki_dir"],
            "repair_type": "broken_link",
            "target": dest,
            "sources": srcs,
        })

    return items


async def _repair_broken_link(
    wiki_dir: Path,
    target: str,
    sources: list[str],
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> tuple[list[str], list[str]]:
    """Create a stub page for a broken wikilink target.

    Uses the LLM to generate a brief description for the stub page.  Guards
    against concurrent stub creation for the same slug using _creating_pages.
    Does nothing if the target file already exists.

    Args:
        wiki_dir: Root wiki directory where the stub will be written.
        target: The raw link target string (unslugified).
        sources: Page stems that contain the broken link.
        cfg: Pipeline config providing the repair prompt.
        llm: Active LLM client.
        llm_logger: Logger for the stub-generation call.

    Returns:
        A tuple (repaired, errors) where repaired is a list with the target
        name if successful, and errors is a list of error strings on failure.
    """
    slug = _safe_slug(target)
    dest_path = wiki_dir / f"{slug}.md"

    with _creating_pages_lock:
        if slug in _creating_pages or dest_path.exists():
            return [], []
        _creating_pages.add(slug)

    try:
        # Try to find similar existing page
        source_links = "\n".join(f"- [[{s}]]" for s in sources[:20])
        system = cfg.prompt_lint.read_text(encoding="utf-8")
        user = f"Link quebrado: **{target}**\n\nReferenciado em:\n{source_links}"

        t0 = llm_logger.start_call()
        try:
            resp = await llm.call(system, user)
            llm_logger.record(
                system=system, user=user, output=resp.text,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cached_tokens=resp.cached_tokens, model_id=resp.model_id,
                stage="repair.broken_link", elapsed=time.monotonic() - t0,
            )
            title_safe = target.replace('"', "'")
            fm = (
                "---\n"
                f'title: "{title_safe}"\n'
                "tipo: stub\n"
                "fonte: agente_reparo\n"
                "---\n"
            )
            content = f"{fm}\n# {target}\n\n{resp.text.strip()}\n\n"
            content += "## Referenciado em\n\n" + source_links + "\n"
            _write_atomic(dest_path, content, skip_if_exists=True)
            return [target], []
        except Exception as exc:  # noqa: BLE001
            return [], [f"broken_link:{target}:{exc}"]
    finally:
        with _creating_pages_lock:
            _creating_pages.discard(slug)


async def _repair_orphan(
    wiki_dir: Path,
    target: str,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> tuple[list[str], list[str]]:
    """Find pages that should link to an orphan and add those links.

    Shows the orphan page content and a list of candidate page IDs to the LLM
    and asks it to identify which ones should link to the orphan.  Then adds
    ``[[target]]`` to the bottom of each suggested page using _add_link_to_page.

    Args:
        wiki_dir: Root wiki directory to search for the orphan and candidates.
        target: The page stem of the orphan page.
        cfg: Pipeline config providing the repair prompt.
        llm: Active LLM client.
        llm_logger: Logger for the orphan-resolution call.

    Returns:
        A tuple (repaired, errors).  repaired contains the target if at least
        one backlink was added; errors contains a string on LLM failure.
    """
    path = _find_page(wiki_dir, target)
    if path is None:
        return [], [f"orphan:{target}:file not found"]

    # List all page ids as candidates
    candidate_ids = [p.stem for p in wiki_dir.rglob("*.md") if p.name not in _SYSTEM_PAGES and p.stem != target]
    candidates_text = "\n".join(f"- {c}" for c in candidate_ids[:100])

    system = cfg.prompt_lint.read_text(encoding="utf-8")
    user = (
        f"Orphan page: **[[{target}]]**\n\n"
        f"Content (first 500 chars):\n{path.read_text(encoding='utf-8')[:500]}\n\n"
        f"Pages that could reference this one:\n{candidates_text}"
    )

    t0 = llm_logger.start_call()
    try:
        resp = await llm.call(system, user)
        llm_logger.record(
            system=system, user=user, output=resp.text,
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            cached_tokens=resp.cached_tokens, model_id=resp.model_id,
            stage="repair.orphan", elapsed=time.monotonic() - t0,
        )
        # Extract page ids from LLM response
        suggested = re.findall(r"\b([a-z0-9_-]{3,})\b", resp.text.lower())
        added: list[str] = []
        for sid in suggested[:5]:
            if sid in {c.lower() for c in candidate_ids}:
                if _add_link_to_page(wiki_dir, sid, target):
                    added.append(sid)
        return ([target] if added else []), []
    except Exception as exc:  # noqa: BLE001
        return [], [f"orphan:{target}:{exc}"]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def run_repair(
    repair_state: RepairState,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> RepairState:
    """Execute the LangGraph repair agent and return an updated RepairState.

    Builds a LangGraph workflow with a dispatch node (Send fan-out) and
    a repair_item node that handles each broken link or orphan page in
    parallel.  Returns gracefully with the original state if LangGraph is
    not installed.

    Args:
        repair_state: The RepairState produced by run_lint containing the
            lists of orphans and broken links to fix.
        cfg: Active WikiConfig providing prompt paths and wiki_dir.
        llm: Active LLM client.
        llm_logger: Logger for all repair-related LLM calls.

    Returns:
        The input RepairState updated with repaired and errors lists.
    """
    try:
        from langgraph.graph import StateGraph, END, START  # noqa: PLC0415
        from langgraph.constants import Send  # noqa: PLC0415
    except ImportError:
        logger.warning("langgraph not installed — automatic repair disabled")
        return repair_state

    wiki_dir = repair_state.wiki_dir

    def dispatch(state: RepairGlobalState):  # noqa: ANN202
        items = _build_item_states(state)
        if not items:
            return [END]
        return [Send("repair_item", item) for item in items]

    async def repair_item(item: RepairItemState) -> dict[str, Any]:
        wdir = Path(item["wiki_dir"])
        if item["repair_type"] == "broken_link":
            repaired, errors = await _repair_broken_link(wdir, item["target"], item["sources"], cfg, llm, llm_logger)
        else:
            repaired, errors = await _repair_orphan(wdir, item["target"], cfg, llm, llm_logger)
        return {"repaired": repaired, "errors": errors}

    builder: StateGraph = StateGraph(RepairGlobalState)
    builder.add_node("repair_item", repair_item)
    builder.add_conditional_edges(START, dispatch, ["repair_item", END])
    builder.add_edge("repair_item", END)
    graph = builder.compile()

    initial_state: RepairGlobalState = {
        "wiki_dir": str(wiki_dir),
        "orphans": repair_state.orphans,
        "broken_links": repair_state.broken_links,
        "repaired": [],
        "errors": [],
    }

    final = await graph.ainvoke(initial_state)
    repair_state.repaired = final.get("repaired", [])
    repair_state.errors = final.get("errors", [])
    logger.info("Repair: %d repaired, %d errors", len(repair_state.repaired), len(repair_state.errors))
    return repair_state
