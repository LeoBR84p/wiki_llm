"""BM25-based RAG chat engine over the generated wiki.

Strategy:
1. Build an in-memory catalogue (slug -> chunked content) from wiki/*.md
2. Index all chunks with rank_bm25, using markdown_hero.extract_chunks(purpose="rag")
   to produce section-aware splits that preserve heading context
3. Retrieve the top-K pages by BM25 score for each user query
4. Call the LLM with the retrieved context + conversation history

No vector store or embeddings are required: BM25 is fast, fully offline, and
surprisingly effective for navigating structured knowledge bases where the
user's vocabulary closely matches the document vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_hero import extract_chunks

from ..llm.base import BaseLLMClient, LLMResponse
from ..models.config import WikiConfig

_STOPWORDS = frozenset({
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "a", "o", "e", "que", "com", "para", "por", "se", "ou", "um", "uma",
    "como", "qual", "quais", "sobre", "entre", "tem", "deve", "pode",
})

_SYSTEM_PAGES = {"index.md", "lint_report.md"}


def _tokenize(text: str) -> list[str]:
    """Tokenize Markdown text for BM25 indexing.

    Lowercases, extracts Unicode word tokens, and removes stopwords and very
    short tokens.  The stopword list covers common Portuguese and English
    function words so that the BM25 score focuses on meaningful terms.

    Args:
        text: Raw Markdown or plain text to tokenize.

    Returns:
        A list of lowercase token strings with stopwords and 1-2 char tokens removed.
    """
    tokens = re.findall(r"[a-záéíóúãõàâêô\w]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


class ChatEngine:
    """BM25 retrieval engine that answers questions about the wiki.

    Maintains an in-memory BM25 index built from all non-system wiki pages.
    Each page is split into section-aware chunks by markdown_hero before
    indexing.  Conversation history is kept in memory for the lifetime of
    the engine instance; call clear_history() to start a fresh session.
    """

    def __init__(self, cfg: WikiConfig) -> None:
        """Initialize the engine without yet building the index.

        The BM25 index is built lazily on the first call to retrieve() or
        ask(), or eagerly by calling build_index() explicitly (recommended
        at server startup to avoid latency on the first user query).

        Args:
            cfg: Active WikiConfig; uses wiki_dir to locate Markdown files.
        """
        self._cfg = cfg
        self._corpus: list[str] = []
        self._slugs: list[str] = []
        self._bm25 = None
        self._history: list[dict[str, str]] = []

    def build_index(self) -> None:
        """Build (or rebuild) the BM25 index from all wiki .md files.

        Reads every non-system Markdown file under wiki_dir, extracts RAG
        chunks via markdown_hero.extract_chunks, tokenizes the combined text,
        and initializes a BM25Okapi instance.  Safe to call multiple times;
        each call replaces the previous index so that new pages are picked up
        after a pipeline run without restarting the server.

        Raises:
            RuntimeError: If rank_bm25 is not installed.
        """
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("rank_bm25 não instalado. Execute: pip install rank-bm25") from exc

        self._corpus = []
        self._slugs = []
        tokenized: list[list[str]] = []

        for p in self._cfg.wiki_dir.rglob("*.md"):
            if p.name in _SYSTEM_PAGES or p.stem.startswith("lint_"):
                continue
            text = p.read_text(encoding="utf-8")
            # Extract RAG chunks (headers + content)
            try:
                chunks = extract_chunks(text, purpose="rag")
                combined = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in chunks)
            except Exception:  # noqa: BLE001
                combined = text

            self._slugs.append(p.stem)
            self._corpus.append(combined)
            tokenized.append(_tokenize(combined))

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Return the top-K most relevant (slug, content) pairs for a query.

        Builds the index on first call if it has not been built yet.  Scores
        all corpus documents using BM25 and returns the highest-scoring ones.
        Documents with a zero BM25 score are excluded from the results.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of documents to return.

        Returns:
            A list of (slug, content) tuples ordered by descending relevance,
            with at most top_k entries.
        """
        if self._bm25 is None:
            self.build_index()
        if not self._slugs:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._slugs[i], self._corpus[i]) for i in top_indices if scores[i] > 0]

    async def ask(
        self,
        question: str,
        llm: BaseLLMClient,
        *,
        top_k: int = 5,
        max_history: int = 6,
    ) -> str:
        """Answer a question using BM25 retrieval and an LLM.

        Retrieves the most relevant wiki pages, renders them as a context
        block, prepends recent conversation history, and calls the LLM with
        the combined prompt.  Appends the question and answer to the in-memory
        history for multi-turn continuity.

        Args:
            question: The user's natural-language question.
            llm: The active LLM backend client.
            top_k: Number of wiki pages to retrieve as context.
            max_history: Maximum number of prior turn pairs to include in the
                prompt (older turns are dropped to stay within context limits).

        Returns:
            The model's answer as a Markdown-formatted string.
        """
        relevant = self.retrieve(question, top_k=top_k)

        context_parts: list[str] = []
        for slug, content in relevant:
            context_parts.append(f"### [[{slug}]]\n{content[:3000]}")
        context = "\n\n---\n\n".join(context_parts) or "(sem contexto relevante)"

        system_tpl = self._cfg.prompt_chat.read_text(encoding="utf-8")
        from jinja2 import Template  # noqa: PLC0415
        system = Template(system_tpl).render(wiki_name=self._cfg.wiki_name)

        # Build conversation
        messages_user = f"**Contexto da wiki:**\n\n{context}\n\n**Pergunta:**\n{question}"

        # Include history in system for simplicity (avoids complex message threading)
        if self._history:
            history_text = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in self._history[-max_history:]
            )
            messages_user = f"**Histórico:**\n{history_text}\n\n{messages_user}"

        resp = await llm.call(system, messages_user)
        answer = resp.text

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        """Reset the conversation history to an empty state.

        Call this when the user wants to start a fresh session without
        restarting the server.  The BM25 index is not affected.
        """
        self._history.clear()
