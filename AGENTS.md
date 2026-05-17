# Wiki Agent

> **Single source of truth.** All agent instruction files in this repo point here.
> Compatible with: Claude (Anthropic), GitHub Copilot, OpenAI Codex/Assistants, VS Code Agent Mode.

---

<!-- MEMORY — the agent fills this block during setup; do not edit manually -->
<!--
CONFIG_START
wiki_name:
input_folder:
output_folder:
entity_types:
language: English
CONFIG_END
-->

---

## Table of Contents

1. [Language Policy](#language-policy)
2. [Setup (First Run)](#setup-first-run)
3. [Modes of Operation](#modes-of-operation)
4. [Stage 1 — Read](#stage-1--read)
5. [Stage 2 — Generate](#stage-2--generate)
6. [Stage 3 — Topics](#stage-3--topics)
7. [Stage 4 — Groups](#stage-4--groups)
8. [Stage 5 — Index](#stage-5--index)
9. [Stage 6 — Consolidate](#stage-6--consolidate)
10. [Stage 7 — Lint](#stage-7--lint)
11. [Stage 8 — Repair](#stage-8--repair)
12. [Quick Reference](#quick-reference)
13. [Conventions](#conventions)

---

## Language Policy

All output (wiki pages, reports, messages to the user) is written in **English by default**.

If the user writes in another language or explicitly asks to switch (e.g. "switch to Portuguese", "responda em português"), immediately:
1. Update `language` in the CONFIG block above.
2. Write all subsequent output — including wiki content, lint reports, and conversational replies — in the requested language.

---

## Setup (First Run)

Before starting any work, read the CONFIG block at the top of this file.  
If `wiki_name` is empty, run the following wizard **before any other action**.

Ask the user, one question at a time:

1. **"What should the wiki be named?"**
2. **"Where are the input documents? (provide a folder path)"**
3. **"Where should the wiki pages be saved? (default: `wiki/`)"**
4. **"What entity types organize your content?"**  
   Examples: `Products, People, Projects, Processes`.  
   If the user is unsure, say: *"I can auto-detect entity types from the documents — type 'auto' to do that."*
5. **"What language should the wiki content be written in? (default: English)"**

After collecting answers, update the CONFIG block in this file:

```
<!--
CONFIG_START
wiki_name: {answer}
input_folder: {answer}
output_folder: {answer}
entity_types: {answer}
language: {answer}
CONFIG_END
-->
```

Then create the output folder if it does not exist.  
Confirm to the user: *"Setup complete. Config saved. Ready to generate your wiki."*

---

## Modes of Operation

### Incremental Mode (default)

**Triggered by:** user mentions a new file, pastes document content, or says "process this file".

Run in order:
1. [Stage 1 — Read](#stage-1--read) (for that file only)
2. [Stage 2 — Generate](#stage-2--generate) (for that file only)
3. [Stage 5 — Index](#stage-5--index) (full rebuild)
4. [Stage 7 — Lint](#stage-7--lint) (new page only)

Report: page created at `{path}`, any lint warnings, links suggested.

### Full Pipeline

**Triggered by:** user says "generate the wiki", "run all", "process everything", "build the wiki", or equivalent.

Run all 8 stages in order:

```
Read → Generate → Topics → Groups → Index → Consolidate → Lint → Repair
```

After each stage, print a one-line status: `✓ Stage N complete — {summary}`.  
If a stage produces errors, report them and ask the user whether to continue or stop.

---

## Stage 1 — Read

**Goal:** Discover all documents in `input_folder` and extract their text.

For each file found:

| Extension | Extraction method |
|---|---|
| `.md`, `.txt` | Read as-is |
| `.pdf` | Extract text page by page; note page numbers |
| `.docx` | Extract paragraphs and headings; preserve heading hierarchy |
| `.xlsx`, `.csv` | Render as pipe-delimited markdown table |
| `.pptx` | Extract slide titles and body text, one section per slide |

Build an internal document list:
```
[{ filename, title_guess, raw_text, extension, path }]
```

If a file cannot be read, log `[READ ERROR: {filename} — {reason}]` and continue.  
Do not move, rename, or delete any source file.

---

## Stage 2 — Generate

**Goal:** Transform each document into one polished wiki page using a three-pass loop.

For each document:

### Pass 1 — Writer

Draft a wiki page in Markdown using this structure:

```markdown
# {Title}

## Summary
{One paragraph: what this page is about.}

## Key Points
- {Bullet list of main facts, decisions, or concepts.}

## Details
{Expanded content. Use H3 subheadings to organize naturally.}

## Related
- [[{Linked Page}]]

## References
- Source: {original filename}
```

Infer the entity type from the content (or use the configured types).  
Use `[[Double Bracket]]` wikilinks when referencing other likely pages.

### Pass 2 — Evaluator

Review the draft silently. Score it 1–5 on:
- Coverage (does it capture all key points from the source?)
- Clarity (encyclopedic, neutral tone)
- Structure (all sections present, correct heading levels)

If score < 4, note specific improvements.

### Pass 3 — Editor

Apply the improvements. Output the final page.

Save to: `{output_folder}/{entity_type}/{slugified-title}.md`  
If entity type is unknown, save to: `{output_folder}/general/{slugified-title}.md`

---

## Stage 3 — Topics

**Goal:** Build a normalized taxonomy of key terms across all wiki pages.

1. Read all pages in `output_folder`.
2. Extract key terms, names, acronyms, and concepts.
3. Normalize: merge synonyms, fix inconsistent capitalization, expand abbreviations.
4. Write `{output_folder}/topics.md`:

```markdown
# Topics

## {Canonical Term}
Aliases: {synonym1}, {synonym2}
Pages: [[Page A]], [[Page B]]
```

---

## Stage 4 — Groups

**Goal:** Create category pages that aggregate related content.

Using the entity types from config and the topics from Stage 3:

1. Identify logical groupings.
2. For each group, write a page:

```markdown
# {Group Name}

{One paragraph description of this group.}

## Pages
- [[Page A]]
- [[Page B]]
```

Save to: `{output_folder}/groups/{group-name}.md`

---

## Stage 5 — Index

**Goal:** Rebuild the global `index.md`.

1. Scan all `.md` files in `output_folder` recursively (exclude `index.md` and `topics.md`).
2. Group by the immediate subfolder (entity type).
3. Write `{output_folder}/index.md`:

```markdown
# {wiki_name} — Index

## {Entity Type 1}
- [[Page A]]
- [[Page B]]

## {Entity Type 2}
- [[Page C]]
```

---

## Stage 6 — Consolidate

**Goal:** Detect and merge near-duplicate pages.

For each pair of pages, check for:
- Same or very similar title
- Content overlap above ~70% (shared concepts, sentences, or bullet points)

If duplicates are found:
1. Keep the more complete version as the primary page.
2. Merge unique content from the other page into the primary.
3. Delete the redundant page.
4. Update all wikilinks in other pages that pointed to the deleted file.

Ask the user to confirm before any deletion.

After all merges, re-run [Stage 5 — Index](#stage-5--index).

---

## Stage 7 — Lint

**Goal:** Validate structural and semantic quality of the entire wiki.

Check every page for:

| Check | Description |
|---|---|
| Broken wikilinks | `[[Name]]` with no matching file in output folder |
| Orphan pages | Pages with no incoming links from other pages |
| Missing sections | Pages without Summary, Key Points, or Details |
| Skipped headings | e.g. `##` directly followed by `####` |
| Duplicate content | Paragraphs or bullets that repeat verbatim across pages |

Write `{output_folder}/lint_report.md`:

```markdown
# Lint Report

## Broken Links
- `[[Missing Page]]` in `file.md` (line N)

## Orphan Pages
- `orphan.md` — no incoming links

## Quality Warnings
- `file.md` — missing Summary section
- `file.md` — skipped heading level (## → ####)
```

Report the count to the user: *"Lint complete: N broken links, N orphans, N warnings."*

---

## Stage 8 — Repair

**Goal:** Automatically fix issues found in Stage 7.

### Broken wikilinks

For each broken `[[Link]]`:
- Find the closest matching page title (fuzzy/semantic match).
- If confidence > 80%: replace with the correct `[[Correct Title]]`.
- If confidence ≤ 80%: replace with plain text and add `<!-- TODO: verify link -->`.

### Orphan pages

For each orphan page:
- Identify the most topically relevant existing page.
- Add a link to the orphan in that page's **Related** section.

After all repairs, re-run [Stage 7 — Lint](#stage-7--lint).  
Report remaining issues to the user and stop — do not loop indefinitely.

---

## Quick Reference

| User says | Action |
|---|---|
| "generate the wiki" / "run all" / "build the wiki" | Full Pipeline |
| "process this file" / "new document" | Incremental Mode |
| "update the index" | Stage 5 only |
| "find duplicates" / "consolidate" | Stage 6 only |
| "lint the wiki" / "check quality" | Stage 7 only |
| "fix broken links" / "repair" | Stage 8 only |
| "switch to Portuguese" / any language request | Update CONFIG, switch language |
| "update config" / "change settings" | Re-run setup for changed fields |

---

## Conventions

- All wikilinks use `[[Double Bracket]]` format.
- File names are slugified: lowercase, hyphens, no special characters or spaces.
- Headings use sentence case (not Title Case).
- Never delete, rename, or move source documents from `input_folder`.
- Never delete a wiki page without explicit user confirmation.
- When in doubt about scope, ask — do not assume.
