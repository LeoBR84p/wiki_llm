# PROMPT

You are a wiki author writing a theme summary page for the taxonomy "{{ taxonomy }}".

Language: {{ language }} — write the entire page in {{ language }}. Do NOT translate technical terms, proper nouns, or product names.

Theme: {{ term }}
Related pages: {{ links }}

Write a concise Markdown page for this theme. All headings must be in {{ language }}. Structure:

- `# {{ term }}` — heading
- `## Visão Geral` — 2–3 sentences describing what this theme covers across the wiki.
- `## Páginas Relacionadas` — bullet list using the links provided: {{ links }}

Do NOT include YAML frontmatter. Output only the Markdown content.
