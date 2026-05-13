# PROMPT

You are a wiki author writing a theme summary page for the taxonomy "{{ taxonomy }}".

Language: {{ language }} — write the entire page in {{ language }}. Do NOT translate technical terms, proper nouns, or product names.

Theme: {{ term }}
Related pages: {{ links }}

Write a concise Markdown page for this theme. Structure:

- `# {{ term }}` — heading
- `## Overview` — 2–3 sentences describing what this theme covers across the wiki.
- `## Related Pages` — bullet list using the links provided: {{ links }}

Do NOT include YAML frontmatter. Output only the Markdown content.
