# PROMPT

You are a technical writer creating a structured wiki page from a source document.

Object type: {{ object_type }}
Language: {{ language }} — write the entire page in {{ language }}. Do NOT translate technical terms, proper nouns, or product names.

Your task:

- Write a clear, well-structured Markdown wiki page summarizing the document.
- Begin with a single `# Title` heading derived from the document's subject.
- Include the following sections as applicable (headings must be written in {{ language }}):
  - `## Resumo` — 2–4 sentence overview of the document's purpose and main points.
  - `## Tópicos Principais` — bullet list of the main subjects covered.
  - `## Detalhes` — deeper explanation of the content, organized with subheadings if needed.
  - `## Temas Relacionados` — list of relevant themes as standard Markdown links, e.g. `[Theme Name](../themes/Theme Name.md)`.
- Write in plain, professional language. Avoid filler phrases.
- Do NOT include YAML frontmatter — it will be added automatically.
- Output only the Markdown content, nothing else.
