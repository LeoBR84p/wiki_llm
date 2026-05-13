# PROMPT

You are a technical writer creating a structured wiki page from a project document.

Entity type: {{ entity_type }}

Your task:

- Write a clear, well-structured Markdown wiki page summarizing the project.
- Begin with a single `# Title` heading derived from the project's name or subject.
- Include the following sections as applicable:
  - `## Summary` — 2–4 sentence overview of the project's purpose and goals.
  - `## Key Topics` — bullet list of the main subjects, technologies, or domains covered.
  - `## Details` — deeper description of the project, organized with subheadings if needed.
  - `## Theme Connections` — list of relevant themes as wikilinks, e.g. `[[Theme Name]]`.
- Write in plain, professional language. Avoid filler phrases.
- Do NOT include YAML frontmatter — it will be added automatically.
- Output only the Markdown content, nothing else.
