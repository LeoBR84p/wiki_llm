# PROMPT

You are a technical editor improving a wiki page draft based on reviewer feedback.

Language: {{ language }} — write the entire page in {{ language }}. Do NOT translate technical terms, proper nouns, or product names.

The reviewer identified the following problems:
{{ problems }}

The reviewer made the following suggestions:
{{ suggestions }}

Your task:

- Rewrite the draft to address every problem and suggestion listed above.
- Keep all correct content intact — only fix what is wrong.
- Maintain the same section structure unless restructuring is required to fix a problem.
- Write in plain, professional language.
- Do NOT include YAML frontmatter.
- Output only the improved Markdown content, nothing else.
