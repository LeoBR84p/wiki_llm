# PROMPT

You are a wiki editor. Given a list of wiki page titles, identify groups of semantically duplicate or near-duplicate pages that should be merged into one.

Return ONLY a JSON array (no markdown, no explanation) where each element has:
  { "canonical": "<best canonical title>", "duplicates": ["<dup1>", "<dup2>", ...] }

Only include groups with at least one duplicate. If there are no duplicates, return [].

Example: [{"canonical": "Credit Risk", "duplicates": ["Credit Risks", "Risk of Credit"]}]
