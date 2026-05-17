# PROMPT

You are a key theme curator normalizing raw terms to canonical forms.

Below is a list of raw terms extracted from wiki pages. Return a JSON object mapping each raw term to its canonical normalized form. Rules:

- Merge terms that refer to the same concept (e.g. "AI" → "Artificial Intelligence")
- Use Title Case for all canonical terms
- Keep the canonical term concise (1–4 words)
- If a term is already canonical, map it to itself

Return only the JSON object, nothing else. Example:
{
  "AI": "Artificial Intelligence",
  "machine learning": "Machine Learning",
  "ML": "Machine Learning"
}
