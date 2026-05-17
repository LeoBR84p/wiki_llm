# Wiki Agent — Claude Instructions

> Full instructions live in [`AGENTS.md`](./AGENTS.md).  
> This file is the Claude Code / claude.ai entry point. Read `AGENTS.md` first, then follow its stages.

## Claude-specific notes

- Use the `Read` tool to extract text from documents in `input_folder`.
- Use the `Write` tool to save wiki pages to `output_folder`.
- Use the `Edit` tool when updating existing pages (Consolidate, Repair stages).
- Use the `Glob` tool to scan `output_folder` for all `.md` files (Index, Lint stages).
- Store the CONFIG block update in `AGENTS.md` using the `Edit` tool after setup.
- Never use `Bash` to run project code — operate entirely through file tools.
