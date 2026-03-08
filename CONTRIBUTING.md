# Contributing to SuperML

Thanks for your interest in contributing! SuperML is an open-source ML engineering plugin backed by the [Leeroopedia](https://leeroopedia.com) knowledge base.

## Ways to Contribute

- **Report bugs** — open an issue with steps to reproduce
- **Suggest improvements** — skills, agent prompts, URL registry entries
- **Add framework coverage** — new URLs in the web mode registry, new pitfall patterns
- **Fix docs** — typos, unclear instructions, outdated examples
- **Test on new platforms** — verify install docs work on your setup

## Getting Started

1. Fork and clone the repo:
   ```bash
   git clone https://github.com/leeroo-ai/superml.git
   cd superml
   ```

2. Test locally with Claude Code:
   ```bash
   claude --plugin-dir .
   ```

3. Make your changes, then verify the plugin loads and skills trigger correctly.

## Project Structure

```
superml/
├── .claude-plugin/plugin.json   # Plugin manifest
├── .mcp.json                    # MCP server config (remote HTTP)
├── skills/                      # ML workflow skills (SKILL.md files)
├── agents/                      # Custom subagent definitions
├── hooks/                       # SessionStart hook for context injection
├── .codex/                      # Codex install docs
├── .opencode/                   # OpenCode install docs and plugin
└── docs/                        # Design docs and implementation guides
```

## Editing Skills

Skills live in `skills/<skill-name>/SKILL.md`. Each has YAML frontmatter and markdown instructions.

When editing a skill:
- Keep the frontmatter `description` under 1024 chars — it's what triggers auto-invocation
- Test with `claude --plugin-dir .` and try prompts that should trigger the skill
- Check that KB mode and web mode paths both work

## Editing the Session Hook

The `hooks/session-start` script injects context at conversation start. If you modify it:
- Test with API key set: `LEEROOPEDIA_API_KEY=test bash hooks/session-start | python3 -c "import sys,json; json.load(sys.stdin); print('OK')"`
- Test without API key: `unset LEEROOPEDIA_API_KEY && bash hooks/session-start | python3 -c "import sys,json; json.load(sys.stdin); print('OK')"`
- Output must be valid JSON in both cases

## Pull Requests

1. Create a branch from `main`
2. Make focused changes — one concern per PR
3. Test that the plugin loads without errors
4. Write a clear PR description explaining what changed and why

## Code Style

- Skills use markdown with structured sections (phases, gates, anti-pattern tables)
- Shell scripts use `set -euo pipefail` and `${VAR:-default}` for safety
- Config files use the platform's native format (JSON for Claude Code/Cursor, TOML for Codex)

## Reporting Issues

Open an issue at [github.com/leeroo-ai/superml/issues](https://github.com/leeroo-ai/superml/issues) with:
- Your platform (Claude Code, Cursor, Codex, OpenCode, Gemini CLI)
- Plugin version (from `.claude-plugin/plugin.json`)
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
