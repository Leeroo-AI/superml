# SuperML

An ML engineering plugin for AI coding agents.

It adds two things your agent doesn't have out of the box:

**Process** — seven skills that cover how ML work actually gets done. Plan the run, verify the config, debug the failure, iterate on results, track experiments across sessions. Basically the same steps you'd follow yourself, just encoded so your agent follows them too.

**Memory** — powered by [Leeroopedia](https://leeroopedia.com), 27k+ pages of best practices and hard-won lessons from 1000+ ML/AI frameworks. When your agent recommends a config or a fix, it can point to where it learned it.

Works with Claude Code, Cursor, Codex, OpenCode, and Gemini CLI.

## How It Works

1. **A session hook** kicks in automatically — no manual setup per conversation.
2. **Skills** walk your agent through ML workflows — before launching a training run, it checks the config; when something breaks, it debugs by root cause; after results, it logs what worked.
3. **MCP tools** connect to the Leeroopedia knowledge base — your agent looks things up and cites real framework docs.
4. **A persistent ML agent** (`ml-expert`) picks up heavier tasks and keeps track of your hardware, experiments, and lessons learned across sessions.

## Prerequisites

### 1. Get an API Key

Head to [app.leeroopedia.com](https://app.leeroopedia.com/dashboard), grab an API key from the dashboard.

- Keys look like `kpsk_...`
- **$20 free credit** on signup, no credit card needed

### 2. Set the Environment Variable

```bash
export LEEROOPEDIA_API_KEY=kpsk_your_key_here
```

Add it to your shell profile (`~/.bashrc`, `~/.zshrc`, `~/.config/fish/config.fish`) so it sticks.

### 3. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

### Claude Code

```bash
claude plugin add --from-github leeroo-ai/superml
```

Or install manually:

```bash
git clone https://github.com/leeroo-ai/superml.git
claude plugin add ./superml
```

### Cursor

```bash
git clone https://github.com/leeroo-ai/superml.git
# Cursor auto-detects .cursor-plugin/plugin.json
```

### Codex

See [.codex/INSTALL.md](.codex/INSTALL.md).

### OpenCode

See [.opencode/INSTALL.md](.opencode/INSTALL.md).

### Gemini CLI

```bash
git clone https://github.com/leeroo-ai/superml.git
gemini extension add ./superml/gemini-extension.json
```

### Alternative: Remote MCP (no local install)

If you just want the knowledge base without the full plugin:

```bash
claude mcp add --transport http leeroopedia "https://mcp.leeroopedia.com/mcp?token=YOUR_KEY"
```

You get the MCP tools (memory) but not the workflow skills (process).

### Verify Installation

Start a conversation and try something like:

```
Verify my LoRA config: r=64, alpha=16, lr=5e-3, target_modules=q_proj,v_proj
```

If it's working, your agent will call Leeroopedia tools and cite sources with `[PageID]` tags in its response.

## What's Inside

### Skills

<!-- BEGIN_SKILLS_TABLE -->
| Skill | What it does |
|-------|-------------|
| [ml-plan](skills/ml-plan/SKILL.md) | Plan training runs, architectures, and multi-step pipelines |
| [ml-verify](skills/ml-verify/SKILL.md) | Check configs, code, and math before you burn GPU hours |
| [ml-debug](skills/ml-debug/SKILL.md) | Debug OOM, NaN, divergence, crashes, bad throughput |
| [ml-iterate](skills/ml-iterate/SKILL.md) | Ranked next steps when results aren't where you want them |
| [ml-experiment](skills/ml-experiment/SKILL.md) | Track experiments — hypotheses, results, and learnings across sessions |
| [ml-research](skills/ml-research/SKILL.md) | Deep-dive into ML topics, compare approaches, survey frameworks |
| [using-superml](skills/using-superml/SKILL.md) | Loaded at session start — wires up skills to KB tools and sets quality standards |
<!-- END_SKILLS_TABLE -->

### Agent

**ml-expert** — a persistent ML engineer agent for the bigger stuff: pipeline reviews, deep analysis, framework deep-dives. It remembers your hardware setup, past experiments, and lessons learned across sessions.

### MCP Tools

Eight tools that talk to the Leeroopedia knowledge base:

| Tool | What it does |
|------|-------------|
| `search_knowledge` | Look up best practices, configs, framework details |
| `build_plan` | Get a KB-grounded implementation plan |
| `review_plan` | Spot risks and gaps in an existing plan |
| `verify_code_math` | Check code or config against documented behavior |
| `diagnose_failure` | Match errors against known framework failure patterns |
| `propose_hypothesis` | Ranked alternatives when you're stuck |
| `query_hyperparameter_priors` | Recommended parameter ranges for your specific setup |
| `get_page` | Pull up the full page behind a `[PageID]` citation |

## Links

- [Leeroopedia](https://leeroopedia.com) — the ML/AI knowledge base behind the memory
- [leeroopedia-mcp](https://github.com/leeroo-ai/leeroopedia-mcp) — MCP server repo

## License

Apache-2.0
