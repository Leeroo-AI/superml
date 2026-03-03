# Leeroopedia Plugin

An AI coding agent plugin that gives Claude Code (and Cursor, Codex, OpenCode, Gemini CLI) access to **Leeroopedia** — an ML/AI knowledge base with **27,667 pages** from **1000+ repos**.

Instead of guessing about framework internals, config formats, and best practices, your agent searches a curated knowledge base and gives you grounded, cited answers.

## Why

LLMs hallucinate ML/AI details. Framework APIs change, config formats vary, and "standard" hyperparameters depend on model size, hardware, and task. Leeroopedia fixes this by grounding agent responses in documented best practices from real framework repos.

## Prerequisites

### 1. Get an API Key

Sign up at [app.leeroopedia.com](https://app.leeroopedia.com) → Dashboard → API Keys → Copy key.

- Format: `kpsk_...`
- **$20 free credit** on signup, no credit card required

### 2. Set the Environment Variable

```bash
export LEEROOPEDIA_API_KEY=kpsk_your_key_here
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, `~/.config/fish/config.fish`) for persistence.

### 3. Install `uv` (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

### Claude Code

```bash
claude plugin add --from-github leeroo-ai/leeroopedia-plugin
```

Or install manually:

```bash
git clone https://github.com/leeroo-ai/leeroopedia-plugin.git
claude plugin add ./leeroopedia-plugin
```

### Cursor

```bash
git clone https://github.com/leeroo-ai/leeroopedia-plugin.git
# Cursor auto-detects .cursor-plugin/plugin.json
```

### Codex

See [.codex/INSTALL.md](.codex/INSTALL.md).

### OpenCode

See [.opencode/INSTALL.md](.opencode/INSTALL.md).

### Gemini CLI

```bash
git clone https://github.com/leeroo-ai/leeroopedia-plugin.git
gemini extension add ./leeroopedia-plugin/gemini-extension.json
```

### Alternative: Remote MCP (no Python/uv required)

```bash
claude mcp add --transport http leeroopedia "https://mcp.leeroopedia.com/mcp?token=YOUR_KEY"
```

## Skills

<!-- BEGIN_SKILLS_TABLE -->
| Skill | Description |
|-------|-------------|
| [ml-debug](skills/ml-debug/SKILL.md) | Use when something is failing in ML/AI work — OOM, NaN, divergence, crashes, bad throughput, wrong outputs, dependency conflicts |
| [ml-iterate](skills/ml-iterate/SKILL.md) | Use when the user is stuck, needs ranked next steps, or wants alternatives after initial experiments with ML/AI systems |
| [ml-plan](skills/ml-plan/SKILL.md) | Use when the user wants an implementation plan, architecture design, or multi-step ML pipeline grounded in Leeroopedia KB |
| [ml-research](skills/ml-research/SKILL.md) | Use when the user wants to understand an ML/AI topic, compare approaches, or survey framework capabilities via Leeroopedia KB |
| [ml-verify](skills/ml-verify/SKILL.md) | Use when the user wants to verify code correctness, config validity, math/logic accuracy, or API usage against Leeroopedia KB |
| [using-leeroopedia](skills/using-leeroopedia/SKILL.md) | Use when starting any conversation involving ML/AI — establishes how to use Leeroopedia KB tools and workflow skills |
<!-- END_SKILLS_TABLE -->

### Agent

**ml-expert** — A senior ML/AI engineer agent for heavy-lift tasks: pipeline reviews, deep analysis, framework deep-dives. Maintains persistent memory of your hardware setup, experiments, and lessons learned across sessions.

## How It Works

1. **SessionStart hook** injects the bootstrap skill into every conversation
2. **Bootstrap skill** (`using-leeroopedia`) teaches the agent when and how to use the KB
3. **Workflow skills** (`ml-plan`, `ml-debug`, etc.) guide specific task types with structured tool sequences
4. **8 MCP tools** (`search_knowledge`, `build_plan`, `review_plan`, `verify_code_math`, `diagnose_failure`, `propose_hypothesis`, `query_hyperparameter_priors`, `get_page`) connect to the Leeroopedia knowledge base
5. **ml-expert agent** handles complex multi-step tasks with persistent memory

## Links

- [Leeroopedia](https://leeroopedia.com) — ML/AI knowledge base
- [leeroopedia-mcp](https://github.com/leeroo-ai/leeroopedia-mcp) — MCP server
- [Benchmarks](https://leeroopedia.com/benchmarks) — Knowledge base coverage and accuracy

## License

Apache-2.0
