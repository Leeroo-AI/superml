# SuperML

Give your AI coding agent ML engineering superpowers.

It adds two things your coding agent doesn't have:

**ML Pipeline**: Seven skills that encode the workflow you already follow. Plan against real framework docs. Catch config mistakes before they cost you GPU hours. Debug OOM, NaN, and divergence by root cause, not by guessing. Get ranked next steps when metrics plateau. An agentic experiment memory carries hypotheses, results, and lessons across sessions — your agent stops repeating failed experiments and starts compounding what works.

**Memory**: Backed by [Leeroopedia](https://leeroopedia.com), 27k+ pages across 1000+ ML/AI frameworks. Config references, debugging heuristics, implementation patterns, and battle-tested defaults from vLLM to DeepSpeed to LangChain. Built by the [Leeroo](https://leeroo.com) continuous learning system, structured as a browsable wiki, and continuously updated by AI and human engineers. When your agent recommends a config, it points to the page it learned it from.

Works with Claude Code, Cursor, Codex, OpenCode, and Gemini CLI.

## How It Works

1. **A session hook** loads automatically, zero setup per conversation.
2. **Skills** guide the ML workflow, verify before launch, debug by root cause, iterate on results, track what worked.
3. **MCP tools** connect to the Leeroopedia knowledge base, your agent looks things up and cites real docs instead of guessing.
4. **A persistent ML agent** (`ml-expert`) handles deeper tasks and remembers your hardware, experiments, and lessons across sessions.

## Results

We gave 38 ML tasks to Claude Code — once with SuperML, once without — and had an independent LLM judge rate both. Each response is scored out of 15 across correctness, specificity, mistake prevention, actionability, and grounding. Tasks cover QLoRA fine-tuning, distributed training, LLM inference optimization, alignment (DPO/GRPO), RAG pipelines, model merging, quantization, and more.

| | With SuperML | Without |
|--|--------|----------|
| ML task average | **13.2 / 15** | 8.3 / 15 |
| ML task win rate | **91%** | 9% |

See [TESTED_TASKS.md](TESTED_TASKS.md) for the full list of tasks and scores.

## Prerequisites

### API Key (optional, recommended)

The plugin works without an API key — skills use web search to ground answers. With a key, your agent gets access to the Leeroopedia knowledge base (27k+ pages, faster and more precise lookups). The plugin will tell you if it's running without a key.

To get a key: [app.leeroopedia.com](https://app.leeroopedia.com/dashboard) — $20 free credit on signup, no credit card.

```bash
export LEEROOPEDIA_API_KEY=kpsk_your_key_here
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`) so it persists.

## Installation

### Claude Code

Register the marketplace, then install the plugin:

```
/plugin marketplace add leeroo-ai/leeroo-marketplace
/plugin install superml@leeroo-marketplace
```

Or install directly from GitHub:

```bash
claude plugin add --from-github leeroo-ai/superml
```

### Cursor

In Cursor Agent chat:

```
/add-plugin superml
```

Or clone into your project — Cursor auto-detects `.cursor-plugin/plugin.json`:

```bash
git clone https://github.com/leeroo-ai/superml.git
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

If you just want the knowledge base without the full plugin, see [leeroopedia-mcp](https://github.com/Leeroo-AI/leeroopedia-mcp) for setup instructions.

You get the MCP tools (memory) but not the workflow skills (process).

### Verify Installation

Start a conversation and try something like:

```
I'm fine-tuning Llama 3.1 8B on 50k instruction pairs with 1xA100 80GB.
Set up the full training config — QLoRA, proper chat template, loss masking on prompts.
```

If it's working, your agent will ground its answer in documentation (KB citations or web sources), catch common pitfalls before they waste a training run, and give you a runnable config.

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
