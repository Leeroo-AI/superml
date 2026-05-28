# superml-cli

A fast, low-token CLI for ML engineering workflows, powered by Claude Code.

Runs the same workflows as the [SuperML plugin](https://github.com/leeroo-ai/superml) (`plan`, `debug`, `research`, `verify`, `iterate`, `experiment`) but as direct terminal commands — no plugin overhead, no session-start hook injection, no leeroopedia auth required.

## Why

Using SuperML skills inside Claude Code injects ~3,000–5,000 tokens of context per session (the full `using-superml` SKILL.md + session-start hook). This CLI bypasses that entirely: each command is a single `claude -p` call with a ~300-token compact prompt.

| | SuperML plugin | superml-cli |
|---|---|---|
| Auth required | leeroopedia API key for KB mode | Claude Code (OAuth) |
| Tokens per call | ~5,000–10,000 (plugin context + hooks) | ~500–1,500 |
| Invocation | `/ml-plan` inside Claude Code | `superml plan "..."` in terminal |
| Speed | Session startup + hook injection | Direct API call, streams immediately |

## Requirements

- [Claude Code](https://claude.ai/code) installed and authenticated (`claude --version`)
- Bash

No Python, no `pip install`, no API key beyond your existing Claude subscription.

## Install

```bash
git clone https://github.com/BlackhatShiftey/superml-cli
cd superml-cli
bash install.sh
```

Or one-liner:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/BlackhatShiftey/superml-cli/main/install.sh)
```

## Usage

```bash
superml <skill> "<task>" [--model haiku|sonnet|opus]
```

### Skills

| Skill | When to use | Example |
|---|---|---|
| `plan` | Starting a new ML project or feature | `superml plan "fine-tune Llama 3.1 8B with QLoRA on 1xA100"` |
| `debug` | Something broke (OOM, NaN, crash, slow) | `superml debug "CUDA OOM, batch_size=8, seq_len=2048, A100 80GB"` |
| `research` | Understand a framework or technique | `superml research "how does vLLM chunked prefill work"` |
| `verify` | Check code/config for bugs | `superml verify "$(cat train_config.yaml)"` |
| `iterate` | Improve results after an experiment | `superml iterate "tried rank-8 LoRA, loss 0.35, not converging after 2k steps"` |
| `experiment` | Design a reproducible experiment | `superml experiment "compare LoRA rank 8 vs 16 on MMLU 5-shot"` |

### Examples

```bash
# Planning
superml plan "multi-node DeepSpeed ZeRO-3 training on 8xH100 with gradient checkpointing"

# Debugging
superml debug "loss NaN after step 200, LR=3e-4, grad_clip=1.0, bf16, Llama-2-13B"

# Research
superml research "flash attention v2 vs SDPA — when does each win"

# Verify a config file
superml verify "$(cat axolotl_config.yaml)"

# Iteration
superml iterate "QLoRA rank=8 alpha=16, eval_loss=0.41 at 1k steps, baseline=0.38"

# Use sonnet for harder tasks
superml plan --model sonnet "production vLLM serving with autoscaling on AWS EKS"
```

### Model selection

Default model is `haiku` (fastest, cheapest). Override per-command or globally:

```bash
# Per-command
superml debug --model sonnet "..."

# Global override
export SUPERML_MODEL=sonnet
superml plan "..."
```

## Response format

Every response includes these sections:

- **Main content** (Plan / Diagnosis / Answer / etc.) — concrete, runnable output
- **Verify** — exact command to confirm it worked
- **References** — 3+ links to official docs
- **Pitfalls** — 3+ specific failure modes with exact fixes

## Contributing

The skill prompts live in `skills/`. Each is a compact (~300-token) system prompt that captures the essential rules of the corresponding SuperML workflow skill.

To improve a skill:
1. Edit `skills/<skill>.md`
2. Test: `superml <skill> "a representative task"`
3. Check that the output includes all required sections and cites real sources
4. Submit a PR

## Relation to SuperML

This project is a companion to the [SuperML plugin](https://github.com/leeroo-ai/superml). The skill workflows (`plan → verify → experiment → iterate`) are the same. The difference is invocation: plugin skills run inside Claude Code with full context; this CLI runs standalone with a minimal prompt.

If you have a leeroopedia API key, the plugin's KB mode gives richer grounding. Without one, this CLI is the faster path.

## License

MIT
