# Self-Refine

Test and improve SuperML skills for your specialized ML domain. Generate a test suite, run it against the plugin, and let the judge-refine loop optimize the skills for your use case.

## Prerequisites

- Python 3.10+
- PyYAML: `pip install pyyaml`
- Claude CLI installed and authenticated
- Optional: `LEEROOPEDIA_API_KEY` for KB mode (web mode works without it)

## Quick Start

```bash
# 1. Generate a test suite for your domain
python self-refine/generate_suite.py "biomedical fine-tuning with clinical NLP"

# 2. Run the suite
python self-refine/run.py --suite self-refine/suites/biomedical-fine-tuning-with-clinical-nlp.yaml

# 3. Skills are automatically refined — check the logs
ls self-refine/logs/round1/
```

## Generate a Test Suite

Describe your domain and the generator creates test prompts that exercise all skill types (plan, debug, verify, iterate, research):

```bash
python self-refine/generate_suite.py "RAG pipelines for legal documents with Qdrant"
python self-refine/generate_suite.py "LLM serving optimization for production" -o suites/serving.yaml
python self-refine/generate_suite.py "multimodal training" --existing suites/default.yaml
```

Or write your own `suites/my-domain.yaml`:

```yaml
suite: my-domain
description: What this suite covers

tests:
  - id: my_first_test
    prompt: "I'm fine-tuning X for Y with Z hardware..."
  - id: my_second_test
    prompt: "Debug this OOM error when training..."
```

Only `id` and `prompt` are required. Optional fields: `category`, `expected_skill`, `needs_gpu`, `expected_agent`.

## Run Tests

```bash
python self-refine/run.py                                           # default suite (38 tests)
python self-refine/run.py --suite suites/my-domain.yaml             # custom suite
python self-refine/run.py --suite suites/my-domain.yaml --no-kb     # web mode (no API key)
python self-refine/run.py --suite suites/my-domain.yaml test_01     # specific test only
```

## How It Works

For each test:
1. **Baseline** — runs Claude without the plugin
2. **Plugin** — runs Claude with SuperML skills loaded
3. **Judge** — 3 independent LLM judges score both on correctness, specificity, mistake prevention, actionability, and grounding (0-3 each, 15 max)
4. **Refine** — analyzes weak scores, generates targeted edits to skill files, retests, reverts regressions

Two refine passes per round. Results saved to `self-refine/logs/roundN/`.
