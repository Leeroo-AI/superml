# Self-Refine

Optimize SuperML skills for your specialized ML domain. Describe what you work on, generate a targeted test suite, and let the automated judge-refine loop improve the skills for your use case.

## How It Works

```
Describe your domain ──► Generate test suite ──► Run tests ──► Judge scores ──► Refine skills
         │                      │                    │              │               │
  "biomedical NLP"      12-18 prompts         baseline vs     3 judges per    edit skills,
                        covering all           plugin A/B      test, median    retest, revert
                        skill types            comparison      scores          regressions
```

**The loop runs automatically.** You provide the domain, it does the rest: generates tests, runs baselines, runs with the plugin, judges both, identifies weak spots, edits the skill files, retests to confirm improvement, and reverts anything that regresses.

## Prerequisites

- Python 3.10+
- PyYAML: `pip install pyyaml`
- Claude CLI installed and authenticated
- Optional: `LEEROOPEDIA_API_KEY` for KB mode (works without it using web search)

## Quick Start

```bash
# 1. Generate a test suite for your domain
python self-refine/generate_suite.py "biomedical fine-tuning with clinical NLP"

# 2. Run the suite — tests, judges, and refines automatically
python self-refine/run.py --suite self-refine/suites/biomedical-fine-tuning-with-clinical-nlp.yaml

# 3. Check results
ls self-refine/logs/round1/
```

## Generating a Test Suite

Describe your ML domain in natural language. The generator calls Claude to create 12-18 test prompts that cover all skill types:

```bash
python self-refine/generate_suite.py "biomedical fine-tuning with clinical NLP"
python self-refine/generate_suite.py "RAG pipelines for legal documents with Qdrant"
python self-refine/generate_suite.py "LLM serving optimization for production"
```

**Options:**

```bash
# Custom output path
python self-refine/generate_suite.py "RAG for legal docs" -o suites/legal.yaml

# Extend an existing suite (fills coverage gaps)
python self-refine/generate_suite.py "multimodal training" --existing suites/my-tests.yaml
```

The generator ensures coverage across skill types:

| Skill | What the generator creates |
|-------|---------------------------|
| ml-plan | Architecture and setup questions for your domain |
| ml-debug | Debugging scenarios with domain-specific errors and symptoms |
| ml-verify | Configs with deliberate issues relevant to your use case |
| ml-iterate | Past experiment results needing next-step recommendations |
| ml-research | Framework comparisons and deep-dives for your stack |

## Writing Your Own Suite

Create a YAML file in `suites/`. Only `id` and `prompt` are required:

```yaml
suite: my-domain
description: What this suite covers

tests:
  - id: my_first_test
    prompt: |
      I'm fine-tuning Llama-3-8B for clinical NER on 20k annotated
      medical records. Hardware: 1xA100 80GB. Walk me through the
      complete QLoRA setup with the right target modules and chat template.

  - id: my_debug_test
    prompt: |
      Training loss spikes to NaN at step 200 when fine-tuning
      BioBERT on radiology reports. Config: lr=5e-4, bf16, batch_size=16.
      What's causing this?

  - id: my_verify_test
    prompt: |
      Check this config before I start training:
      model: microsoft/BioGPT-Large
      lora_r: 256
      learning_rate: 1e-2
      max_seq_length: 4096
      Hardware: 1xT4 16GB
```

**Optional fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `category` | `uncategorized` | For grouping in reports |
| `expected_skill` | none | Verify this skill fires (e.g., `ml-debug`) |
| `needs_gpu` | `false` | Tells the judge to score actionability based on config quality, not execution |
| `expected_agent` | none | Verify the ml-expert agent is invoked |

## Running Tests

```bash
# Run the default suite (38 built-in ML tasks)
python self-refine/run.py

# Run a custom suite
python self-refine/run.py --suite self-refine/suites/biomed.yaml

# Web mode (no Leeroopedia API key needed)
python self-refine/run.py --suite self-refine/suites/biomed.yaml --no-kb

# Run specific tests only
python self-refine/run.py --suite self-refine/suites/biomed.yaml biomed_01 biomed_03
```

## What Happens During a Run

For each test in the suite:

1. **Baseline** — runs the prompt through Claude without the plugin
2. **Plugin** — runs the same prompt with SuperML skills loaded
3. **Judge** — 3 independent LLM judges score both responses (median scores for consistency)
4. **Efficiency** — counts KB citations and tool call efficiency

After all tests:

5. **Refine pass 1** — analyzes judge feedback, generates targeted edits to skill files, retests modified tests, reverts regressions
6. **Refine pass 2** — same on remaining weaknesses

### Scoring

Each response is scored on 5 dimensions (0-3 each, 15 max):

| Dimension | What it measures |
|-----------|-----------------|
| Correctness | Right configs, parameters, API calls |
| Specificity | Framework-specific details, not generic advice |
| Mistake prevention | Catches pitfalls with concrete fixes |
| Actionability | Runnable code/config, not abstract discussion |
| Grounding | Cites sources, version-specific details |

### Output

Results go to `self-refine/logs/roundN/`:

```
logs/round1/
├── summary.json                    # Full round results
├── t01_test_id.baseline.json       # Baseline response
├── t01_test_id.plugin.json         # Plugin response
├── t01_test_id.judge.json          # Judge scores
├── t01_test_id.efficiency.json     # Citation efficiency
├── refine_pass1.json               # Refiner edits
├── refine_pass1_retest/            # Retest results
├── refine_pass2.json
└── ml-debug.pass1.backup.md        # Skill backups (for revert)
```

## Tips

- **Start small** — generate a suite, run 2-3 tests first to check quality, then run the full suite
- **Iterate** — run multiple rounds. Each round builds on previous baselines (cached) and checks for cross-round regressions
- **Review edits** — check `git diff` after a run to see what the refiner changed in the skill files
- **Commit good refinements** — if the scores improve, commit the skill edits to lock in the gains
- **Web mode first** — use `--no-kb` if you don't have an API key. Skills fall back to web search grounding

## Directory Structure

```
self-refine/
├── generate_suite.py      # Suite generator (calls Claude)
├── run.py                 # Test harness (baseline → plugin → judge → refine)
├── judge_prompt.md        # Judge scoring rubric
├── suites/
│   └── default.yaml       # 38 built-in ML tests
└── logs/                  # Round results (gitignored)
```
