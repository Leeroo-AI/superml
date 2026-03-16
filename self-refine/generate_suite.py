#!/usr/bin/env python3
"""
Generate a test suite YAML for a specialized ML domain.
Calls Claude to create domain-specific test prompts that exercise SuperML skills.

Usage:
    python self-refine/generate_suite.py "biomedical fine-tuning with clinical NLP"
    python self-refine/generate_suite.py "RAG for legal docs" -o suites/legal.yaml
    python self-refine/generate_suite.py "LLM serving" --existing suites/default.yaml
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

SELF_REFINE_DIR = Path(__file__).resolve().parent

SKILLS = [
    {"name": "ml-plan", "triggers": "planning, architecture, pipeline design, setup"},
    {"name": "ml-verify", "triggers": "config review, pre-launch checks, math verification"},
    {"name": "ml-debug", "triggers": "OOM, NaN, divergence, crashes, latency, bad throughput"},
    {"name": "ml-iterate", "triggers": "plateau, next steps, experiment results analysis"},
    {"name": "ml-research", "triggers": "topic deep-dive, framework comparison, how does X work"},
    {"name": "ml-experiment", "triggers": "experiment tracking, hypothesis logging"},
]


def build_generator_prompt(domain: str,
                           existing_tests: list[dict] | None = None) -> str:
    """Build the prompt for Claude to generate test cases."""
    skills_desc = "\n".join(
        f"- **{s['name']}**: {s['triggers']}" for s in SKILLS
    )

    existing_section = ""
    if existing_tests:
        existing_summary = "\n".join(
            f"  - {t.get('id', '?')}: {t.get('prompt', '')[:80]}..."
            for t in existing_tests[:20]
        )
        existing_section = (
            f"\n## Existing tests (fill coverage gaps, don't duplicate):\n"
            f"{existing_summary}\n"
            f"Total existing: {len(existing_tests)}\n"
        )

    target_count = "5-10 additional" if existing_tests else "12-18"

    return f"""You are generating a test suite for evaluating ML workflow skills in a specialized domain.

## Domain
{domain}

## Available skills to exercise:
{skills_desc}

{existing_section}

## Task
Generate {target_count} test prompts for this domain. Each test should be a realistic ML engineering question that someone working in this domain would ask their AI coding assistant.

## Requirements
1. Each test needs:
   - `id`: lowercase, underscores, descriptive (e.g., `biomed_qlora_setup`, `legal_rag_chunking`)
   - `prompt`: 3-10 sentences with specific details — model names, dataset sizes, hardware specs, error messages where relevant
2. Cover all skill types: at least 2 tests for plan, 2 for debug, 1 for verify, 1 for iterate, 1 for research
3. Mix of complexity: some straightforward single-topic, some multi-faceted
4. Include realistic details specific to the domain:
   - Specific model names relevant to the domain
   - Realistic dataset sizes and descriptions
   - Plausible hardware setups
   - For debug tests: specific error messages, loss values, GPU memory numbers
   - For verify tests: a config with 2-3 deliberate issues to catch
   - For iterate tests: previous experiment results with specific metrics

## Output format — return ONLY valid JSON:
```json
{{
  "suite": "short-kebab-name",
  "description": "One line describing the suite",
  "tests": [
    {{
      "id": "descriptive_test_id",
      "prompt": "The full detailed test prompt..."
    }}
  ]
}}
```

Make prompts read like messages from a real ML engineer working in this domain, not generic templates. Include framework-specific details, version numbers, and concrete numbers wherever possible."""


def generate_suite(domain: str, existing_path: str | None = None,
                   output_path: str | None = None) -> Path:
    """Call Claude to generate a test suite."""
    existing_tests = None
    if existing_path:
        ep = Path(existing_path)
        if not ep.is_absolute():
            ep = Path.cwd() / ep
        with open(ep, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            existing_tests = data.get("tests", [])
        print(f"Loaded {len(existing_tests)} existing tests from {ep.name}")

    prompt = build_generator_prompt(domain, existing_tests)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["PATH"] = (
        f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}"
        f":{env.get('PATH', '')}"
    )

    print(f"Generating test suite for: {domain}")
    print(f"Calling Claude...")

    start = time.time()
    try:
        result = subprocess.run(
            ["claude", "--dangerously-skip-permissions",
             "-p", prompt, "--max-turns", "1",
             "--output-format", "json"],
            capture_output=True, text=True, timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Claude timed out after 120s")
        sys.exit(1)

    elapsed = time.time() - start

    try:
        output_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse Claude output")
        print(f"stdout: {result.stdout[:500]}")
        print(f"stderr: {result.stderr[:500]}")
        sys.exit(1)

    response_text = output_data.get("result", "")
    cost = output_data.get("total_cost_usd", 0)

    # Extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        print(f"ERROR: Claude did not return valid JSON")
        print(f"Response: {response_text[:500]}")
        sys.exit(1)

    try:
        suite_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in response: {e}")
        print(f"Match: {json_match.group()[:300]}")
        sys.exit(1)

    # Validate minimum structure
    tests = suite_data.get("tests", [])
    valid_tests = [t for t in tests if "id" in t and "prompt" in t]
    if not valid_tests:
        print(f"ERROR: No valid tests generated")
        print(f"Raw tests: {json.dumps(tests[:3], indent=2)}")
        sys.exit(1)

    suite_data["tests"] = valid_tests

    # Merge with existing tests if provided
    if existing_tests:
        existing_ids = {t["id"] for t in existing_tests}
        new_tests = [t for t in valid_tests if t["id"] not in existing_ids]
        suite_data["tests"] = existing_tests + new_tests
        print(f"Merged: {len(existing_tests)} existing + {len(new_tests)} new")

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        safe_name = re.sub(r'[^a-z0-9]+', '-',
                           domain.lower().strip())[:40].strip('-')
        out = SELF_REFINE_DIR / "suites" / f"{safe_name}.yaml"

    if not out.is_absolute():
        out = Path.cwd() / out

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(suite_data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False, width=120)

    test_count = len(suite_data.get("tests", []))
    print(f"\nGenerated {test_count} tests in {elapsed:.1f}s (${cost:.3f})")
    print(f"Saved to: {out}")
    print(f"\nRun with:")
    print(f"  python self-refine/run.py --suite {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate a test suite for a specialized ML domain")
    parser.add_argument(
        "domain",
        help="Domain description (e.g., 'biomedical fine-tuning with clinical NLP')")
    parser.add_argument(
        "-o", "--output",
        help="Output YAML path (default: suites/<domain-slug>.yaml)")
    parser.add_argument(
        "--existing",
        help="Existing suite YAML to extend (fills coverage gaps)")
    args = parser.parse_args()

    generate_suite(args.domain, existing_path=args.existing,
                   output_path=args.output)


if __name__ == "__main__":
    main()
