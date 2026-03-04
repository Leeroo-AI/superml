# Quality Score A/B + Judge + Refiner — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the boolean checklist quality score with an A/B test + LLM judge + auto-refiner pipeline that proves whether the plugin makes ML code better.

**Architecture:** `test_interactive.py` gains 3 new phases (baseline run, judge, efficiency parse) and a refiner loop. The judge prompt lives in a separate file. The TESTS list and file structure are reused; run_test is refactored into run_baseline/run_plugin helpers.

**Tech Stack:** Python 3.13, subprocess (claude CLI), json, re, random (for judge position randomization)

---

### Task 1: Create the Judge Prompt

**Files:**
- Create: `scripts/judge_prompt.md`

**Step 1: Write the judge prompt file**

```markdown
You are a senior ML engineer reviewing two responses to the same ML/AI question.

## Task

You will see:
- **Question**: The user's original ML/AI question
- **Response A**: One answer to the question
- **Response B**: Another answer to the same question

Score EACH response independently on these 4 dimensions (0-3 scale):

### Correctness (0-3)
- 0: Wrong configs, parameters, or API calls that would cause failures
- 1: Mostly correct but has 1-2 meaningful errors
- 2: Correct with only minor gaps or omissions
- 3: Correct AND catches edge cases or subtle issues

### Specificity (0-3)
- 0: Generic advice with no framework-specific details
- 1: Mentions relevant frameworks but stays surface-level
- 2: Framework-specific configs, code, and API calls
- 3: Version-specific gotchas, exact parameter values with rationale

### Mistake Prevention (0-3)
- 0: No warnings about potential issues
- 1: Generic "be careful" or "watch out" without specifics
- 2: Flags specific pitfalls with concrete fixes
- 3: Catches non-obvious issues that the other response missed entirely

### Actionability (0-3)
- 0: Abstract discussion, nothing runnable
- 1: Some code snippets but incomplete
- 2: Runnable code/config with explanation
- 3: Copy-paste ready with validation/verification steps included

## Output Format

Return ONLY valid JSON with this exact structure:

```json
{
  "response_a": {
    "correctness": {"score": 0, "reasoning": "..."},
    "specificity": {"score": 0, "reasoning": "..."},
    "prevention": {"score": 0, "reasoning": "..."},
    "actionability": {"score": 0, "reasoning": "..."}
  },
  "response_b": {
    "correctness": {"score": 0, "reasoning": "..."},
    "specificity": {"score": 0, "reasoning": "..."},
    "prevention": {"score": 0, "reasoning": "..."},
    "actionability": {"score": 0, "reasoning": "..."}
  },
  "winner": "a|b|tie",
  "winner_reasoning": "2-3 sentences explaining which response you would rather use in production and why."
}
```

Score each response on its own merits. Do not let one response's quality inflate or deflate the other's score.
Keep each reasoning to 1-3 sentences focused on specific evidence from the response.
```

**Step 2: Verify file created**

Run: `cat scripts/judge_prompt.md | head -5`
Expected: First 5 lines of the prompt

**Step 3: Commit**

```bash
git add scripts/judge_prompt.md
git commit -m "feat: add judge prompt for A/B quality scoring"
```

---

### Task 2: Refactor test_interactive.py — Core Helpers

**Files:**
- Modify: `scripts/test_interactive.py`

This task replaces the single `run_test` function with separate `run_baseline`, `run_plugin`, and shared helpers. Keep the existing `TESTS` list and `assess_quality` function intact.

**Step 1: Add new imports and constants at the top**

After the existing imports, add:

```python
import random

BASELINE_MAX_TURNS = 15
PLUGIN_MAX_TURNS = 20
BASELINE_TIMEOUT = 600  # 10 min, no tools to wait for
PLUGIN_TIMEOUT = 900    # 15 min, tool calls take time
JUDGE_TIMEOUT = 120     # 2 min for judge eval
```

**Step 2: Write `run_claude` helper**

This is the low-level runner that both baseline and plugin modes use. Replace the existing `run_test` function body with a shared helper:

```python
def run_claude(prompt: str, test_id: str, mode: str, max_turns: int,
               timeout: int, use_plugin: bool, round_dir: Path) -> dict:
    """Run claude CLI and return parsed output.
    mode: 'baseline' or 'plugin' — used for file naming only.
    """
    env = os.environ.copy()
    env["CLAUDECODE"] = ""
    env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}:{env.get('PATH', '')}"

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", prompt,
        "--max-turns", str(max_turns),
        "--output-format", "json",
    ]

    if use_plugin:
        cmd.extend(["--plugin-dir", str(PLUGIN_DIR)])
        cmd.extend(["--allowedTools",
                     "mcp__leeroopedia__*,mcp__plugin_leeroopedia_leeroopedia__*,"
                     "Read,Write,Edit,Bash,Glob,Grep,Agent,Skill"])

    out_file = round_dir / f"{test_id}.{mode}.json"

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env, cwd=str(PLUGIN_DIR),
        )
        elapsed = time.time() - start

        output_data = None
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

        report = {
            "test_id": test_id,
            "mode": mode,
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 1),
            "raw_stdout": result.stdout,
        }

        if output_data and isinstance(output_data, dict):
            report["result"] = output_data.get("result", "")
            report["num_turns"] = output_data.get("num_turns", 0)
            report["cost_usd"] = round(output_data.get("total_cost_usd", 0), 3)
            report["usage"] = output_data.get("usage", {})

        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        report = {"test_id": test_id, "mode": mode, "error": "timeout",
                  "elapsed_sec": round(elapsed, 1)}
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as e:
        report = {"test_id": test_id, "mode": mode, "error": str(e)}
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
```

**Step 3: Verify syntax**

Run: `python3 -c "import scripts.test_interactive"`
Expected: No errors

**Step 4: Commit**

```bash
git add scripts/test_interactive.py
git commit -m "refactor: extract run_claude helper for A/B testing"
```

---

### Task 3: Add Judge Function

**Files:**
- Modify: `scripts/test_interactive.py`

**Step 1: Write `judge_responses` function**

```python
def judge_responses(question: str, baseline_result: str, plugin_result: str,
                    test_id: str, round_dir: Path) -> dict:
    """Call Claude as judge to compare baseline vs plugin responses."""
    judge_prompt_path = PLUGIN_DIR / "scripts" / "judge_prompt.md"
    judge_system = judge_prompt_path.read_text(encoding="utf-8")

    # Randomize position to avoid bias
    coin = random.random() > 0.5
    if coin:
        response_a, response_b = baseline_result, plugin_result
        mapping = {"a": "baseline", "b": "plugin"}
    else:
        response_a, response_b = plugin_result, baseline_result
        mapping = {"a": "plugin", "b": "baseline"}

    user_prompt = (
        f"## Question\n\n{question}\n\n"
        f"## Response A\n\n{response_a}\n\n"
        f"## Response B\n\n{response_b}"
    )

    env = os.environ.copy()
    env["CLAUDECODE"] = ""
    env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}:{env.get('PATH', '')}"

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", user_prompt,
        "--system-prompt", judge_system,
        "--max-turns", "1",
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=JUDGE_TIMEOUT, env=env, cwd=str(PLUGIN_DIR),
        )

        output_data = json.loads(result.stdout)
        judge_text = output_data.get("result", "")

        # Extract JSON from judge response (may have markdown wrapping)
        json_match = re.search(r'\{[\s\S]*\}', judge_text)
        if json_match:
            raw_scores = json.loads(json_match.group())
        else:
            return {"test_id": test_id, "error": "judge returned no JSON",
                    "raw": judge_text}

        # Unmap positions back to baseline/plugin
        scores = {}
        for role in ["baseline", "plugin"]:
            pos = [k for k, v in mapping.items() if v == role][0]
            key = f"response_{pos}"
            scores[role] = raw_scores.get(key, {})

        # Compute totals
        for role in ["baseline", "plugin"]:
            total = sum(
                dim.get("score", 0) if isinstance(dim, dict) else 0
                for dim in scores[role].values()
            )
            scores[f"{role}_score"] = total

        scores["value_add"] = scores["plugin_score"] - scores["baseline_score"]

        # Map winner back
        raw_winner = raw_scores.get("winner", "tie")
        if raw_winner in mapping:
            scores["winner"] = mapping[raw_winner]
        else:
            scores["winner"] = "tie"
        scores["winner_reasoning"] = raw_scores.get("winner_reasoning", "")
        scores["position_mapping"] = mapping
        scores["test_id"] = test_id
        scores["judge_cost_usd"] = round(output_data.get("total_cost_usd", 0), 3)

        out_file = round_dir / f"{test_id}.judge.json"
        out_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return scores

    except Exception as e:
        return {"test_id": test_id, "error": f"judge failed: {e}"}
```

**Step 2: Verify syntax**

Run: `python3 -c "import scripts.test_interactive"`
Expected: No errors

**Step 3: Commit**

```bash
git add scripts/test_interactive.py
git commit -m "feat: add judge_responses function for A/B evaluation"
```

---

### Task 4: Add Efficiency Parser

**Files:**
- Modify: `scripts/test_interactive.py`

**Step 1: Write `parse_efficiency` function**

This parses the raw plugin stdout to find KB tool calls and match them against citations in the final response.

```python
def parse_efficiency(plugin_report: dict, test_id: str,
                     round_dir: Path) -> dict:
    """Analyze KB call efficiency from plugin run."""
    result_text = plugin_report.get("result", "")
    raw_stdout = plugin_report.get("raw_stdout", "")

    # Find all citations in final response
    response_citations = set(re.findall(r'\[[\w]+/[\w_]+\]', result_text))

    # Parse tool calls from raw JSON output
    # The claude JSON output includes usage info but not individual tool calls.
    # We count KB calls from the num_turns and citation presence as proxy.
    # For detailed call-level analysis, we'd need --verbose or conversation log.

    # Proxy: count unique citation patterns as a lower bound on useful calls
    total_citations = len(response_citations)

    # Estimate calls from turns (each turn with KB tool = ~1-2 calls)
    num_turns = plugin_report.get("num_turns", 0)
    # Conservative estimate: half of turns involve a KB call
    estimated_kb_calls = max(num_turns // 2, 1)

    efficiency = {
        "test_id": test_id,
        "citations_in_response": total_citations,
        "unique_citations": list(response_citations)[:20],  # cap for readability
        "estimated_kb_calls": estimated_kb_calls,
        "efficiency_ratio": round(total_citations / max(estimated_kb_calls, 1), 2),
        "assessment": "",
    }

    if total_citations == 0:
        efficiency["assessment"] = "No citations in response — KB calls may not have contributed."
    elif efficiency["efficiency_ratio"] >= 1.0:
        efficiency["assessment"] = "Good efficiency — multiple citations per estimated KB call."
    elif efficiency["efficiency_ratio"] >= 0.5:
        efficiency["assessment"] = "Moderate efficiency — most calls contributed."
    else:
        efficiency["assessment"] = "Low efficiency — many calls may have been redundant."

    out_file = round_dir / f"{test_id}.efficiency.json"
    out_file.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    return efficiency
```

**Step 2: Verify syntax**

Run: `python3 -c "import scripts.test_interactive"`
Expected: No errors

**Step 3: Commit**

```bash
git add scripts/test_interactive.py
git commit -m "feat: add parse_efficiency for KB call analysis"
```

---

### Task 5: Add Refiner Function

**Files:**
- Modify: `scripts/test_interactive.py`

**Step 1: Write `run_refiner` function**

```python
def run_refiner(judge_results: list, pass_num: int,
                round_dir: Path) -> dict:
    """Analyze judge results, generate skill patches, apply them."""
    # Find weak tests: plugin_score < 10 or value_add <= 0
    weak = [j for j in judge_results
            if not j.get("error")
            and (j.get("plugin_score", 0) < 10 or j.get("value_add", 0) <= 0)]

    if not weak:
        return {"pass": pass_num, "patterns_found": 0,
                "message": "No weak tests to refine."}

    # Group weaknesses by skill
    by_skill = {}
    for j in weak:
        test_id = j["test_id"]
        test_def = next((t for t in TESTS if t["id"] == test_id), None)
        if not test_def:
            continue
        skill = test_def.get("expected_skill", "unknown")
        if skill not in by_skill:
            by_skill[skill] = []

        # Collect weak dimensions
        for role_key in ["plugin"]:
            dims = j.get(role_key, {})
            for dim_name, dim_data in dims.items():
                if isinstance(dim_data, dict) and dim_data.get("score", 3) < 2:
                    by_skill[skill].append({
                        "test_id": test_id,
                        "dimension": dim_name,
                        "score": dim_data.get("score", 0),
                        "reasoning": dim_data.get("reasoning", ""),
                    })

    if not any(by_skill.values()):
        return {"pass": pass_num, "patterns_found": 0,
                "message": "No actionable weakness patterns found."}

    # For each skill with weaknesses, generate a patch
    patches = []
    for skill_name, weaknesses in by_skill.items():
        if not weaknesses:
            continue

        skill_path = PLUGIN_DIR / "skills" / skill_name / "SKILL.md"
        if not skill_path.exists():
            # Try using-leeroopedia for general issues
            skill_path = PLUGIN_DIR / "skills" / "using-leeroopedia" / "SKILL.md"
        if not skill_path.exists():
            continue

        current_content = skill_path.read_text(encoding="utf-8")
        current_lines = len(current_content.splitlines())
        max_lines = int(current_lines * 1.2)  # 20% growth cap

        weakness_summary = "\n".join(
            f"- {w['test_id']}: {w['dimension']} scored {w['score']}/3 — {w['reasoning']}"
            for w in weaknesses
        )

        refiner_prompt = (
            f"You are refining an ML workflow skill to fix test failures.\n\n"
            f"## Current skill file ({skill_name}):\n\n"
            f"```\n{current_content}\n```\n\n"
            f"## Weaknesses found by judge:\n\n{weakness_summary}\n\n"
            f"## Constraints:\n"
            f"- Max {max_lines} total lines (currently {current_lines})\n"
            f"- Do NOT delete Iron Laws, phase gates, or anti-pattern tables\n"
            f"- Only ADD or MODIFY content to address the specific weaknesses\n"
            f"- Be surgical — fix the gap, don't rewrite the skill\n\n"
            f"## Output:\n"
            f"Return ONLY the complete updated skill file content, nothing else.\n"
            f"No markdown wrapping, no explanation — just the raw file content."
        )

        env = os.environ.copy()
        env["CLAUDECODE"] = ""
        env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}:{env.get('PATH', '')}"

        try:
            result = subprocess.run(
                ["claude", "--dangerously-skip-permissions",
                 "-p", refiner_prompt, "--max-turns", "1",
                 "--output-format", "json"],
                capture_output=True, text=True, timeout=120,
                env=env, cwd=str(PLUGIN_DIR),
            )
            output_data = json.loads(result.stdout)
            new_content = output_data.get("result", "")

            # Strip markdown code fences if present
            new_content = re.sub(r'^```\w*\n', '', new_content)
            new_content = re.sub(r'\n```$', '', new_content.rstrip())

            new_lines = len(new_content.splitlines())
            if new_lines > max_lines:
                patches.append({
                    "file": str(skill_path.relative_to(PLUGIN_DIR)),
                    "status": "skipped",
                    "reason": f"Patch too large: {new_lines} lines > {max_lines} max",
                    "weaknesses": weaknesses,
                })
                continue

            if not new_content.startswith("---"):
                patches.append({
                    "file": str(skill_path.relative_to(PLUGIN_DIR)),
                    "status": "skipped",
                    "reason": "Patch missing YAML frontmatter",
                    "weaknesses": weaknesses,
                })
                continue

            # Save backup and apply
            backup_path = round_dir / f"{skill_name}.backup.md"
            backup_path.write_text(current_content, encoding="utf-8")
            skill_path.write_text(new_content, encoding="utf-8")

            patches.append({
                "file": str(skill_path.relative_to(PLUGIN_DIR)),
                "status": "applied",
                "old_lines": current_lines,
                "new_lines": new_lines,
                "weaknesses": weaknesses,
                "backup": str(backup_path.relative_to(PLUGIN_DIR)),
            })

        except Exception as e:
            patches.append({
                "file": str(skill_path.relative_to(PLUGIN_DIR)),
                "status": "error",
                "reason": str(e),
                "weaknesses": weaknesses,
            })

    retest_ids = list(set(w["test_id"] for ws in by_skill.values() for w in ws))

    refiner_result = {
        "pass": pass_num,
        "patterns_found": sum(len(ws) for ws in by_skill.values()),
        "patches": patches,
        "retest_needed": retest_ids,
    }

    out_file = round_dir / f"refine_pass{pass_num}.json"
    out_file.write_text(json.dumps(refiner_result, indent=2), encoding="utf-8")
    return refiner_result
```

**Step 2: Verify syntax**

Run: `python3 -c "import scripts.test_interactive"`
Expected: No errors

**Step 3: Commit**

```bash
git add scripts/test_interactive.py
git commit -m "feat: add run_refiner for auto-patching skills"
```

---

### Task 6: Rewrite main() — Full Pipeline

**Files:**
- Modify: `scripts/test_interactive.py`

**Step 1: Replace `main()` with the full A/B + judge + refiner pipeline**

```python
def run_full_pipeline(test_ids: list[str] | None = None):
    """Run the full A/B + judge + refiner pipeline."""
    tests = [t for t in TESTS if t["id"] in test_ids] if test_ids else TESTS

    # Create round directory
    existing = list(LOG_DIR.glob("round*"))
    round_num = max((int(re.search(r'\d+', d.name).group())
                     for d in existing
                     if re.search(r'\d+', d.name)), default=5) + 1
    round_dir = LOG_DIR / f"round{round_num}"
    round_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ROUND {round_num} — A/B Quality Evaluation")
    print(f"{'='*70}")

    all_judge_results = []

    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]

        print(f"\n--- {test_id} [{test.get('category')}] ---")

        # Phase 1: Baseline
        print(f"  [1/4] Baseline run (no plugin)...")
        baseline = run_claude(
            prompt, test_id, "baseline",
            max_turns=BASELINE_MAX_TURNS,
            timeout=BASELINE_TIMEOUT,
            use_plugin=False, round_dir=round_dir,
        )
        b_chars = len(baseline.get("result", ""))
        print(f"        {b_chars} chars, {baseline.get('num_turns', '?')} turns, "
              f"${baseline.get('cost_usd', 0):.2f}")

        # Phase 2: Plugin
        print(f"  [2/4] Plugin run...")
        plugin = run_claude(
            prompt, test_id, "plugin",
            max_turns=PLUGIN_MAX_TURNS,
            timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=round_dir,
        )
        p_chars = len(plugin.get("result", ""))
        print(f"        {p_chars} chars, {plugin.get('num_turns', '?')} turns, "
              f"${plugin.get('cost_usd', 0):.2f}")

        # Phase 3: Judge
        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            print(f"  [3/4] Judge evaluation...")
            judge = judge_responses(
                prompt, baseline_text, plugin_text,
                test_id, round_dir,
            )
            if not judge.get("error"):
                print(f"        Baseline: {judge.get('baseline_score', '?')}/12 | "
                      f"Plugin: {judge.get('plugin_score', '?')}/12 | "
                      f"Value-add: {judge.get('value_add', '?'):+d} | "
                      f"Winner: {judge.get('winner', '?')}")
                print(f"        Reason: {judge.get('winner_reasoning', '')[:120]}")
            else:
                print(f"        Judge error: {judge.get('error')}")
        else:
            judge = {"test_id": test_id, "error": "missing baseline or plugin result"}
            print(f"  [3/4] Judge skipped — missing response")

        # Phase 4: Efficiency
        if plugin_text:
            print(f"  [4/4] Efficiency analysis...")
            eff = parse_efficiency(plugin, test_id, round_dir)
            print(f"        {eff.get('citations_in_response', 0)} citations, "
                  f"efficiency={eff.get('efficiency_ratio', 0):.2f}, "
                  f"{eff.get('assessment', '')}")
            judge["efficiency"] = eff.get("efficiency_ratio", 0)
        else:
            print(f"  [4/4] Efficiency skipped — no plugin result")

        all_judge_results.append(judge)

    # Print summary
    print_summary(all_judge_results, round_dir)

    # Refiner pass 1
    print(f"\n{'='*70}")
    print("REFINER PASS 1")
    print(f"{'='*70}")
    refine1 = run_refiner(all_judge_results, 1, round_dir)
    print(f"  Patterns: {refine1.get('patterns_found', 0)} | "
          f"Patches: {len(refine1.get('patches', []))} | "
          f"Retest: {refine1.get('retest_needed', [])}")

    if refine1.get("retest_needed"):
        print("  Re-running weak tests...")
        retest_dir = round_dir / "refine_pass1_retest"
        retest_dir.mkdir(exist_ok=True)
        retest1_judges = retest_weak(refine1["retest_needed"], retest_dir)
        # Check for regressions
        check_regressions(all_judge_results, retest1_judges, refine1, round_dir)
    else:
        retest1_judges = []

    # Refiner pass 2
    print(f"\n{'='*70}")
    print("REFINER PASS 2")
    print(f"{'='*70}")
    combined = [r for r in retest1_judges if not r.get("error")] or all_judge_results
    refine2 = run_refiner(combined, 2, round_dir)
    print(f"  Patterns: {refine2.get('patterns_found', 0)} | "
          f"Patches: {len(refine2.get('patches', []))} | "
          f"Retest: {refine2.get('retest_needed', [])}")

    if refine2.get("retest_needed"):
        print("  Re-running weak tests...")
        retest_dir = round_dir / "refine_pass2_retest"
        retest_dir.mkdir(exist_ok=True)
        retest2_judges = retest_weak(refine2["retest_needed"], retest_dir)
        check_regressions(
            retest1_judges or all_judge_results, retest2_judges, refine2, round_dir)

    # Final summary
    save_round_summary(all_judge_results, refine1, refine2, round_dir, round_num)


def retest_weak(test_ids: list[str], retest_dir: Path) -> list[dict]:
    """Re-run only the weak tests after a refiner pass."""
    tests = [t for t in TESTS if t["id"] in test_ids]
    results = []
    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]
        print(f"  Retesting {test_id}...")

        plugin = run_claude(
            prompt, test_id, "plugin",
            max_turns=PLUGIN_MAX_TURNS, timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=retest_dir,
        )

        # We need a baseline for judge — read from parent round dir
        parent_dir = retest_dir.parent
        baseline_file = parent_dir / f"{test_id}.baseline.json"
        if baseline_file.exists():
            baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        else:
            baseline = {"result": ""}

        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            judge = judge_responses(
                prompt, baseline_text, plugin_text,
                test_id, retest_dir,
            )
            print(f"    Plugin: {judge.get('plugin_score', '?')}/12 | "
                  f"Value-add: {judge.get('value_add', '?'):+d}")
        else:
            judge = {"test_id": test_id, "error": "missing result"}

        results.append(judge)
    return results


def check_regressions(old_judges: list, new_judges: list,
                      refine_result: dict, round_dir: Path):
    """If any test regressed, revert the patch."""
    old_by_id = {j["test_id"]: j for j in old_judges if not j.get("error")}
    for new_j in new_judges:
        tid = new_j.get("test_id")
        if new_j.get("error") or tid not in old_by_id:
            continue
        old_score = old_by_id[tid].get("plugin_score", 0)
        new_score = new_j.get("plugin_score", 0)
        if new_score < old_score:
            print(f"  REGRESSION on {tid}: {old_score} -> {new_score}. Reverting patch.")
            # Restore from backup
            for patch in refine_result.get("patches", []):
                backup = patch.get("backup")
                if backup and patch.get("status") == "applied":
                    backup_path = PLUGIN_DIR / backup
                    target_path = PLUGIN_DIR / patch["file"]
                    if backup_path.exists():
                        target_path.write_text(
                            backup_path.read_text(encoding="utf-8"),
                            encoding="utf-8")
                        print(f"    Reverted {patch['file']}")


def print_summary(judge_results: list, round_dir: Path):
    """Print the A/B comparison summary table."""
    print(f"\n{'='*90}")
    print("A/B COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Test':<30} {'Base':>5} {'Plugin':>7} {'Delta':>6} {'Winner':<8} {'Eff':>5}")
    print("-" * 90)

    total_base = 0
    total_plugin = 0
    count = 0

    for j in judge_results:
        if j.get("error"):
            print(f"{j.get('test_id', '?'):<30} {'ERROR':>5}")
            continue
        b = j.get("baseline_score", 0)
        p = j.get("plugin_score", 0)
        d = j.get("value_add", 0)
        w = j.get("winner", "?")
        e = j.get("efficiency", 0)
        print(f"{j['test_id']:<30} {b:>4}/12 {p:>5}/12 {d:>+5} {w:<8} {e:>5.2f}")
        total_base += b
        total_plugin += p
        count += 1

    if count:
        print("-" * 90)
        print(f"{'AVERAGE':<30} {total_base/count:>5.1f} {total_plugin/count:>7.1f} "
              f"{(total_plugin-total_base)/count:>+5.1f}")


def save_round_summary(judge_results: list, refine1: dict, refine2: dict,
                       round_dir: Path, round_num: int):
    """Save the complete round summary."""
    summary = {
        "round": round_num,
        "tests": len(judge_results),
        "judge_results": judge_results,
        "refine_pass1": refine1,
        "refine_pass2": refine2,
    }
    out_file = round_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRound {round_num} results saved to {round_dir}/")


def main():
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]
        # Check for valid test IDs
        valid = [t["id"] for t in TESTS]
        invalid = [t for t in test_ids if t not in valid]
        if invalid:
            print(f"Unknown test IDs: {invalid}")
            print(f"Available: {valid}")
            sys.exit(1)
        run_full_pipeline(test_ids)
    else:
        run_full_pipeline()
```

**Step 2: Verify syntax**

Run: `python3 -c "import scripts.test_interactive; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/test_interactive.py
git commit -m "feat: full A/B + judge + refiner pipeline in main()"
```

---

### Task 7: Smoke Test — Single Test End-to-End

**Files:** None (test run only)

**Step 1: Run a single test through the full pipeline**

Run:
```bash
LEEROOPEDIA_API_KEY=kpsk_46eb0b96_46eb0b961b639556a16121962660b126 \
  python3 scripts/test_interactive.py t12_preflight_check
```

Expected:
- Creates `test-logs/round6/` directory
- Creates `t12_preflight_check.baseline.json`
- Creates `t12_preflight_check.plugin.json`
- Creates `t12_preflight_check.judge.json`
- Creates `t12_preflight_check.efficiency.json`
- Prints A/B comparison with scores
- Refiner may or may not find patterns (1 test is fine)

**Step 2: Inspect judge output**

Run: `cat test-logs/round6/t12_preflight_check.judge.json | python3 -m json.tool`
Expected: JSON with baseline_score, plugin_score, value_add, winner, and reasoning per dimension

**Step 3: If errors, fix and re-run. If passes, commit.**

```bash
git add scripts/test_interactive.py
git commit -m "test: smoke test passes for A/B pipeline"
```

---

### Task 8: Full Run — All 12 Tests

**Files:** None (test run only)

**Step 1: Run full pipeline**

Run:
```bash
LEEROOPEDIA_API_KEY=kpsk_46eb0b96_46eb0b961b639556a16121962660b126 \
  python3 scripts/test_interactive.py
```

Expected: ~60-90 min runtime. Creates full round6 directory with all results.

**Step 2: Review summary**

Run: `cat test-logs/round6/summary.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Tests: {d[\"tests\"]}'); [print(f'  {j[\"test_id\"]}: base={j.get(\"baseline_score\",\"?\")}/12 plugin={j.get(\"plugin_score\",\"?\")}/12 delta={j.get(\"value_add\",\"?\")}') for j in d['judge_results'] if not j.get('error')]"`

**Step 3: Review refiner patches**

Run: `cat test-logs/round6/refine_pass1.json | python3 -m json.tool`

Check: Did patches get applied? Did retests improve?

**Step 4: Commit round results**

```bash
git add test-logs/round6/
git commit -m "results: round 6 A/B evaluation with judge + refiner"
```

---

### Task 9: Iterate on Failures

This task is conditional — only needed if the smoke test or full run reveals bugs.

**Common issues to watch for:**

1. **Judge returns non-JSON**: Add more robust extraction in `judge_responses` — try stripping markdown fences, handle partial JSON.
2. **Baseline times out**: Increase `BASELINE_TIMEOUT` or reduce `BASELINE_MAX_TURNS`.
3. **Refiner bloats skill files**: Tighten the `max_lines` constraint or make the refiner prompt more specific.
4. **Regressions after refinement**: The `check_regressions` function should catch these — verify reverts work.
5. **Python 3.13 regex issues**: All patterns should use `re.IGNORECASE` parameter, not `(?i)` inline flags.

For each issue: fix, re-run the failing test only, verify, commit.
