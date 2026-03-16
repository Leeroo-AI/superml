#!/usr/bin/env python3
"""
Self-refine test harness.
Runs A/B quality evaluation: baseline vs plugin, then auto-refines skills.

Usage:
    python self-refine/run.py                                     # run default suite
    python self-refine/run.py --suite suites/biomed.yaml          # custom suite
    python self-refine/run.py --suite suites/biomed.yaml --no-kb  # web mode
    python self-refine/run.py t01_agent_orchestration t02_qlora   # specific tests
"""

import argparse
import json
import os
import random
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
PLUGIN_DIR = SELF_REFINE_DIR.parent
LOG_DIR = SELF_REFINE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Global flag: when True, strip LEEROOPEDIA_API_KEY from plugin runs
NO_KB = False

BASELINE_MAX_TURNS = 15
PLUGIN_MAX_TURNS = 20
BASELINE_TIMEOUT = 600   # 10 min
PLUGIN_TIMEOUT = 900     # 15 min
JUDGE_TIMEOUT = 300      # 5 min
N_JUDGES = 3


# ---------------------------------------------------------------------------
# Suite loader
# ---------------------------------------------------------------------------


def load_suite(suite_path: Path) -> list[dict]:
    """Load test suite from YAML file."""
    with open(suite_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tests = data.get("tests", [])
    valid = []
    for t in tests:
        if "id" not in t or "prompt" not in t:
            print(f"WARNING: test missing id or prompt, skipping: {t}")
            continue
        t.setdefault("category", "uncategorized")
        t.setdefault("needs_gpu", False)
        t.setdefault("expected_skill", None)
        t.setdefault("expected_agent", None)
        valid.append(t)

    return valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


INLINE_SUFFIX = (
    "\n\nIMPORTANT: Present your COMPLETE solution directly in your response "
    "text. Include all code, configs, and commands inline. Do NOT create or "
    "write to separate files."
)


def _make_env() -> dict:
    """Build a clean env for Claude CLI calls."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["PATH"] = (
        f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}"
        f":{env.get('PATH', '')}"
    )
    return env


def run_claude(prompt: str, test_id: str, mode: str, max_turns: int,
               timeout: int, use_plugin: bool, round_dir: Path,
               agent: str | None = None) -> dict:
    """Run claude CLI and return parsed output."""
    full_prompt = prompt + INLINE_SUFFIX

    env = _make_env()
    if NO_KB and use_plugin:
        env.pop("LEEROOPEDIA_API_KEY", None)

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", full_prompt,
        "--max-turns", str(max_turns),
        "--output-format", "json",
    ]

    if agent:
        cmd.extend(["--agent", agent])

    if use_plugin:
        cmd.extend(["--plugin-dir", str(PLUGIN_DIR)])
        cmd.extend([
            "--allowedTools",
            "mcp__leeroopedia__*,mcp__plugin_leeroopedia_leeroopedia__*,"
            "Read,Glob,Grep,Agent,Skill",
        ])

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
        }

        if output_data and isinstance(output_data, dict):
            report["result"] = output_data.get("result", "")
            report["num_turns"] = output_data.get("num_turns", 0)
            report["cost_usd"] = round(output_data.get("total_cost_usd", 0), 3)

        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        report = {
            "test_id": test_id, "mode": mode, "error": "timeout",
            "elapsed_sec": round(elapsed, 1),
        }
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as e:
        report = {"test_id": test_id, "mode": mode, "error": str(e)}
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------


def get_or_run_baseline(test: dict, round_dir: Path) -> dict:
    """Reuse baseline from a previous round if available, else run fresh."""
    test_id = test["id"]
    prompt = test["prompt"]

    for prev_dir in sorted(LOG_DIR.glob("round*"), reverse=True):
        if prev_dir == round_dir:
            continue
        baseline_file = prev_dir / f"{test_id}.baseline.json"
        if baseline_file.exists():
            try:
                cached = json.loads(
                    baseline_file.read_text(encoding="utf-8"))
                if cached.get("result") and not cached.get("error"):
                    dest = round_dir / f"{test_id}.baseline.json"
                    dest.write_text(
                        baseline_file.read_text(encoding="utf-8"),
                        encoding="utf-8")
                    cached["cached_from"] = str(prev_dir.name)
                    return cached
            except (json.JSONDecodeError, OSError):
                continue

    return run_claude(
        prompt, test_id, "baseline",
        max_turns=BASELINE_MAX_TURNS,
        timeout=BASELINE_TIMEOUT,
        use_plugin=False, round_dir=round_dir,
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_responses(question: str, baseline_result: str, plugin_result: str,
                    test_id: str, round_dir: Path,
                    needs_gpu: bool = False) -> dict:
    """Call Claude as judge to compare baseline vs plugin responses."""
    judge_prompt_path = SELF_REFINE_DIR / "judge_prompt.md"
    judge_system = judge_prompt_path.read_text(encoding="utf-8")

    coin = random.random() > 0.5
    if coin:
        response_a, response_b = baseline_result, plugin_result
        mapping = {"a": "baseline", "b": "plugin"}
    else:
        response_a, response_b = plugin_result, baseline_result
        mapping = {"a": "plugin", "b": "baseline"}

    gpu_note = (
        "\n\n**Note:** This question requires GPU hardware that was not "
        "available during generation. Score actionability based on whether "
        "the code/config would work given the specified hardware, not "
        "whether it was actually executed.\n"
        if needs_gpu else ""
    )

    full_prompt = (
        f"{judge_system}\n\n---\n\n"
        f"## Question\n\n{question}{gpu_note}\n\n"
        f"## Response A\n\n{response_a}\n\n"
        f"## Response B\n\n{response_b}"
    )

    env = _make_env()

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", full_prompt,
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

        json_match = re.search(r'\{[\s\S]*\}', judge_text)
        if not json_match:
            err = {"test_id": test_id, "error": "judge returned no JSON",
                   "raw": judge_text[:500],
                   "stderr": result.stderr[:500] if result.stderr else "",
                   "exit_code": result.returncode}
            err_file = round_dir / f"{test_id}.judge_error.json"
            err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
            return err

        raw_scores = json.loads(json_match.group())

        scores: dict = {}
        for role in ("baseline", "plugin"):
            pos = [k for k, v in mapping.items() if v == role][0]
            key = f"response_{pos}"
            scores[role] = raw_scores.get(key, {})

        for role in ("baseline", "plugin"):
            total = sum(
                dim.get("score", 0) if isinstance(dim, dict) else 0
                for dim in scores[role].values()
            )
            scores[f"{role}_score"] = total

        scores["value_add"] = scores["plugin_score"] - scores["baseline_score"]

        raw_winner = raw_scores.get("winner", "tie")
        if raw_winner in mapping:
            scores["winner"] = mapping[raw_winner]
        else:
            scores["winner"] = raw_winner
        scores["winner_reasoning"] = raw_scores.get("winner_reasoning", "")
        scores["position_mapping"] = mapping
        scores["test_id"] = test_id
        scores["judge_cost_usd"] = round(
            output_data.get("total_cost_usd", 0), 3)

        out_file = round_dir / f"{test_id}.judge.json"
        out_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return scores

    except json.JSONDecodeError as e:
        err = {"test_id": test_id, "error": f"judge JSON parse error: {e}",
               "stdout": result.stdout[:500] if result.stdout else ""}
        err_file = round_dir / f"{test_id}.judge_error.json"
        err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
        return err
    except subprocess.TimeoutExpired:
        err = {"test_id": test_id, "error": "judge timeout"}
        err_file = round_dir / f"{test_id}.judge_error.json"
        err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
        return err
    except Exception as e:
        return {"test_id": test_id, "error": f"judge failed: {e}"}


# ---------------------------------------------------------------------------
# Multi-judge consistency
# ---------------------------------------------------------------------------


def _median(values: list[int | float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _median_scores(all_scores: list[dict], test_id: str) -> dict:
    result: dict = {"test_id": test_id}
    for role in ("baseline", "plugin"):
        dims: dict[str, list] = {}
        for scores in all_scores:
            role_data = scores.get(role, {})
            for dim_name, dim_data in role_data.items():
                if isinstance(dim_data, dict) and "score" in dim_data:
                    dims.setdefault(dim_name, []).append(dim_data["score"])
        role_result: dict = {}
        for dim_name, score_list in dims.items():
            med = _median(score_list)
            best_reasoning = ""
            for scores in all_scores:
                rd = scores.get(role, {}).get(dim_name, {})
                if isinstance(rd, dict) and rd.get("score", -1) == med:
                    best_reasoning = rd.get("reasoning", "")
                    break
            if not best_reasoning and all_scores:
                rd = all_scores[0].get(role, {}).get(dim_name, {})
                best_reasoning = rd.get("reasoning", "") if isinstance(
                    rd, dict) else ""
            role_result[dim_name] = {
                "score": med,
                "reasoning": best_reasoning,
            }
        result[role] = role_result
        result[f"{role}_score"] = sum(
            d["score"] for d in role_result.values()
        )

    result["value_add"] = (
        result.get("plugin_score", 0) - result.get("baseline_score", 0))

    winners = [s.get("winner", "tie") for s in all_scores]
    winner_counts: dict[str, int] = {}
    for w in winners:
        winner_counts[w] = winner_counts.get(w, 0) + 1
    result["winner"] = max(winner_counts, key=winner_counts.get)  # type: ignore[arg-type]
    result["winner_reasoning"] = next(
        (s.get("winner_reasoning", "") for s in all_scores
         if s.get("winner") == result["winner"]),
        "",
    )
    result["judge_runs"] = len(all_scores)
    result["judge_cost_usd"] = round(
        sum(s.get("judge_cost_usd", 0) for s in all_scores), 3)

    return result


def judge_responses_robust(question: str, baseline: str, plugin: str,
                           test_id: str, round_dir: Path,
                           needs_gpu: bool = False,
                           n_judges: int = N_JUDGES) -> dict:
    """Run judge N times and take median scores for consistency."""
    all_scores = []
    for i in range(n_judges):
        scores = judge_responses(
            question, baseline, plugin,
            f"{test_id}_j{i}", round_dir, needs_gpu,
        )
        if not scores.get("error"):
            all_scores.append(scores)

    if not all_scores:
        return {"test_id": test_id, "error": "all judges failed"}

    if len(all_scores) == 1:
        all_scores[0]["test_id"] = test_id
        all_scores[0]["judge_runs"] = 1
        return all_scores[0]

    result = _median_scores(all_scores, test_id)

    out_file = round_dir / f"{test_id}.judge.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Skill-trigger verification
# ---------------------------------------------------------------------------


def check_skill_triggered(plugin_report: dict,
                          expected_skill: str | None) -> bool | None:
    if expected_skill is None:
        return None

    result = plugin_report.get("result", "")
    if not result:
        return False

    result_lower = result.lower()
    markers = [
        f"/{expected_skill}",
        f"skill: \"{expected_skill}\"",
        f"using {expected_skill}",
        f"**{expected_skill}**",
        expected_skill.replace("-", " "),
    ]

    return any(marker.lower() in result_lower for marker in markers)


def check_agent_invoked(plugin_report: dict,
                        expected_agent: str) -> bool:
    result = plugin_report.get("result", "")
    if not result:
        return False

    result_lower = result.lower()
    agent_lower = expected_agent.lower()
    markers = [
        f"subagent_type=\"{expected_agent}\"",
        f'subagent_type="{expected_agent}"',
        f"agent: {agent_lower}",
        f"agent={agent_lower}",
        f"--agent {agent_lower}",
        "delegat",
        "ml expert",
        "ml-expert",
    ]

    return any(m in result_lower for m in markers)


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def parse_efficiency(plugin_report: dict, test_id: str,
                     round_dir: Path) -> dict:
    result_text = plugin_report.get("result", "")

    response_citations = set(re.findall(r'\[[\w]+/[\w_]+\]', result_text))
    total_citations = len(response_citations)

    num_turns = plugin_report.get("num_turns", 0)
    kb_calls = 0

    messages = plugin_report.get("messages", [])
    if messages:
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    if "leeroopedia" in tc.get("name", "").lower():
                        kb_calls += 1

    if kb_calls == 0:
        kb_call_patterns = re.findall(
            r'(?:mcp__(?:plugin_)?leeroopedia__\w+|'
            r'leeroopedia__\w+)',
            result_text, re.IGNORECASE)
        kb_calls = len(set(kb_call_patterns))

    actual_kb_calls = kb_calls if kb_calls > 0 else max(num_turns // 2, 1)

    efficiency = {
        "test_id": test_id,
        "citations_in_response": total_citations,
        "unique_citations": sorted(response_citations)[:20],
        "actual_kb_calls": actual_kb_calls,
        "kb_calls_method": "parsed" if kb_calls > 0 else "estimated",
        "efficiency_ratio": round(
            total_citations / max(actual_kb_calls, 1), 2),
    }

    if total_citations == 0:
        efficiency["assessment"] = (
            "No citations in response — KB may not have contributed.")
    elif efficiency["efficiency_ratio"] >= 1.0:
        efficiency["assessment"] = (
            "Good — multiple citations per KB call.")
    elif efficiency["efficiency_ratio"] >= 0.5:
        efficiency["assessment"] = (
            "Moderate — most calls contributed.")
    else:
        efficiency["assessment"] = (
            "Low — many calls may have been redundant.")

    out_file = round_dir / f"{test_id}.efficiency.json"
    out_file.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    return efficiency


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------


def parse_sections(content: str) -> list[dict]:
    lines = content.splitlines()
    sections = []
    current: dict | None = None

    for i, line in enumerate(lines):
        if re.match(r'^#{1,3}\s', line):
            if current:
                current["end"] = i
                sections.append(current)
            current = {"name": line.strip("# ").strip(),
                       "header": line, "start": i, "end": len(lines)}
        elif i == 0 and line.startswith("---"):
            current = {"name": "frontmatter",
                       "header": "---", "start": 0, "end": len(lines)}

    if current:
        current["end"] = len(lines)
        sections.append(current)

    return sections


def run_refiner(judge_results: list, pass_num: int,
                round_dir: Path, tests: list[dict]) -> dict:
    """Analyze judge results, generate targeted skill edits."""
    weak = [
        j for j in judge_results
        if not j.get("error")
        and (j.get("plugin_score", 0) < 15 or j.get("value_add", 0) <= 0)
    ]

    if not weak:
        return {"pass": pass_num, "patterns_found": 0,
                "edits": [],
                "message": "All tests at 15/15 — nothing to refine."}

    by_skill: dict[str, list] = {}
    target_is_agent: dict[str, str] = {}

    for j in weak:
        test_id = j["test_id"]
        test_def = next((t for t in tests if t["id"] == test_id), None)
        if not test_def:
            continue

        if test_def.get("category") == "agent-direct" and test_def.get("expected_agent"):
            target = f"agent:{test_def['expected_agent']}"
            target_is_agent[target] = test_def["expected_agent"]
        elif test_def.get("category") == "agent-delegation":
            target = test_def.get("expected_skill") or "unknown"
            if not j.get("agent_delegated", True):
                agent_target = f"agent:{test_def['expected_agent']}"
                target_is_agent[agent_target] = test_def["expected_agent"]
                if agent_target not in by_skill:
                    by_skill[agent_target] = []
        else:
            target = test_def.get("expected_skill") or "unknown"

        if target not in by_skill:
            by_skill[target] = []

        plugin_file = round_dir / f"{test_id}.plugin.json"
        if not plugin_file.exists():
            plugin_file = round_dir / f"{test_id}.agent.json"
        if plugin_file.exists():
            try:
                plugin_data = json.loads(
                    plugin_file.read_text(encoding="utf-8"))
                response_snippet = plugin_data.get("result", "")[:2000]
            except (json.JSONDecodeError, OSError):
                response_snippet = ""
        else:
            response_snippet = ""

        plugin_dims = j.get("plugin", {})
        for dim_name, dim_data in plugin_dims.items():
            if isinstance(dim_data, dict) and dim_data.get("score", 3) < 3:
                by_skill[target].append({
                    "test_id": test_id,
                    "dimension": dim_name,
                    "score": dim_data.get("score", 0),
                    "reasoning": dim_data.get("reasoning", ""),
                    "response_snippet": response_snippet,
                })

        if (test_def.get("category") == "agent-delegation"
                and not j.get("agent_delegated", True)
                and test_def.get("expected_agent")):
            agent_target = f"agent:{test_def['expected_agent']}"
            by_skill[agent_target].append({
                "test_id": test_id,
                "dimension": "delegation",
                "score": 0,
                "reasoning": (
                    f"Main Claude did not delegate to "
                    f"{test_def['expected_agent']} agent. The agent's "
                    f"description may need to better match this task type."),
                "response_snippet": response_snippet,
            })

    if not any(by_skill.values()):
        return {"pass": pass_num, "patterns_found": 0,
                "edits": [],
                "message": "No actionable weakness patterns found."}

    dim_counts: dict[str, int] = {}
    for skill_name_tmp, weaknesses_tmp in by_skill.items():
        for w in weaknesses_tmp:
            dim = w["dimension"]
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
    systemic_patterns = {
        dim: count for dim, count in dim_counts.items() if count >= 3
    }
    systemic_note = ""
    if systemic_patterns:
        systemic_note = (
            "## SYSTEMIC PATTERNS (priority — these recur across tests):\n"
            + "\n".join(
                f"- {dim}: weak in {count} tests — this is likely a "
                f"structural issue in the skill, not a one-off gap"
                for dim, count in sorted(
                    systemic_patterns.items(), key=lambda x: -x[1])
            )
            + "\n\n"
        )

    all_edits = []
    modified_skills: set[str] = set()

    for skill_name, weaknesses in by_skill.items():
        if not weaknesses:
            continue

        if skill_name in target_is_agent:
            agent_name = target_is_agent[skill_name]
            skill_path = PLUGIN_DIR / "agents" / f"{agent_name}.md"
            if not skill_path.exists():
                continue
        else:
            skill_path = PLUGIN_DIR / "skills" / skill_name / "SKILL.md"
            if not skill_path.exists():
                skill_path = (
                    PLUGIN_DIR / "skills" / "using-leeroopedia" / "SKILL.md")
            if not skill_path.exists():
                continue

        current_content = skill_path.read_text(encoding="utf-8")
        sections = parse_sections(current_content)

        section_list = "\n".join(
            f"  - [{s['start']+1}-{s['end']}] {s['name']}"
            for s in sections
        )

        MAX_WEAKNESSES_PER_BATCH = 6
        batches = [weaknesses[i:i + MAX_WEAKNESSES_PER_BATCH]
                    for i in range(0, len(weaknesses), MAX_WEAKNESSES_PER_BATCH)]

        skill_applied_all = []
        skill_failed_all = []
        skill_skipped_all = []
        modified = current_content
        backup_path = None

        for batch_idx, batch in enumerate(batches):
            weakness_summary = "\n".join(
                f"- {w['test_id']}: {w['dimension']} scored {w['score']}/3 "
                f"— {w['reasoning']}"
                for w in batch
            )

            seen_snippets: set[str] = set()
            snippet_block = ""
            for w in batch:
                snip = w.get("response_snippet", "")
                if snip and w["test_id"] not in seen_snippets:
                    seen_snippets.add(w["test_id"])
                    snippet_block += (
                        f"\n### Plugin output for {w['test_id']} "
                        f"(first 2000 chars):\n{snip}\n"
                    )

            is_agent_file = skill_name in target_is_agent
            file_type = "agent definition" if is_agent_file else "skill"
            file_label = (
                target_is_agent[skill_name] if is_agent_file
                else skill_name
            )

            refiner_prompt = (
                f"You are improving an ML workflow {file_type} file based "
                f"on test feedback.\n\n"
                + (f"This is an AGENT file (agents/{file_label}.md) — it "
                   f"defines the system prompt and behavior for a subagent "
                   f"that handles complex ML tasks. Edits should improve "
                   f"the agent's instructions, KB usage patterns, and "
                   f"response quality directives.\n\n"
                   if is_agent_file else "")
                + f"## {'Agent' if is_agent_file else 'Skill'} file: "
                f"{file_label}\n\n"
                f"```\n{modified}\n```\n\n"
                f"## Sections in this file:\n{section_list}\n\n"
                f"## Weaknesses found by judge:\n\n{weakness_summary}\n\n"
                + (f"## Actual plugin responses:\n{snippet_block}\n\n"
                   if snippet_block else "")
                + (systemic_note if systemic_note else "")
                + f"## Your task:\n"
                f"For each weakness, decide:\n"
                f"1. Is this fixable by editing the skill instructions? "
                f"(e.g., adding a warning to Anti-Patterns, adding a step "
                f"to a Phase, adding a tool call reminder)\n"
                f"2. Or is this a general code quality issue that skill "
                f"instructions can't fix? (e.g., syntax errors in generated "
                f"code, formatting issues) — if so, skip it.\n\n"
                f"For each fixable weakness, produce a targeted edit.\n\n"
                f"3. Is any existing content in the skill file redundant, "
                f"outdated, or never triggered by tests? If so, propose a "
                f"'delete' action to remove it.\n\n"
                f"## Output format — return ONLY valid JSON:\n"
                f"```json\n"
                f'{{"edits": [\n'
                f'  {{\n'
                f'    "section": "section name",\n'
                f'    "action": "add_after | replace | delete",\n'
                f'    "find": "exact line(s) to find in the file",\n'
                f'    "content": "new line(s) to add after find / '
                f'replace find with (ignored for delete)",\n'
                f'    "reason": "what this fixes"\n'
                f'  }}\n'
                f'],\n'
                f'"skipped": [\n'
                f'  {{"weakness": "...", '
                f'"reason": "not skill-fixable because..."}}\n'
                f']\n'
                f'}}\n'
                f"```\n\n"
                f"Rules:\n"
                f"- `find` must be an EXACT substring from the current "
                f"file\n"
                f"- `add_after`: inserts `content` on the line after "
                f"`find`\n"
                f"- `replace`: replaces `find` with `content`\n"
                f"- `delete`: removes `find` entirely\n"
                f"- Keep edits small — 1-3 lines each\n"
                f"- Do NOT touch the YAML frontmatter (---)\n"
                f"- Do NOT delete Iron Laws or phase structure"
            )

            env = _make_env()

            try:
                result = subprocess.run(
                    ["claude", "--dangerously-skip-permissions",
                     "-p", refiner_prompt, "--max-turns", "1",
                     "--output-format", "json"],
                    capture_output=True, text=True, timeout=180,
                    env=env, cwd=str(PLUGIN_DIR),
                )
                output_data = json.loads(result.stdout)
                refiner_text = output_data.get("result", "")

                json_match = re.search(r'\{[\s\S]*\}', refiner_text)
                if not json_match:
                    if batch_idx == 0 and len(batches) == 1:
                        all_edits.append({
                            "file": str(
                                skill_path.relative_to(PLUGIN_DIR)),
                            "status": "error",
                            "reason": "refiner returned no JSON",
                            "raw": refiner_text[:300],
                            "weaknesses": weaknesses,
                        })
                    continue

                edit_plan = json.loads(json_match.group())
                edits = edit_plan.get("edits", [])
                skipped = edit_plan.get("skipped", [])
                skill_skipped_all.extend(skipped)

                for edit in edits:
                    find_str = edit.get("find", "")
                    content_str = edit.get("content", "")
                    action = edit.get("action", "add_after")
                    reason = edit.get("reason", "")

                    if not find_str or find_str not in modified:
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": "find string not found in file",
                        })
                        continue

                    if action == "delete":
                        new_modified = modified.replace(find_str, "", 1)
                    elif action == "replace":
                        new_modified = modified.replace(
                            find_str, content_str, 1)
                    elif action == "add_after":
                        new_modified = modified.replace(
                            find_str, find_str + "\n" + content_str, 1)
                    else:
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": f"unknown action: {action}",
                        })
                        continue

                    new_lines = len(new_modified.splitlines())
                    orig_lines = len(current_content.splitlines())
                    if new_lines > orig_lines * 1.2:
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": "would exceed 20% growth limit",
                        })
                        continue

                    modified = new_modified
                    skill_applied_all.append({
                        "action": action,
                        "section": edit.get("section", ""),
                        "reason": reason,
                        "lines_added": (
                            len(modified.splitlines())
                            - len(current_content.splitlines())),
                    })

            except Exception as e:
                if batch_idx == 0 and len(batches) == 1:
                    all_edits.append({
                        "file": str(
                            skill_path.relative_to(PLUGIN_DIR)),
                        "status": "error",
                        "reason": str(e),
                        "weaknesses": weaknesses,
                    })

        if skill_applied_all:
            safe_name = skill_name.replace(":", "_")
            backup_path = (
                round_dir / f"{safe_name}.pass{pass_num}.backup.md")
            backup_path.write_text(current_content, encoding="utf-8")
            skill_path.write_text(modified, encoding="utf-8")
            modified_skills.add(skill_name)

        all_edits.append({
            "file": str(skill_path.relative_to(PLUGIN_DIR)),
            "status": ("applied" if skill_applied_all
                        else "no_valid_edits"),
            "applied": skill_applied_all,
            "failed": skill_failed_all,
            "skipped": skill_skipped_all,
            "backup": str(backup_path) if skill_applied_all else None,
            "weaknesses": weaknesses,
        })

    retest_ids = list({
        w["test_id"] for skill_name, ws in by_skill.items()
        for w in ws
        if skill_name in modified_skills
    })

    refiner_result = {
        "pass": pass_num,
        "patterns_found": sum(len(ws) for ws in by_skill.values()),
        "edits": all_edits,
        "retest_needed": retest_ids,
        "modified_skills": sorted(modified_skills),
    }

    out_file = round_dir / f"refine_pass{pass_num}.json"
    out_file.write_text(json.dumps(refiner_result, indent=2), encoding="utf-8")
    return refiner_result


# ---------------------------------------------------------------------------
# Retest + regression check
# ---------------------------------------------------------------------------


def retest_weak(test_ids: list[str], retest_dir: Path,
                parent_dir: Path, tests: list[dict]) -> list[dict]:
    """Re-run only the weak tests after a refiner pass."""
    test_list = [t for t in tests if t["id"] in test_ids]
    results = []
    for test in test_list:
        test_id = test["id"]
        prompt = test["prompt"]
        is_agent_direct = test.get("category") == "agent-direct"
        agent_name = (
            test.get("expected_agent") if is_agent_direct else None
        )
        mode = "agent" if is_agent_direct else "plugin"

        print(f"    Retesting {test_id}"
              f"{' (agent: ' + agent_name + ')' if agent_name else ''}...")

        plugin = run_claude(
            prompt, test_id, mode,
            max_turns=PLUGIN_MAX_TURNS, timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=retest_dir,
            agent=agent_name,
        )

        baseline_file = parent_dir / f"{test_id}.baseline.json"
        if baseline_file.exists():
            baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        else:
            baseline = {"result": ""}

        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            judge = judge_responses_robust(
                prompt, baseline_text, plugin_text,
                test_id, retest_dir,
                needs_gpu=test.get("needs_gpu", False),
            )
            if not judge.get("error"):
                delta = judge.get("value_add", 0)
                print(f"      Plugin: {judge.get('plugin_score', '?')}/15 | "
                      f"Delta: {int(delta):+d}")
            else:
                print(f"      Judge error: {judge.get('error')}")
        else:
            judge = {"test_id": test_id, "error": "missing result"}

        results.append(judge)
    return results


def cross_round_regression_check(current_judges: list,
                                 round_dir: Path) -> list[tuple]:
    best_scores: dict[str, float] = {}
    for prev_dir in sorted(LOG_DIR.glob("round*")):
        if prev_dir == round_dir:
            continue
        summary = prev_dir / "summary.json"
        if summary.exists():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                for j in data.get("judge_results", []):
                    tid = j.get("test_id")
                    score = j.get("plugin_score", 0)
                    if tid:
                        best_scores[tid] = max(
                            best_scores.get(tid, 0), score)
            except (json.JSONDecodeError, OSError):
                continue

    regressions = []
    for j in current_judges:
        if j.get("error"):
            continue
        tid = j.get("test_id")
        current = j.get("plugin_score", 0)
        if tid in best_scores and current < best_scores[tid] - 2:
            regressions.append((tid, best_scores[tid], current))

    return regressions


def check_regressions(old_judges: list, new_judges: list,
                      refine_result: dict, tests: list[dict]):
    REGRESSION_THRESHOLD = 3

    target_to_edit = {}
    for edit_group in refine_result.get("edits", []):
        if edit_group.get("status") == "applied" and edit_group.get("backup"):
            file_path = edit_group["file"]
            parts = file_path.split("/")
            if parts[0] == "agents" and len(parts) >= 2:
                agent_name = parts[1].replace(".md", "")
                target_to_edit[f"agent:{agent_name}"] = edit_group
            elif len(parts) >= 2:
                target_to_edit[parts[1]] = edit_group

    old_by_id = {j["test_id"]: j for j in old_judges if not j.get("error")}
    regressed_targets: set[str] = set()

    for new_j in new_judges:
        tid = new_j.get("test_id")
        if new_j.get("error") or tid not in old_by_id:
            continue

        test_def = next((t for t in tests if t["id"] == tid), None)
        if not test_def:
            continue
        if (test_def.get("category") == "agent-direct"
                and test_def.get("expected_agent")):
            test_target = f"agent:{test_def['expected_agent']}"
        else:
            test_target = test_def.get("expected_skill") or "unknown"

        if test_target not in target_to_edit:
            continue

        old_score = old_by_id[tid].get("plugin_score", 0)
        new_score = new_j.get("plugin_score", 0)
        drop = old_score - new_score

        if drop >= REGRESSION_THRESHOLD:
            print(f"    REGRESSION on {tid}: {old_score} -> {new_score} "
                  f"(drop={drop}). Reverting {test_target}.")
            regressed_targets.add(test_target)

    for target_name in regressed_targets:
        edit_group = target_to_edit.get(target_name)
        if not edit_group:
            continue
        backup_path = Path(edit_group["backup"])
        target_path = PLUGIN_DIR / edit_group["file"]
        if backup_path.exists():
            target_path.write_text(
                backup_path.read_text(encoding="utf-8"),
                encoding="utf-8")
            print(f"      Reverted {edit_group['file']}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_summary(judge_results: list):
    print(f"\n{'='*100}")
    print("A/B COMPARISON SUMMARY")
    print(f"{'='*100}")
    header = (f"{'Test':<30} {'Base':>5} {'Plugin':>7} {'Delta':>6} "
              f"{'Winner':<8} {'Skill?':>6} {'Eff':>5} {'Reason'}")
    print(header)
    print("-" * 100)

    total_base = 0
    total_plugin = 0
    wins = {"baseline": 0, "plugin": 0, "tie": 0}
    count = 0

    for j in judge_results:
        if j.get("error"):
            print(f"{j.get('test_id', '?'):<30} ERROR: {j['error'][:50]}")
            continue
        b = j.get("baseline_score", 0)
        p = j.get("plugin_score", 0)
        d = j.get("value_add", 0)
        w = j.get("winner", "?")
        sk = "Y" if j.get("skill_triggered") else ("N" if j.get("skill_triggered") is False else "?")
        e = j.get("efficiency", 0)
        reason = j.get("winner_reasoning", "")[:40]
        print(f"{j['test_id']:<30} {b:>4}/15 {p:>5}/15 {d:>+5} "
              f"{w:<8} {sk:>6} {e:>5.2f} {reason}")
        total_base += b
        total_plugin += p
        wins[w] = wins.get(w, 0) + 1
        count += 1

    if count:
        print("-" * 100)
        print(f"{'AVERAGE':<30} {total_base/count:>5.1f} "
              f"{total_plugin/count:>7.1f} "
              f"{(total_plugin - total_base)/count:>+5.1f}")
        print(f"Plugin wins: {wins.get('plugin', 0)} | "
              f"Baseline wins: {wins.get('baseline', 0)} | "
              f"Ties: {wins.get('tie', 0)}")


def save_round_summary(judge_results: list, refine1: dict, refine2: dict,
                       round_dir: Path, round_num: int,
                       cross_regressions: list | None = None):
    summary = {
        "round": round_num,
        "mode": "web" if NO_KB else "kb",
        "tests": len(judge_results),
        "judge_results": judge_results,
        "refine_pass1": refine1,
        "refine_pass2": refine2,
    }
    if cross_regressions:
        summary["cross_round_regressions"] = [
            {"test_id": tid, "best_ever": best, "current": cur}
            for tid, best, cur in cross_regressions
        ]
    out_file = round_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRound {round_num} results saved to {round_dir}/")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(tests: list[dict], no_kb: bool = False,
                      test_ids: list[str] | None = None):
    global NO_KB
    NO_KB = no_kb

    if test_ids:
        tests = [t for t in tests if t["id"] in test_ids]

    existing = list(LOG_DIR.glob("round*"))
    round_num = max(
        (int(m.group())
         for d in existing
         if (m := re.search(r'\d+', d.name))),
        default=0,
    ) + 1
    round_dir = LOG_DIR / f"round{round_num}"
    round_dir.mkdir(exist_ok=True)

    mode_label = "WEB MODE (no KB)" if NO_KB else "KB MODE"
    print(f"\n{'='*70}")
    print(f"ROUND {round_num} — A/B Quality Evaluation — {mode_label}")
    print(f"{'='*70}")

    all_judge_results = []

    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]
        gpu_note = " [GPU]" if test.get("needs_gpu") else " [E2E]"

        print(f"\n--- {test_id} [{test.get('category')}]{gpu_note} ---")

        # Phase 1: Baseline
        print(f"  [1/4] Baseline run...")
        baseline = get_or_run_baseline(test, round_dir)
        b_chars = len(baseline.get("result", ""))
        b_err = baseline.get("error", "")
        cached_from = baseline.get("cached_from")
        if b_err:
            print(f"        ERROR: {b_err}")
        elif cached_from:
            print(f"        CACHED from {cached_from} — "
                  f"{b_chars} chars")
        else:
            print(f"        {b_chars} chars, "
                  f"{baseline.get('num_turns', '?')} turns, "
                  f"${baseline.get('cost_usd', 0):.2f}")

        # Phase 2: Plugin run
        is_agent_direct = test.get("category") == "agent-direct"
        agent_name = test.get("expected_agent") if is_agent_direct else None

        if is_agent_direct:
            print(f"  [2/4] Direct agent run (--agent {agent_name})...")
        else:
            print(f"  [2/4] Plugin run...")

        plugin = run_claude(
            prompt, test_id, "agent" if is_agent_direct else "plugin",
            max_turns=PLUGIN_MAX_TURNS,
            timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=round_dir,
            agent=agent_name,
        )
        p_chars = len(plugin.get("result", ""))
        p_err = plugin.get("error", "")
        if p_err:
            print(f"        ERROR: {p_err}")
        else:
            print(f"        {p_chars} chars, "
                  f"{plugin.get('num_turns', '?')} turns, "
                  f"${plugin.get('cost_usd', 0):.2f}")

        # Phase 3: Judge
        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            print(f"  [3/4] Judge evaluation ({N_JUDGES} runs)...")
            judge = judge_responses_robust(
                prompt, baseline_text, plugin_text,
                test_id, round_dir,
                needs_gpu=test.get("needs_gpu", False),
            )
            if not judge.get("error"):
                runs = judge.get("judge_runs", 1)
                print(
                    f"        Baseline: "
                    f"{judge.get('baseline_score', '?')}/15 | "
                    f"Plugin: {judge.get('plugin_score', '?')}/15 | "
                    f"Delta: {int(judge.get('value_add', 0)):+d} | "
                    f"Winner: {judge.get('winner', '?')} "
                    f"({runs} judge{'s' if runs > 1 else ''})")
                reason = judge.get("winner_reasoning", "")
                if reason:
                    print(f"        {reason[:120]}")
            else:
                print(f"        Judge error: {judge.get('error')}")
        else:
            judge = {"test_id": test_id,
                     "error": "missing baseline or plugin result"}
            print(f"  [3/4] Judge skipped — missing response")

        # Negative control check
        if (test.get("category") == "negative-control"
                and not judge.get("error")):
            if judge.get("baseline_score", 0) > judge.get("plugin_score", 0):
                print(f"        NEGATIVE CONTROL FAIL: plugin degraded "
                      f"response ({judge['baseline_score']} > "
                      f"{judge['plugin_score']})")

        # Skill trigger check
        expected_skill = test.get("expected_skill")
        if plugin_text and expected_skill is not None:
            triggered = check_skill_triggered(plugin, expected_skill)
            judge["skill_triggered"] = triggered
            if not triggered:
                print(f"        WARNING: Expected skill '{expected_skill}' "
                      f"did NOT fire")

        # Agent delegation check
        expected_agent = test.get("expected_agent")
        if (plugin_text and expected_agent
                and test.get("category") == "agent-delegation"):
            delegated = check_agent_invoked(plugin, expected_agent)
            judge["agent_delegated"] = delegated
            if delegated:
                print(f"        Agent delegation: '{expected_agent}' "
                      f"was invoked")
            else:
                print(f"        WARNING: Expected delegation to "
                      f"'{expected_agent}' did NOT happen")

        # Phase 4: Efficiency
        if plugin_text:
            print(f"  [4/4] Efficiency analysis...")
            eff = parse_efficiency(plugin, test_id, round_dir)
            print(f"        {eff.get('citations_in_response', 0)} citations, "
                  f"eff={eff.get('efficiency_ratio', 0):.2f}, "
                  f"{eff.get('assessment', '')}")
            judge["efficiency"] = eff.get("efficiency_ratio", 0)
        else:
            print(f"  [4/4] Efficiency skipped — no plugin result")

        all_judge_results.append(judge)

    # Summary
    print_summary(all_judge_results)

    # Cross-round regression check
    cross_regs = cross_round_regression_check(
        all_judge_results, round_dir)
    if cross_regs:
        print(f"\n{'='*70}")
        print("CROSS-ROUND REGRESSIONS (vs best-ever scores)")
        print(f"{'='*70}")
        for tid, best, current in cross_regs:
            print(f"  {tid}: best={best}, now={current} "
                  f"(dropped {best - current})")

    # Refiner pass 1
    print(f"\n{'='*70}")
    print("REFINER PASS 1")
    print(f"{'='*70}")
    refine1 = run_refiner(all_judge_results, 1, round_dir, tests)
    applied1 = sum(
        len(e.get("applied", []))
        for e in refine1.get("edits", [])
    )
    print(f"  Patterns: {refine1.get('patterns_found', 0)} | "
          f"Edits applied: {applied1} | "
          f"Retest: {refine1.get('retest_needed', [])}")
    for e in refine1.get("edits", []):
        for a in e.get("applied", []):
            print(f"    + {e['file']} [{a['section']}]: {a['reason']}")
        for s in e.get("skipped", []):
            print(f"    ~ skipped: {s.get('reason', '')[:80]}")

    retest1_judges = []
    if refine1.get("retest_needed") and applied1 > 0:
        print("  Re-running weak tests after pass 1...")
        retest1_dir = round_dir / "refine_pass1_retest"
        retest1_dir.mkdir(exist_ok=True)
        retest1_judges = retest_weak(
            refine1["retest_needed"], retest1_dir, round_dir, tests)
        check_regressions(all_judge_results, retest1_judges, refine1, tests)

    # Merge retest results
    retested_ids = {
        j["test_id"] for j in retest1_judges if not j.get("error")}
    merged_judges = []
    for j in all_judge_results:
        retest_ver = next(
            (r for r in retest1_judges
             if r.get("test_id") == j.get("test_id")
             and not r.get("error")),
            None,
        )
        merged_judges.append(retest_ver if retest_ver else j)

    # Refiner pass 2
    print(f"\n{'='*70}")
    print("REFINER PASS 2")
    print(f"{'='*70}")
    refine2 = run_refiner(merged_judges, 2, round_dir, tests)
    applied2 = sum(
        len(e.get("applied", []))
        for e in refine2.get("edits", [])
    )
    print(f"  Patterns: {refine2.get('patterns_found', 0)} | "
          f"Edits applied: {applied2} | "
          f"Retest: {refine2.get('retest_needed', [])}")
    for e in refine2.get("edits", []):
        for a in e.get("applied", []):
            print(f"    + {e['file']} [{a['section']}]: {a['reason']}")
        for s in e.get("skipped", []):
            print(f"    ~ skipped: {s.get('reason', '')[:80]}")

    if refine2.get("retest_needed") and applied2 > 0:
        print("  Re-running weak tests after pass 2...")
        retest2_dir = round_dir / "refine_pass2_retest"
        retest2_dir.mkdir(exist_ok=True)
        retest2_judges = retest_weak(
            refine2["retest_needed"], retest2_dir, round_dir, tests)
        check_regressions(
            merged_judges, retest2_judges, refine2, tests)

    # Save final summary
    save_round_summary(
        all_judge_results, refine1, refine2, round_dir, round_num,
        cross_regressions=cross_regs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-refine: test and improve SuperML skills")
    parser.add_argument(
        "--suite",
        default=str(SELF_REFINE_DIR / "suites" / "default.yaml"),
        help="Path to test suite YAML (default: suites/default.yaml)")
    parser.add_argument(
        "--no-kb", action="store_true",
        help="Strip LEEROOPEDIA_API_KEY — skills use web fallback")
    parser.add_argument(
        "test_ids", nargs="*",
        help="Optional: run only these test IDs from the suite")
    return parser.parse_args()


def main():
    args = parse_args()
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = Path.cwd() / suite_path

    if not suite_path.exists():
        print(f"Suite not found: {suite_path}")
        sys.exit(1)

    tests = load_suite(suite_path)
    print(f"Loaded {len(tests)} tests from {suite_path.name}")

    if args.test_ids:
        valid = [t["id"] for t in tests]
        invalid = [t for t in args.test_ids if t not in valid]
        if invalid:
            print(f"Unknown test IDs: {invalid}")
            print(f"Available: {valid}")
            sys.exit(1)

    run_full_pipeline(tests, no_kb=args.no_kb, test_ids=args.test_ids)


if __name__ == "__main__":
    main()
