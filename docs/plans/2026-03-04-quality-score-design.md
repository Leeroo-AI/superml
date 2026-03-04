# Quality Score Redesign: A/B with LLM Judge + Auto-Refiner

**Date**: 2026-03-04
**Status**: Approved

## Problem

Current quality score (0-7 boolean checklist) measures output formatting, not whether the plugin actually makes ML code better or triggers at the right times. Need to answer:

1. Did calling the KB produce better ML code than not calling it?
2. Did the plugin trigger at the right moments — not too many, not too few calls?
3. Where specifically does it help or hurt, so we can refine skills automatically?

## Design

### Test Phases

Each of the 12 test scenarios runs in 4 phases:

```
Phase 1: Baseline run     (no plugin, vanilla Claude, 15 max turns)
Phase 2: Plugin run        (with plugin, 20 max turns)
Phase 3: Judge evaluation  (Claude compares both, 4 dimensions + reasoning)
Phase 4: Efficiency parse  (analyze KB call transcript)
```

Both baseline and plugin get the same prompt. Baseline has no access to KB tools.

### Output Structure

```
test-logs/round6/
  t01.baseline.json          # raw claude output, no plugin
  t01.plugin.json            # raw claude output, with plugin
  t01.judge.json             # judge scores + qualitative reasoning
  t01.efficiency.json        # KB call analysis
  ...
  refine_pass1.json          # refiner patches + analysis
  refine_pass1_retest/       # re-run results after pass 1
  refine_pass2.json          # refiner patches + analysis
  refine_pass2_retest/       # re-run results after pass 2
  summary.json               # full round comparison
```

### Judge Rubric

Judge receives both responses anonymized as "Response A" / "Response B" (randomized order to avoid position bias). Scores each independently on 4 dimensions (0-3):

| Dimension | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| **Correctness** | Wrong configs/params/APIs | Mostly right, 1-2 errors | Correct, minor gaps | Correct, catches edge cases |
| **Specificity** | Generic advice, no framework details | Some framework mentions | Framework-specific configs and code | Version-specific gotchas, exact API calls |
| **Mistake Prevention** | No warnings | Generic "be careful" | Flags specific pitfalls with fixes | Catches non-obvious issues other missed |
| **Actionability** | Abstract discussion | Some code snippets | Runnable code/config with explanation | Copy-paste ready, includes validation steps |

Each dimension produces BOTH a numeric score AND qualitative reasoning explaining why.

### Judge Output Per Test

```json
{
  "baseline_score": 7,
  "plugin_score": 11,
  "value_add": 4,
  "winner": "plugin",
  "efficiency": 0.82,
  "dimensions": {
    "correctness": {
      "baseline": 2, "plugin": 3,
      "reasoning": "Baseline recommends lr=2e-4 which is reasonable but plugin caught that alpha/r=0.125 suppresses updates 8x..."
    },
    "specificity": {
      "baseline": 1, "plugin": 3,
      "reasoning": "Baseline says 'use a lower learning rate' generically. Plugin cites exact scaling formula..."
    },
    "prevention": {
      "baseline": 1, "plugin": 2,
      "reasoning": "Baseline mentions overfitting risk. Plugin additionally flags zero warmup causing step-1 gradient spikes..."
    },
    "actionability": {
      "baseline": 3, "plugin": 3,
      "reasoning": "Both provide runnable configs. Tie."
    }
  }
}
```

### Efficiency Metric

Parsed from tool-use transcript:

```json
{
  "kb_calls": [
    {"tool": "search_knowledge", "query": "...", "citations_returned": 4, "citations_used": 3},
    ...
  ],
  "total_calls": 4,
  "citations_used_in_response": 7,
  "efficiency": 0.70,
  "wasted_calls": 0,
  "assessment": "All 4 calls contributed at least 1 citation."
}
```

- `efficiency = citations_used_in_response / total_kb_calls`
- `wasted_calls` = calls with 0 citations making it to final response
- `missed_opportunities` = baseline got things wrong but plugin didn't call KB

### Auto-Refiner (2 passes)

Runs after initial test round. Same logic both passes:

1. **Aggregate** judge reasoning across failing/weak tests, group by failure pattern
2. **Map** failures to responsible skill files via `expected_skill` field
3. **Generate** targeted patches by feeding failure patterns + current skill to Claude
4. **Apply** patches with guardrails:
   - Skill file can't grow more than 20% per iteration
   - Can only ADD or MODIFY lines, not delete Iron Laws/phases/gates
   - If re-test score drops on any previously-passing test, automatic revert
5. **Re-run** only the failing/weak tests
6. Both `using-leeroopedia` and per-workflow skills are patchable

### Refiner Output

```json
{
  "patterns_found": 3,
  "patches_proposed": [
    {
      "file": "skills/ml-plan/SKILL.md",
      "reason": "2 tests failed on memory estimation",
      "diff": "...",
      "expected_impact": "t04, t02 correctness +1"
    }
  ],
  "auto_applied": true,
  "retest_needed": ["t04", "t02"]
}
```

### Files to Create/Modify

| File | Change |
|------|--------|
| `scripts/test_interactive.py` | Rewrite: baseline run, judge phase, efficiency parsing, refiner loop |
| `scripts/judge_prompt.md` | New: judge system prompt with rubric |

### Success Criteria

- Plugin wins or ties on 10+ of 12 tests
- Average value_add > 0 across all tests
- Efficiency > 0.5 (more than half of KB calls contribute)
- Refiner improves at least 1 weak test per pass without regressing others
