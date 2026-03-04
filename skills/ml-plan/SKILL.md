---
name: ml-plan
description: Use when the user wants an implementation plan, architecture design, or multi-step ML pipeline — "build X", "implement X", "design X", "set up X"
---

# ML Planning

Turn goals into validated, runnable implementation plans grounded in framework documentation.

## The Iron Law

```
NO IMPLEMENTATION WITHOUT A VALIDATED PLAN FIRST
```

A plan that hasn't been reviewed against the KB is a guess. Guesses waste GPU hours.

## Phases

### Phase 1: Understand — Build the Plan

Call `build_plan(goal, constraints?)` IMMEDIATELY with the user's stated goal.

- Use the user's exact words as the goal
- Include any hardware, framework, latency, or scale constraints they mentioned
- Do NOT wait for more information — use what you have now

**Gate**: You have a KB-grounded plan with numbered steps and validation criteria before proceeding.

### Phase 2: Validate — Review and Gap-Fill

1. Call `review_plan(proposal, goal)` with the plan from Phase 1 to catch risks
2. Identify the 2-4 most uncertain steps
3. Call `search_knowledge` in **parallel** for each gap:
   - Framework-specific API details
   - Config format requirements
   - Known pitfalls or gotchas
   - Memory/compute estimation for the specific hardware

**Gate**: Every step in the plan has either KB confirmation or an explicit "verify during dry-run" flag.

### Phase 3: Present — Structured Plan with Validation

Compose the final plan:

```
## Plan: [Goal]

### Overview
[1-2 sentences: what we're building, why this approach]

### Prerequisites
- [ ] Hardware: [specific GPUs, RAM]
- [ ] Dependencies: [packages with versions]
- [ ] Data: [format, size, location]

### Steps
1. **[Step name]** — [description] [PageID]
   - Config: `key: value`
   - Validate: [how to verify this step worked]
   - Time estimate: [wall time]

### Risks & Mitigations
- Risk: [what could go wrong] → Mitigation: [specific action] [PageID]

### Execution Strategy
- Dry-run: [what to test on 1% of data first]
- Checkpoint: [when to evaluate before continuing]
- Success criteria: [specific metrics or behaviors]
```

## After This

Execute in phases: **dry-run on 1% data → 1 epoch → full run.**

- Before running → invoke **ml-verify** to catch config mistakes
- Log the experiment → invoke **ml-experiment** to track hypothesis and results
- If any phase fails → invoke **ml-debug** with the error and plan context
- After results → invoke **ml-iterate** if metrics aren't at target

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Planning without memory estimation | "We'll figure out OOM at runtime" | Estimate GPU memory in the plan. Include per-step memory budget. |
| Missing evaluation step | "We'll evaluate after training" | Build evaluation into the plan — what metrics, what threshold, what data |
| Wrong parallelism for model size | "Just use FSDP for everything" | Check KB for model-size-specific parallelism recommendations |
| No dry-run phase | "The config looks right" | Always plan a 10-step dry-run before committing GPU hours |
| Skipping review_plan | "build_plan gave a good result" | review_plan catches risks that build_plan misses. Always run both. |

## Examples

**"Fine-tune Qwen2.5-7B with QLoRA on 2xA100"**
1. `build_plan("QLoRA fine-tuning Qwen2.5-7B", "2xA100 80GB, instruction tuning dataset")`
2. `review_plan(plan_output, "QLoRA fine-tuning Qwen2.5-7B")`
3. Parallel: `search_knowledge("Qwen2.5 QLoRA target_modules config Axolotl")`, `search_knowledge("QLoRA memory estimation 7B model 2xA100")`

**"Design a RAG system with hybrid retrieval"**
1. `build_plan("RAG system with hybrid vector+BM25 retrieval", "FastAPI, ChromaDB, production-ready")`
2. `review_plan(plan_output, "Hybrid RAG system")`
3. Parallel: `search_knowledge("ChromaDB hybrid retrieval BM25 integration")`, `search_knowledge("RAG evaluation metrics recall@k RAGAS")`
