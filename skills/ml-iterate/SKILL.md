---
name: ml-iterate
description: Use when the user is stuck, needs ranked next steps, or wants alternatives after initial experiments — "I tried X and got Y, what next?"
---

# ML Iteration

Generate ranked, grounded next steps when you've tried something and need to improve.

## The Iron Law

```
NO NEW EXPERIMENT WITHOUT REVIEWING WHAT YOU ALREADY TRIED
```

Re-running a failed approach with minor tweaks is the most common waste of GPU time. Check your history first.

## Phases

### Phase 1: Review History

1. Read `experiments/journal.md` if it exists — what's been tried, what worked, what didn't
2. Call `propose_hypothesis(current_status, recent_experiments?)` with:
   - **current_status**: Where things stand — metrics, what's working, what's not
   - **recent_experiments**: What's been tried and what happened (prevents re-suggesting dead ends)

**Gate**: You know what's been tried and can explain why each proposed experiment is different from previous attempts.

### Phase 2: Rank Options

For the top 2-3 hypotheses from Phase 1:
1. Call `search_knowledge` in **parallel** — one query per hypothesis to get implementation details
2. If any hypothesis involves tuning, call `query_hyperparameter_priors` for recommended ranges

**Gate**: Each option has KB-grounded implementation details, not just a name.

### Phase 3: Design Next Experiment

Present ranked alternatives:

```
## Iteration Options

**Current state**: [summary of metrics and what's been tried]

### Option 1: [name] — Expected impact: HIGH/MEDIUM/LOW
- **What**: [specific change — one variable]
- **Why**: [KB-grounded rationale] [PageID]
- **How**: [implementation steps with config/code]
- **Effort**: quick fix / half day / multi-day
- **Risk**: [what could go wrong]

### Option 2: ...

### Recommended Next Experiment
[Which option to try first and why. Include: what metric to watch, how many steps before deciding, what "success" looks like.]
```

## After This

- **Log the hypothesis** in `experiments/journal.md` with **ml-experiment** before running
- After results come in → log the outcome → back to **Phase 1** if not at target
- If the experiment fails → invoke **ml-debug** with the error
- If you've exhausted obvious options → invoke **ml-research** to explore less common approaches

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Changing 5 things at once | "Let me try everything that might help" | One variable per experiment. You need to know what caused the improvement. |
| Running too few steps to see effect | "10 steps should be enough to tell" | Most training effects need 100-500 steps to stabilize. Check KB for task-specific guidance. |
| Re-trying something that already failed | "Maybe it'll work this time with slight tweaks" | Read the journal. If it failed before, explain specifically what's different now. |
| Only trying hyperparameter changes | "Let me just sweep LR" | Data quality, data mix, architecture changes, and loss functions often matter more than HP tuning. |
| Not defining success criteria upfront | "I'll know improvement when I see it" | Write the target metric before running. Otherwise you'll rationalize mediocre results. |

## Examples

**"BM25 gets 30% recall, vector gets 35%, fusion gets 36%"**
1. Read `experiments/journal.md` for history
2. `propose_hypothesis("RAG: BM25 recall@5=30%, vector recall@5=35%, naive fusion=36%", "Tried larger chunk overlap - marginal gains")`
3. Parallel: `search_knowledge("reciprocal rank fusion vs distribution-based score fusion RAG")`, `search_knowledge("query expansion techniques for RAG retrieval")`

**"LoRA fine-tune gives worse eval than base model"**
1. Read `experiments/journal.md`
2. `propose_hypothesis("LoRA fine-tune of Llama-3 8B, eval dropped below base model", "Used lora_r=16, lr=2e-4, 3 epochs, 10k examples")`
3. `query_hyperparameter_priors("LoRA rank and learning rate for instruction tuning Llama-3 8B, preventing catastrophic forgetting")`
4. `search_knowledge("LoRA fine-tuning catastrophic forgetting prevention strategies")`
