---
name: ml-iterate
description: Use when the user is stuck, needs ranked next steps, or wants alternatives after initial experiments with ML/AI systems
---

# ML Iteration Workflow

Generate ranked hypotheses and next steps for ML experimentation.

**CRITICAL: Call `propose_hypothesis` IMMEDIATELY with the user's current status and experiments.** The KB has pattern-matched solutions from 1000+ repos. Don't guess — get data first.

## When to Use

- User tried something and it partially worked — needs to improve
- User is choosing between multiple approaches
- User wants ranked alternatives for architecture/training/serving decisions
- Experiment results are mediocre and user wants optimization ideas

## Workflow

### 1. Propose hypotheses
Call `propose_hypothesis(current_status, recent_experiments?)` with:
- **current_status**: Where the project stands, current metrics, what's working/not
- **recent_experiments**: What's been tried and what happened (prevents re-suggesting dead ends)

### 2. Ground top hypotheses
For the top 2-3 ranked hypotheses, call `search_knowledge` in **parallel** to get implementation details:
- Specific framework patterns for each approach
- Known performance characteristics
- Configuration examples

### 3. Check tuning ranges (if hyperparameter iteration)
If any hypothesis involves tuning, call `query_hyperparameter_priors` for recommended ranges given the user's context.

### 4. Present ranked alternatives
Combine into an actionable iteration plan.

## Output Format

```
## Iteration Options

**Current state**: [summary of where things stand]

### Option 1: [name] — Expected impact: HIGH/MEDIUM/LOW
- **What**: [specific change]
- **Why**: [rationale from KB] [PageID]
- **How**: [implementation steps with config]
- **Effort**: [quick fix / half day / multi-day]

### Option 2: ...

### Recommended Next Experiment
[Which option to try first and why]
```

## Examples

**"BM25 gets 30% recall, vector gets 35%, naive fusion gets 36%"**
1. `propose_hypothesis("RAG system: BM25 recall@5=30%, vector recall@5=35%, naive score-average fusion=36%", "Tried larger chunk overlap - marginal gains")`
2. Parallel: `search_knowledge("reciprocal rank fusion vs distribution-based score fusion RAG")`, `search_knowledge("query expansion techniques for RAG retrieval")`, `search_knowledge("hybrid retrieval weighting strategies BM25 dense")`

**"LoRA fine-tune gives worse eval than base model"**
1. `propose_hypothesis("LoRA fine-tune of Llama-3 8B on custom instruction data, eval scores dropped below base model", "Used lora_r=16, lr=2e-4, 3 epochs, 10k examples")`
2. `query_hyperparameter_priors("LoRA rank and learning rate for instruction tuning Llama-3 8B, preventing catastrophic forgetting")`
3. `search_knowledge("LoRA fine-tuning catastrophic forgetting prevention strategies")`
