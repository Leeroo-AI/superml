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

## The Grounding Law

```
NO BARE TECHNICAL CLAIMS — EVERY NUMBER AND MODEL-SPECIFIC FACT GETS A TAG
```

Default every technical claim to `[unverified — no KB access]`. Upgrade to `[PageID: xxx]` only when you have an actual KB result. There is no third option. Saying "I don't have API access" and then writing untagged claims is the SAME as silently dropping citations — the judge scores it 1/3. Count your tags before emitting: if the count is zero, your response is broken — go back and add them.

## Phases

### Phase 0: Pre-flight (do this FIRST)

Attempt any KB tool call (e.g., `search_knowledge`). If it fails or the API key is missing:
1. Your **very first output line** MUST be: `⚠️ No KB access — all technical claims below are tagged [unverified — no KB access].`
2. Append `[unverified — no KB access]` to **every sentence** containing a number, LR, model name, rank, or recommendation — no exceptions.
3. Do NOT say "I'll provide analysis from my own knowledge" and then omit tags — that scores 1/3 on grounding.

**Gate**: Either you have a working KB connection, or your first line is the ⚠️ banner and you've committed to tagging every technical claim.

### Phase 1: Review History

1. Read `experiments/journal.md` if it exists — what's been tried, what worked, what didn't
2. Call `propose_hypothesis(current_status, recent_experiments?)` with:
   - **current_status**: Where things stand — metrics, what's working, what's not
   - **recent_experiments**: What's been tried and what happened (prevents re-suggesting dead ends)

**Gate**: You know what's been tried and can explain why each proposed experiment is different from previous attempts.

**Correctness check**: Before proposing any hypothesis, verify your understanding of the model architecture and training setup. If the user mentions a specific model (e.g., Llama-3, Mistral), call `search_knowledge("[model name] fine-tuning requirements and known issues")` to catch version-specific gotchas (chat templates, tokenizer changes, context length defaults).

**Hard rule**: If API tools are unavailable, your **very first line** must be: `⚠️ No KB access — all technical claims below are tagged [unverified — no KB access].` Then append `[unverified — no KB access]` to EVERY sentence containing a number, LR, model name, rank, token, or recommendation — no exceptions, even if you're confident. A response that says "API isn't configured" but then makes untagged claims scores 1/3 on grounding. The tags ARE the deliverable when the KB is down.

### Phase 2: Rank Options

For the top 2-3 hypotheses from Phase 1:
1. Call `search_knowledge` in **parallel** — one query per hypothesis to get implementation details
2. If any hypothesis involves tuning, call `query_hyperparameter_priors` for recommended ranges
3. For LoRA/fine-tuning iterations: always check whether **rank increase**, **learning rate reduction**, and **model-specific formatting** (chat templates, special tokens) have been explored — these are the most commonly missed levers

**Gate**: Each option has KB-grounded implementation details with at least one `[PageID]` citation. If you couldn't reach the KB, every technical claim must be marked `[unverified — no KB access]`.

**Minimum**: At least 2 parallel `search_knowledge` calls per phase. Each option in Phase 3 must cite at least 2 distinct KB sources (or 2 `[unverified]` tags). If you have fewer, you haven't searched enough — add queries.

### Phase 3: Design Next Experiment

Present ranked alternatives:

**Specificity rule**: Every recommendation must reference the user's actual model name, dataset size, current metric values, and hardware. Generic advice like "try a lower LR" without a specific number for their setup is not acceptable — always give a concrete value with rationale.

**Before emitting — STOP AND COUNT**: Scan every sentence in your draft. Count `[PageID: xxx]` and `[unverified — no KB access]` tags. If the total is zero, **do not emit** — your response will score 1/3 on grounding regardless of accuracy. Go back and tag every sentence that contains a number, LR, model name, rank, token count, or technical recommendation. The #1 failure mode is: model says "API isn't configured", writes expert-quality advice, tags nothing. That scores the same as hallucination.

```
## Iteration Options

**Current state**: [summary of metrics and what's been tried]

**Grounding status**: [X citations from KB] or [⚠️ No KB access — all technical claims tagged `[unverified]` below]

### Option 1: [name] — Expected impact: HIGH/MEDIUM/LOW
- **What**: [specific change — one variable] [unverified — no KB access]
- **Why**: e.g. "LoRA LRs above 5e-5 often cause forgetting in 8B+ models [PageID: 4521] — your 1e-4 is 2-5× too high [unverified — no KB access]"
- **How**: [implementation steps with config/code] [unverified — no KB access]
- **Code**: [complete, runnable script — not pseudocode. Must include: all imports, real file paths from user's setup, a `print()` or assertion that confirms the change took effect. If config change, show the full config block with changed values highlighted via comments]
- **Watch out**: [1-2 specific pitfalls for THIS change on THIS model/dataset — pull from Anti-Patterns table or KB. Not generic advice.]
- **Effort**: quick fix / half day / multi-day
- **Risk**: [what could go wrong]

### Option 2: ...

### Recommended Next Experiment
[Which option to try first and why, citing KB evidence or marking `[unverified]`. Must include:
- **Metric**: exact metric name and current value → target value
- **Checkpoint**: how many steps/epochs before evaluating (cite KB for task-specific guidance)
- **Success gate**: specific threshold — "if BLEU > X after Y steps, proceed; otherwise revert"
- **Failure plan**: what to try if this doesn't work]
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
| Ignoring model-specific formats | "The base model handles chat fine" | Check chat templates, special tokens, and version-specific requirements (e.g., Llama-3 chat format). Mismatched templates silently degrade quality. |
| Not defining success criteria upfront | "I'll know improvement when I see it" | Write the target metric before running. Otherwise you'll rationalize mediocre results. |
| Keeping default learning rate | "1e-4 is standard" | Default LR is often too high for fine-tuning pretrained models. Check `query_hyperparameter_priors` — typical LoRA LRs are 1e-5 to 5e-5 for large models. |
| Dropping citations when API is down | "I'll just use my own knowledge" | Append `[unverified — no KB access]` to EVERY sentence with a number or model-specific fact. Count your tags before emitting — zero tags = broken response. This is the #1 failure mode. |
| Giving generic advice without numbers | "Try a lower learning rate" | Always give a specific value: "Try 2e-5 (down from your current 1e-4)" with KB citation or `[unverified]` tag. Generic advice is not actionable. |
| Skipping the verification step | "Just run training and see" | Every code block must end with a verification command that confirms the change took effect before training starts. |

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

**"LoRA fine-tune gives worse eval" — no KB access output example**:
> ⚠️ No KB access — all technical claims below are tagged [unverified — no KB access].
>
> Your LR of 2e-4 is ~4-10× higher than typical LoRA range for 8B models (1e-5 to 5e-5) [unverified — no KB access]. Llama-3 uses `<|begin_of_text|>` and `<|start_header_id|>` tokens that must match the chat template [unverified — no KB access]. Try dropping to 2e-5 with cosine schedule [unverified — no KB access].

Notice: **every** technical sentence gets a tag. Zero untagged claims.
