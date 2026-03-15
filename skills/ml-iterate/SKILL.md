---
name: ml-iterate
description: Use when the user is stuck, needs ranked next steps, or wants alternatives after initial experiments — "I tried X and got Y, what next?"
---

# ML Iteration

Generate ranked, grounded next steps when you've tried something and need to improve.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode**.

**KB mode:** Call `propose_hypothesis` → `search_knowledge` → `query_hyperparameter_priors`. Cite as `[PageID]`.

**Web mode:** WebFetch GitHub issues for similar problems → WebFetch framework tuning guides → WebFetch published configs/ablations. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs.`

**Web mode URL registry:**
- HF Transformers/PEFT/TRL: `https://huggingface.co/docs/{transformers,peft,trl}`
- Axolotl: `https://github.com/axolotl-ai-cloud/axolotl`
- DeepSpeed: `https://www.deepspeed.ai/docs`
- vLLM: `https://docs.vllm.ai`
- Model cards: `https://huggingface.co/{org}/{model}` (always fetch for the user's specific model)
- PyTorch: `https://pytorch.org/docs/stable`
- Weights & Biases reports: `https://wandb.ai/site/articles` (for published ablation studies)

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

Attempt a `search_knowledge` call. If it succeeds, you're in KB mode. If it fails:

**YOU ARE NOW IN WEB MODE. Execute these WebFetch calls before writing ANY text:**
1. `WebFetch` the user's model card: `https://huggingface.co/{org}/{model}` (e.g., `https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`)
2. `WebFetch` the framework docs for their training setup (e.g., `https://huggingface.co/docs/trl` or `https://huggingface.co/docs/peft`)
3. `WebFetch` at least one more relevant source (GitHub issues, ablation studies, or config examples)
4. **Extract numbers immediately**: After each WebFetch, write down specific values (LR, rank, batch size, warmup) found in the source. These become your citation anchors — quote them in Phase 3. A URL without an extracted number is not a useful citation.

**DO NOT WRITE A SINGLE SENTENCE OF ADVICE until you have completed at least 2 WebFetch calls.** If you catch yourself writing advice without having called WebFetch, STOP and go back.

Only if ALL WebFetch calls fail with network errors:
1. Your **very first output line** MUST be: `⚠️ No documentation access — all technical claims below are tagged [unverified].`
2. Append `[unverified]` to **every sentence** containing a number, LR, model name, rank, or recommendation.

**Gate**: You have either KB access, web access, or you've committed to tagging every technical claim.

**Hard rule**: If you wrote "KB tools aren't authenticated" or similar and have not yet called WebFetch, STOP GENERATING. Go back and call WebFetch NOW. Do not continue writing advice. This is the #1 scoring failure.

**Grounding completeness**: After all WebFetch/KB calls, verify you have citations covering: (1) model architecture/requirements, (2) framework-specific API/config, AND (3) the specific failure mode the user reported (e.g., hallucination, overfitting, divergence). If any of these three angles is missing a citation, do one more lookup before writing.

### Phase 1: Review History

1. Read `experiments/journal.md` if it exists — what's been tried, what worked, what didn't
2. **KB mode:** Call `propose_hypothesis(current_status, recent_experiments?)` with current metrics and what's been tried
   **Web mode:** WebFetch framework docs and GitHub issues for the specific model/task to find documented tuning strategies and known issues

**Gate**: You know what's been tried and can explain why each proposed experiment is different from previous attempts.

**Data quality check**: If the user reports hallucinations, factual errors, or wrong entity names, ALWAYS include a data-quality hypothesis before hyperparameter changes. Hallucinated facts often mean the training data itself contains errors, inconsistent formatting, or missing grounding context — no amount of LR tuning fixes bad data. Recommend: (1) sample 50-100 training examples and audit for correctness, (2) check if entity names/facts in training data match expected outputs, (3) consider adding grounding context (e.g., product catalog) to each training example.

**Correctness check**: Before proposing any hypothesis, verify your understanding of the model architecture and training setup. If the user mentions a specific model, look up its fine-tuning requirements and known issues (KB: `search_knowledge`; Web: WebFetch the model card and framework docs). Explicitly state what you verified: "Confirmed: Llama-3-8B uses GQA with 8 KV heads, BOS token is <|begin_of_text|> [source]".

**Model variant check**: If the user is fine-tuning, verify they're using the right base: Instruct models are for chat/instruction tasks, base models for continued pretraining or domain adaptation. If their choice seems mismatched (e.g., fine-tuning base model for chat, or instruct model for domain pretraining), flag it as Option 0 before other recommendations. Cite the model card for variant differences.

**Specificity rule**: Every recommendation must use the user's actual numbers AND show the reasoning. "Try a lower LR" → "Try 5e-5 (halving your current 1e-4 — a conservative first step) [source or unverified]". Every option in Phase 3 must name their model, dataset size, current metric, and hardware.

**Proportional change rule (HARD GATE)**: Never recommend changing a hyperparameter by more than 3× in a single experiment unless a fetched source explicitly recommends a larger jump for this scenario AND you quote that source. Before emitting any option, compute the ratio: `proposed / current`. If ratio > 3× or < 1/3×, either split into two experiments or find a source justifying the jump. Show the math: "Current: 1e-4 → Proposed: 5e-5 (2× reduction)". A 5× reduction (e.g., 1e-4 → 2e-5) violates this rule — use 3e-5 or 5e-5 instead.

### Phase 2: Rank Options

For the top 2-3 hypotheses from Phase 1:

**KB mode:**
1. Call `search_knowledge` in **parallel** — one query per hypothesis to get implementation details
2. If any hypothesis involves tuning, call `query_hyperparameter_priors` for recommended ranges

**Web mode:**
1. WebFetch official docs in **parallel** — one doc page per hypothesis. Call WebFetch NOW, not later. Example: `WebFetch("https://huggingface.co/docs/trl/sft_trainer")`, `WebFetch("https://huggingface.co/docs/peft/conceptual_guides/lora")`
2. If any hypothesis involves tuning, WebFetch known-good configs (e.g., Axolotl example configs on GitHub)
3. For LoRA/fine-tuning iterations: always check whether **rank increase**, **learning rate reduction**, and **model-specific formatting** (chat templates, special tokens) have been explored
4. For every hypothesis: WebFetch the **model card** to verify architecture details and known limitations
5. **After WebFetch calls complete**: extract specific numbers and settings from the fetched content. Cite as `[source](URL)`. Do not paraphrase from memory — quote or reference the actual fetched content.

**Gate**: Each option has documentation-grounded implementation details with at least one citation. KB mode: `[PageID]`. Web mode: `[source](URL)`. If neither is available, mark every technical claim `[unverified]`.

**Minimum**: At least 3 parallel lookups per phase. Each option in Phase 3 must cite at least 2 distinct sources AND cross-reference them (e.g., "TRL docs recommend 2e-5 [source1], consistent with the Axolotl config default of 2e-5 [source2]"). If sources disagree, say so explicitly and recommend the more conservative value. If you have fewer than 3 lookups, you haven't searched enough — add queries.

**Quote, don't paraphrase**: For each citation, include a 5-15 word direct quote or specific number from the source. "[source](URL)" alone is insufficient — write "TRL docs state 'learning_rate=2e-4 for adapters' [source](URL)". This proves you read the source and prevents hallucinated citations.

**Cross-check config keys**: After WebFetch, grep the fetched content for the exact parameter names you plan to use. If a parameter name does not appear verbatim in any fetched source, do NOT use it in code — find the real name or mark it `# VERIFY`. This prevents hallucinated API parameters (e.g., writing `completion_only_loss=True` when the real API is `DataCollatorForCompletionOnlyLM`).

### Phase 3: Design Next Experiment

Present ranked alternatives:

**Root cause rule**: If symptoms include hallucination, factual errors, or wrong entity names, Option 1 MUST address data quality (audit, filtering, or grounding context injection), not hyperparameters. HP options go in Option 2+.

**Specificity rule**: Every recommendation must reference the user's actual model name, dataset size, current metric values, and hardware. Generic advice like "try a lower LR" without a specific number for their setup is not acceptable — always give a concrete value with rationale.

**Before emitting — STOP AND COUNT**: Scan every sentence. Count ALL tags (`[PageID: xxx]`, `[source](URL)`, or `[unverified — no KB access]`). **Minimum 6 tags total or do not emit.** Tag every sentence with a number, LR, model name, rank, token count, or recommendation. Zero-tag responses score 1/3 — same as hallucination. If you said "KB tools aren't authenticated" anywhere and have zero `[unverified]` tags, your response is broken — go back and tag NOW.

**Before emitting — VERIFY CORRECTNESS**: For each concrete number you recommend (LR, rank, batch size, warmup ratio), confirm it appears in at least one fetched source OR is derived from the user's own reported numbers. If a number came from your training data rather than a fetched source, tag it `[unverified]`. Do not present training-data knowledge as if it came from a citation.

**Before emitting — VERIFY PARAMETER NAMES**: For every config key, argument name, or API parameter in your code blocks, confirm the exact spelling and module path against fetched docs. Common traps: TRL uses `DataCollatorForCompletionOnlyLM` (not `completion_only_loss=True`), PEFT uses `use_rslora` (not `rslora`), `neftune_noise_alpha` is the SFTTrainer arg (not `neftune_alpha`). If you cannot find the exact parameter name in a fetched source, mark the code line with `# VERIFY: param name unconfirmed` and tell the user to check.

```
## Iteration Options

**Current state**: [summary of metrics and what's been tried]

**Grounding status**: [X citations from KB] or [X web citations with URLs] or [⚠️ No KB/web access — all technical claims tagged `[unverified]` below]
**Tag count**: [N tags in this response — MUST be ≥ 6]

### Option 1: [name] — Expected impact: HIGH/MEDIUM/LOW
- **What**: [specific change — one variable] [unverified — no KB access]
- **Why**: e.g. "LoRA LRs above 5e-5 often cause forgetting in 8B+ models [PageID: 4521] — your 1e-4 is 2-5× too high [unverified — no KB access]"
- **Evidence**: [direct quote from fetched source, 10-30 words, with citation. Example: PEFT docs state "rsLoRA uses rank-proportional scaling, stabilizing training at higher ranks" [source](URL). If no source found, write: "No direct source found — recommendation based on [reasoning] [unverified]"]
- **How**: [implementation steps with config/code] [unverified — no KB access]
- **Code**: [complete, runnable script. Requirements: (1) all imports, (2) exact model ID, (3) user's actual dataset path, (4) hardware-appropriate batch size with VRAM math comment, (5) `# CHANGED: old_value → new_value` comment on every modified line, (6) a print/assert that confirms the change before training starts. If config change, show full config with `# CHANGED` markers.]
- **Watch out**: [2-3 specific pitfalls. Each MUST follow this format: "At {param}={user_value} with {context}, watch for {metric} {direction} after {N} steps/epochs because {mechanism} [citation]." Example: "At lr=5e-5 with r=32 on 50k examples, watch for eval loss rising after epoch 2 because cosine decay may undershoot — monitor every 500 steps [unverified]." Generic warnings like 'be careful with learning rate' score 0.]
- **Effort**: quick fix / half day / multi-day
- **Risk**: [what could go wrong]

### Option 2: ...

### Recommended Next Experiment
[Which option to try first and why, citing KB evidence or marking `[unverified]`. Must include:
- **Metric**: exact metric name and current value → target value
- **Checkpoint**: how many steps/epochs before evaluating (cite KB for task-specific guidance)
- **Success gate**: specific threshold — "if BLEU > X after Y steps, proceed; otherwise revert"
- **Failure plan**: what to try if this doesn't work — name the specific Option number from above]
- **Citation count**: [N tags total — count `[PageID]` + `[source](URL)` + `[unverified]`. MUST be ≥ 6 or STOP and add more.]
- **WebFetch calls made**: [list the URLs you actually fetched — if this list is empty and you're in web mode, STOP and go back to Phase 0]
- **Numbers sourced vs unsourced**: [list every concrete number in your response and whether it came from a fetched source or is unverified. Example: "lr=5e-5 (from TRL docs), r=64 (unverified), warmup=0.05 (from Axolotl config)". This is your correctness audit.]
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
| Writing "no KB access" then untagged advice | "I acknowledged the limitation" | Acknowledging != compliance. Every technical sentence STILL needs `[unverified]` or a web citation. Count your tags — if zero, you failed. |
| Citing a URL without quoting its content | "The link is the evidence" | A URL proves you fetched, not that you read it. Include a 5-15 word quote or specific number from the source next to every `[source](URL)` tag. |
| Recommending >3× parameter change in one step | "Go big or go home" | Large jumps make it impossible to diagnose what worked. Halve or double — never 5× in one experiment. Show the math. |
| Using invented parameter names in code | "I'm pretty sure the flag is called X" | Every config key and API parameter in code blocks must match fetched docs verbatim. If you can't find it, mark `# VERIFY` and tell the user. Hallucinated param names silently break training. |
| Treating hallucination as only an HP problem | "Lower LR will fix the made-up facts" | Hallucinated entities (wrong names, invented products) usually signal bad training data — audit 50-100 examples for correctness before tuning LR/rank. Data fixes beat HP sweeps for factual grounding. |

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
