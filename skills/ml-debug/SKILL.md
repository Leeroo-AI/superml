---
name: ml-debug
description: Use when something is failing in ML/AI work — OOM, NaN, divergence, crashes, bad throughput, wrong outputs, dependency conflicts
---

# ML Debugging

Systematically diagnose ML failures using framework-specific knowledge, not guesswork.

## The Iron Law

```
NO FIX WITHOUT UNDERSTANDING THE ROOT CAUSE FIRST
```

Applying fixes without diagnosis leads to fix-on-fix layering. The third "fix" usually breaks something the first fix was hiding.

```
NO CLAIM WITHOUT A [PageID] CITATION — OR MARK IT [no KB]
```

Every factual claim about framework internals requires either a `[PageID]` citation or an explicit `[no KB]` tag. There is no middle ground. "I have deep knowledge" is not a citation.

```
IF KB CALL FAILS → EVERY SECTION HEADER GETS ⚠️, EVERY CLAIM GETS [no KB], EVERY CONFIDENCE IS "Low"
```
This is an Iron Law, not a suggestion. A response that looks normal but has zero `[PageID]` and zero `[no KB]` tags is the #1 grounding failure.

## Phases

**Hard rule: You MUST actually call `diagnose_failure()` or `search_knowledge()` before writing any analysis.** Do not assume KB is unavailable — make the call and let it fail. If it fails, switch to the **KB-Unavailable response format** below — no exceptions, no "but I know this topic well" overrides. Skipping the call and writing authoritative-sounding analysis is the #1 failure mode of this skill.

**STOP — read this before writing anything after a KB failure:** If any KB call returns an error (API key issue, timeout, empty result), you are in **degraded mode**. You MUST use the KB-Unavailable response format for your ENTIRE response. You may NOT write a normal-looking diagnosis that omits citations. The phrase "but I have deep knowledge" is not an escape hatch — it is the exact failure mode this rule prevents. A response without `[PageID]` tags that also lacks `[no KB]` tags on every claim is a broken response, full stop.

### Phase 1: Gather Evidence — Diagnose Immediately

Call `diagnose_failure(symptoms, logs)` with everything available:
- **symptoms**: What's failing, what was expected, when it started
- **logs**: Error lines, stack trace, unexpected output, metrics timeline

Do NOT guess at the cause. The KB has framework-specific failure patterns — use them.

**Gate**: You have a KB-grounded diagnosis with a root cause hypothesis before proposing any fix.

**Non-negotiable**: Every diagnosis MUST include at least one `[PageID]` citation from a KB tool call. If `diagnose_failure` returns no results, call `search_knowledge()` with the framework + error type. If both fail, prefix your entire response with `⚠️ KB Unavailable` and mark ALL confidence as "Low (no KB grounding)". Do NOT present ungrounded expertise as authoritative — no "I have deep knowledge of X internals" framing. Without citations, you are guessing.

**KB-Unavailable response format** (you MUST use this exact structure if ANY KB call fails or returns errors):
```
⚠️ KB Unavailable — confidence is Low for all claims below

## ⚠️ Diagnosis
**Root cause**: [one sentence] [no KB]
**Confidence**: Low (no KB grounding)
**Version**: [framework version — state if unverified] [no KB]
**Evidence**: [what in the logs/symptoms points to this] [no KB]

### ⚠️ Fix
1. [specific action] [no KB]
2. [verification step] [no KB]

### ⚠️ If That Doesn't Work
- [alternative] [no KB]
```
Every section header must include the ⚠️ marker. Every factual claim must end with `[no KB]`. Every confidence must be "Low". No exceptions — not even "Medium". A response that looks like a normal diagnosis but lacks both `[PageID]` and `[no KB]` tags is the #1 grounding failure. Self-check: before sending, count your `[no KB]` tags. If the count is zero and you have no `[PageID]` citations, your response is broken.

### Phase 2: Confirm the Diagnosis

Ask yourself before fixing:
- Is this **deterministic** (happens every time) or **intermittent** (timing/race condition)?
- Does it happen on **first step** (config/setup issue) or **step N** (accumulation/overflow)?
- Is it **one GPU** or **all GPUs** (distributed-specific vs general)?
- What **changed** since it last worked?

If the diagnosis is ambiguous:
1. Call `propose_hypothesis(current_status, recent_experiments?)` for ranked alternatives
2. Call `query_hyperparameter_priors(query)` if the diagnosis points to config values

**Gate**: You can explain the root cause in one sentence and say why the proposed fix addresses it.

**Confidence calibration**: Only mark "High" confidence when the root cause is a well-documented, widely-reproduced failure mode with direct log evidence AND you have a KB citation confirming the mechanism. If the mechanism is speculative or you're inferring from timing correlation alone, mark "Medium" or "Low" — even if it *sounds* plausible. Plausible ≠ confirmed.

**Causal mechanism rule**: If you claim "X causes Y because of Z" (e.g., "eager mode causes memory fragmentation via caching allocator"), you need evidence for the *mechanism*, not just for X and Y independently. Without a `[PageID]`, mark the mechanism as speculative: "X may cause Y (speculative — no KB confirmation) `[no KB]`". Do NOT present speculative mechanisms in a "Why it hurts" or "What it does" explainer format that implies certainty — use conditional language ("may", "can", "in some configurations") and tag `[no KB]`.

**Version pinning rule**: State the exact framework version (e.g., `vLLM 0.6.0`, not just "vLLM") in both Diagnosis and Fix. Config keys, CLI flags, and defaults change across versions — unversioned advice is unverifiable and hurts specificity.

**Hard ceiling**: No `[PageID]` → max confidence "Medium". No KB at all → max confidence "Low", every claim tagged `[no KB]`. Do not use hedging language ("likely", "in practice", "typically") to present ungrounded claims as authoritative.

Before choosing a fix, ask: **What is the least destructive intervention?** Prefer targeted fixes (reset one component, adjust one parameter) over broad ones (restart from scratch, lower all learning rates). If resuming from a checkpoint after a failure (e.g., expert collapse, NaN), check whether optimizer state or specific weights need resetting — a full restart may discard recoverable work.

When multiple hypotheses exist, **order by diagnostic cost**: test the hypothesis that takes minutes (e.g., check `git diff`, inspect data, do arithmetic on step counts) before the one that requires a full training run. If simple arithmetic (e.g., steps × batch_size = dataset_size) explains the symptom, lead with that — don't bury it as a fallback behind a speculative mechanism. **Verify your arithmetic end-to-end**: if you claim "epoch ends at step N, so step M is the boundary", check that M actually equals N — off-by-one or rounding errors in your own math undermine the diagnosis.

### Phase 3: Fix + Verify

1. Apply the fix — provide specific config changes, code patches, or commands
2. Include a verification step: "Run 10 steps and confirm [specific behavior]"
3. For serving/inference fixes: use a structured load test (not just manual curl) to validate latency/throughput claims
4. Double-check generated CLI commands for duplicate flags, deprecated options, and version-specific syntax
5. For config-heavy fixes (DeepSpeed, vLLM, Megatron): call `search_knowledge()` to verify the exact config key names and structure for the user's version — config schemas change across versions and wrong keys silently do nothing
3. If the fix involves hyperparameters, include the recommended range from KB
4. Every fix action MUST cite a `[PageID]` — if you cannot cite one, call `search_knowledge()` for that specific fix before presenting it
5. Pin the framework version in every fix AND diagnosis: state the exact version tested (e.g., `vLLM 0.4.1`, `transformers 4.41.0`) in both the Diagnosis and Fix sections — config keys, CLI flags, and internal behaviors change across versions, and unversioned advice is unverifiable. In KB-Unavailable mode, still state the version but mark it `[no KB]` if you cannot verify it against documentation
4. For serving/inference fixes, also check: KV cache dtype (FP8 KV cache as a lighter alternative to full model quantization), chunked prefill settings, and prefix caching — these are orthogonal optimizations that may solve the problem with less risk than full-model quantization
5. For OOM fixes, also check secondary memory optimizations: `double_quant` for QLoRA, gradient checkpointing granularity, FSDP as alternative to DeepSpeed — mention at least one alternative approach
6. When providing inspection/debugging code, use the actual framework APIs (e.g., `trainer.get_train_dataloader()` not a manually reconstructed DataLoader) — generic recreations won't reproduce the exact behavior

**Pre-output self-check (mandatory before writing the response below)**:
1. Count your `[PageID]` citations. If zero → you MUST use the KB-Unavailable format from Phase 1. Stop and rewrite.
2. Count your `[no KB]` tags. If both `[PageID]` and `[no KB]` counts are zero → your response is broken. Stop and rewrite.
3. Check every "Why it hurts" or "What it does" explanation — does it have a `[PageID]`? If not, add `[no KB]` and use conditional language.
4. Check that you stated an exact framework version (e.g., `vLLM 0.6.0`) in both Diagnosis and Fix sections.

Present the result:

```
## Diagnosis

**Root cause**: [one sentence] [PageID or `[no KB]` if unavailable]
**Confidence**: High / Medium / Low (no `[PageID]` → max Medium; no KB at all → must be Low)
**Version**: [exact framework version, e.g., vLLM 0.6.0] [PageID or `[no KB]`]
**Evidence**: [what in the logs/symptoms points to this]

### Fix
1. [specific action with exact config/code] [PageID or `[no KB]`]
2. [verification step — how to confirm the fix worked]

### Consolidated Config
```
[copy-paste-ready final config with all changes applied — not scattered across explanation]
```

### If That Doesn't Work
- Alternative cause: [what else it could be] → Try: [next diagnostic step] [PageID]

### Prevention
- [how to catch this earlier next time] [PageID]
```

## After This

- **Log the fix** in `experiments/lessons.md` — include the symptom, root cause, and fix so it's findable next time
- If the fix didn't work → back to **Phase 1** with the new evidence (what the fix changed, what happened)
- If the fix worked → invoke **ml-verify** to confirm the config is solid before continuing
- If the issue revealed a config problem → invoke **ml-verify** on the full config

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Blaming batch size for every OOM | It's the most common OOM cause, but not the only one | Check: is it activation memory, optimizer states, or KV cache? Different fixes. |
| Changing LR when data is the problem | LR is easy to change; data quality is hard to inspect | Look at loss curves shape first. Plateau ≠ LR issue. Noisy loss = data issue. |
| Fixing symptoms not causes | "Add gradient clipping" without asking why gradients explode | Trace back: bad initialization? LR too high? Data outliers? Mixed precision overflow? |
| Applying multiple fixes at once | "Let me reduce batch size AND add grad clipping AND lower LR" | One fix at a time. Otherwise you can't attribute improvement. |
| Applying a global fix when a targeted one exists | "Lower the global LR" when only the router LR is wrong | Identify which component is misbehaving. Adjust only that component's config (e.g., router LR, not global LR). |
| Not checking what changed | "It was working yesterday" | `git diff`, env changes, package updates, data changes. Find the delta. |
| Restarting from early checkpoint when a surgical reset is possible | "Restart from step 1000" discards thousands of steps of valid training | Check if you can reset only the broken component (e.g., router weights/optimizer state) while keeping the rest. Less destructive = better. |
| Using framework-specific config keys without version check | Config keys change across versions; wrong key silently does nothing | Verify config keys against your installed version. Pin the version in your fix. |
| Claiming quadratic memory scaling without checking for flash attention | Vanilla attention is O(n²) in sequence length, but flash attention (default in Llama 3+, Mistral, etc.) changes this | Always check whether the model uses flash attention before citing quadratic memory scaling. State the assumption explicitly. |
| Recommending quantization without checking hardware support | FP8 needs SM89+ (H100), AWQ/GPTQ need specific kernel support, not all formats work on all GPUs | Before recommending a quantization format, verify the target GPU supports it. FP8 → H100+, not A100. State the hardware requirement explicitly and lead with a compatible option. |
| **Presenting general knowledge as KB-grounded analysis** | KB call was skipped or failed, response says "I have deep knowledge" and proceeds without citations | You MUST attempt the KB call — don't assume it will fail. If it does fail, use the **KB-Unavailable response format** from Phase 1 exactly. Every claim gets `[no KB]`, every confidence is "Low", every section header gets ⚠️. **BANNED phrases** without `[PageID]`: "I have deep knowledge of X", "from my understanding of X internals", "X typically causes Y". These are confidence-laundering — they make guesses sound authoritative. |
| **Writing a normal-format response when KB is down** | KB call failed but the response uses the standard Diagnosis/Fix format without `[no KB]` tags or ⚠️ markers — looks authoritative but has zero grounding | If you have zero `[PageID]` citations, you MUST be in KB-Unavailable format. Self-check before sending: search your response for `[no KB]` — if count is 0 and `[PageID]` count is also 0, rewrite in KB-Unavailable format. |

## Examples

**"QLoRA OOM on A100 40GB, batch size 4, seq len 4096"**
1. `diagnose_failure("OOM during QLoRA fine-tuning on A100 40GB, batch_size=4, seq_len=4096, Qwen2.5-7B", "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")`
2. `query_hyperparameter_priors("QLoRA memory-safe batch size and seq length for 7B model on A100 40GB")`

**"Training loss stuck at 2.3, not decreasing"**
1. `diagnose_failure("Training loss plateaued at 2.3 after 100 steps, not decreasing", "Step 100: loss=2.31, Step 200: loss=2.29, Step 300: loss=2.30")`
2. `propose_hypothesis("Loss plateau at 2.3 during SFT of Llama-3 8B", "Tried lr=2e-5, cosine schedule, warmup 10%")`
3. `query_hyperparameter_priors("Learning rate and schedule for SFT Llama-3 8B on instruction data")`

**"NCCL timeout during distributed training"**
1. `diagnose_failure("NCCL timeout after 30 seconds during DDP init, 4 nodes 8xH100", "RuntimeError: NCCL communicator was aborted... NCCL_TIMEOUT")`
2. `search_knowledge("NCCL timeout debugging multi-node distributed training InfiniBand")`
