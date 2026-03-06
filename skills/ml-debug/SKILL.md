---
name: ml-debug
description: Use when something is failing in ML/AI work — OOM, NaN, divergence, crashes, bad throughput, wrong outputs, dependency conflicts
---

# ML Debugging

Systematically diagnose ML failures using framework-specific knowledge, not guesswork.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode**.

**KB mode:** Call `diagnose_failure` → `query_hyperparameter_priors` → `search_knowledge`. Cite as `[PageID]`.

**Web mode:** WebFetch GitHub issues for the error message → WebFetch framework troubleshooting docs → WebFetch config references. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs and GitHub issues.`

**Web mode URL registry:**
- PyTorch issues: `https://github.com/pytorch/pytorch/issues`
- HF Transformers issues: `https://github.com/huggingface/transformers/issues`
- DeepSpeed issues: `https://github.com/microsoft/DeepSpeed/issues`
- vLLM issues: `https://github.com/vllm-project/vllm/issues`
- PEFT docs: `https://huggingface.co/docs/peft`
- Axolotl issues: `https://github.com/axolotl-ai-cloud/axolotl/issues`

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

**Hard rule: You MUST look things up before writing any analysis.** In KB mode, call `diagnose_failure()` or `search_knowledge()`. In Web mode, WebFetch GitHub issues and framework docs for the error. Do not skip grounding — looking things up first is the #1 differentiator between useful and useless debugging advice.

**If KB tools fail:** Switch to Web mode. WebFetch GitHub issues for the exact error message, then WebFetch framework troubleshooting docs. Cite as `[source](URL)`. If both KB and web fail (e.g., no internet), use the **Ungrounded response format** below and mark ALL claims with `[unverified]`.

### Phase 1: Gather Evidence — Diagnose Immediately

**KB mode:** Call `diagnose_failure(symptoms, logs)` with everything available:
- **symptoms**: What's failing, what was expected, when it started
- **logs**: Error lines, stack trace, unexpected output, metrics timeline

**Web mode:** Search for the error using WebFetch:
1. WebFetch GitHub issues for the framework + exact error message (e.g., `https://github.com/huggingface/transformers/issues?q=<error>`)
2. WebFetch the framework's troubleshooting/FAQ page
3. If config-related, WebFetch the framework's config documentation

Do NOT guess at the cause. Look it up first — framework-specific failure patterns are well-documented.

**Gate**: You have a documentation-grounded diagnosis with a root cause hypothesis before proposing any fix. Every diagnosis MUST include at least one citation — `[PageID]` in KB mode, `[source](URL)` in Web mode.

**Ungrounded response format** (use ONLY if both KB and web search fail):
```
⚠️ Ungrounded — no documentation sources available. All claims below are [unverified].

## Diagnosis
**Root cause**: [one sentence] [unverified]
**Confidence**: Low
**Version**: [framework version] [unverified]
**Evidence**: [what in the logs/symptoms points to this]

### Fix
1. [specific action] [unverified]
2. [verification step] [unverified]

### If That Doesn't Work
- [alternative] [unverified]
```

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

**Grounding depth rule**: When citing KB sources, prefer citations that include version-specific details (changelogs, config schemas, API signatures) over general principle citations. If a KB result contains a specific version note or API detail, surface it in your response — e.g., "parameter `X` was renamed to `Y` in v0.5.0 [PageID]" is stronger grounding than "see [PageID] for details". Cross-reference multiple KB sources when available to strengthen confidence.

**Citation extraction rule**: When a KB call returns results, extract and quote the specific relevant detail — don't just append a `[PageID]`. BAD: "Speculative decoding helps here [PageID]" GOOD: "Speculative decoding generates N tokens per draft step, reducing decode passes by ~4× for acceptance rate >0.7 [PageID]". The citation must add information the reader can verify, not just authority.

### Phase 3: Fix + Verify

1. Apply the fix — provide specific config changes, code patches, or commands
2. Include a **runnable verification script** — not just prose instructions. For training: a code block that runs N steps and prints the metric to check. For serving: an async load-test script that measures p50/p95/p99 under concurrent load (not just a single curl). Single-request latency checks are insufficient for serving fixes.
3. **Correctness gate**: Before presenting any fix, verify every API call, import, and config key against KB results. Call `search_knowledge()` for any function signature or parameter you're unsure about. Wrong API calls (e.g., nonexistent methods, swapped argument order, deprecated parameters) are worse than no fix — they waste the user's time debugging YOUR code.
4. For serving/inference fixes: use a structured load test (not just manual curl) to validate latency/throughput claims
5. Double-check generated CLI commands for duplicate flags, deprecated options, and version-specific syntax
6. For config-heavy fixes (DeepSpeed, vLLM, Megatron): call `search_knowledge()` to verify the exact config key names and structure for the user's version — config schemas change across versions and wrong keys silently do nothing
3. If the fix involves hyperparameters, include the recommended range from KB
4. Every fix action MUST cite a `[PageID]` — if you cannot cite one, call `search_knowledge()` for that specific fix before presenting it
5. Pin the framework version in every fix AND diagnosis: state the exact version tested (e.g., `vLLM 0.4.1`, `transformers 4.41.0`) in both the Diagnosis and Fix sections — config keys, CLI flags, and internal behaviors change across versions, and unversioned advice is unverifiable. In KB-Unavailable mode, still state the version but mark it `[no KB]` if you cannot verify it against documentation
4. For serving/inference fixes, also check: KV cache dtype (FP8 KV cache as a lighter alternative to full model quantization), chunked prefill settings, and prefix caching — these are orthogonal optimizations that may solve the problem with less risk than full-model quantization
5. For OOM fixes, also check secondary memory optimizations: `double_quant` for QLoRA, gradient checkpointing granularity, FSDP as alternative to DeepSpeed — mention at least one alternative approach
6. When providing inspection/debugging code, use the actual framework APIs (e.g., `trainer.get_train_dataloader()` not a manually reconstructed DataLoader) — generic recreations won't reproduce the exact behavior
7. **Parameter class verification**: Before referencing any config parameter (e.g., `max_seq_length`), verify which class it belongs to (e.g., `SFTTrainer` vs `TrainingArguments`). Misattributing a parameter to the wrong class is a correctness error — the user will put it in the wrong place and it will silently do nothing.
8. **Memory arithmetic for OOM fixes**: Always include explicit per-GPU memory math (model params + optimizer states + activations + KV cache + overhead) so the user can verify the fix will actually fit. Show the calculation, don't just assert "this will fit".

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
1. [specific action with exact config/code — include exact values, not ranges] [PageID or `[no KB]`]
2. [verification step — how to confirm the fix worked]

**Actionability rule**: Every fix step must be copy-paste-ready. "Increase X" is not a fix — "Set X=128 (was 64)" is. "Try a smaller batch size" is not a fix — "Set per_device_train_batch_size=2 with gradient_accumulation_steps=8" is. If you write a fix step that contains the word "try" or "consider" without a concrete value, rewrite it.

### Consolidated Config
```
[copy-paste-ready final config with ALL changes applied in one block — user should need to copy exactly ONE config block, not assemble pieces from multiple sections. Include comments showing what changed and why: `# was 256, lowered to prevent over-batching during speculation`]
```

### If That Doesn't Work
- Alternative cause: [what else it could be] → Try: [next diagnostic step] [PageID]

### Prevention
- [specific monitoring command with exact threshold — e.g., `assert torch.cuda.max_memory_allocated() < 0.95 * torch.cuda.get_device_properties(0).total_memory` or `if expert_util.min() < 0.05: alert()`] [PageID]
- [runnable guard script or config flag that catches recurrence — e.g., `--log-expert-utilization --alert-threshold=0.05`] [PageID]

**Prevention quality gate**: Each prevention item must include either (a) a specific numeric threshold, (b) a runnable command/script, or (c) a config flag with its exact value. "Monitor loss curves" and "use checkpoints regularly" are NOT prevention — they are generic advice that would appear in any debugging guide. Prevention must be specific to THIS failure mode.
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
| **Skipping prevention or writing generic prevention** | Prevention section says "monitor loss" or "use checkpoints" — advice so generic it adds zero value | Prevention must cite a specific tool, metric threshold, or config flag that would have caught this failure earlier. E.g., "add `--log-expert-utilization` every 50 steps; if any expert drops below 5%, trigger alert" — not "monitor expert utilization". |
| **Verification script that doesn't match the failure mode** | Serving fix verified with single curl; OOM fix verified with "run and see if it crashes" | Match verification to the failure: serving → concurrent load test with latency percentiles; OOM → memory profiling with `torch.cuda.max_memory_allocated()`; convergence → loss curve over N steps with expected trajectory. |
| **Fix steps without concrete values** | "Try a lower learning rate" or "reduce batch size" without saying to what | Every fix step needs an exact value with reasoning: "Set lr=1e-5 (was 3e-4, ~30× reduction to let aux loss compete with LM loss gradient)". Ranges are acceptable only with a recommended starting point. |

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
