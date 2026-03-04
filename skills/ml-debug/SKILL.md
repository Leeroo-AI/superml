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

## Phases

### Phase 1: Gather Evidence — Diagnose Immediately

Call `diagnose_failure(symptoms, logs)` with everything available:
- **symptoms**: What's failing, what was expected, when it started
- **logs**: Error lines, stack trace, unexpected output, metrics timeline

Do NOT guess at the cause. The KB has framework-specific failure patterns — use them.

**Gate**: You have a KB-grounded diagnosis with a root cause hypothesis before proposing any fix.

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

### Phase 3: Fix + Verify

1. Apply the fix — provide specific config changes, code patches, or commands
2. Include a verification step: "Run 10 steps and confirm [specific behavior]"
3. If the fix involves hyperparameters, include the recommended range from KB

Present the result:

```
## Diagnosis

**Root cause**: [one sentence] [PageID]
**Confidence**: High / Medium / Low
**Evidence**: [what in the logs/symptoms points to this]

### Fix
1. [specific action with exact config/code] [PageID]
2. [verification step — how to confirm the fix worked]

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
| Not checking what changed | "It was working yesterday" | `git diff`, env changes, package updates, data changes. Find the delta. |

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
