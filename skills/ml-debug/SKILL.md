---
name: ml-debug
description: Use when something is failing in ML/AI work — OOM, NaN, divergence, crashes, bad throughput, wrong outputs, dependency conflicts
---

# ML Debugging Workflow

Systematically diagnose ML/AI failures using Leeroopedia KB.

**CRITICAL: Call `diagnose_failure` IMMEDIATELY with whatever error info you have.** Do NOT guess at the cause first. The KB has framework-specific failure patterns you don't know about.

## When to Use

- Training crashes (OOM, CUDA errors, NCCL timeouts)
- Loss divergence, NaN/Inf values
- Bad model outputs after training
- Inference serving failures or poor throughput
- Dependency conflicts or version mismatches
- Any "it doesn't work" in ML context

## Workflow

### 1. Diagnose IMMEDIATELY
Call `diagnose_failure(symptoms, logs)` with:
- **symptoms**: Clear description of what's failing and what was expected
- **logs**: The most relevant error lines, stack trace, or unexpected output

### 2. Check configuration (if config-related)
If the diagnosis points to hyperparameters or config, call `query_hyperparameter_priors(query)` for recommended values given the user's setup.

### 3. Propose alternatives (if ambiguous)
If multiple root causes are plausible, call `propose_hypothesis(current_status, recent_experiments?)` for ranked hypotheses.

### 4. Get implementation details
Call `search_knowledge` for the specific fix — framework-specific config changes, memory optimizations, version-specific workarounds.

## Output Format

Every key claim MUST include a `[PageID]` citation from the KB response:

```
## Diagnosis

**Root cause**: [most likely cause] [PageID]
**Confidence**: High/Medium/Low

### Fix
1. [specific action with config/code] [PageID]
2. ...

### If that doesn't work
- Alternative cause: ... → Try: ... [PageID]

### Prevention
- [how to avoid this in future] [PageID]
```

## Examples

**"QLoRA OOM on A100 40GB, batch size 4, seq len 4096"**
1. `diagnose_failure("OOM during QLoRA fine-tuning on A100 40GB, batch size 4, seq len 4096, Qwen2.5-7B", "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")`
2. `query_hyperparameter_priors("QLoRA memory-safe batch size and seq length for 7B model on A100 40GB")`

**"Training loss stuck at 2.3, not decreasing"**
1. `diagnose_failure("Training loss plateaued at 2.3 after 100 steps, not decreasing", "Step 100: loss=2.31, Step 200: loss=2.29, Step 300: loss=2.30")`
2. `propose_hypothesis("Loss plateau at 2.3 during SFT of Llama-3 8B", "Tried lr=2e-5, cosine schedule, warmup 10%")`
3. `query_hyperparameter_priors("Learning rate and schedule for SFT Llama-3 8B on instruction data")`
