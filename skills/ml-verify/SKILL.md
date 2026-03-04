---
name: ml-verify
description: Use when the user wants to verify code, config, or math before running — or proactively before any expensive training job or deployment
---

# ML Verification

Catch mistakes before they waste GPU hours. Verify configs, code, and math against documented framework behavior.

## The Iron Law

```
NO TRAINING RUN WITHOUT VERIFICATION FIRST
```

An hour of verification saves days of debugging failed runs. Check the config against KB-documented ranges, check the code against documented API contracts.

## Phases

### Phase 1: Check Against KB

**For code/math:** Call `verify_code_math(code_snippet, concept_name)` with:
- The code, formula, or config to check
- What it's implementing (e.g., "LoRA scaling factor", "gradient accumulation with DDP")

**For configs/hyperparameters:** Call `query_hyperparameter_priors(query)` with:
- Model size, task type, hardware, and framework context
- The specific parameters you're checking

**For full training configs:** Call `review_plan(proposal, goal)` with:
- The complete config as the proposal
- The training objective as the goal

Run whichever combination fits. When in doubt, run all applicable checks in parallel.

**Gate**: Every parameter and code path has been checked against KB documentation.

### Phase 2: Dry Run Checklist

Before the real run, verify these can complete without error:
- [ ] Model loads on target hardware (no OOM on init)
- [ ] Data pipeline produces correctly shaped batches
- [ ] Forward + backward pass completes (1 step, no crash)
- [ ] Loss is a reasonable initial value (not 0, not NaN, not 1e6)
- [ ] Gradient norms are in expected range
- [ ] Checkpoint save/load works

Present the result:

```
## Verification: [what was checked]

**Verdict**: PASS / FAIL / WARNING

### Findings
| Check | Status | Detail |
|-------|--------|--------|
| [item] | PASS/FAIL/WARN | [explanation] [PageID] |

### Issues Found (if any)
1. **[Issue]**: [what's wrong] [PageID]
   - Current: `param = value`
   - Recommended: `param = value` [PageID]
   - Why: [one sentence explanation]

### Corrected Version (if FAIL)
```[language]
[corrected code or config]
```

### Dry-Run Command
```bash
[command to run 1-step verification]
```
```

## After This

- **PASS** → Proceed to training. Log the experiment with **ml-experiment**.
- **WARNING** → Proceed with monitoring. Watch the flagged parameters closely.
- **FAIL** → Fix before running. If the fix is non-obvious, invoke **ml-debug**.

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| alpha/r ratio inverted | LoRA alpha=16 with r=64 gives 0.25x scaling — often too low | Check: alpha/r should typically be 1-2x. KB has per-framework defaults. |
| LR 100x too high for PEFT | Using full fine-tuning LR (1e-3) with LoRA | PEFT typically needs 1e-4 to 5e-5. Check KB for model-specific ranges. |
| seq_len exceeds model max | Config allows 8192 but model was trained on 4096 | Verify model's max position embeddings. RoPE scaling needed beyond native context. |
| Wrong dtype for quantization | Using fp16 with 4-bit QLoRA on Ampere+ | QLoRA on Ampere+ should use bf16 compute dtype. fp16 causes instability. |
| Missing warmup for large LR | Jumping to peak LR on step 1 | Use 3-10% warmup. More important with larger LR or smaller datasets. |

## Examples

**"Is this LoRA config correct? lora_r=64, lora_alpha=16, lr=5e-5"**
1. `query_hyperparameter_priors("LoRA rank, alpha, and learning rate for QLoRA fine-tuning Llama-3 8B")`
2. `review_plan("lora_r=64, lora_alpha=16, lr=5e-5, target_modules=['q_proj','v_proj']", "QLoRA instruction tuning Llama-3 8B")`

**"Check my vLLM serving config before deployment"**
1. `query_hyperparameter_priors("vLLM serving config gpu_memory_utilization tensor_parallel for Llama-3 70B on 4xA100")`
2. `review_plan("[user's vLLM config]", "Serve Llama-3 70B with vLLM on 4xA100 80GB")`

**"Is my gradient accumulation math right?"**
1. `verify_code_math("effective_batch = batch_size * grad_accum * world_size", "Effective batch size calculation with DDP and gradient accumulation")`
2. `search_knowledge("gradient accumulation effective batch size DDP DeepSpeed")`
