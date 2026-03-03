---
name: ml-verify
description: Use when the user wants to verify code correctness, config validity, math/logic accuracy, or API usage against Leeroopedia KB
---

# ML Verification Workflow

Verify ML code, configs, and math against Leeroopedia KB documentation.

**CRITICAL: Call the verification tool IMMEDIATELY with the user's code/config.** Do NOT eyeball it first. The KB catches framework-specific issues you might miss.

## When to Use

- User asks "Is this code correct?"
- User wants to validate a training config before running
- User needs to check math/algorithm implementation
- User asks about correct API usage for a framework
- Before running an expensive training job (proactive verification)

## Mode A: Code/Math Verification

### 1. Verify against KB
Call `verify_code_math(code_snippet, concept_name)` with:
- **code_snippet**: The code or formula to check
- **concept_name**: What it's implementing (e.g., "LoRA scaling factor", "gradient accumulation with DDP")

### 2. Fill in framework details
If verification depends on framework-specific behavior, call `search_knowledge` for the exact API contract or documented behavior.

### 3. Present result
Pass/Fail with specific discrepancies and corrected code.

## Mode B: Config Verification

### 1. Check hyperparameters
Call `query_hyperparameter_priors(query)` with the model size, task, and hardware context.

### 2. Review the full config
Call `review_plan(proposal, goal)` with the config as the proposal and the training goal.

### 3. Present result
Flag any values outside recommended ranges with KB-grounded alternatives. **Before sending:** scan your draft — every `##` section MUST have at least one `[Category/Page_Name]` citation from tool results.

## Output Format

```
## Verification: [what was checked]

**Verdict**: PASS / FAIL / WARNING

### Findings
- [item]: [status] — [explanation] [PageID]
- ...

### Corrected Version (if FAIL)
```[language]
[corrected code or config]
```

### Rationale
[Why the correction is needed, citing KB]
```

## Examples

**"Is this LoRA config correct? lora_r=64, lora_alpha=16, lr=5e-5"**
1. `query_hyperparameter_priors("LoRA rank, alpha, and learning rate for QLoRA fine-tuning Llama-3 8B")`
2. `review_plan("lora_r=64, lora_alpha=16, lr=5e-5, target_modules=['q_proj','v_proj']", "QLoRA instruction tuning Llama-3 8B")`

**"Is my gradient accumulation math right?"**
1. `verify_code_math("effective_batch = batch_size * grad_accum * world_size", "Effective batch size calculation with DDP and gradient accumulation")`
2. `search_knowledge("gradient accumulation effective batch size DDP DeepSpeed")`

**"Check my vLLM serving config before deployment"**
1. `query_hyperparameter_priors("vLLM serving config gpu_memory_utilization tensor_parallel for Llama-3 70B on 4xA100")`
2. `review_plan("[user's vLLM config]", "Serve Llama-3 70B with vLLM on 4xA100 80GB")`
