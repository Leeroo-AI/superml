---
name: ml-verify
description: Use when the user wants to verify code, config, or math before running — or proactively before any expensive training job or deployment
---

# ML Verification

Catch mistakes before they waste GPU hours. Verify configs, code, and math against documented framework behavior.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode**.

**KB mode:** Call `verify_code_math` / `query_hyperparameter_priors` / `review_plan`. Cite as `[PageID]`.

**Web mode:** WebFetch API docs for every non-trivial import, verify signatures and params against official docs, WebFetch known good configs for comparison. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs.`

**Web mode URL registry:**
- HF PEFT: `https://huggingface.co/docs/peft`
- HF Transformers: `https://huggingface.co/docs/transformers`
- HF TRL: `https://huggingface.co/docs/trl`
- DeepSpeed: `https://www.deepspeed.ai/docs/config-json`
- vLLM: `https://docs.vllm.ai`
- PyTorch: `https://pytorch.org/docs/stable`

## The Iron Law

```
NO TRAINING RUN WITHOUT VERIFICATION FIRST
```

An hour of verification saves days of debugging failed runs. Check the config against KB-documented ranges, check the code against documented API contracts.

## Phases

### Phase 1: Check Against Documentation

**KB mode:**

Call the appropriate KB tools:
- **For code/math:** `verify_code_math(code_snippet, concept_name)`
- **For configs/hyperparameters:** `query_hyperparameter_priors(query)` with model size, task type, hardware, and framework context
- **For full training configs:** `review_plan(proposal, goal)` with the complete config

Run whichever combination fits. When in doubt, run all applicable checks in parallel. Cite as `[PageID]`.

**Web mode:**

WebFetch the relevant documentation for each check:
- **For code/math:** WebFetch the API docs for the framework. Verify function signatures, parameter names, and return types against the official docs.
- **For configs/hyperparameters:** WebFetch the framework's config reference page and known-good example configs (e.g., Axolotl examples, HF training examples). Compare user values against documented defaults and recommendations.
- **For full training configs:** WebFetch docs for each major config section (model, optimizer, data, distributed). Cross-check all values.

Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs.`

**If BOTH KB and web are unavailable:**
1. First line: `⚠️ WARNING: This verification is ungrounded. All recommendations below are best-effort. Verify independently.`
2. Every row in the findings table ends with `**UNGROUNDED**`
3. Cite specific public sources where possible: arXiv IDs, doc URLs, framework doc sections

**Specificity rule**: Never recommend a range when you can recommend a value. Pick the single best value from your sources and cite why.

**Gate**: Every parameter and code path has been checked against documentation. If any check couldn't be verified, flag it in the findings table.

### Phase 2: Dry Run Checklist

Before the real run, verify these can complete without error:
- [ ] Model loads on target hardware (no OOM on init)
- [ ] Data pipeline produces correctly shaped batches
- [ ] Forward + backward pass completes (1 step, no crash)
- [ ] Loss is a reasonable initial value (cross-entropy on vocab: expect `ln(vocab_size)` ≈ 10-11 for 32k vocab; flag if <1.0 or >15.0 or NaN)
- [ ] Gradient norms are in expected range (0.1–10.0 for LLM fine-tuning; flag if >100 or exactly 0.0)
- [ ] Checkpoint save/load works
- [ ] Estimate total VRAM with exact formula:
- [ ] Verify every FAIL/WARN fix is copy-paste ready (run the Verify command from Issues Found)
  - QLoRA 4-bit: `(params × 0.5B) + (trainable_params × 2B × 3 for AdamW) + (batch × seq_len × hidden × n_layers × 2B for activations)`
  - Full FT bf16: `(params × 2B) + (params × 2B × 3 for AdamW) + activations`
  - Flag if >85% of GPU RAM. Show the arithmetic.

Present the result:

**STOP-CHECK before writing output**: Count your PageID citations. If the count is zero:
1. Your response MUST start with the `⚠️ WARNING:` banner — not a verdict, not a heading, not any other text
2. Every table row MUST end with `**UNGROUNDED**`
3. Scan your draft for "but I can", "manual review", "thorough review", any form of "but" followed by an offer to review — delete the entire sentence
3b. Scan your draft for rows ending with only `**UNGROUNDED**` and no `[public-ref:...]` — add a specific public reference (doc URL, arXiv ID, or framework doc section) to each
4. Scan for any row that ends with just an explanation (no PageID, no `**UNGROUNDED**`) — append `**UNGROUNDED**`
Do NOT proceed to the template below until all four checks pass.

```
## Verification: [what was checked]

**Verdict**: PASS / FAIL / WARNING

### Findings
| Check | Status | Detail | Fix |
|-------|--------|--------|
| [item] | PASS/FAIL/WARN | [explanation with exact math + exact numbers, never just ranges] — [PageID:title] ([public-ref:URL]) or **UNGROUNDED** [public-ref:URL] | [copy-paste fix: exact config line or command; PASS rows say '—'] | ← EVERY row needs ALL 4 columns + citation |

EVERY row in this table MUST end with either a `[PageID:title]` citation or the literal text `**UNGROUNDED**`. No exceptions. No row may have just an explanation.

**Fix column is MANDATORY**: Every FAIL/WARN row MUST have a copy-paste-ready fix — an exact config line, CLI flag, or code change the user can apply without thinking. `Fix: —` is only allowed for PASS rows. If your table is missing the Fix column, you have failed the skill.

**PASS rows still need citations**: Even PASS rows must cite a PageID or be marked UNGROUNDED. "Correct" is not self-evident — the reader needs to verify WHY it's correct.

**Citation count**: [N] PageIDs cited.

> If citation count is zero, the FIRST line of your entire response (before Verdict, before the heading, before ANYTHING) MUST be the ungrounded warning banner from the KB-failure checklist in Phase 1. Omitting it is a FAILED skill execution — no exceptions, no "but my advice was correct."

### Issues Found (if any)
1. **[Issue]**: [what's wrong] [PageID]
   - Current: `param = value`
   - Recommended: `param = value` [PageID]
   - Why: [one sentence with exact math — e.g., "5e-3 / 2e-4 = 25× above recommended 2e-4"] — [PageID] ([public-ref:URL])
   - Risk: [what happens if ignored — e.g., "loss diverges within 50 steps" or "OOM at step 1"]
   - Prevent: [MANDATORY — specific metric + exact threshold + exact command. E.g., "Monitor `grad_norm` — if >1.0 in first 100 steps, halve LR. Check: `grep grad_norm logs/`"]
   - Verify: [one command to confirm the fix worked — e.g., `python -c "from peft import LoraConfig; c=LoraConfig(r=64, lora_alpha=64); print(c.lora_alpha/c.r)"`]

### Corrected Version (if FAIL)
```[language]
[corrected code or config — with inline comments showing the math for each changed value]
```
The corrected version MUST be complete and copy-paste ready. Do not use `...` or `# rest unchanged`. Show every line.

### Dry-Run Command
```bash
# Always include ALL of these (adapt framework):
[1-step train command, e.g.: accelerate launch train.py --max_steps=1 --logging_steps=1]
[VRAM check, e.g.: nvidia-smi --query-gpu=memory.used --format=csv]
[gradient check, e.g.: add `print(f"grad_norm={model.get_grad_norm()}")` after backward]
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
| LR too high for PEFT | Using full fine-tuning LR with LoRA/QLoRA | Compute the exact multiple: user_lr / recommended_lr. QLoRA typical range is 1e-4 to 2e-4. Say "X× too high" with the real number. Always include the fix: `learning_rate: [corrected value]`. |
| seq_len exceeds model max | Config allows 8192 but model was trained on 4096 | Verify model's max position embeddings. RoPE scaling needed beyond native context. |
| Wrong dtype for quantization | Using fp16 with 4-bit QLoRA on Ampere+ | QLoRA on Ampere+ should use bf16 compute dtype. fp16 causes instability. |
| Missing warmup for large LR | Jumping to peak LR on step 1 | Use 3-10% warmup. More important with larger LR or smaller datasets. |
| Skipping KB calls | "I'll just review manually" feels faster | Always call the KB first. Ungrounded advice sounds confident but may be wrong. Mark every uncited claim UNGROUNDED. |
| Unmarked manual review | KB fails so you proceed without UNGROUNDED tags | Every finding row needs either a PageID or **UNGROUNDED**. Confident tone without citations is the most dangerous output. |
| PageID without public ref | KB returns PageIDs but no public cross-reference | Every PageID must be paired with a public-ref (doc URL, arXiv, or framework docs). Internal-only citations can't be verified by the user. |
| Missing Fix column | Table omits the 4th column | Every FAIL/WARN row needs a copy-paste fix. Rebuild the table if the Fix column is missing. |
| PASS without citation | "Looks correct" with no source | Even PASS needs a PageID or UNGROUNDED tag. The user can't verify "correct" without a source. |
| Range instead of value | "Use 1e-4 to 3e-4" | Pick one value and cite why. Ranges defer the decision to the user — that's our job. |
| "Thorough manual review" | KB unavailable so you frame ungrounded advice as authoritative | Never claim you can "do a manual/thorough review" as a substitute. Say "KB unavailable — all findings UNGROUNDED" and tag every row. Ungrounded ≠ authoritative. |

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

**"Check my QLoRA config" (when KB is unavailable)**

CORRECT first line of output (non-negotiable):
```
⚠️ WARNING: This verification is entirely ungrounded — zero KB citations. All recommendations below are best-effort and may be wrong. Verify independently before trusting.
```
CORRECT table row:
```
| LoRA alpha/r ratio | FAIL | alpha=16 / r=128 = 0.125x scaling — too low | **UNGROUNDED** [public-ref: QLoRA paper — arXiv:2305.14314 §4] [public-ref: HuggingFace PEFT docs — peft.readthedocs.io/en/latest/conceptual_guides/lora] |
```
WRONG (instant fail — grounding score = 0): `"The Leeroopedia KB isn't available (API key not configured), but I can do a thorough manual review."`
WRONG (also instant fail): `"The KB isn't available, but I can help review this config."`
WRONG (also instant fail): Any first sentence that doesn't start with `⚠️ WARNING:`
