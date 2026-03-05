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

> **ZERO-CITATION GATE** — If KB calls fail: (1) first token = `⚠️`, (2) every table row ends `**UNGROUNDED**`, (3) never say "but I can review". No exceptions. See Phase 1 for full rules.

### Phase 1: Check Against KB

> **CRITICAL FAILURE PATH**: If ANY KB call returns "API key not configured" or empty results:
> 1. Your FIRST output token MUST be `⚠️` — not a verdict, not "The KB isn't available but...", not ANY other text
> 2. Follow the 4-rule KB-failure checklist below exactly
> 3. These exact phrases are BANNED after KB failure — delete the sentence if you catch yourself writing any of them: "but I can do a", "but let me review", "I can do a thorough", "let me review manually", "isn't available but", "not configured, but", "thorough manual review", "but I can help"
> 4. **Self-test**: Re-read your first sentence. Does it start with `⚠️ WARNING: This verification is entirely ungrounded`? If not, delete everything and restart.

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

**MANDATORY**: Always attempt KB calls first — even if you suspect they'll fail. Never skip to "manual review." If calls fail, follow the KB-failure checklist below exactly. A response with zero PageID citations that lacks the warning banner and UNGROUNDED tags is a FAILED skill execution.

**If KB calls fail or return nothing (including "API key not configured") — STOP and do this exactly:**
1. Your FIRST line of output MUST be: `⚠️ WARNING: This verification is entirely ungrounded — zero KB citations. All recommendations below are best-effort and may be wrong. Verify independently before trusting.`
2. EVERY row in the findings table MUST end with `**UNGROUNDED**`
3. You MUST NOT say "manual review", "thorough review", or any phrase implying you can substitute for the KB. If your draft contains ANY form of "but" after acknowledging KB failure, delete that entire sentence. The ONLY acceptable first sentence is the warning banner — nothing else, no preamble, no softening. **Test**: if your opening doesn't start with `⚠️ WARNING:`, you have already failed.
4. Set citation count to 0

Violating ANY of these four rules is a FAILED skill execution, even if the technical advice is correct.

**Grounding rule**: Every claim in your output MUST cite a specific PageID from KB results. If a KB call returns no results, say "no KB match found" for that check — do NOT substitute general knowledge without flagging it as ungrounded. Vague references like "the QLoRA paper" or "typically" without a PageID are not grounded.

**Gate**: Every parameter and code path has been checked against KB documentation. If any KB call failed or returned nothing, flag those checks as `UNGROUNDED` in the findings table.

### Phase 2: Dry Run Checklist

Before the real run, verify these can complete without error:
- [ ] Model loads on target hardware (no OOM on init)
- [ ] Data pipeline produces correctly shaped batches
- [ ] Forward + backward pass completes (1 step, no crash)
- [ ] Loss is a reasonable initial value (not 0, not NaN, not 1e6)
- [ ] Gradient norms are in expected range
- [ ] Checkpoint save/load works

Present the result:

**STOP-CHECK before writing output**: Count your PageID citations. If the count is zero:
1. Your response MUST start with the `⚠️ WARNING:` banner — not a verdict, not a heading, not any other text
2. Every table row MUST end with `**UNGROUNDED**`
3. Scan your draft for "but I can", "manual review", "thorough review", any form of "but" followed by an offer to review — delete the entire sentence
4. Scan for any row that ends with just an explanation (no PageID, no `**UNGROUNDED**`) — append `**UNGROUNDED**`
Do NOT proceed to the template below until all four checks pass.

```
## Verification: [what was checked]

**Verdict**: PASS / FAIL / WARNING

### Findings
| Check | Status | Detail |
|-------|--------|--------|
| [item] | PASS/FAIL/WARN | [explanation] — [PageID:title] or **UNGROUNDED** |  ← EVERY row, no exceptions |

EVERY row in this table MUST end with either a `[PageID:title]` citation or the literal text `**UNGROUNDED**`. No exceptions. No row may have just an explanation.

**Citation count**: [N] PageIDs cited.

> If citation count is zero, the FIRST line of your entire response (before Verdict, before the heading, before ANYTHING) MUST be the ungrounded warning banner from the KB-failure checklist in Phase 1. Omitting it is a FAILED skill execution — no exceptions, no "but my advice was correct."

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
| Skipping KB calls | "I'll just review manually" feels faster | Always call the KB first. Ungrounded advice sounds confident but may be wrong. Mark every uncited claim UNGROUNDED. |
| Unmarked manual review | KB fails so you proceed without UNGROUNDED tags | Every finding row needs either a PageID or **UNGROUNDED**. Confident tone without citations is the most dangerous output. |
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
| LoRA alpha/r ratio | FAIL | alpha=16 / r=128 = 0.125x scaling — too low | **UNGROUNDED** |
```
WRONG (instant fail — grounding score = 0): `"The Leeroopedia KB isn't available (API key not configured), but I can do a thorough manual review."`
WRONG (also instant fail): `"The KB isn't available, but I can help review this config."`
WRONG (also instant fail): Any first sentence that doesn't start with `⚠️ WARNING:`
