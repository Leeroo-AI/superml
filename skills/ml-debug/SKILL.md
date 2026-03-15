---
name: ml-debug
description: Use when something is failing in ML/AI work — OOM, NaN, divergence, crashes, bad throughput, wrong outputs, dependency conflicts
---

# ML Debugging

Systematically diagnose ML failures using framework-specific knowledge, not guesswork.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode**.

**HARD RULE: You MUST ground before writing analysis.** If KB fails, you MUST WebFetch at least 2 URLs before writing ANY diagnosis. Writing from memory without fetching is the #1 failure mode of this skill — it produces zero-citation responses that score 0/3 on grounding. "I know X well" is NOT a substitute for fetching documentation.

**KB mode:** Call `diagnose_failure` → `query_hyperparameter_priors` → `search_knowledge`. Cite as `[PageID]`.

**KB mode grounding supplement (MANDATORY):** After KB calls, you MUST WebFetch at least 2 public URLs (official docs, PyPI, GitHub issues/source) and cite them as `[source](URL)` alongside `[PageID]` citations. KB-only responses score 2/3 max on grounding because reviewers cannot verify proprietary page IDs. Pattern: KB call for diagnosis → WebFetch PyPI for version → WebFetch GitHub/docs for config verification → cite BOTH KB and public URLs in every section. **Self-test**: if your response has zero `[source](URL)` citations, you will lose a grounding point regardless of KB citation count.

**Web mode:** WebFetch GitHub issues for the error message → WebFetch framework troubleshooting docs → WebFetch config references. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs and GitHub issues.`

**Web mode grounding targets by response section** (aim for these counts):
- Diagnosis root cause: 1+ citation (to a specific doc section or GitHub source line, NOT a top-level page)
- Each "Why it matters" explanation: 1+ citation or `[no KB]`
- Each fix step: 1+ citation for the specific API/config being changed
- Each quantitative claim ("X× faster"): 1 citation or `[no KB]`
- Prevention items: 1+ citation for the metric/tool referenced
Target: 5+ total citations in web mode, each linking to a specific doc section or source line (not top-level domain pages). Below 3 is a grounding failure. Generic page links (e.g., `huggingface.co/docs/transformers`) score lower than specific section links (e.g., `huggingface.co/docs/transformers/model_doc/mixtral#MixtralConfig`).

**PyPI pages are version-only citations**: A PyPI link confirms a version number but contributes zero technical content. Do NOT count PyPI links toward your citation minimum. You need 3+ citations that contain *technical claims* (API signatures, config defaults, known failure modes). PyPI fetches are a prerequisite step, not a citation source.

 When citing GitHub source files, use a tagged release URL (e.g., `github.com/huggingface/transformers/blob/v4.45.0/src/...`) NOT the `main` branch. **MANDATORY first WebFetch in web mode**: fetch the framework's PyPI page (`https://pypi.org/project/<package>/`) to get the current stable version — then use that version tag in ALL subsequent GitHub URLs. **Do NOT fabricate version numbers** — if you haven't fetched the PyPI page, you don't know the current version. Writing "DeepSpeed 0.18.7" or "Transformers v5.3.0" without fetching is fabrication.
**Multi-package rule**: If your diagnosis involves N frameworks, you need N PyPI fetches — one per package. Fetching DeepSpeed's version does NOT tell you the Transformers version. Each `**Version**:` line must link to the specific PyPI page it came from. If you cannot fetch any URLs, use the Ungrounded response format. **Self-test**: before writing your response, count your `[source](URL)` citations. If < 2, you have not met the minimum bar — fetch more or switch to Ungrounded format.

**Web mode response template** (use this structure when in web mode):
```
> Grounding: Web mode — citations from official docs and GitHub issues.
> Sources fetched: [URL1], [URL2]

## Diagnosis
**Root cause**: [one sentence] [source](URL)
**Version**: [exact version from fetched docs] [source](URL)
...
```

**Web mode URL registry:**
- PyTorch issues: `https://github.com/pytorch/pytorch/issues`
- HF Transformers issues: `https://github.com/huggingface/transformers/issues`
- DeepSpeed issues: `https://github.com/microsoft/DeepSpeed/issues`
- vLLM issues: `https://github.com/vllm-project/vllm/issues`
- PEFT docs: `https://huggingface.co/docs/peft`
- Axolotl issues: `https://github.com/axolotl-ai-cloud/axolotl/issues`
- vLLM docs: `https://docs.vllm.ai/en/latest/`
- PyTorch docs: `https://pytorch.org/docs/stable/`
- DeepSpeed docs: `https://www.deepspeed.ai/docs/config-json/`
- PyPI (version lookup): `https://pypi.org/project/<package>/` (use to find current stable version)
- HF Transformers releases: `https://github.com/huggingface/transformers/releases`
- vLLM releases: `https://github.com/vllm-project/vllm/releases`

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

```
NO WRITING FROM MEMORY — IF KB FAILS, YOU MUST WEBFETCH BEFORE WRITING
```
"I know X well" or "I have deep knowledge of X" followed by ungrounded analysis is the #1 anti-pattern. If KB fails: WebFetch docs first, then write. No exceptions.



## Phases

**Hard rule: You MUST look things up before writing any analysis.**

**WHEN KB FAILS — DO THESE 3 THINGS IMMEDIATELY (no prose first):**
1. `WebFetch` the framework's PyPI page to get the current version
2. `WebFetch` GitHub issues for the exact error message
3. `WebFetch` the framework's config/API docs for the feature involved
Then write your response using the Web mode template. If WebFetch also fails, use the Ungrounded format.

**If you write ANY text before completing these 3 WebFetch calls, your response scores 0.**

### Phase 1: Gather Evidence — Diagnose Immediately

**MANDATORY FIRST ACTION**: KB fails → call WebFetch 3× (see checklist above) → THEN write. No prose before fetching.



**KB mode:** Call `diagnose_failure(symptoms, logs)` with everything available:
- **symptoms**: What's failing, what was expected, when it started
- **logs**: Error lines, stack trace, unexpected output, metrics timeline

**Web mode:** Search for the error using WebFetch (you MUST actually call WebFetch — do not skip this and write from memory):

**CHECKPOINT: If KB failed and you have not yet called WebFetch, STOP HERE. Call WebFetch NOW. Do not write diagnosis text. Do not explain why KB failed. Do not claim expertise. Call WebFetch.**
1. WebFetch GitHub issues for the framework + exact error message (e.g., `https://github.com/huggingface/transformers/issues?q=<error>`)
2. WebFetch the framework's official docs page for the relevant feature/config
3. If config-related, WebFetch the framework's config documentation or changelog for the user's version
4. Extract specific facts: version-specific defaults, config key names, known bug numbers — these become your citations
4b. If the problem involves a specific model, WebFetch its `config.json` from HuggingFace (e.g., `https://huggingface.co/<org>/<model>/raw/main/config.json`) — extract `num_key_value_heads`, `num_hidden_layers`, `hidden_size` for any memory/KV math
5. WebFetch the framework's latest release/tag page to pin the exact version — all subsequent citations must reference this version, not `main` branch
6. **Multi-framework version rule**: If the user's setup involves multiple frameworks (e.g., DeepSpeed + Transformers + PEFT), WebFetch the PyPI page for EACH framework separately. Do NOT reuse a version fetched for one package as if it applies to another, and do NOT extrapolate version numbers. Every version number in your response must trace to a specific fetched page. **FABRICATION TRAP**: If you write a version number that did NOT appear in a fetched page, it is fabricated — even if it "looks right". Common fabrications: `transformers 5.x` (does not exist as of 2025, it's 4.x), `deepspeed 0.18+` (verify the actual latest). If you catch yourself writing a version from memory, STOP and fetch the PyPI page.

Do NOT guess at the cause. Look it up first — framework-specific failure patterns are well-documented.

**Gate**: You have a documentation-grounded diagnosis with a root cause hypothesis before proposing any fix. Every diagnosis MUST include at least one citation — `[PageID]` in KB mode, `[source](URL)` in Web mode. **If you have zero citations at this point, STOP — go back and WebFetch something. Do not proceed to Phase 2 without at least one grounded citation.**

**KB mode gate**: If using KB mode, you must ALSO have at least one `[source](URL)` citation from a public URL by this point. If not, WebFetch now before proceeding. KB citations alone are insufficient for full grounding credit.

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

**Confidence calibration**: Only mark "High" confidence when ALL of: (1) the root cause is a well-documented, widely-reproduced failure mode, (2) you have direct log evidence, (3) you have a KB or public doc citation confirming the exact mechanism, AND (4) the mechanism is unconditional (not "if X is set wrong, then Y"). If ANY condition in your causal chain is speculative or requires an assumption about the user's config that you haven't verified, mark "Medium" max. Conditional hypotheses ("this happens IF max_steps is set incorrectly") are NEVER High confidence. Plausible ≠ confirmed.

**Causal mechanism rule**: If you claim "X causes Y because of Z", you need evidence for the *mechanism*. Without a `[PageID]` or `[source](URL)`, mark it speculative: "X may cause Y (speculative) `[no KB]`". Do NOT present speculative mechanisms in a "Why it hurts" format that implies certainty — use conditional language and tag `[no KB]`.

**Arithmetic verification rule**: If your diagnosis includes ANY math (memory budgets, KV cache sizes, batch calculations, step counts), you MUST show the full calculation with labeled inputs. Every input number must come from a fetched source (model config, framework docs) or be marked `[assumed]`. Do NOT use round numbers from memory — e.g., "Mistral has 32 KV heads" is wrong (it has 8); fetch `config.json` to verify.
**Quantitative claim sourcing rule**: If you state a ratio or magnitude (e.g., "30× stronger gradient", "4× memory reduction"), you MUST either (a) show the arithmetic derivation with sourced inputs, or (b) cite a source that states the ratio directly. Unsourced ratios presented as facts are a grounding violation — tag them `[no KB]` if you cannot derive or cite them.

**Version pinning rule**: State the exact framework version (e.g., `vLLM 0.6.0`, not just "vLLM") in both Diagnosis and Fix. Config keys, CLI flags, and defaults change across versions — unversioned advice is unverifiable and hurts specificity.

**Hard ceiling**: No `[PageID]` → max confidence "Medium". No KB at all → max confidence "Low", every claim tagged `[no KB]`. Do not use hedging language ("likely", "in practice", "typically") to present ungrounded claims as authoritative.

Before choosing a fix, ask: **What is the least destructive intervention?** Prefer targeted fixes (reset one component, adjust one parameter) over broad ones (restart from scratch, lower all learning rates). If resuming from a checkpoint after a failure (e.g., expert collapse, NaN), check whether optimizer state or specific weights need resetting — a full restart may discard recoverable work.

When multiple hypotheses exist, **order by diagnostic cost**: test the hypothesis that takes minutes (e.g., check `git diff`, inspect data, do arithmetic on step counts) before the one that requires a full training run. If simple arithmetic (e.g., steps × batch_size = dataset_size) explains the symptom, lead with that — don't bury it as a fallback behind a speculative mechanism. **Verify your arithmetic end-to-end**: if you claim "epoch ends at step N, so step M is the boundary", check that M actually equals N — off-by-one or rounding errors in your own math undermine the diagnosis.

**Causal chain precision rule**: If your math shows event X at step N but the symptom is at step M (where M ≠ N), do NOT invent a speculative bridging mechanism ("corruption propagates N-M steps"). Instead: (1) check if your math is slightly off, (2) check for off-by-one in logging vs optimizer steps, (3) if the gap remains unexplained, state it honestly: "The arithmetic predicts step N; the spike at step M is ~K steps later, suggesting [an additional factor / logging offset] that needs investigation." Confidence drops to Medium when the predicted and observed steps don't match.

**Grounding depth rule**: When citing KB sources, prefer citations that include version-specific details (changelogs, config schemas, API signatures) over general principle citations.

**Contradictory framing check**: Before presenting your diagnosis, re-read it for internal contradictions. Common traps:
- Stating a default value, calling the user's value "too high", then recommending an even higher value
- Saying a coefficient is "10× too high relative to default" AND "too weak to be effective" in the same paragraph — pick ONE framing: either it's too high (recommend lowering) or too weak (recommend raising). If the value is high relative to default but weak relative to competing gradients, frame it as: "Despite being above default, the effective signal is too weak because [competing gradient reason]" — do NOT call it "too high" if you're about to recommend raising it further.
For any parameter change, state clearly: (1) the framework default, (2) the user's current value, (3) your recommended value, (4) why the direction of change is correct. If a KB result contains a specific version note or API detail, surface it in your response — e.g., "parameter `X` was renamed to `Y` in v0.5.0 [PageID]" is stronger grounding than "see [PageID] for details". Cross-reference multiple KB sources when available to strengthen confidence.

**Direction-of-change sanity check**: After deciding on a parameter change, verify in one sentence: "The current value is X, the default is Y, I'm recommending Z, which moves it [higher/lower] because [mechanism]." If your mechanism says the value is "too high" but your fix raises it further, you have a contradiction — resolve it before writing. Similarly, if your mechanism says the value is "too weak" but your fix lowers it, stop and reconsider.

**Specificity rule**: Every diagnosis and fix must name the user's exact model, GPU type, framework version, and relevant config values — not generic placeholders. "Llama-3-70B on 4×A100-80GB with vLLM 0.6.0" not "large model on multi-GPU". Reviewers score specificity by counting concrete details that match the user's setup.

**Mandatory correctness checks before writing diagnosis:**
1. If computing KV cache: verify `num_key_value_heads` (NOT `num_attention_heads`) — GQA models have 4-8× fewer KV heads
2. If citing memory numbers: show the full arithmetic (params × bytes + optimizer × bytes + activations + KV cache)
3. If citing config keys: verify the key exists in the stated framework version
4. If citing improvement percentages: cite a source or mark `[no KB]` — never assert "30-50% improvement" without a citation

**Citation extraction rule**: When a KB call returns results, extract and quote the specific relevant detail — don't just append a `[PageID]`. BAD: "Speculative decoding helps here [PageID]" GOOD: "Speculative decoding generates N tokens per draft step, reducing decode passes by ~4× for acceptance rate >0.7 [PageID]". The citation must add information the reader can verify, not just authority.

### Phase 3: Fix + Verify

1. Apply the fix — provide specific config changes, code patches, or commands
2. Include a **runnable verification script** — not just prose instructions. For training: a code block that runs N steps and prints the metric to check. For serving: an async load-test script using `asyncio.gather()` with N concurrent requests matching the user's stated concurrency — sequential loops do NOT test concurrency. Measure and print p50/p95/p99. For OOM: include `torch.cuda.max_memory_allocated()` check. Single-request latency checks are insufficient for serving fixes. The script must test the SPECIFIC failure that was diagnosed — not a generic health check.
3. **Correctness gate**: Before presenting any fix, verify every API call, import, and config key against KB results or fetched docs. In web mode, WebFetch the framework's API reference for any function you're about to recommend. Wrong API calls (e.g., nonexistent methods, deprecated parameters) are worse than no fix. Every config value must include the exact key path (e.g., `engine_args.gpu_memory_utilization`, not just "gpu_memory_utilization").

**Unverified flag protocol**: If you cannot verify a CLI flag or config key exists in fetched docs, you MUST: (1) mark it `[unverified flag]` inline, (2) provide a fallback command using only verified flags, (3) include a one-liner to test the flag: `<tool> --help | grep <flag>`. Never present an unverified flag as the primary recommendation — always lead with verified alternatives.

**CLI flag and config key verification**: Before writing ANY CLI command or config dict, WebFetch the framework's arg parser source or config schema to verify exact names. Common errors: inventing JSON-style flags when the framework uses separate flags, using deprecated names, or **fabricating config keys that don't exist** (e.g., `use_router_z_loss` is NOT a standard DeepSpeed MoE key). If you cannot verify a key/flag exists in the framework source, mark it `[unverified key]` — do NOT present fabricated keys as real config options. Fabricated config keys silently do nothing, which is worse than no fix.


3. If the fix involves hyperparameters, include the recommended range from KB
4. Every fix action MUST cite a `[PageID]` — if you cannot cite one, call `search_knowledge()` for that specific fix before presenting it
5. Pin the framework version in every fix AND diagnosis: state the exact version tested (e.g., `vLLM 0.6.0`, `transformers 4.41.0`) in both the Diagnosis and Fix sections — config keys, CLI flags, and internal behaviors change across versions, and unversioned advice is unverifiable. In Web mode, WebFetch the framework's latest release/changelog to get the current version — do not guess. In Ungrounded mode, state the version but mark it `[no KB]`.
4. For serving/inference fixes, also check: KV cache dtype (FP8 KV cache as a lighter alternative to full model quantization), chunked prefill settings, and prefix caching — these are orthogonal optimizations that may solve the problem with less risk than full-model quantization

6. **Config key existence check**: Before writing your response, list every config key/parameter you recommend. For each one, confirm it appeared in a fetched doc or KB result. If you cannot confirm it exists, either (a) WebFetch the config schema, or (b) mark it `[unverified key]`. Never present an unverified config key as a definitive fix.

**Default value check**: Before recommending any parameter value, WebFetch or check KB for the framework's DEFAULT value for that parameter. If your recommended value equals the default, do NOT present it as a fix — the user already has that value. Example: `max_grad_norm=1.0` is the HuggingFace Trainer default; recommending it as a "fix" is a non-fix that wastes the user's time and erodes trust.
6. When providing inspection/debugging code, use the actual framework APIs (e.g., `trainer.get_train_dataloader()` not a manually reconstructed DataLoader) — generic recreations won't reproduce the exact behavior

**Correctness self-test**: Before finalizing, re-read each fix step and ask: "If the user copy-pastes this exactly, will it work?" Check: (a) import statements present, (b) variable names match user's code, (c) config keys verified against fetched docs, (d) no placeholder values like `<your_value>`. If any check fails, fix it before outputting.
**No stub code rule**: Never leave `pass`, `# TODO`, or `...` in code you present as a fix. Every function body must be implemented. If you cannot implement a step (e.g., optimizer state surgery), either (a) write the full implementation using documented APIs, or (b) explicitly state "this requires manual inspection of your checkpoint format" — do NOT present a stub as if it's a working script.
7. **Parameter class verification**: Before referencing any config parameter (e.g., `max_seq_length`), verify which class it belongs to (e.g., `SFTTrainer` vs `TrainingArguments`). Misattributing a parameter to the wrong class is a correctness error — the user will put it in the wrong place and it will silently do nothing.
8. **Memory arithmetic for OOM fixes**: Always include explicit per-GPU memory math (model params + optimizer states + activations + KV cache + overhead) so the user can verify the fix will actually fit. Show the calculation, don't just assert "this will fit".


**MANDATORY for any KV cache calculation**: WebFetch the model's `config.json` from HuggingFace and extract `num_key_value_heads`. Common GQA values: Mistral-7B=8, Llama-3-8B=8, Llama-3-70B=8, Qwen2.5-7B=4. Do NOT use `num_attention_heads`. If you haven't fetched config.json, you CANNOT write KV math.

**Pre-output self-check (mandatory before writing the response below)**:
1. Count your `[PageID]` or `[source](URL)` citations. If zero → you MUST use the Ungrounded format from Phase 1. Stop and rewrite.
2. Count your `[no KB]` tags. If both citation count and `[no KB]` count are zero → your response is broken. Stop and rewrite.
3. Check every "Why it hurts" or "What it does" explanation — does it have a citation? If not, add `[no KB]` and use conditional language.
4. Check that you stated an exact framework version (e.g., `vLLM 0.6.0`) in both Diagnosis and Fix sections, and that all GitHub source links use a version tag (not `main` branch).
5a. Check every quantitative claim ("X× faster", "Y% improvement") has a citation or `[no KB]` tag.
5b. Check every config key/parameter name you recommend — did it appear in a fetched doc or KB result? If not, mark `[unverified key]`.
5. Scan all CLI commands for duplicate flags — same flag appearing twice is a correctness error.
5a. **Consolidation check**: If your response contains multiple config/launch blocks (e.g., "Phase 1" and "Phase 2"), verify they don't set the SAME flag to DIFFERENT values. Merge into ONE final launch command in the Consolidated Config section. Duplicate or contradictory flags across blocks is the #1 serving-fix correctness error.
6. Check every fix step has a concrete value (not "try reducing X" but "set X=N").
7. Check every recommended parameter value against the framework default — if they match, you're recommending a non-fix. Remove it or pick a different value.
8. If you computed KV cache math, verify you used `num_key_value_heads` (not `num_attention_heads`) — GQA models differ by 4-8×.
9. Check that every version number you cited came from a fetched source (PyPI, releases page) — not from memory.
10. Check that you fetched a SEPARATE PyPI/releases page for EACH framework in your response. If your diagnosis mentions DeepSpeed AND Transformers but you only fetched DeepSpeed's PyPI, the Transformers version is fabricated — go fetch it now.

Present the result:

```
## Diagnosis

**Root cause**: [one sentence] [PageID or `[no KB]` if unavailable]
**Confidence**: High / Medium / Low (no `[PageID]` → max Medium; no KB at all → must be Low)
**Version**: [exact framework version, e.g., vLLM 0.6.0] [PageID or `[no KB]`]
**Evidence**: [what in the logs/symptoms points to this]
**User setup**: [echo back: model name, GPU type/count, framework + version, batch size, seq len — from user's message]

### Fix
1. [specific action with exact config/code — include exact values, not ranges] [PageID or `[no KB]`]
2. [verification step — how to confirm the fix worked]

**Actionability rule**: Every fix step must be copy-paste-ready. "Increase X" is not a fix — "Set X=128 (was 64)" is. "Try a smaller batch size" is not a fix — "Set per_device_train_batch_size=2 with gradient_accumulation_steps=8" is. If you write a fix step that contains the word "try" or "consider" without a concrete value, rewrite it.

**Before→after rule**: Every parameter change MUST show: `# was: <old_value> → now: <new_value> (reason)`. This applies in BOTH the per-step explanation AND the consolidated config block. Reviewers check actionability by counting concrete before/after pairs — implicit changes score 0.

**Actionability scoring**: Reviewers count: (1) concrete numeric values in fix steps, (2) before→after pairs, (3) copy-paste-ready code blocks. Each fix step missing a concrete value costs you points. "Reduce X" = 0 points. "Set X=Y (was Z)" = 1 point. Target: every fix step scores 1.

### Consolidated Config
```
[copy-paste-ready final config with ALL changes applied in one block — user should need to copy exactly ONE config block, not assemble pieces from multiple sections. Include comments showing what changed and why: `# was 256, lowered to prevent over-batching during speculation`]
```

### If That Doesn't Work
- Alternative cause: [what else it could be] → Try: [next diagnostic step] [PageID]

### Prevention
- [runnable guard #1 — exact threshold, copy-pasteable] [PageID]
- [runnable guard #2 — different failure vector] [PageID]
- **Minimum 2 prevention items, each with a numeric threshold.** "Monitor X" without a threshold scores 0 on prevention.

**Serving-specific prevention**: For serving/inference fixes, MUST include: (1) a load-test command showing concurrent request handling, (2) a memory fragmentation guard (e.g., periodic `torch.cuda.empty_cache()` or `gc.collect()`), (3) a latency monitoring threshold (e.g., `assert p99 < 3 * p50`). For vLLM specifically: check chunked prefill interaction with speculative decoding, and log `gpu_cache_usage_perc` with an alert at >85%.

**Prevention quality gate**: Each prevention item MUST be a runnable code snippet with an exact numeric threshold AND a citation. Template: `if <metric> <op> <threshold>: <action>  # [source](URL) or [PageID]`. Reviewers score prevention by counting: (1) runnable snippets (not prose), (2) numeric thresholds, (3) citations. All three required per item. "Monitor loss curves" = 0/3. "`if loss > 2 * rolling_mean: save_checkpoint()` [PageID]" = 3/3. Each item must be copy-pasteable and grounded.

**Prevention examples by failure type** (use these as templates):
- OOM: `assert torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory < 0.90, "Memory usage >90%"`
- Expert collapse: `if expert_util.min() < 0.05: save_checkpoint(); raise Alert("Expert {i} utilization <5%")`
- Loss spike: `if loss > 2 * rolling_avg_loss[-100:].mean(): save_checkpoint(); log.warning("Loss spike at step {step}")`
- Serving latency: `assert p99_latency_ms < 2 * p50_latency_ms, "Tail latency ratio >2× — check prefill interference"`
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
| **Presenting general knowledge as KB-grounded analysis** | KB call failed, model says "I have deep expertise" and writes from memory | BANNED patterns (instant 0 score): "I have deep knowledge", "Let me give you the full analysis directly", "I know X well", "I'm very familiar with X", "KB is unavailable but". This remains the #1 failure mode — t_agent2 used "I have deep knowledge of vLLM internals" and scored 1/3 correctness. STOP → WebFetch 2 URLs → restart with `> Grounding: Web mode`. |

| **Skipping prevention or writing generic prevention** | Prevention section says "monitor loss" or "use checkpoints" — advice so generic it adds zero value | Prevention must cite a specific tool, metric threshold, or config flag that would have caught this failure earlier. E.g., "add `--log-expert-utilization` every 50 steps; if any expert drops below 5%, trigger alert" — not "monitor expert utilization". |
| **Verification script that doesn't match the failure mode** | Serving fix verified with single curl; OOM fix verified with "run and see if it crashes" | Match verification to the failure: serving → concurrent load test with latency percentiles; OOM → memory profiling with `torch.cuda.max_memory_allocated()`; convergence → loss curve over N steps with expected trajectory. |
| **Using total attention heads for KV cache math in GQA models** | Mistral/Llama use GQA with fewer KV heads than query heads — using `num_attention_heads` overcounts KV cache by 2-8× | Always use `num_key_value_heads` from model config. Mistral-7B: 8 KV heads, not 32. Llama-3-8B: 8 KV heads, not 32. |
| **Fix steps without concrete values** | "Try a lower learning rate" or "reduce batch size" without saying to what | Every fix step needs an exact value with reasoning: "Set lr=1e-5 (was 3e-4, ~30× reduction to let aux loss compete with LM loss gradient)". Ranges are acceptable only with a recommended starting point. |

| **Recommending framework defaults as fixes** | "Set max_grad_norm=1.0" when that's already the default | Before recommending ANY parameter value, verify the framework default. If your recommendation equals the default, it's a non-fix. WebFetch the framework's TrainingArguments or config docs to confirm defaults before writing fix steps. |
| **Reducing global LR when a component-specific LR exists** | MoE router collapse, LoRA instability — global LR reduction slows ALL training when only one component needs adjustment | Check if the framework supports component-specific LR (e.g., `router_lr`, `lora_alpha` scaling, parameter group overrides). Prefer targeted LR changes over global ones. "Set router LR=1e-3 (global LR stays 3e-4)" beats "Lower global LR from 3e-4 to 1e-4". |
| **Using version-specific CLI flag syntax without verification** | vLLM `--speculative-config '{JSON}'` only works in recent versions; older versions use `--speculative-model` + `--num-speculative-tokens` as separate flags | WebFetch the framework's CLI arg parser or `--help` output for the user's installed version. When multiple flag syntaxes exist across versions, show BOTH the modern and legacy forms with version cutoffs: "vLLM ≥0.6.2: `--speculative-config`; vLLM <0.6.2: `--speculative-model` + `--num-speculative-tokens`". |
| **Verifying serving fixes with synchronous requests** | Script sends requests one at a time, so it tests single-user latency, not the concurrent load that triggered the problem | Serving verification MUST use `asyncio.gather()` or equivalent to send N concurrent requests matching the user's stated concurrency (e.g., 50 users → 50 concurrent requests). Measure p50/p95/p99 across the batch. A sequential loop that hits the endpoint 50 times in series will show ~1/50th of the real tail latency. |
| **Setting `max-num-seqs` higher than actual concurrency for latency optimization** | Higher `max-num-seqs` means more requests batched together, which INCREASES p99 latency per request due to longer decode iterations | For latency-sensitive serving, set `max-num-seqs` ≤ 2× actual concurrent users (e.g., 50 users → `--max-num-seqs 64-100`, NOT 256). Higher values optimize throughput at the cost of tail latency. State the tradeoff explicitly. |

**Serving benchmark template** (adapt to user's concurrency and endpoint):
```python
import asyncio, aiohttp, time, numpy as np
async def bench(url, n=50, payload={"prompt": "Hello", "max_tokens": 128}):
    async with aiohttp.ClientSession() as s:
        async def req():
            t0 = time.perf_counter()
            async with s.post(url, json=payload) as r: await r.read()
            return time.perf_counter() - t0
        lats = await asyncio.gather(*[req() for _ in range(n)])
    lats_ms = np.array(lats) * 1000
    print(f"p50={np.percentile(lats_ms,50):.0f}ms p95={np.percentile(lats_ms,95):.0f}ms p99={np.percentile(lats_ms,99):.0f}ms")
    assert np.percentile(lats_ms, 99) < 3000, "p99 > 3s SLA breach"
asyncio.run(bench("http://localhost:8000/v1/completions"))
```
Include this template (with user's actual concurrency level and SLA) in EVERY serving fix response. Adjust `n` to match the user's stated concurrent users.


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
