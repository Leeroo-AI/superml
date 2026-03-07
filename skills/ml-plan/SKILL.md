---
name: ml-plan
description: Use when the user wants an implementation plan, architecture design, or multi-step ML pipeline — "build X", "implement X", "design X", "set up X"
---

# ML Planning

Turn goals into validated, runnable implementation plans grounded in framework documentation.

## Grounding

**Detect mode:** Try one KB call (`build_plan` or `search_knowledge`). If it returns results → **KB mode**. If it fails for ANY reason (auth error, no results, timeout) → **Web mode**. There is no third option.



**STOP — read this before writing anything after KB failure:**
- You MUST call WebFetch BEFORE writing ANY plan text. Not after. Not alongside. BEFORE.
- Banned phrases (if you write any of these, you have failed): "established practice", "deep familiarity", "from my training data", "based on known conventions", "well-known", "widely used", "standard approach", "common practice", "I know this from", "I have expertise in", "I know these frameworks", "I'm familiar with", "based on my knowledge", "from experience", "deep knowledge", "I have deep", "I have extensive", "I'm knowledgeable", "I'm well-versed"
- If you catch yourself writing plan content without a WebFetch URL in hand, STOP and fetch.
- **Hard gate: zero external calls = zero plan output. No exceptions, no rationalizations.**

**KB mode:** Call `build_plan` → `review_plan` → `search_knowledge` for gaps. Cite as `[PageID]`.

**Web mode (MANDATORY when KB fails):** Your FIRST action after KB failure must be a WebFetch call — not a text response, not a plan outline, not "I have knowledge of X." Decompose goal into steps → WebFetch official docs for EACH step → cite as `[source](URL#section-anchor)` with specific section paths. **Minimum: 1 WebFetch per plan step.** Start response with: `> Grounding: Web mode — citations from official docs.`

**Self-check after KB failure:** Count your WebFetch calls before writing ANY plan text. If the count is 0, you are about to fail. Stop and fetch. "I know this topic well" is NOT a substitute for WebFetch — it is the exact failure mode this rule prevents.

**Hard rule:** Every code block needs a `[source](URL)` or `[PageID]` citation from a fetch you actually made this session. No exceptions. Every `Class(kwarg=...)` must cite the doc page confirming that kwarg exists. If you write `Agent(input_description=...)`, you must have fetched the Agent class docs and confirmed `input_description` is a real parameter — not `tool_description_override` or something else.

**Architecture diagram rule (web mode):** Do NOT draw architecture diagrams, flow charts, or system designs until you have fetched docs for every component in the diagram. An architecture diagram without grounding is a guess dressed up as a plan. Fetch first, diagram second.

**Citation enforcement (both modes):** Every code block that calls a library API MUST have an inline comment citing the source: `# [PageID]` or `# [source](URL)`. Every class instantiation must cite the doc page where its kwargs are listed. Uncited API calls are treated as unverified guesses. When citing, always include the **library version** (e.g., `peft==0.12.0 [PageID]`). **Cross-reference rule:** When a plan combines multiple libraries (e.g., PEFT + Transformers, RAGAS + LangChain), verify version compatibility between them — fetch each library's install docs to confirm compatible version ranges. State the verified combination explicitly in Prerequisites.

**Web mode URL registry:**
**Citation anchor rule (web mode):** Link to the specific API class/function section, NOT the library homepage. Use `#anchor` paths — e.g., `https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments` not `https://docs.vllm.ai`. A homepage link is not a citation.
- HF Transformers/PEFT/TRL: `https://huggingface.co/docs/{transformers,peft,trl}`
- Axolotl: `https://github.com/axolotl-ai-cloud/axolotl`
- DeepSpeed: `https://www.deepspeed.ai/docs`
- vLLM: `https://docs.vllm.ai`
- Model cards: `https://huggingface.co/{org}/{model}` — ALWAYS fetch for architecture-specific layer names, config keys, and training recipes
- Anthropic/Claude API: `https://docs.anthropic.com` — NOT `platform.claude.com` (does not exist). SDK reference: `https://docs.anthropic.com/en/docs/build-with-claude`
- OpenAI: `https://platform.openai.com/docs/api-reference`
- LangChain/LangGraph: `https://python.langchain.com/docs`, `https://langchain-ai.github.io/langgraph`

## The Iron Law

```
NO IMPLEMENTATION WITHOUT A VALIDATED PLAN FIRST
```

A plan that hasn't been reviewed against documentation is a guess. Guesses waste GPU hours.

## Phases

### Phase 1: Understand — Build the Plan

**KB mode:** Call `build_plan(goal, constraints?)` IMMEDIATELY with the user's stated goal.

**Web mode:** Do NOT write any step content yet. First:
⚠️ **If you skipped WebFetch and started writing plan text, STOP NOW. Go back and fetch docs. No amount of domain knowledge substitutes for a URL citation.**
1. List the frameworks/libraries needed (one line each)
2. WebFetch the API reference page for EACH library — do ALL fetches BEFORE writing any plan text
3. For each `Class(kwarg=...)` you plan to use, find its `__init__` signature in the fetched docs and copy the exact parameter names
4. NOW write steps using ONLY the fetched parameter names — if a param isn't in the fetched docs, it doesn't exist
5. **Deprecation check**: For each API pattern you plan to use, verify it's not deprecated in the pinned version. Common traps: `@app.on_event("startup")` → use `lifespan` context manager in FastAPI ≥0.93; `model.generate()` kwargs change across transformers versions. If the fetched docs show a deprecation warning, use the replacement.
5. For multi-provider plans (e.g., Claude + OpenAI + local models), fetch EACH provider's SDK docs SEPARATELY — do NOT assume shared API patterns

**Web mode minimum calls:** Count your steps. You must make AT LEAST that many WebFetch calls. If you have 6 steps, make 6+ fetches. Do NOT batch multiple steps into one fetch unless they use the exact same doc page.



In both modes:
- Use the user's exact words as the goal
- Include any hardware, framework, latency, or scale constraints they mentioned
- Do NOT wait for more information — use what you have now
- For every SDK/framework in the plan, pin the version and cite where you verified its API. KB mode: `search_knowledge("[library] [version] API")`. Web mode: WebFetch the library's changelog or API reference for that version.

**Gate**: You have a documentation-grounded plan with numbered steps and validation criteria before proceeding.

> **Import verification rule**: For every `import` or `from X import Y` in the plan, verify the exact import path against the library's **installed version** docs. Do NOT present uncertain imports — if you cannot confirm the path, WebFetch or `search_knowledge` the module's API reference. If still uncertain, provide the verified fallback import AND a one-line check: `python -c "from X import Y"` so the user catches it before running the full script.

> **Citation rule**: Every step in the plan MUST include at least one citation. KB mode: `[PageID]` from `build_plan` output. Web mode: `[source](URL)` from the doc page you verified against. If a step has no citation, look it up before presenting.
>
> **Version rule**: When citing a library or framework, include the **pinned version** next to the citation. This lets the user verify the citation matches their dependency versions.

### Phase 2: Validate — Review and Gap-Fill

**KB mode:**
1. Call `review_plan(proposal, goal)` with the plan from Phase 1 to catch risks
2. Identify the 2-4 most uncertain steps
3. Call `search_knowledge` in **parallel** for each gap (cite every result as `[PageID]` in the final plan)

**Web mode:**
1. Self-review: walk through each step and ask "would this actually work on the stated hardware?"
2. Identify the 2-4 most uncertain steps
3. WebFetch official docs in **parallel** for each gap — framework API details, config formats, known pitfalls, memory estimates

In both modes, verify for each gap:
   - Framework-specific API details and correct import paths
   - Config format requirements
   - Known pitfalls or gotchas
   - **Current API version** — verify exact function signatures for the pinned version
   - Memory/compute estimation for the specific hardware
   - **Compatibility caveats** — features that are version-dependent or have known workarounds. For every `kwarg` or `scheduler_kwargs` dict passed to a Trainer/config, verify it exists in the **pinned version** — not just the latest. If uncertain, provide a version check: `assert version.parse(lib.__version__) >= version.parse("X.Y.Z")`
   - **API method existence** — for EVERY `client.method_name()` call, verify that method exists in the SDK docs for the pinned version. Do NOT assume methods from one provider exist on another (e.g., OpenAI's `messages.parse()` does not exist in Anthropic's SDK). WebFetch the SDK reference page and find the exact method signature.

   - **Internal consistency** — verify that model names, variable names, and config values are identical between comments and code, between different steps, and between budget tables and implementation. If a comment says `gpt-4o-mini` but the code uses `claude-sonnet-4-20250514`, that's a bug. Scan all code blocks for mismatches before presenting.
   - **Stateful logic verification** — for any retry counter, conversation history, or accumulated state: trace the identity/key used to track it. `id(obj)` changes per object creation — use a stable key (tool name string, step index). Verify state is never silently discarded: if a loop resets a list or dict, confirm that's intentional, not a bug that loses prior context.
   - **Framework performance flags** — for each framework, WebFetch the performance tuning guide and include ALL recommended flags. E.g., Megatron-LM: `--sequence-parallel`, `--use-distributed-optimizer`, `--overlap-grad-reduce`, `--overlap-param-gather`. Missing one flag can halve throughput on long runs. Do NOT rely on memory for flag names — fetch the docs.
   - **Numerical estimate verification** — for ANY throughput, memory, or time estimate, show the full arithmetic. Cross-check against published benchmarks (fetch them). Common trap: underestimating tokens/sec/GPU for smaller models on powerful hardware (e.g., 3B on H100 processes 8,000-15,000 tok/s/GPU, not 3,500). **Self-consistency check**: if you claim X% MFU in text, verify your tokens/sec numbers actually imply that MFU. Formula: `MFU = (6 × params × tokens_per_sec) / (GPU_FLOPS × num_GPUs)`. If the numbers don't match, fix them before presenting.
   - **KV cache math**: Always compute per-token KV cache as `2 × num_layers × num_kv_heads × head_dim × bytes_per_param`. For GQA/MQA models, use the actual KV head count (not query heads). For MoE models, attention layers are shared across experts — use the full layer count, not expert count. Show the per-token size AND total budget in the plan.

**Gate**: Every step in the plan has either documentation confirmation or an explicit "verify during dry-run" flag.

> **Hardware-fit gate**: Before presenting, verify quantization/precision choices AND throughput estimates match the hardware. Memory: `model_params × bytes_per_param + optimizer_overhead + activation_memory ≤ 0.85 × total_VRAM`. If bf16 fits, do NOT use QLoRA/4-bit. Throughput: fetch published benchmarks for the specific GPU+model-size combination — do NOT estimate tokens/sec from first principles without a benchmark anchor. Show all math in Prerequisites.

> **Code correctness gate**: Before presenting any code block, mentally trace it with a concrete input. Check: (1) variable names match across lines, (2) return types match what the caller expects, (3) no duplicate class/function definitions across steps, (4) callback/metric functions return the type the framework expects (e.g., `output_schema` ≠ `output_format`), (5) **error handling**: every call to `Runner.run()`, `trainer.train()`, or any external API must be wrapped in try/except with a specific recovery action — not bare `except:`, (6) **state persistence**: if the design needs variables/results to survive across calls (REPL-style execution, multi-turn agents), verify the execution model actually preserves state — `subprocess.run()` per call does NOT persist variables, you need a persistent process or shared store, (7) **security layer verification**: for every sandbox/restriction (import blocking, path restriction, read-only mode), write down one concrete bypass and add a defense — e.g., `__import__` override is bypassed by `importlib`; use `ast.parse` + node-type whitelist instead. If you spot an inconsistency, fix it before presenting.

> **Scale/range consistency gate**: When combining scores from different sources (e.g., deterministic 0-1 metrics with LLM judge 1-5 scores), verify the ranges align. A linear map from [0,1] to [0,5] is NOT the same as [1,5]. Show the mapping formula explicitly and trace one example value through it. Also: when extracting structured scores (e.g., `[[score]]` regex), add a fallback for parse failures — log the raw output and assign a default, don't silently drop the sample.

> **Kwarg verification gate (BLOCKING)**: Before presenting ANY `Class(kwarg=...)` call, you MUST have fetched the class docs page this session. Copy the exact parameter names from the docs. Common traps: `data_collator` not `data_collate_fn`, `tool_description_override` not `input_description`, `compute_metrics` not `metric_fn`. If you cannot confirm a kwarg from a fetched doc, mark it `[UNVERIFIED]`. One wrong kwarg silently ignored = hours wasted.

> **Correctness gate**: For any step that involves an SDK or library API, you MUST verify the import path, function signature, and **exact parameter names**. KB mode: call `search_knowledge("[library] [function] API signature [version]")`. Web mode: WebFetch the API docs page. Do NOT guess APIs from memory.

> **Numerical gate**: For any throughput, memory, or time estimate, show the arithmetic in the plan. Ground estimates in documented benchmarks, not intuition.

### Phase 3: Present — Structured Plan with Validation

Compose the final plan:

```
## Plan: [Goal]

### Overview
[1-2 sentences: what we're building, why this approach]

### Prerequisites
- [ ] Hardware: [specific GPUs, RAM]
- [ ] Dependencies: [packages with **pinned** versions — use `==`, not `>=`. For each dependency, cite the doc page confirming the API you use exists in that version. Verify: import paths, function signatures, and kwarg names change across versions.]
- [ ] Install commands: `pip install` (or equivalent) with ALL packages. Include model download command if applicable (e.g., `huggingface-cli download`, `vllm serve --download-dir`). The user should be able to go from bare machine to running in one copy-paste.
- [ ] Dependencies: minimize — don't add a library for a single function (e.g., `sklearn` for one metric). Inline or use stdlib when trivial.
- [ ] Data: [format, size, location]

### Steps
1. **[Step name]** — [description] [PageID]
   - Code: ```[language]\n[complete, runnable snippet — all imports, all args, all config. User should copy-paste and run with zero edits except paths/keys. For multi-step agents/pipelines: show how state (variables, history, connections) flows between steps — never silently discard intermediate results. NO stubs, NO `NotImplementedError`, NO `pass` placeholders, NO `# TODO` — if you can't write the full implementation, call `search_knowledge` until you can.]\n```
   - Code must handle edge cases inline: empty result sets, dedup of repeated items, type mismatches between steps. If a node appends to a list via reducer, include dedup logic in the node itself — don't defer it to "pitfalls" text.
   - Concrete params: [every hyperparameter, threshold, batch size, model name as a specific value — not "tune this" or "adjust as needed". Show why that value: e.g., `batch_size=4  # 7B model × 4 = ~72GB on 2×A100-80GB`]

0. **Data Setup** — [ingestion, index creation, collection setup — don't assume infra exists]
   - Preprocessing: `[exact shell command or script to transform raw data → training format]`
   - Validate: [query test data, verify row counts / index health — write the actual validation function, not a placeholder]
   - Config: `key: value`
   - Validate: [how to verify this step worked]
   - Eval thresholds: [per-benchmark numeric thresholds — e.g., `MMLU ≥ 0.60, PubMedQA ≥ 0.72, BoolQ ≥ 0.80` — not just "evaluate on standard benchmarks"]
   - Warnings: [what can silently go wrong — e.g., sycophantic judges, silent dtype downcasts, version-specific API breaks, **silently ignored kwargs**, runtime failures without error handling. **Minimum 2 warnings** per step that calls an external API or runs training — one about silent failures, one about correctness risks.]
   - Warnings MUST cover: (a) what happens when the external call returns unexpected format (parse failures, empty responses), (b) what happens when numeric values are at boundary conditions (zero scores, empty sets, division by zero). Don't just warn about big risks — warn about the mundane failures that waste debugging hours.
   - Try/except: ```[language]\ntry:\n    [the operation]\nexcept [SpecificError] as e:\n    [concrete recovery — e.g., checkpoint resume, batch size reduction, fallback model]\n``` — REQUIRED for every step that calls an SDK, runs training, or hits an external service. Not optional.
   - Error handling: [specific exception class to catch, exact recovery action — e.g., `except torch.cuda.OutOfMemoryError: reduce batch_size by half and retry`]
   - Limitations: [what this step does NOT handle — at least one honest limitation per step that calls an external API or uses a framework feature]
   - Time estimate: [wall time + throughput estimate with **show-your-work math** — e.g., `3B params × 6 FLOPs/param × 200B tokens ÷ (32 GPUs × 312 TFLOPS × 0.45 MFU) = X hours`. Never state throughput without the calculation backing it. Include tokens/sec/GPU estimate anchored to a fetched benchmark.]

... [repeat for each step]

N-1. **Export & Deploy** — [merge adapters / export model / package artifacts]
   - Validate: [load exported model, run inference on test input]
   - Artifacts: [what files are produced, where they go]

N. **Inference Smoke Test** — [complete runnable inference script that loads the exported artifact and runs on 2-3 example inputs]
   - Code: ```[language]\n[full inference script — load model, run prediction, print output. Not a stub.]\n```
   - Validate: [compare output against expected baseline — exact string or metric threshold]

**VRAM estimate requirement**: The Prerequisites section MUST include a per-GPU VRAM breakdown: `model_params × bytes_per_param + optimizer_states + activation_memory + KV_cache = total`. Show the math, not just the conclusion. If the total exceeds 85% of GPU VRAM, flag it and adjust the config (reduce batch size, add gradient checkpointing, switch precision).

**Post-assembly citation audit**: Before presenting, count citations. KB mode: minimum 1 `[PageID]` per step AND per code block. Web mode: minimum 1 `[source](URL#specific-section)` per step AND per code block — every URL MUST include a `#fragment` or `/path/to/specific-page` pointing to the exact class, function, or parameter list. Page-level URLs like `https://sbert.net/docs/training/overview.html` fail — use `https://sbert.net/docs/training/overview.html#data-formats` instead. Any step below the floor → fetch/search before presenting. Mark truly uncitable steps `[UNVERIFIED — test in dry-run]`. Also verify: every `Class(kwarg=...)` has a citation confirming that kwarg exists in the pinned version.

**Grounding density check**: After the citation audit, verify: (1) every `Class(kwarg=...)` cites the docs URL confirming those kwargs exist — cite the **specific version's API page**, not the library homepage (e.g., `scipy==1.14.0 bootstrap` → link to `scipy/1.14.0/reference/generated/scipy.stats.bootstrap.html`, not `docs.scipy.org`), (2) every numerical estimate shows the formula AND cites the benchmark, (3) every model ID cites the model card. If any are missing, WebFetch before presenting. A plan with code but no doc links is an unverified guess.

**Launcher note**: Provide launch commands for **both** Slurm (`sbatch`/`srun`) and bare-metal (`torchrun`/`accelerate launch`) when the plan involves multi-node or multi-GPU execution.

**Completeness audit**: Before presenting, verify the plan includes ALL of: (1) data preparation with validation, (2) training/building with error handling, (3) evaluation with numeric thresholds, (4) export/merge of artifacts, (5) inference smoke test with runnable script, (6) deployment artifacts if the plan is for a service (Dockerfile, requirements.txt with pinned versions, environment variables). Any missing stage → add it. An incomplete pipeline wastes more time than a thorough plan.



**Concurrency audit**: If ANY step makes ≥10 independent network calls (LLM judge calls, API requests, batch embeddings), the code MUST use `asyncio.gather` / `concurrent.futures` with a configurable concurrency limit. Serial LLM judge loops over hundreds of samples are unacceptably slow — flag and fix before presenting.

**Specificity audit**: Scan every step for vague language: "adjust as needed", "tune this", "configure appropriately", "use a suitable value". Replace each with a concrete value and show-your-work justification. If you truly cannot determine the value, state the range with a recommended default: `lr=2e-5  # range 1e-5 to 5e-5; 2e-5 is standard for 7B LoRA per [source]`. Also scan for placeholder variables left undefined (e.g., `corpus = ...`, `data = your_data_here`) — either provide a concrete example value or show the exact loading code.

**Throughput/latency claims audit**: Every tokens/sec, latency, or QPS number MUST cite a benchmark source (fetched this session). Format: `~40 tok/s [source](URL#benchmark-section)`. If no benchmark exists for the exact config, state it as a range with upper/lower bounds from the closest benchmarks you did fetch, and mark `[ESTIMATED]`.

### Risks & Mitigations
- Risk: [what could go wrong] → Mitigation: [specific action] [PageID]

### Execution Strategy
- Dry-run: [what to test on 1% of data first]
- Checkpoint: [when to evaluate before continuing]
- Success criteria: [specific metrics with numeric thresholds — e.g., "recall@10 ≥ 0.85 on held-out test set" not just "good retrieval quality"]
```

## After This

Execute in phases: **dry-run on 1% data → 1 epoch → full run.**

- Before running → invoke **ml-verify** to catch config mistakes
- Log the experiment → invoke **ml-experiment** to track hypothesis and results
- If any phase fails → invoke **ml-debug** with the error and plan context
- After results → invoke **ml-iterate** if metrics aren't at target

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Planning without memory estimation | "We'll figure out OOM at runtime" | Estimate GPU memory in the plan. Include per-step memory budget. |
| Missing evaluation step | "We'll evaluate after training" | Build evaluation into the plan — what metrics, what threshold, what data |
| Wrong parallelism for model size | "Just use FSDP for everything" | Check KB for model-size-specific parallelism recommendations |
| No dry-run phase | "The config looks right" | Always plan a 10-step dry-run before committing GPU hours |
| Skipping review_plan | "build_plan gave a good result" | review_plan catches risks that build_plan misses. Always run both. |
| Custom data pipeline when native exists | "I'll write my own Dataset class" | Use the framework's native data loading first (e.g., LLaVA's pipeline, Axolotl's YAML datasets). Custom only when native can't handle your format. Call `search_knowledge("[framework] native data pipeline")` to confirm the native path before writing custom code. |

| No error handling in execution steps | "We'll deal with errors when they happen" | Every step that calls an API or runs training needs try/except with a recovery action (retry, checkpoint resume, graceful exit). |

| Single-launcher execution scripts | "Everyone uses Slurm" | Provide launcher commands for both Slurm (`srun`/`sbatch`) and bare-metal (`torchrun`) setups. Not all clusters use the same scheduler. |
| String parsing for routing decisions | "I'll use StrOutputParser for yes/no" | Use structured output (Pydantic models, tool calls, or JSON mode) for any LLM decision that controls flow — routing, grading, gating. String parsing breaks on minor output variations. |
| Hardcoded thresholds and magic numbers | "I'll tune them later" | Put thresholds, retry limits, and quality gates in a config file (YAML/JSON) from the start. Hardcoded values resist tuning and A/B testing. |
| Statistical methods with wrong assumptions | "temperature=0 measures self-consistency" | Verify that your methodology actually tests what you claim. Near-deterministic sampling can't measure variance; paired vs pooled statistics answer different questions. State assumptions explicitly in the plan. |
| Unverified model/endpoint names | "rerank-v4.0-pro should exist" | Call `search_knowledge("[provider] [model] available models")` for every model name, endpoint, or API version. If the KB can't confirm it exists, mark it `[UNVERIFIED]` and provide a fallback (e.g., `rerank-v3.5` as known-good). |


| Inconsistent model/config names across steps | "I'll fix it later" / "The user will notice" | After writing all steps, scan for model name, variable name, and config value consistency. If Step 2 says `gpt-4o-mini` and Step 3's code uses `claude-sonnet-4-20250514`, that's a bug — fix it before presenting. Budget tables must match the actual API calls in code. |
| Missing edge-case handling in node/step logic | "I mentioned it in the warnings" | Warnings don't fix code. If a retrieval step can return duplicates, the code must deduplicate — `seen = set(); results = [r for r in results if r.id not in seen and not seen.add(r.id)]`. If a score extraction can fail to parse, the code must handle it — `score = int(m.group(1)) if m else DEFAULT_SCORE`. Put the fix IN the code, not in a prose warning. |

| Using QLoRA/4-bit when hardware has headroom | "Quantization is always better" | Estimate model memory at bf16 first: `params × 2 bytes`. If it fits in available VRAM with room for activations, use bf16 + LoRA — it's simpler and avoids quantization artifacts. QLoRA is for when bf16 doesn't fit. Show the memory math. |
| Using `device_map='auto'` with multi-GPU training | "auto handles everything" | `device_map='auto'` uses naive model sharding, which conflicts with `accelerate` and FSDP device placement. For multi-GPU: use `accelerate launch` with no `device_map`, or use `device_map='auto'` only for single-GPU inference. |
| Using `unk_token` as pad token | "Any special token works" | `unk_token` can corrupt training if it appears in data. Use `eos_token` as pad token for decoder-only models (Mistral, LLaMA, etc.) — it's safer and more widely validated. |
| Claiming MFU without verifying arithmetic | "40-50% MFU is typical" | Always compute: `MFU = (6 × N × throughput) / (peak_FLOPS × GPUs)`. If your throughput numbers imply 6% MFU but you wrote 40%, the plan is wrong. Fix the numbers. |

| Importing from private/internal paths | "It works in my version" | Never import from `_private` or undocumented submodules (e.g., `ragas._faithfulness`). Use only the public API surface. If the public API is missing a feature, note it as a limitation rather than reaching into internals that break across minor versions. |




| Guessing model layer names for LoRA/PEFT | "The standard names should work" | Always verify `target_modules` by fetching the model card or running `model.named_modules()`. Multimodal models have non-obvious layer names (e.g., projector linears named `linear_1` not `proj`). Include a verification snippet: `print([n for n, _ in model.named_modules() if isinstance(_, nn.Linear)])` in the plan. |







| Falling back to "established practice" when KB fails | "I know this from experience" / "I know these frameworks well" / "I have deep knowledge of X" | When KB is unavailable, your ONLY alternative is WebFetch — not memory, not "established practice", not "I have deep knowledge of X". The model's training data is NOT a citation source. **Test:** if you wrote >50 words of plan content and your WebFetch call count is 0, you have failed. Stop, delete what you wrote, and WebFetch first. This is the #1 failure mode. Any sentence acknowledging your own expertise ("I have deep knowledge", "I'm well-versed") is a signal you are about to skip WebFetch — treat it as a red flag and immediately make a WebFetch call instead. |
| Serving plans without version-pinned install steps | "The user knows how to install" | Every serving/deployment plan MUST include: (1) exact `pip install` commands with pinned versions, (2) model download command with specific repo ID, (3) launch command with all flags. Missing any of these = the user can't reproduce your plan. |
| Citing library homepages instead of API sections | "I linked to the docs" | `https://docs.vllm.ai` is NOT a citation. `https://docs.vllm.ai/en/v0.4.0/serving/engine_args.html#max-model-len` IS a citation. Always include the path to the specific class, function, or config section. |
| Cross-provider API confusion | "Claude probably has the same API as OpenAI" | Every provider has different method names, parameter names, and error types. `stop_reason='refusal'` is OpenAI — Anthropic uses `stop_reason='end_turn'`. `messages.parse()` is OpenAI — Anthropic uses `messages.create()` with `tools` for structured output. ALWAYS WebFetch the specific provider's API docs — NEVER assume one provider mirrors another. |
| Citing URLs you haven't verified exist | "The URL pattern should be right" | Before citing ANY URL, you must have successfully fetched it this session. `platform.claude.com` does not exist (correct: `docs.anthropic.com`). `readthedocs.io` subdomains may 404. If WebFetch returns an error, do NOT cite that URL — find the real one. |

## Examples

**"Fine-tune Qwen2.5-7B with QLoRA on 2xA100"**

KB mode:
1. `build_plan("QLoRA fine-tuning Qwen2.5-7B", "2xA100 80GB, instruction tuning dataset")`
2. `review_plan(plan_output, "QLoRA fine-tuning Qwen2.5-7B")`
3. Parallel: `search_knowledge("Qwen2.5 QLoRA target_modules config Axolotl")`, `search_knowledge("QLoRA memory estimation 7B model 2xA100")`

Web mode:
1. Decompose: environment setup → data prep → QLoRA config → training → eval → export
2. WebFetch `https://huggingface.co/docs/peft` for LoRA config params
3. WebFetch `https://github.com/axolotl-ai-cloud/axolotl` for Qwen2.5 example configs
4. Self-review: verify memory math, check target_modules against model architecture
5. Present plan with `[source](URL)` citations per step

**"Design a RAG system with hybrid retrieval"**

KB mode:
1. `build_plan("RAG system with hybrid vector+BM25 retrieval", "FastAPI, ChromaDB, production-ready")`
2. `review_plan(plan_output, "Hybrid RAG system")`
3. Parallel: `search_knowledge("ChromaDB hybrid retrieval BM25 integration")`, `search_knowledge("RAG evaluation metrics recall@k RAGAS")`

Web mode:
1. Decompose: data ingestion → index creation → retrieval → reranking → generation → eval
2. WebFetch ChromaDB docs for hybrid search API
3. WebFetch RAGAS docs for evaluation metrics
4. WebFetch LangChain docs for retriever integration patterns
5. Present plan with runnable code per step — no pseudocode, no "fill in here"
