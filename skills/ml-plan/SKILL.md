---
name: ml-plan
description: Use when the user wants an implementation plan, architecture design, or multi-step ML pipeline — "build X", "implement X", "design X", "set up X"
---

# ML Planning

Turn goals into validated, runnable implementation plans grounded in framework documentation.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools (`search_knowledge`, etc.) are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode** for the rest of this conversation.

**KB mode:** Call `build_plan` → `review_plan` → `search_knowledge` for gaps. Cite as `[PageID]`.

**Web mode:** Decompose goal into steps manually → WebFetch official framework docs per step to verify APIs, configs, and params → self-review against docs. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs.`

**Web mode URL registry:**
- HF Transformers/PEFT/TRL: `https://huggingface.co/docs/{transformers,peft,trl}`
- Axolotl: `https://github.com/axolotl-ai-cloud/axolotl`
- DeepSpeed: `https://www.deepspeed.ai/docs`
- vLLM: `https://docs.vllm.ai`
- LangChain/LangGraph: `https://python.langchain.com/docs`, `https://langchain-ai.github.io/langgraph`

## The Iron Law

```
NO IMPLEMENTATION WITHOUT A VALIDATED PLAN FIRST
```

A plan that hasn't been reviewed against documentation is a guess. Guesses waste GPU hours.

## Phases

### Phase 1: Understand — Build the Plan

**KB mode:** Call `build_plan(goal, constraints?)` IMMEDIATELY with the user's stated goal.

**Web mode:** Decompose the goal into numbered steps yourself, then WebFetch the official docs for each step's framework to verify APIs, configs, and parameters. For each step:
1. Identify the framework/library involved
2. WebFetch its documentation page for the specific feature
3. Verify API signatures, config format, and required parameters
4. Write the step with verified information

In both modes:
- Use the user's exact words as the goal
- Include any hardware, framework, latency, or scale constraints they mentioned
- Do NOT wait for more information — use what you have now

**Gate**: You have a documentation-grounded plan with numbered steps and validation criteria before proceeding.

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
   - **Compatibility caveats** — features that are version-dependent or have known workarounds

**Gate**: Every step in the plan has either documentation confirmation or an explicit "verify during dry-run" flag.

> **Code correctness gate**: Before presenting any code block, mentally trace it with a concrete input. Check: (1) variable names match across lines, (2) return types match what the caller expects, (3) set/list operations use the right identity check (`==` not `id()`), (4) no duplicate class/function definitions across steps, (5) callback/metric functions return the type the framework expects (e.g., DSPy metrics must return `bool` when `trace is not None` for bootstrapping; structured output params must match the SDK — `output_schema` ≠ `output_format`). If you spot an inconsistency, fix it before presenting.

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
- [ ] Dependencies: [packages with **pinned** versions — use `==`, not `>=`. Verify API compatibility: import paths and function signatures change across major versions]
- [ ] Dependencies: minimize — don't add a library for a single function (e.g., `sklearn` for one metric). Inline or use stdlib when trivial.
- [ ] Data: [format, size, location]

### Steps
1. **[Step name]** — [description] [PageID]
   - Code: ```[language]\n[complete, runnable snippet — all imports, all args, all config. User should copy-paste and run with zero edits except paths/keys. For multi-step agents/pipelines: show how state (variables, history, connections) flows between steps — never silently discard intermediate results. NO stubs, NO `NotImplementedError`, NO `pass` placeholders, NO `# TODO` — if you can't write the full implementation, call `search_knowledge` until you can.]\n```
   - Concrete params: [every hyperparameter, threshold, batch size, model name as a specific value — not "tune this" or "adjust as needed". Show why that value: e.g., `batch_size=4  # 7B model × 4 = ~72GB on 2×A100-80GB`]

0. **Data Setup** — [ingestion, index creation, collection setup — don't assume infra exists]
   - Preprocessing: `[exact shell command or script to transform raw data → training format]`
   - Validate: [query test data, verify row counts / index health — write the actual validation function, not a placeholder]
   - Config: `key: value`
   - Validate: [how to verify this step worked]
   - Warnings: [what can silently go wrong — e.g., sycophantic judges, silent dtype downcasts, version-specific API breaks. At least one warning per step that calls an external API or runs training.]
   - Error handling: [specific exception class to catch, exact recovery action — e.g., `except torch.cuda.OutOfMemoryError: reduce batch_size by half and retry`]
   - Limitations: [what this step does NOT handle — e.g., "blocklist does not prevent all write methods", "sandbox has no internet access". At least one honest limitation per step that involves security, sandboxing, or external services.]
   - Time estimate: [wall time + throughput estimate with **show-your-work math** — e.g., `3B params × 2 FLOPs/param × 200B tokens ÷ (32 GPUs × 312 TFLOPS × 0.45 MFU) = X hours`. Never state throughput without the calculation backing it.]

... [repeat for each step]

N. **Export & Deploy** — [merge adapters / export model / package artifacts]
   - Validate: [load exported model, run inference on test input]
   - Artifacts: [what files are produced, where they go]

**Launcher note**: Provide launch commands for **both** Slurm (`sbatch`/`srun`) and bare-metal (`torchrun`/`accelerate launch`) when the plan involves multi-node or multi-GPU execution.

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
| Using outdated or misremembered APIs | "The tutorial code works" / "I've seen this param before" | `search_knowledge` for the **pinned version's** API. Verify exact parameter names, not just function names — e.g., `output_schema` ≠ `output_format`. If two initialization paths exist (e.g., `Trainer` vs manual loop), pick ONE and remove the other. |
| No error handling in execution steps | "We'll deal with errors when they happen" | Every step that calls an API or runs training needs try/except with a recovery action (retry, checkpoint resume, graceful exit). |
| Single-launcher execution scripts | "Everyone uses Slurm" | Provide launcher commands for both Slurm (`srun`/`sbatch`) and bare-metal (`torchrun`) setups. Not all clusters use the same scheduler. |
| String parsing for routing decisions | "I'll use StrOutputParser for yes/no" | Use structured output (Pydantic models, tool calls, or JSON mode) for any LLM decision that controls flow — routing, grading, gating. String parsing breaks on minor output variations. |
| Hardcoded thresholds and magic numbers | "I'll tune them later" | Put thresholds, retry limits, and quality gates in a config file (YAML/JSON) from the start. Hardcoded values resist tuning and A/B testing. |
| Statistical methods with wrong assumptions | "temperature=0 measures self-consistency" | Verify that your methodology actually tests what you claim. Near-deterministic sampling can't measure variance; paired vs pooled statistics answer different questions. State assumptions explicitly in the plan. |
| Unverified model/endpoint names | "rerank-v4.0-pro should exist" | Call `search_knowledge("[provider] [model] available models")` for every model name, endpoint, or API version. If the KB can't confirm it exists, mark it `[UNVERIFIED]` and provide a fallback (e.g., `rerank-v3.5` as known-good). |
| Presenting code sketches as plans | "The user can fill in the rest" | Every code block in the plan must be complete and runnable. If you can't write the full code, call `search_knowledge` until you can. Incomplete snippets waste more time than thorough planning. |

| Importing from private/internal paths | "It works in my version" | Never import from `_private` or undocumented submodules (e.g., `ragas._faithfulness`). Use only the public API surface. If the public API is missing a feature, note it as a limitation rather than reaching into internals that break across minor versions. |
| Plan steps without citations | "The KB didn't have this topic" | If `build_plan` didn't cite a step, call `search_knowledge` for it. If the KB truly has no info, mark the step `[UNVERIFIED — test in dry-run]` so the user knows the risk. |

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
