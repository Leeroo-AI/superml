---
name: ml-research
description: Use when the user wants to understand an ML/AI topic, compare approaches, or survey framework capabilities — "how does X work?", "compare X vs Y"
---

# ML Research

Deep-dive into ML topics using verified framework documentation, not stale training data.

## Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools are available. If they return results, use **KB mode**. If unavailable or auth fails, use **Web mode**.

**CRITICAL SELF-CHECK before Phase 1**: If KB auth fails, your FIRST action must be WebFetch — not text output. Do NOT write any prose, explanation, or "I know this topic" before completing at least 3 WebFetch calls. The phrase "I know this topic well" appeared in a real failure and is a disqualifying error.

**KB mode:** Call `search_knowledge` (2-4 parallel queries) → `get_page` on top results → synthesize. Cite as `[PageID]`.

**Web mode:** WebFetch official docs (3-5 pages) → WebFetch GitHub README/examples → WebFetch model cards for each model mentioned → synthesize. Cite as `[source](URL)`. Start response with: `> Grounding: Web mode — citations from official docs.`

**Web mode is NOT optional.** If KB auth fails, you MUST immediately switch to web mode and fetch real documentation. Do not proceed without at least 3 WebFetch calls returning real content.

**ZERO-TOLERANCE RULE**: If your first instinct is "I know this topic well enough" — that is the exact moment you MUST ground. The phrases "well within my training knowledge", "I can answer this from memory", and "this is a well-known topic" are BANNED. They are the #1 failure mode of this skill. Every response requires fetched sources, no exceptions.

**Web mode URL registry:**
- HF Transformers/PEFT/TRL: `https://huggingface.co/docs/{transformers,peft,trl}`
- vLLM: `https://docs.vllm.ai`
- DeepSpeed: `https://www.deepspeed.ai/docs`
- LangChain: `https://python.langchain.com/docs`
- LangGraph: `https://langchain-ai.github.io/langgraph`
- RAGAS: `https://docs.ragas.io`
- PyTorch: `https://pytorch.org/docs/stable`
- vLLM (speculative decoding, serving): `https://docs.vllm.ai/en/latest/features/spec_decode.html`
- HF TGI: `https://huggingface.co/docs/text-generation-inference`
- mergekit: `https://github.com/arcee-ai/mergekit` (README + `mergekit/_data/`)  
- FAISS: `https://github.com/facebookresearch/faiss/wiki`
- sentence-transformers: `https://sbert.net/docs`

## The Iron Law

```
NO CLAIMS WITHOUT KB VERIFICATION
NO RESPONSE WITHOUT GROUNDING — KB OR WEB
```

Your training data is months old. The KB has current framework docs. When the two disagree, the KB wins.

**Hard stop**: If KB auth fails, switch to web mode IMMEDIATELY — do not tell the user grounding is unavailable unless WebFetch also fails. Only if BOTH KB and WebFetch fail should you tell the user and refuse to answer. Never answer from training data alone.

**Enforcement sequence**: KB fails → WebFetch 3+ doc pages → synthesize. There is no path from "KB fails" to "answer from memory".

## Phases

### Phase 1: Multi-Angle Search

**KB mode:** Launch 2-4 `search_knowledge` calls in **parallel** with different angles:
- Core concept / mechanism
- Framework-specific implementation
- Configuration and usage patterns
- Performance characteristics or tradeoffs

**Web mode:** WebFetch 3-5 official documentation pages covering different angles:
- The framework's main docs page for the feature
- GitHub README or examples directory
- Any dedicated guide or tutorial page
- Comparison or migration guide (if applicable)
- **Model cards or API references** for each specific model/tool being discussed — one WebFetch per model minimum

**Web mode URL construction**: Use the URL registry above. Build specific URLs by appending path segments (e.g., `https://docs.vllm.ai/en/latest/features/spec_decode.html`). If a URL 404s, try the parent path or search the docs index page. Do NOT invent URLs — only fetch URLs you constructed from the registry or found linked in a fetched page.

**Gate**: You have sources covering at least 2 distinct angles on the topic before synthesizing. **For comparison tasks**: you MUST fetch a source for EACH item being compared (each model, each method, each library). A single overview page is insufficient — individual model cards, method docs, or API references are required.

**Hard gate**: If you have ZERO fetched sources (no KB results AND no WebFetch results), you MUST NOT proceed to Phase 2. Go back and fetch sources. Count your citations — if the count is 0, you have not done Phase 1.

**Self-audit**: Before moving to Phase 2, count your tool calls. In KB mode: at least 2 `search_knowledge` calls completed. In Web mode: at least 3 `WebFetch` calls completed with real content returned. If you have not met these minimums, STOP and make more calls. Do not proceed based on what you "already know".

> **Citation density target**: Aim for 10+ unique citations in the final answer. KB mode: `[PageID]` citations, call `get_page` on at least 3 top results. Web mode: `[source](URL)` citations from distinct doc pages. **If you have fewer than 5 unique citations after Phase 2, STOP and fetch more sources before synthesizing.**

### Phase 2: Expand and Resolve

**KB mode:**
1. Call `get_page` on the most relevant `[PageID]` citations — prioritize pages with code examples, edge cases, or quantitative comparisons
2. If sources disagree, note which is newer or more framework-specific
3. If there's a gap, run one more targeted `search_knowledge`

**Web mode:**
1. For the most relevant pages, WebFetch specific sub-pages (API reference, config reference, changelog)
2. If sources disagree, check the changelog or release notes for the latest info
3. If there's a gap, WebFetch GitHub issues or discussions for the topic

**Gate**: Every factual claim, config value, and table cell has a citation. CLI flags, parameter names, and config keys must appear verbatim in a documentation source — do not invent flags or options from memory.

**Numeric claims gate**: Any quantitative claim (perplexity scores, memory usage, throughput numbers, speedup ratios) MUST have a citation to a benchmark or doc page. Do not estimate numeric values from training data — if you cannot find a source for the number, write `[needs benchmark]` instead of guessing.

> **Speedup and performance tables**: Tables with columns like "Expected Speedup" or "Throughput" are HIGH-HALLUCINATION zones. Every numeric cell in such tables must either (1) cite a specific benchmark paper/blog with the number, or (2) be marked `[needs benchmark]`. Do not present ranges like "2.0–2.5x" without a cited source — these look precise but are invented.

> **Identifier verification (GATE — do not proceed past Phase 2 without this)**: Before including any model name, API method signature, CLI flag, or config field name, confirm the **exact string** appears in a KB page or fetched doc. If you cannot find verbatim confirmation, call `search_knowledge` or WebFetch the relevant API reference. If still unconfirmed, flag it as `⚠️ UNVERIFIED — check docs` in bold rather than presenting it as fact. **Never invent parameter names like `int8_mask` or assume a model works with a wrapper (e.g., SentenceTransformer) without confirming compatibility in the docs.**

> **Model spec verification (GATE)**: Context window sizes, embedding dimensions, supported languages, and max token limits are HIGH-HALLUCINATION fields. You MUST confirm these numbers from a fetched source — do not rely on training data for model specs. If a model card or doc page was fetched, extract the exact number. If not fetched, WebFetch the model card before including the spec. A wrong context window (e.g., listing 1024 when it's 8192, or 32K when it's 16K) invalidates the entire comparison table.

> **Config schema verification (GATE)**: Before showing any tool/library config file (mergekit YAML, DeepSpeed JSON, vLLM args), WebFetch or search_knowledge for the config schema or a working example. Verify: (1) top-level keys vs nested keys — don't put a top-level field inside a list, (2) required vs optional fields, (3) exact field names — no invented fields. Example failure: mergekit's `base_model` is a top-level key, NOT an entry in the `models:` array.

### Phase 3: Synthesize

Compose a structured answer:

**Completeness rules:**
- Code must be **runnable as-is**: include imports, CLI arg parsing, and shebang lines — never reference a hypothetical `train.py` without providing it
- Pin versions: `pip install mergekit==0.3.1`, not `pip install mergekit`
- Include **deployment scaffolding** when the use case implies production: Dockerfile, docker-compose.yml, or systemd unit — not just the Python code
- When recommending a library/model, verify the **integration path** works (e.g., does model X actually load with library Y's API? Check the docs, don't assume)
- When showing **comparison tables**, every cell with a number (context window, dimensions, cost, throughput) must have a citation. If a cell has no citation, the number is suspect — fetch a source or mark `⚠️ UNVERIFIED`
- If showing a multi-file pipeline, include an orchestration script that ties files together
- Every table cell containing a factual claim needs a `[PageID]`
- Every **parameter name, CLI flag, and config key** in tables must be copy-paste-correct from docs — one wrong flag wastes hours
- Web mode citations must include the **doc version or date**: `[source](URL)` is insufficient — use `[source: vLLM v0.4.1 docs](URL)` or `[source: retrieved 2026-03](URL)` so the user knows the freshness of the info
- **Citation authenticity**: Only cite URLs you actually fetched via WebFetch in this session. Do not reconstruct citations from training data memory — if you didn't WebFetch it, you cannot cite it. Fabricated citations (URLs that look plausible but weren't fetched) are worse than no citations.
- Config files must be **complete** — do not omit fields. If a config has 8 required fields, show all 8 with KB-sourced values
- Config files must be **structurally correct** — fetch a working example from docs/GitHub before writing configs. Verify top-level vs nested keys, list vs scalar values, and required field placement
- Include a **validation/test snippet** (curl test, benchmark script, or smoke test) so the user can verify their setup works

**Final correctness check (GATE — do not output response without this)**:
1. Re-read every parameter name, CLI flag, and config key in your response. Is each one copied verbatim from a fetched source? If not, mark `⚠️ UNVERIFIED`.
2. Re-read every number in tables. Does each have a `[PageID]` or `[source](URL)` citation? If not, add `[needs benchmark]`.
3. Count unique citations. If fewer than 5, you have not grounded sufficiently — go back to Phase 1.

```
## [Topic]

### How It Works
[Core mechanism — what it does and why] [PageID]

### Implementation
[Framework-specific details with code examples] [PageID]
```[language]
[runnable code snippet]
```

### Key Configuration
| Parameter | Recommended Value | Why This Value | Source |
|-----------|-------------------|----------------|--------|
| param | exact_value (not a range) | [rationale with numbers] | [PageID] |

> Fill every row with a **single concrete value**, not a range like "0.1–0.5". If the best value depends on context, show 2 rows (e.g., "for 7B" and "for 70B").

### Version & Compatibility
- _(filled by lines above)_
- Known breaking changes: [if any] [PageID]

### Tradeoffs
| Approach | Pros | Cons | Best for |
|----------|------|------|----------|
| A | ... [PageID] | ... | [use case] |
| B | ... [PageID] | ... | [use case] |

### Pitfalls & Prevention
_(Mandatory — minimum 7 items, each MUST include: 1) the specific failure symptom, 2) the root cause, 3) the exact fix command or config change. A pitfall without a concrete fix is useless. At least 4 pitfalls must cite a fetched source — doc page, GitHub issue, or changelog entry. Generic pitfalls from training data are filler; prioritize pitfalls you found IN the fetched docs.)_
- [Common mistake and how to avoid it — include the fix, not just the warning] [PageID]
- [Version-specific gotcha or breaking change — state exact versions affected] [PageID]
- [Resource/cost trap and how to estimate before committing] [PageID]
- [Silent misconfiguration that produces wrong results without errors] [PageID]
- [Scaling surprise: what changes when moving from toy to production data/traffic] [PageID]
- [Config field that is often set incorrectly — show wrong vs right YAML/JSON side by side] [PageID]
- [Dependency or version conflict between libraries in the recommended stack] [PageID]

### Further Reading
- [PageID]: [one-line description of what the page covers]

### Warnings
- [What will silently break or degrade if misconfigured] [PageID]
- [Scaling or cost surprises at production volume] [PageID]
- [Hardware/driver/version incompatibility that causes subtle failures] [PageID]
```

## After This

- If researching to **make a decision** → invoke **ml-plan** to build the implementation plan
- If researching to **debug an issue** → invoke **ml-debug** with the new understanding
- If researching to **improve results** → invoke **ml-iterate** with the new options
- If the user needs to **verify a specific claim** → invoke **ml-verify**

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Answering from memory when KB has the actual docs | "I know how attention works" | You know the concept. The KB knows the specific framework's implementation and config. |
| Single-query research | "One search should cover it" | Topics have multiple angles. 2-4 parallel queries catches what one misses. |
| Treating all sources as equal | "This page says X, that page says Y" | Check: which is framework-specific vs generic? Which is newer? Prefer specific + recent. |
| Skipping code examples | "The explanation is sufficient" | If the KB has runnable code, include it. Users implement faster with examples. |
| Not noting version specifics | "This works with transformers" | Which version? Document the version when KB mentions it — saves debugging later. |
| Stub code with "TODO" placeholders | "I'll show the structure" | If the KB has implementation details, fill them in. Stubs aren't actionable — users need code they can run now. |
| Inventing CLI flags or config keys | "This flag probably exists" | Only include flags/params that appear verbatim in KB sources. One hallucinated flag wastes hours of debugging. |
| Showing partial configs with "..." or omitted fields | "The important parts are here" | Users copy-paste configs wholesale. One missing required field = cryptic runtime error. Show the complete config. |
| Giving ranges instead of values | "Use learning rate 1e-5 to 5e-5" | Pick a specific default. If it depends on scale, show a table with one value per scenario. |
| Answering without ANY grounding when KB fails | "I can answer from deep knowledge" | Switch to web mode immediately. If web mode also fails, tell the user — never answer ungrounded. |
| Assuming library X works with model Y without checking | "SentenceTransformer probably wraps this" | Verify integration paths in docs. Many models need custom loading code. |
| Answering ungrounded when KB auth fails | "This is well within my training knowledge" | Switch to web mode immediately. Fetch 3+ doc pages. There is no excuse for zero citations. |
| Citing numbers without sources | "Perplexity is roughly 5.5" | Every number needs a benchmark citation. If no source exists, write `[needs benchmark]`. Training-data numbers are often wrong. |
| Trusting training data for model specs | "StarEncoder supports 1024 tokens" | Context windows, embedding dims, and supported languages change across model versions. Fetch the model card — your training data number is probably wrong. |
| Putting top-level config keys in nested structures | "base_model goes in the models list" | Always fetch a real config example for the tool. Schema mistakes (nesting, wrong key names) cause silent failures. |
| Starting response with prose when KB fails instead of immediately WebFetching | "KB is unavailable, but I know this topic well" | Your FIRST action after KB failure must be a WebFetch call, not text. Write zero words until you have fetched sources. |
| Presenting training-data knowledge as if it were fetched | "I have strong grounding" (but zero WebFetch calls made) | Count your actual tool calls. If WebFetch count is 0, you have no grounding regardless of confidence. |
| Using unverified config parameters | "`int8_mask: true` should work" | If a parameter doesn't appear verbatim in fetched docs or examples, it doesn't exist. Flag as ⚠️ UNVERIFIED. |

## Examples

**"How does vLLM handle tensor parallelism?"**
1. Parallel: `search_knowledge("vLLM tensor parallelism architecture implementation")`, `search_knowledge("vLLM kv-cache memory management tensor parallel")`, `search_knowledge("vLLM tensor parallel configuration multi-GPU setup")`
2. `get_page` on the most relevant cited pages
3. Synthesize: architecture → config → memory implications → gotchas

**"Compare LoRA vs QLoRA vs full fine-tuning for 7B models"**
1. Parallel: `search_knowledge("LoRA fine-tuning memory quality tradeoffs 7B")`, `search_knowledge("QLoRA 4-bit quantization fine-tuning memory savings quality")`, `search_knowledge("full fine-tuning vs parameter-efficient methods comparison")`, `search_knowledge("LoRA rank selection guidelines quality compute tradeoff")`
2. `get_page` on comparison pages
3. Synthesize: comparison table with memory, quality, speed, and use-case fit
