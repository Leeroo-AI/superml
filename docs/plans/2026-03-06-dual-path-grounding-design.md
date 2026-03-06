# Dual-Path Grounding: KB + Web Fallback

**Goal:** Make the plugin fully functional without a Leeroopedia API key by falling back to web search, so users get the same structured workflow experience regardless of whether KB tools are connected.

**Architecture:** Each skill and the agent detect KB availability at first use, then follow either the KB path (current behavior) or the web path (simulated equivalents using WebFetch). The user experience — phases, output format, citations, anti-patterns — stays identical in both modes. Only the knowledge source changes.

---

## Design Decisions

1. **Dual-path in each skill** — every skill is self-contained; no dependency on a detection skill running first
2. **Simulate all KB tools** with web search equivalents — same UX, lower fidelity accepted
3. **Per-skill detection** — try KB tool first, fall back to web on auth error or tool unavailability
4. **Bite-sized plan granularity** (inspired by superpowers/writing-plans) — applies to both modes
5. **Agent (`ml-expert.md`) also gets dual-path**
6. **Test harness gets `--no-kb` flag** for tracking web-mode quality separately
7. **Rename `using-leeroopedia`** — deferred to later
8. **Citation format**: KB mode = `[PageID]`, Web mode = `[source](URL)`
9. **Known-good URL registry** per skill domain

---

## Detection Logic

Each skill detects mode independently on first grounding call:

```
If search_knowledge is available in tool list:
    Try a search_knowledge call
    If success → KB mode for rest of conversation
    If auth error → Web mode
Else:
    Web mode
```

This costs one tool call per conversation (not per skill invocation). Once mode is set, all subsequent grounding follows that path.

---

## KB Tool Simulation Mapping

| KB Tool | Web Simulation | Notes |
|---------|---------------|-------|
| `search_knowledge(query)` | WebFetch 2-3 official doc pages for the query topic. Use domain URL registry. Cite as `[source](URL)`. | Noisier but functional |
| `build_plan(goal, constraints)` | Manual decomposition: break goal into bite-sized steps, WebFetch framework docs per step, validate API signatures. Same output format. | Follows writing-plans granularity |
| `review_plan(proposal, goal)` | Self-review checklist: walk each step, WebFetch to verify claims, flag unverifiable steps as `[unverified — test in dry-run]`. | Manual but structured |
| `diagnose_failure(error)` | WebFetch GitHub issues search for exact error message + official troubleshooting pages. Broaden search if no exact match. | GitHub issues are surprisingly good for this |
| `query_hyperparameter_priors(param)` | WebFetch known config references (HF training examples, Axolotl default configs, published ablation studies). Flag as `[web-sourced]`. | Biggest quality gap — no structured priors |
| `verify_code_math(code)` | WebFetch API docs for every non-trivial import. Manually check signatures, dtypes, shapes against docs. | Slower but achievable |
| `propose_hypothesis()` | Reason from web-sourced context. Search for similar reported issues/solutions on GitHub + forums. | Adequate |
| `get_page(id)` | N/A in web mode — only exists for KB citations | Skip |

---

## URL Registry (per skill domain)

Each skill includes a curated set of doc URLs for web mode:

### Training / Fine-tuning
- HuggingFace Transformers: `https://huggingface.co/docs/transformers`
- HuggingFace PEFT: `https://huggingface.co/docs/peft`
- HuggingFace TRL: `https://huggingface.co/docs/trl`
- Axolotl: `https://github.com/axolotl-ai-cloud/axolotl`
- Unsloth: `https://docs.unsloth.ai`

### Serving
- vLLM: `https://docs.vllm.ai`
- TGI: `https://huggingface.co/docs/text-generation-inference`
- SGLang: `https://sgl-project.github.io`

### Distributed
- DeepSpeed: `https://www.deepspeed.ai/docs`
- PyTorch FSDP: `https://pytorch.org/docs/stable/fsdp.html`
- Megatron-LM: `https://github.com/NVIDIA/Megatron-LM`

### Agents / RAG
- LangChain: `https://python.langchain.com/docs`
- LangGraph: `https://langchain-ai.github.io/langgraph`
- LlamaIndex: `https://docs.llamaindex.ai`

### Evaluation
- RAGAS: `https://docs.ragas.io`
- lm-eval-harness: `https://github.com/EleutherAI/lm-evaluation-harness`

---

## Skill-Level Changes

Each skill gets a **Grounding** section near the top:

```markdown
### Grounding

**Detect mode:** On your first grounding call, check if Leeroopedia KB tools
(`search_knowledge`, etc.) are available. If they return results, use KB mode.
If unavailable or auth fails, use Web mode for the rest of this conversation.

**KB mode:**
- Call `search_knowledge` with 2-4 parallel queries
- Cite as `[PageID]`
- Use `build_plan`, `review_plan`, `diagnose_failure` etc. as documented

**Web mode:**
- WebFetch official documentation pages (see URL registry below)
- Cite as `[source](URL)`
- For each claim, fetch the specific doc page and verify
- Flag anything you cannot verify against docs as `[unverified]`

**URL registry for this skill:**
- [skill-specific URLs from the registry above]
```

The rest of the skill (phases, anti-patterns, examples, Iron Laws) stays unchanged.

---

## Agent Changes (`ml-expert.md`)

Same dual-path approach. The agent's "Ground in KB before responding" section becomes:

- Detect KB availability on first use
- KB mode: current behavior (mandatory KB calls, `[PageID]` citations)
- Web mode: mandatory web lookups before responding (WebFetch official docs), `[source](URL)` citations, same "never silently fall back to training knowledge" rule

The agent's memory structure, self-improvement loop, and execution standards stay unchanged.

---

## Test Harness Changes

Add `--no-kb` flag to `test_interactive.py`:

- When set, run plugin tests WITHOUT `LEEROOPEDIA_API_KEY` in the environment
- MCP tools will be unavailable, forcing skills into web mode
- Track web-mode scores separately in summary
- Can run both modes in same round: `--compare-modes` runs each test twice (KB + web) and reports the delta

This lets us quantify the quality gap and track whether web-mode improvements close it over time.

---

## `build_plan` Specifics (Writing-Plans Inspired)

Both modes produce plans with this granularity:

```markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py`

**Step 1: [Action]**
[Complete code/config — not a sketch]
- Cite: [PageID] or [source](URL)

**Step 2: Validate**
Run: `exact command`
Expected: [specific output]

**Step 3: [Next action]**
...
```

**KB mode:** `build_plan()` returns the structured plan, then format into bite-sized tasks.

**Web mode:**
1. Decompose goal into numbered components
2. For each component, WebFetch the relevant framework docs
3. Write complete code/config per step (verified against fetched docs)
4. Self-review: walk through checking for gaps, version mismatches, missing imports
5. Flag any step without a doc citation as `[unverified — test in dry-run]`

---

## Expected Quality Impact

| Dimension | KB Mode | Web Mode | Gap |
|-----------|---------|----------|-----|
| Correctness | High — verified against curated KB | Medium — web docs are accurate but noisier | Small |
| Specificity | High — version-specific configs from KB | Medium — docs give current version, less historical context | Small |
| Prevention | High — KB has curated pitfall databases | Lower — depends on finding the right GitHub issues | Medium |
| Actionability | High | High — same plan structure, same code quality | Minimal |
| Grounding | High — `[PageID]` citations from structured KB | Medium — `[URL]` citations from web docs | Small |

The biggest gap is in **prevention** (pitfall detection) and **hyperparameter priors** — these are where the curated KB adds the most value. Web mode will still catch common issues but miss the non-obvious ones.

---

## Implementation Order

1. Update `using-leeroopedia` skill with dual-path grounding section + URL registry
2. Update `ml-plan` with dual-path + writing-plans granularity (both modes)
3. Update remaining skills: `ml-debug`, `ml-verify`, `ml-research`, `ml-iterate`, `ml-experiment`
4. Update `ml-expert` agent with dual-path
5. Add `--no-kb` flag to test harness
6. Run comparative test: KB mode vs web mode on same 38 tests
7. Iterate on web-mode skill instructions based on results
