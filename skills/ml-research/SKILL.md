---
name: ml-research
description: Use when the user wants to understand an ML/AI topic, compare approaches, or survey framework capabilities — "how does X work?", "compare X vs Y"
---

# ML Research

Deep-dive into ML topics using verified framework documentation, not stale training data.

## The Iron Law

```
NO CLAIMS WITHOUT KB VERIFICATION
```

Your training data is months old. The KB has current framework docs. When the two disagree, the KB wins.

## Phases

### Phase 1: Multi-Angle Search

Launch 2-4 `search_knowledge` calls in **parallel** with different angles:
- Core concept / mechanism
- Framework-specific implementation
- Configuration and usage patterns
- Performance characteristics or tradeoffs

**Gate**: You have KB results covering at least 2 distinct angles on the topic before synthesizing.

> **Citation density target**: Aim for 15+ unique `[PageID]` citations in the final answer. If your search results yield fewer than 8 distinct pages, add another search angle. Call `get_page` on at least 3 top results — summaries from `search_knowledge` are not enough to ground specific claims.

### Phase 2: Expand and Resolve

1. Call `get_page` on the most relevant `[PageID]` citations — prioritize pages with:
   - Code examples or config references
   - Edge cases or gotchas
   - Quantitative comparisons
2. If sources disagree, note which is newer or more framework-specific
3. If there's a gap, run one more targeted `search_knowledge`

**Gate**: Every factual claim, config value, and table cell has a `[PageID]` citation. CLI flags, parameter names, and config keys must appear verbatim in a KB source — do not invent flags or options from memory. Every version number and compatibility statement must cite the specific KB page that confirms it.

> **Identifier verification**: Before including any model name (e.g., `BAAI/bge-code-v1`), API method signature, or config field name, confirm the **exact string** appears in a KB page. If you cannot find verbatim confirmation, call `search_knowledge` with the identifier. If still unconfirmed, flag it as "unverified — check Hub/docs" rather than presenting it as fact.

### Phase 3: Synthesize

Compose a structured answer:

**Completeness rules:**
- Code must be **runnable as-is**: include imports, CLI arg parsing, and shebang lines — never reference a hypothetical `train.py` without providing it
- Pin versions: `pip install mergekit==0.3.1`, not `pip install mergekit`
- If showing a multi-file pipeline, include an orchestration script that ties files together
- Every table cell containing a factual claim needs a `[PageID]`
- Config files must be **complete** — do not omit fields. If a config has 8 required fields, show all 8 with KB-sourced values
- Include a **validation/test snippet** (curl test, benchmark script, or smoke test) so the user can verify their setup works

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
- [Common mistake and how to avoid it — include the fix, not just the warning] [PageID]
- [Version-specific gotcha or breaking change — state exact versions affected] [PageID]
- [Resource/cost trap and how to estimate before committing] [PageID]
- [Silent misconfiguration that produces wrong results without errors] [PageID]
- [Scaling surprise: what changes when moving from toy to production data/traffic] [PageID]

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

## Examples

**"How does vLLM handle tensor parallelism?"**
1. Parallel: `search_knowledge("vLLM tensor parallelism architecture implementation")`, `search_knowledge("vLLM kv-cache memory management tensor parallel")`, `search_knowledge("vLLM tensor parallel configuration multi-GPU setup")`
2. `get_page` on the most relevant cited pages
3. Synthesize: architecture → config → memory implications → gotchas

**"Compare LoRA vs QLoRA vs full fine-tuning for 7B models"**
1. Parallel: `search_knowledge("LoRA fine-tuning memory quality tradeoffs 7B")`, `search_knowledge("QLoRA 4-bit quantization fine-tuning memory savings quality")`, `search_knowledge("full fine-tuning vs parameter-efficient methods comparison")`, `search_knowledge("LoRA rank selection guidelines quality compute tradeoff")`
2. `get_page` on comparison pages
3. Synthesize: comparison table with memory, quality, speed, and use-case fit
