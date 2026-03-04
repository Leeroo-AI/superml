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

### Phase 2: Expand and Resolve

1. Call `get_page` on the most relevant `[PageID]` citations — prioritize pages with:
   - Code examples or config references
   - Edge cases or gotchas
   - Quantitative comparisons
2. If sources disagree, note which is newer or more framework-specific
3. If there's a gap, run one more targeted `search_knowledge`

**Gate**: Key claims are backed by specific KB pages, not just search snippets.

### Phase 3: Synthesize

Compose a structured answer:

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
| Parameter | Recommended | Why | Source |
|-----------|-------------|-----|--------|
| param | value | [rationale] | [PageID] |

### Tradeoffs
| Approach | Pros | Cons | Best for |
|----------|------|------|----------|
| A | ... [PageID] | ... | [use case] |
| B | ... [PageID] | ... | [use case] |

### Pitfalls
- [Common mistake and how to avoid it] [PageID]

### Further Reading
- [PageID]: [one-line description of what the page covers]
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

## Examples

**"How does vLLM handle tensor parallelism?"**
1. Parallel: `search_knowledge("vLLM tensor parallelism architecture implementation")`, `search_knowledge("vLLM kv-cache memory management tensor parallel")`, `search_knowledge("vLLM tensor parallel configuration multi-GPU setup")`
2. `get_page` on the most relevant cited pages
3. Synthesize: architecture → config → memory implications → gotchas

**"Compare LoRA vs QLoRA vs full fine-tuning for 7B models"**
1. Parallel: `search_knowledge("LoRA fine-tuning memory quality tradeoffs 7B")`, `search_knowledge("QLoRA 4-bit quantization fine-tuning memory savings quality")`, `search_knowledge("full fine-tuning vs parameter-efficient methods comparison")`, `search_knowledge("LoRA rank selection guidelines quality compute tradeoff")`
2. `get_page` on comparison pages
3. Synthesize: comparison table with memory, quality, speed, and use-case fit
