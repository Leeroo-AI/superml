---
name: ml-research
description: Use when the user wants to understand an ML/AI topic, compare approaches, or survey framework capabilities via Leeroopedia KB
---

# ML Research Workflow

Deep-dive into ML/AI topics using Leeroopedia KB for grounded answers.

**CRITICAL: Launch 2-4 parallel `search_knowledge` calls IMMEDIATELY.** Don't think about what you know first — the KB has specific framework details, code examples, and gotchas that memory won't have.

## When to Use

- User asks "How does X work?"
- User wants to compare approaches (e.g., LoRA vs QLoRA vs full fine-tuning)
- User needs a survey of framework capabilities
- User asks about best practices for a technique
- User wants to understand internals of a framework

## Workflow

### 1. Multi-angle search
Call `search_knowledge` with 2-4 **parallel** queries covering different angles:
- Core concept/mechanism
- Framework-specific implementation
- Configuration and usage patterns
- Performance characteristics or tradeoffs

### 2. Expand key citations
For the most relevant `[PageID]` citations, call `get_page` to get full page content. Prioritize pages that:
- Are directly about the user's question
- Contain code examples or config references
- Cover edge cases or gotchas

### 3. Synthesize
Combine into a structured, grounded answer with inline citations. **Before sending:** scan your draft — every `##` section MUST have at least one `[Category/Page_Name]` citation from tool results. If a section has none, find the relevant citation from your tool results and add it.

## Output Format

```
## [Topic]

### How It Works
[Core mechanism explanation] [PageID]

### Implementation
[Framework-specific details with code examples] [PageID]

### Key Configuration
| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| ... | ... | ... [PageID] |

### Tradeoffs
- Pro: ... [PageID]
- Con: ... [PageID]

### Further Reading
- [PageID]: [brief description]
```

## Examples

**"How does vLLM handle tensor parallelism?"**
1. Parallel: `search_knowledge("vLLM tensor parallelism implementation architecture")`, `search_knowledge("vLLM kv-cache memory management tensor parallel")`, `search_knowledge("vLLM tensor parallel configuration multi-GPU")`
2. `get_page` on the most relevant cited pages
3. Synthesize: architecture, config, memory implications, gotchas

**"Compare LoRA vs QLoRA vs full fine-tuning for 7B models"**
1. Parallel: `search_knowledge("LoRA fine-tuning memory and quality tradeoffs 7B models")`, `search_knowledge("QLoRA 4-bit quantization fine-tuning memory savings")`, `search_knowledge("full fine-tuning vs parameter-efficient methods quality comparison")`, `search_knowledge("LoRA rank selection guidelines quality vs compute")`
2. `get_page` on cited comparison pages
3. Synthesize: comparison table with memory, quality, speed, and use-case fit

**"How does the OpenAI Agents SDK handle handoffs?"**
1. Parallel: `search_knowledge("OpenAI Agents SDK handoff mechanism between agents")`, `search_knowledge("OpenAI Agents SDK typed handoffs best practices")`, `search_knowledge("OpenAI Swarm agent handoff patterns")`
2. `get_page` on most relevant results
3. Synthesize: handoff architecture, typed patterns, examples
