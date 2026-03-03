---
name: ml-plan
description: Use when the user wants an implementation plan, architecture design, or multi-step ML pipeline grounded in Leeroopedia KB
---

# ML Planning Workflow

Build KB-grounded implementation plans for ML/AI systems.

**CRITICAL: Call `build_plan` IMMEDIATELY with the user's stated goal.** Do NOT ask clarifying questions first. Do NOT brainstorm first. The KB will give you a grounded plan that you can refine with the user afterward.

## When to Use

- User says "build", "implement", "design", "set up", "create" anything ML/AI
- User wants an end-to-end pipeline (training, serving, evaluation)
- User needs architecture design for an ML system
- User asks for a step-by-step approach to an ML task

## Workflow

### 1. Build the plan IMMEDIATELY
Call `build_plan(goal, constraints?)` with the user's stated goal and any constraints they mentioned.
- Use the user's exact words as the goal
- Include any hardware, framework, latency, or scale constraints they mentioned
- Do NOT wait for more information — use what you have

### 2. Review the plan
Call `review_plan(proposal, goal)` with the plan from step 1 to catch risks and improvements.

### 3. Fill knowledge gaps
Identify the 2-4 most uncertain steps. Call `search_knowledge` in **parallel**:
- Framework-specific API details
- Config format requirements
- Known pitfalls or gotchas
- Performance characteristics

### 4. Present the validated plan
Combine into a final plan with inline `[PageID]` citations:
- **Overview**: What we're building and why this approach
- **Prerequisites**: Dependencies, hardware, API keys
- **Numbered steps**: Each with specific tool/API/config and citations
- **Validation criteria**: How to verify each step
- **Risks**: Known pitfalls with mitigations

## Output Format

```
## Plan: [Goal]

### Overview
[1-2 sentences]

### Prerequisites
- [ ] ...

### Steps
1. **[Step name]** — [description] [PageID]
   - Config: `...`
   - Validate: [how to verify]

2. ...

### Risks & Mitigations
- Risk: ... → Mitigation: ... [PageID]
```

## Examples

**"I want to fine-tune Qwen2.5-7B with QLoRA on 2xA100"**
1. `build_plan("QLoRA fine-tuning Qwen2.5-7B", "2xA100 80GB, instruction tuning dataset")`
2. `review_plan(plan_output, "QLoRA fine-tuning Qwen2.5-7B")`
3. Parallel: `search_knowledge("Qwen2.5 QLoRA config format Axolotl")`, `search_knowledge("QLoRA memory estimation 7B model 2xA100")`

**"Design a RAG system with hybrid retrieval"**
1. `build_plan("RAG system with hybrid vector+BM25 retrieval", "FastAPI, ChromaDB, production-ready")`
2. `review_plan(plan_output, "Hybrid RAG system")`
3. Parallel: `search_knowledge("ChromaDB hybrid retrieval BM25 integration")`, `search_knowledge("RAG evaluation metrics recall@k RAGAS")`
