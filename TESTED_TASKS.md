# Tested Tasks

38 ML/AI tasks evaluated head-to-head: Claude Code with SuperML vs without. Each response scored by 3 independent LLM judges across correctness, specificity, mistake prevention, actionability, and grounding (15 points max).

<p align="center">
  <img src="assets/results-chart.svg" alt="SuperML vs Baseline — 38 ML Tasks" width="1150">
</p>

---

## Detailed Scores

### Fine-Tuning & Training

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| QLoRA multimodal (LLaVA-1.5-7B, medical imaging, 1xA100) | **15** | 8 | +7 |
| DPO alignment (Mistral-7B, TRL, 2xA100) | **15** | 8 | +7 |
| GRPO alignment (Qwen2.5-7B, TRL GRPOTrainer, 2xA100) | **15** | 8 | +7 |
| Distributed pretraining (3B GPT, 4x8xH100, Megatron/DeepSpeed/FSDP) | **15** | 6 | +9 |
| Continual pretraining (Llama-3.1-8B, biomedical, DeepSpeed ZeRO-2) | 14 | 9 | +5 |
| Vision fine-tuning (Qwen2-VL-7B, document understanding, 1xH100) | 14 | 8 | +6 |
| Embedding fine-tuning (BGE-large, legal contracts, InfoNCE) | 13 | 10 | +3 |
| Model distillation (GPT-4o to Llama-3-8B, $300 budget) | 13 | 8 | +5 |
| Synthetic data generation (legal contracts, 100k pairs, $500 budget) | 13 | 9 | +4 |

### Debugging & Verification

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| Preflight config check (QLoRA config with deliberate issues) | **15** | 9 | +6 |
| Iterate after results (Llama-3-8B, 3 LoRA experiments, hallucination) | **15** | 7 | +8 |
| MoE training debug (Mixtral, expert collapse, router imbalance) | 14 | 9 | +5 |
| OOM distributed debug (Llama-3.1-70B, ZeRO-3, 4xA100) | 14 | 7 | +7 |
| Training loss spike (QLoRA, step 501 spike, data-dependent) | 13 | 7 | +6 |

### Inference & Serving

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| Speculative decoding (Llama-3-70B + 8B draft, vLLM) | **15** | 8 | +7 |
| FSDP vs DeepSpeed (Llama-3.1-70B, 8xH100, comparison) | **15** | 8 | +7 |
| Serving optimization (Llama-3-70B, vLLM, p99 latency target) | 14 | 6 | +8 |
| KV cache optimization (Llama-3.1-70B, 32k context, PagedAttention) | 14 | 6 | +8 |
| Quantization comparison (GPTQ/AWQ/GGUF/HQQ/bnb, 70B on 2xA100) | 13 | 9 | +4 |
| MoE serving (Mixtral-8x22B, 4xA100, production) | 7 | **14** | -7 |

### RAG & Retrieval

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| Multimodal RAG (PDFs with tables/charts, ColPali vs pipelines) | **15** | 7 | +8 |
| RAG evaluation (RAGAS, ChromaDB + BGE + GPT-4o) | 13 | 6 | +7 |
| Agentic RAG (LangGraph, Qdrant, self-correction, query decomposition) | 13 | 7 | +6 |

### Architecture & Systems

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| Model merging (3x Llama-3-8B, TIES/DARE/SLERP, mergekit) | 14 | 8 | +6 |
| Embedding pipeline (10M code snippets, sub-100ms search) | 14 | 9 | +5 |
| A/B testing LLMs (production, sequential testing, FastAPI) | 12 | 8 | +4 |
| Prompt optimization (DSPy, 3-module RAG pipeline) | 12 | 9 | +3 |
| Guardrails system (NeMo/Guardrails AI, <200ms latency) | 12 | 9 | +3 |
| Structured output (Pydantic, function calling vs Instructor vs Outlines) | 11 | 10 | +1 |
| LLM eval framework (4-axis, LLM-as-judge, CI integration) | 11 | 10 | +1 |

### Agent Tasks

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| Agent delegation (vLLM serving review via ml-expert) | **15** | 7 | +8 |
| Direct agent review (Llama-3-8B pipeline audit) | 14 | 10 | +4 |
| Agent tool use (data analysis agent, Claude API) | 10 | 9 | +1 |
| Agent orchestration (OpenAI Agents SDK, triage routing) | 8 | 9 | -1 |

### Negative Controls (non-ML tasks)

| Task | Plugin | Baseline | Delta |
|------|:------:|:--------:|:-----:|
| DevOps (GitHub Actions, Docker, ECR, ECS) | 11 | 9 | +2 |
| Basic Python (merge sorted lists) | 7 | 6 | +1 |
| Algorithm (trie + autocomplete) | 6 | 7 | -1 |
| Web dev (FastAPI CRUD + SQLAlchemy) | 6 | **9** | -3 |

---

## Methodology

- Each task is run through Claude Code twice: once with SuperML installed, once without
- 3 independent LLM judges score each response (median scores used)
- Judge positions are randomized to avoid position bias
- Tasks that require GPU hardware are scored on config/code quality, not execution
- Negative controls verify the plugin doesn't degrade non-ML responses
- Full test harness: [self-refine/](self-refine/)
