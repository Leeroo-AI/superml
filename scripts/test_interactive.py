#!/usr/bin/env python3
"""
A/B quality evaluation for leeroopedia plugin.
Runs each test with and without the plugin, judges both via LLM,
then auto-refines skills based on failure patterns.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PLUGIN_DIR / "test-logs"
LOG_DIR.mkdir(exist_ok=True)

BASELINE_MAX_TURNS = 15
PLUGIN_MAX_TURNS = 20
BASELINE_TIMEOUT = 600   # 10 min
PLUGIN_TIMEOUT = 900     # 15 min
JUDGE_TIMEOUT = 300      # 5 min

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = [
    {
        "id": "t01_agent_orchestration",
        "prompt": "I'm building a customer support agent system using the OpenAI Agents SDK. I need a triage agent that routes to 3 specialists: billing, technical, and account. The triage agent should pass structured context (customer tier, issue category, sentiment score) via typed handoffs. Give me the full implementation with proper error handling.",
        "category": "agent-building",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t02_qlora_multimodal",
        "prompt": "I want to fine-tune LLaVA-1.5-7B for medical image analysis using QLoRA. My data is 50k image-text pairs of X-rays with radiologist reports. I have 1xA100 80GB. Walk me through the complete setup: data format, config, training script, and evaluation. What target_modules should I use for the vision encoder vs the language model?",
        "category": "fine-tuning",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t03_serving_optimization",
        "prompt": "I'm serving Llama-3-70B-Instruct with vLLM on 4xA100 80GB in production. Current config: tensor_parallel=4, max_model_len=4096, gpu_memory_utilization=0.9. I'm getting p99 latency of 8 seconds for 2048-token outputs at 50 concurrent users. I need to get this under 3 seconds. What specific optimizations should I make? Show me the exact vLLM launch command and any code changes.",
        "category": "inference-optimization",
        "expected_skill": "ml-debug",
        "needs_gpu": True,
    },
    {
        "id": "t04_distributed_training",
        "prompt": "I need to pre-train a 3B parameter GPT model from scratch on a custom dataset (200B tokens). I have a cluster of 4 nodes, each with 8xH100 80GB connected via InfiniBand. Should I use Megatron-LM, DeepSpeed, or FSDP? Give me the complete training config including parallelism strategy (TP, PP, DP split), batch size, learning rate schedule, and the launch command.",
        "category": "distributed-training",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t05_rag_evaluation",
        "prompt": "My RAG system is in production but users complain answers are sometimes wrong or incomplete. I need to build an automated evaluation pipeline. The system uses ChromaDB + BGE-large embeddings + GPT-4o for generation. I want to measure: retrieval quality (recall, MRR), answer faithfulness, answer relevance, and hallucination rate. Show me how to set this up with RAGAS or a similar framework, including synthetic test set generation.",
        "category": "evaluation",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t06_dpo_alignment",
        "prompt": "I fine-tuned Mistral-7B-v0.3 on my custom instruction data and it works well, but it sometimes generates unsafe content. I want to align it using DPO. I have 5k preference pairs (chosen/rejected). My setup: 2xA100 40GB. Give me the complete TRL DPO training config, explain the key hyperparameters (beta, learning rate, loss type), and tell me what can go wrong. What should my data format look like?",
        "category": "alignment",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t07_embedding_pipeline",
        "prompt": "I'm building a code search engine that needs to index 10M code snippets across Python, JavaScript, and Rust. I need to: (1) choose the right embedding model for code, (2) build an efficient indexing pipeline that can process 10M items in under 4 hours on 2xA100, (3) serve similarity search with sub-100ms latency. Compare code embedding models (CodeBERT vs StarEncoder vs Voyage Code vs OpenAI) and give me the full architecture.",
        "category": "embeddings",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t08_model_merging",
        "prompt": "I have three Llama-3-8B fine-tunes: one for coding, one for math, and one for creative writing. I want to merge them into a single model that's good at all three. Compare merging methods: TIES, DARE, SLERP, linear. Which should I use? Show me the mergekit config and the evaluation strategy to verify the merge didn't degrade any capability.",
        "category": "model-merging",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t09_speculative_decoding",
        "prompt": "I want to speed up inference of Llama-3-70B using speculative decoding. I'm considering using Llama-3-8B as the draft model. How does speculative decoding work exactly? What's the expected speedup? How do I set this up with vLLM? Are there any gotchas with the draft model choice (tokenizer compatibility, acceptance rate)? Give me the complete setup.",
        "category": "inference-optimization",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t10_training_debug_complex",
        "prompt": (
            "I'm training a mixture-of-experts model using Mixtral architecture with "
            "DeepSpeed on 8xA100. After 5000 steps, I see these symptoms:\n"
            "1. Router load balancing loss keeps increasing (started at 0.01, now at 0.8)\n"
            "2. Only 2 of 8 experts are being used (expert utilization histogram is extremely skewed)\n"
            "3. Training loss plateaued at 2.1 and won't decrease\n"
            "4. GPU memory usage is uneven across GPUs (GPU 0: 72GB, others: ~45GB)\n\n"
            "Config: expert_parallel_size=4, num_experts=8, top_k=2, aux_loss_coeff=0.01, lr=3e-4.\n\n"
            "What's happening and how do I fix each issue?"
        ),
        "category": "debugging",
        "expected_skill": "ml-debug",
        "needs_gpu": True,
    },
    {
        "id": "t11_iterate_after_results",
        "prompt": (
            "Here are my experiment results so far for fine-tuning Llama-3-8B on a "
            "customer support dataset:\n\n"
            "Experiment 1: LoRA r=16, lr=2e-4, 3 epochs -> BLEU 32.1, but model "
            "hallucinates product names\n"
            "Experiment 2: LoRA r=32, lr=1e-4, 3 epochs -> BLEU 33.8, hallucinations "
            "slightly reduced\n"
            "Experiment 3: LoRA r=32, lr=1e-4, 5 epochs -> BLEU 31.2, overfitting "
            "(train loss kept dropping but eval got worse)\n\n"
            "I'm targeting BLEU > 38 with minimal hallucination. What should I try next? "
            "I have 2xA100 40GB and 50k training examples."
        ),
        "category": "iteration",
        "expected_skill": "ml-iterate",
        "needs_gpu": False,
    },
    {
        "id": "t12_preflight_check",
        "prompt": (
            "I'm about to start a QLoRA training run. Can you check this config before "
            "I burn GPU hours?\n\n"
            "model: meta-llama/Llama-3-8B-Instruct\n"
            "quantization: 4bit (bnb)\n"
            "lora_r: 128\n"
            "lora_alpha: 16\n"
            "lora_dropout: 0.1\n"
            'target_modules: ["q_proj", "v_proj"]\n'
            "learning_rate: 5e-3\n"
            "batch_size: 8\n"
            "gradient_accumulation_steps: 4\n"
            "max_seq_length: 8192\n"
            "num_epochs: 5\n"
            "warmup_ratio: 0.0\n"
            "bf16: true\n"
            "optimizer: adamw_8bit\n\n"
            "Hardware: 1xA100 40GB\n"
            "Dataset: 20k instruction-response pairs, avg 500 tokens"
        ),
        "category": "verification",
        "expected_skill": "ml-verify",
        "needs_gpu": False,
    },
    {
        "id": "t13_agentic_rag",
        "prompt": (
            "I need to build an agentic RAG system that goes beyond simple "
            "retrieve-and-generate. Requirements:\n\n"
            "1. Query decomposition — break complex questions into sub-queries\n"
            "2. Adaptive retrieval — decide whether to search, re-rank, or "
            "ask for clarification\n"
            "3. Self-correction — detect when retrieved context is insufficient "
            "and re-retrieve with reformulated queries\n"
            "4. Source fusion — combine answers from multiple retrieval rounds "
            "with proper attribution\n\n"
            "Tech stack: LangGraph for orchestration, Qdrant for vector store, "
            "Cohere reranker, GPT-4o for generation. Give me the full "
            "implementation with the state machine, node functions, and "
            "edge conditions. Include the retry/self-correction logic."
        ),
        "category": "agentic-rag",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t14_synthetic_data_gen",
        "prompt": (
            "I want to create a high-quality synthetic instruction-tuning "
            "dataset for a domain-specific LLM (legal contract analysis). "
            "I have 10k real contracts but no instruction-response pairs.\n\n"
            "I've heard of approaches like Self-Instruct, Evol-Instruct "
            "(WizardLM), Magpie, and the Orca/phi approach of distilling "
            "reasoning chains. Which approach should I use? I need:\n\n"
            "1. Pipeline architecture to generate 100k instruction-response "
            "pairs from my raw contracts\n"
            "2. Quality filtering (how to detect and remove low-quality "
            "synthetic samples)\n"
            "3. Diversity enforcement (avoid mode collapse in generated "
            "instructions)\n"
            "4. Contamination checks (ensure no test set leakage)\n\n"
            "Budget: $500 in API calls (GPT-4o or Claude). Give me the full "
            "pipeline with code."
        ),
        "category": "data-engineering",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t15_llm_eval_framework",
        "prompt": (
            "I'm building an automated evaluation framework to compare "
            "fine-tuned LLMs before deploying to production. I need to "
            "evaluate across 4 axes:\n\n"
            "1. Task accuracy — domain-specific benchmarks (I'll provide "
            "gold-standard QA pairs)\n"
            "2. Safety — refusal on harmful prompts, jailbreak resistance\n"
            "3. Hallucination rate — faithfulness to provided context\n"
            "4. Instruction following — format compliance, constraint "
            "adherence\n\n"
            "I want to use LLM-as-judge (GPT-4o) with proper rubrics, "
            "plus deterministic metrics where possible. Need statistical "
            "rigor: confidence intervals, inter-annotator agreement with "
            "human labels, significance testing between model versions.\n\n"
            "Give me the complete framework: rubric design, judge prompts, "
            "scoring pipeline, and the statistical analysis code. I want "
            "to run this as a CI check before any model deployment."
        ),
        "category": "evaluation",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t16_agent_tool_use",
        "prompt": (
            "I'm building a data analysis agent that can autonomously write "
            "and execute Python code, query SQL databases, create "
            "visualizations, and present findings. The agent needs:\n\n"
            "1. Tool schema design — define clean function-calling schemas "
            "for: code execution (with sandboxing), SQL queries (read-only), "
            "chart generation (matplotlib/plotly), and file I/O\n"
            "2. Error recovery — when code execution fails, the agent "
            "should analyze the traceback, fix the code, and retry "
            "(max 3 attempts)\n"
            "3. Safety — prevent SQL injection, sandbox code execution, "
            "limit resource usage\n"
            "4. Conversation memory — maintain context of what data has "
            "been explored and what conclusions were drawn\n\n"
            "Using Claude API with tool_use. Give me the complete "
            "implementation: tool definitions, agent loop, error handling, "
            "and sandboxing setup. Include the system prompt that makes "
            "the agent effective at data analysis."
        ),
        "category": "agent-building",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    # --- Round 2: 16 more tests (t17-t32) ---
    {
        "id": "t17_grpo_alignment",
        "prompt": (
            "I want to align my Qwen2.5-7B-Instruct model using GRPO "
            "(Group Relative Policy Optimization) instead of DPO. I have "
            "10k prompts (no preference pairs — just prompts). Using TRL's "
            "GRPOTrainer on 2xA100 80GB.\n\n"
            "1. Explain how GRPO differs from DPO/PPO — why no reward model "
            "or preference data?\n"
            "2. Give me the complete training config with TRL GRPOTrainer\n"
            "3. What are the key hyperparameters (group_size, kl_coef, "
            "num_generations) and how do they interact?\n"
            "4. What reward function should I use for general instruction "
            "following?\n"
            "5. Common failure modes and how to detect them"
        ),
        "category": "alignment",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t18_vision_finetuning",
        "prompt": (
            "I need to fine-tune Qwen2-VL-7B for document understanding "
            "(invoices, receipts, forms). I have 30k document images with "
            "structured JSON annotations. Hardware: 1xH100 80GB.\n\n"
            "1. What data format does Qwen2-VL expect for supervised "
            "fine-tuning?\n"
            "2. Give me the complete training script using transformers "
            "and the Qwen2-VL processor\n"
            "3. Should I use LoRA or full fine-tuning? What target_modules?\n"
            "4. How do I handle variable-resolution images efficiently?\n"
            "5. Evaluation: how to measure extraction accuracy vs the "
            "JSON ground truth"
        ),
        "category": "fine-tuning",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t19_moe_serving",
        "prompt": (
            "I need to serve Mixtral-8x22B in production on 4xA100 80GB. "
            "Requirements: p99 latency < 5s for 1024-token outputs, "
            "30 concurrent users, 99.9% uptime.\n\n"
            "1. Can this even fit on 4xA100 80GB? What quantization is "
            "needed?\n"
            "2. Compare serving options: vLLM vs TGI vs SGLang for MoE "
            "models specifically\n"
            "3. Give me the exact launch config with optimal tensor "
            "parallel, quantization, and KV cache settings\n"
            "4. How does expert parallelism work in vLLM for MoE?\n"
            "5. Production setup: health checks, graceful degradation, "
            "request queuing"
        ),
        "category": "inference-optimization",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t20_continual_pretraining",
        "prompt": (
            "I want to do continual pre-training of Llama-3.1-8B on a "
            "domain corpus (500M tokens of biomedical papers). Hardware: "
            "8xA100 80GB single node.\n\n"
            "1. How does continual pre-training differ from fine-tuning? "
            "What learning rate schedule should I use (much lower than "
            "from-scratch)?\n"
            "2. Complete training config with DeepSpeed ZeRO-2\n"
            "3. How to prevent catastrophic forgetting of general "
            "capabilities?\n"
            "4. Data mixing strategy: what ratio of domain data vs "
            "replay data?\n"
            "5. How to evaluate: perplexity on domain vs general "
            "benchmarks, and when to stop"
        ),
        "category": "pre-training",
        "expected_skill": "ml-plan",
        "needs_gpu": True,
    },
    {
        "id": "t21_quantization_comparison",
        "prompt": (
            "I need to quantize Llama-3.1-70B for deployment on 2xA100 "
            "40GB. Compare quantization methods:\n\n"
            "1. GPTQ vs AWQ vs GGUF vs HQQ vs bitsandbytes — which "
            "preserves quality best at 4-bit?\n"
            "2. For each method: exact commands to quantize, expected "
            "model size, and known quality impact\n"
            "3. Calibration dataset: how many samples, what kind of data?\n"
            "4. Benchmark plan: how to measure quality degradation "
            "(perplexity, task accuracy, generation quality)\n"
            "5. Can I serve the quantized model with vLLM? Which "
            "quantization formats does vLLM 0.6+ support?"
        ),
        "category": "quantization",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t22_training_loss_spike",
        "prompt": (
            "I'm fine-tuning Llama-3-8B with QLoRA and seeing a "
            "persistent problem: training loss drops normally for the "
            "first 500 steps (from 2.3 to 0.8), then suddenly spikes to "
            "5.0+ at step 501 and never recovers. This happens "
            "consistently at the same step.\n\n"
            "Config: lr=2e-4, cosine schedule, warmup_steps=100, "
            "batch_size=4, grad_accum=8, bf16, bnb 4bit, max_seq_len=2048, "
            "lora_r=16, lora_alpha=32.\n\n"
            "Data: 15k instruction-response pairs, shuffled. Average "
            "length ~800 tokens, but some samples are up to 2048.\n\n"
            "What's causing this and how do I fix it?"
        ),
        "category": "debugging",
        "expected_skill": "ml-debug",
        "needs_gpu": True,
    },
    {
        "id": "t23_kv_cache_optimization",
        "prompt": (
            "I'm serving Llama-3.1-70B on 4xA100 80GB with vLLM. "
            "My KV cache is the bottleneck — I can only handle 8 "
            "concurrent requests with max_model_len=32768.\n\n"
            "1. Explain PagedAttention and how vLLM manages KV cache\n"
            "2. How much KV cache memory does each request use for "
            "Llama-3.1-70B at 32k context?\n"
            "3. Compare optimization strategies: FP8 KV cache, GQA "
            "exploitation, prefix caching, chunked prefill, sliding "
            "window attention\n"
            "4. Give me the vLLM config that maximizes concurrent "
            "requests while keeping quality\n"
            "5. How to monitor KV cache utilization in production"
        ),
        "category": "inference-optimization",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t24_fsdp_vs_deepspeed",
        "prompt": (
            "I need to choose between FSDP and DeepSpeed ZeRO-3 for "
            "fine-tuning Llama-3.1-70B on 8xH100 80GB. The task is "
            "instruction-tuning on 200k examples.\n\n"
            "1. Compare FSDP vs DeepSpeed ZeRO-3 for this specific "
            "use case: memory usage, throughput, ease of setup\n"
            "2. Complete training configs for BOTH approaches (using "
            "Hugging Face Trainer + Accelerate)\n"
            "3. Which is better with QLoRA specifically?\n"
            "4. Activation checkpointing: how to configure for each\n"
            "5. Multi-node: which is easier to scale from 1 to 4 nodes?\n"
            "6. Known bugs or version-specific issues to watch for"
        ),
        "category": "distributed-training",
        "expected_skill": "ml-research",
        "needs_gpu": True,
    },
    {
        "id": "t25_structured_output",
        "prompt": (
            "I need to build a system that reliably extracts structured "
            "data from unstructured text using LLMs. Requirements:\n\n"
            "1. Define Pydantic models for the output schema (nested "
            "objects, enums, optional fields)\n"
            "2. Compare approaches: function calling, JSON mode, "
            "Instructor library, Outlines/guidance for constrained "
            "decoding\n"
            "3. Implement retry logic with progressive prompting when "
            "validation fails\n"
            "4. Handle edge cases: LLM refuses, partial output, "
            "wrong types, hallucinated enum values\n\n"
            "Use case: extracting job postings into structured format "
            "(title, company, salary range, requirements list, location, "
            "remote policy). Show me implementations using both Claude "
            "API and OpenAI API."
        ),
        "category": "applied-llm",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t26_prompt_optimization",
        "prompt": (
            "I'm building a prompt optimization system using DSPy. "
            "My pipeline has 3 modules chained: query_rewriter -> "
            "retriever -> answer_generator. I want to optimize all "
            "3 prompts jointly.\n\n"
            "1. How does DSPy's optimization work (MIPRO, BootstrapFewShot, "
            "etc.)? Which optimizer for my use case?\n"
            "2. Complete implementation: define modules, signatures, "
            "and the optimization loop\n"
            "3. How to define a good metric function for RAG quality\n"
            "4. How many labeled examples do I need for optimization?\n"
            "5. How to evaluate: before vs after optimization comparison\n"
            "6. Known limitations and when DSPy doesn't help"
        ),
        "category": "prompt-engineering",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t27_guardrails_system",
        "prompt": (
            "I need to build a production guardrails system for my "
            "customer-facing LLM application. Requirements:\n\n"
            "1. Input guardrails: detect prompt injection, jailbreak "
            "attempts, PII in user input, off-topic requests\n"
            "2. Output guardrails: detect hallucination, toxic content, "
            "PII leakage, brand-damaging statements\n"
            "3. Compare frameworks: NeMo Guardrails vs Guardrails AI vs "
            "custom classifier approach\n"
            "4. Latency budget: guardrails must add < 200ms to each "
            "request\n"
            "5. Implementation with both rule-based and ML-based "
            "classifiers\n"
            "6. Monitoring: track false positive/negative rates, create "
            "a human review queue for edge cases"
        ),
        "category": "safety",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t28_multimodal_rag",
        "prompt": (
            "I'm building a RAG system that needs to handle PDFs with "
            "mixed content: text, tables, charts, and diagrams. "
            "Requirements:\n\n"
            "1. Document parsing: extract text, tables (as structured "
            "data), and images from PDFs\n"
            "2. Multi-modal embeddings: embed text chunks and images "
            "into the same vector space\n"
            "3. Retrieval: when user asks about a chart, retrieve the "
            "chart image + surrounding text\n"
            "4. Generation: feed retrieved text + images to a "
            "vision-language model for answer generation\n\n"
            "Compare approaches: ColPali-style late interaction vs "
            "separate text/image pipelines vs Unstructured.io + "
            "standard embeddings. Give me the complete implementation "
            "with the recommended approach."
        ),
        "category": "rag",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t29_ab_testing_llm",
        "prompt": (
            "I need to set up A/B testing infrastructure for comparing "
            "LLM versions in production. We deploy new model versions "
            "weekly and need to know if they're better before full "
            "rollout.\n\n"
            "1. Metrics: what to measure (quality, latency, cost, user "
            "satisfaction, task completion rate)\n"
            "2. Statistical framework: sample size calculation, "
            "sequential testing (don't wait for fixed sample), "
            "multiple comparison correction\n"
            "3. Traffic splitting: how to route users to different "
            "model versions with consistent assignment\n"
            "4. LLM-as-judge for automated quality scoring: rubric "
            "design, inter-rater reliability with human labels\n"
            "5. Implementation: FastAPI middleware for routing, "
            "logging pipeline, dashboard queries\n"
            "6. Decision framework: when to promote, when to rollback"
        ),
        "category": "mlops",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t30_embedding_finetuning",
        "prompt": (
            "My RAG system's retrieval quality is poor because the "
            "off-the-shelf embedding model (BGE-large) doesn't understand "
            "my domain (legal contracts). I want to fine-tune it.\n\n"
            "1. How to create training data: mining hard negatives from "
            "my corpus, generating synthetic queries\n"
            "2. Fine-tuning approach: sentence-transformers vs custom "
            "contrastive loss\n"
            "3. Complete training script with InfoNCE loss, hard negative "
            "mining, and in-batch negatives\n"
            "4. Evaluation: how to measure embedding quality (recall@k, "
            "MRR, NDCG) with a held-out test set\n"
            "5. Deployment: how to update embeddings in production "
            "without downtime (re-index strategy)\n"
            "6. Hardware: can I fine-tune BGE-large on 1xA100 40GB?"
        ),
        "category": "embeddings",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
    {
        "id": "t31_debug_oom_distributed",
        "prompt": (
            "I'm getting OOM errors when fine-tuning Llama-3.1-70B "
            "with DeepSpeed ZeRO-3 on 4xA100 80GB. The error occurs "
            "at step 1 during the backward pass.\n\n"
            "Config:\n"
            "- DeepSpeed ZeRO stage 3, offload_param to CPU\n"
            "- bf16, batch_size=1, grad_accum=16\n"
            "- QLoRA with bnb 4bit, lora_r=64\n"
            "- max_seq_length=4096\n"
            "- activation_checkpointing enabled\n\n"
            "nvidia-smi shows all 4 GPUs at 78/80GB just before crash.\n"
            "Error: torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 2.00 GiB\n\n"
            "I thought ZeRO-3 should shard everything. Why is each "
            "GPU using 78GB? How do I fix this?"
        ),
        "category": "debugging",
        "expected_skill": "ml-debug",
        "needs_gpu": False,
    },
    {
        "id": "t32_model_distillation",
        "prompt": (
            "I want to distill GPT-4o's capabilities into a Llama-3-8B "
            "model for my specific task (customer support classification "
            "+ response generation). Budget: $300 in API calls.\n\n"
            "1. Distillation pipeline: generate teacher labels from "
            "GPT-4o, then fine-tune the student\n"
            "2. How to maximize quality per dollar: which prompting "
            "strategy extracts the most useful signal from the teacher?\n"
            "3. What data to generate: just answers, or also chain-of-"
            "thought reasoning?\n"
            "4. Complete pipeline code: data generation, quality "
            "filtering, student training\n"
            "5. Evaluation: how to measure the distilled model against "
            "the teacher\n"
            "6. Legal considerations: is this allowed under OpenAI's "
            "terms of service?"
        ),
        "category": "distillation",
        "expected_skill": "ml-plan",
        "needs_gpu": False,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


INLINE_SUFFIX = (
    "\n\nIMPORTANT: Present your COMPLETE solution directly in your response "
    "text. Include all code, configs, and commands inline. Do NOT create or "
    "write to separate files."
)


def run_claude(prompt: str, test_id: str, mode: str, max_turns: int,
               timeout: int, use_plugin: bool, round_dir: Path) -> dict:
    """Run claude CLI and return parsed output.

    mode: 'baseline' or 'plugin' — used for file naming.
    """
    full_prompt = prompt + INLINE_SUFFIX

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["PATH"] = (
        f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}"
        f":{env.get('PATH', '')}"
    )

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", full_prompt,
        "--max-turns", str(max_turns),
        "--output-format", "json",
    ]

    if use_plugin:
        cmd.extend(["--plugin-dir", str(PLUGIN_DIR)])
        cmd.extend([
            "--allowedTools",
            "mcp__leeroopedia__*,mcp__plugin_leeroopedia_leeroopedia__*,"
            "Read,Glob,Grep,Agent,Skill",
        ])

    out_file = round_dir / f"{test_id}.{mode}.json"

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env, cwd=str(PLUGIN_DIR),
        )
        elapsed = time.time() - start

        output_data = None
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

        report = {
            "test_id": test_id,
            "mode": mode,
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 1),
        }

        if output_data and isinstance(output_data, dict):
            report["result"] = output_data.get("result", "")
            report["num_turns"] = output_data.get("num_turns", 0)
            report["cost_usd"] = round(output_data.get("total_cost_usd", 0), 3)

        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        report = {
            "test_id": test_id, "mode": mode, "error": "timeout",
            "elapsed_sec": round(elapsed, 1),
        }
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as e:
        report = {"test_id": test_id, "mode": mode, "error": str(e)}
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_responses(question: str, baseline_result: str, plugin_result: str,
                    test_id: str, round_dir: Path,
                    needs_gpu: bool = False) -> dict:
    """Call Claude as judge to compare baseline vs plugin responses."""
    judge_prompt_path = PLUGIN_DIR / "scripts" / "judge_prompt.md"
    judge_system = judge_prompt_path.read_text(encoding="utf-8")

    # Randomize position to avoid bias
    coin = random.random() > 0.5
    if coin:
        response_a, response_b = baseline_result, plugin_result
        mapping = {"a": "baseline", "b": "plugin"}
    else:
        response_a, response_b = plugin_result, baseline_result
        mapping = {"a": "plugin", "b": "baseline"}

    gpu_note = (
        "\n\n**Note:** This question requires GPU hardware that was not "
        "available during generation. Score actionability based on whether "
        "the code/config would work given the specified hardware, not "
        "whether it was actually executed.\n"
        if needs_gpu else ""
    )

    full_prompt = (
        f"{judge_system}\n\n---\n\n"
        f"## Question\n\n{question}{gpu_note}\n\n"
        f"## Response A\n\n{response_a}\n\n"
        f"## Response B\n\n{response_b}"
    )

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["PATH"] = (
        f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}"
        f":{env.get('PATH', '')}"
    )

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", full_prompt,
        "--max-turns", "1",
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=JUDGE_TIMEOUT, env=env, cwd=str(PLUGIN_DIR),
        )

        output_data = json.loads(result.stdout)
        judge_text = output_data.get("result", "")

        # Extract JSON from judge response (may have markdown wrapping)
        json_match = re.search(r'\{[\s\S]*\}', judge_text)
        if not json_match:
            err = {"test_id": test_id, "error": "judge returned no JSON",
                   "raw": judge_text[:500],
                   "stderr": result.stderr[:500] if result.stderr else "",
                   "exit_code": result.returncode}
            err_file = round_dir / f"{test_id}.judge_error.json"
            err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
            return err

        raw_scores = json.loads(json_match.group())

        # Unmap positions back to baseline/plugin
        scores: dict = {}
        for role in ("baseline", "plugin"):
            pos = [k for k, v in mapping.items() if v == role][0]
            key = f"response_{pos}"
            scores[role] = raw_scores.get(key, {})

        # Compute totals
        for role in ("baseline", "plugin"):
            total = sum(
                dim.get("score", 0) if isinstance(dim, dict) else 0
                for dim in scores[role].values()
            )
            scores[f"{role}_score"] = total

        scores["value_add"] = scores["plugin_score"] - scores["baseline_score"]

        # Map winner back
        raw_winner = raw_scores.get("winner", "tie")
        if raw_winner in mapping:
            scores["winner"] = mapping[raw_winner]
        else:
            scores["winner"] = raw_winner  # "tie" passes through
        scores["winner_reasoning"] = raw_scores.get("winner_reasoning", "")
        scores["position_mapping"] = mapping
        scores["test_id"] = test_id
        scores["judge_cost_usd"] = round(
            output_data.get("total_cost_usd", 0), 3)

        out_file = round_dir / f"{test_id}.judge.json"
        out_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return scores

    except json.JSONDecodeError as e:
        err = {"test_id": test_id, "error": f"judge JSON parse error: {e}",
               "stdout": result.stdout[:500] if result.stdout else ""}
        err_file = round_dir / f"{test_id}.judge_error.json"
        err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
        return err
    except subprocess.TimeoutExpired:
        err = {"test_id": test_id, "error": "judge timeout"}
        err_file = round_dir / f"{test_id}.judge_error.json"
        err_file.write_text(json.dumps(err, indent=2), encoding="utf-8")
        return err
    except Exception as e:
        return {"test_id": test_id, "error": f"judge failed: {e}"}


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def parse_efficiency(plugin_report: dict, test_id: str,
                     round_dir: Path) -> dict:
    """Analyze KB call efficiency from plugin run."""
    result_text = plugin_report.get("result", "")

    # Find all citations in final response
    response_citations = set(re.findall(r'\[[\w]+/[\w_]+\]', result_text))
    total_citations = len(response_citations)

    # Estimate KB calls from turn count (proxy — each KB-using turn is ~1-2 calls)
    num_turns = plugin_report.get("num_turns", 0)
    estimated_kb_calls = max(num_turns // 2, 1)

    efficiency = {
        "test_id": test_id,
        "citations_in_response": total_citations,
        "unique_citations": sorted(response_citations)[:20],
        "estimated_kb_calls": estimated_kb_calls,
        "efficiency_ratio": round(
            total_citations / max(estimated_kb_calls, 1), 2),
    }

    if total_citations == 0:
        efficiency["assessment"] = (
            "No citations in response — KB may not have contributed.")
    elif efficiency["efficiency_ratio"] >= 1.0:
        efficiency["assessment"] = (
            "Good — multiple citations per estimated KB call.")
    elif efficiency["efficiency_ratio"] >= 0.5:
        efficiency["assessment"] = (
            "Moderate — most calls contributed.")
    else:
        efficiency["assessment"] = (
            "Low — many calls may have been redundant.")

    out_file = round_dir / f"{test_id}.efficiency.json"
    out_file.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    return efficiency


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------


def parse_sections(content: str) -> list[dict]:
    """Parse a skill file into named sections for targeted editing."""
    lines = content.splitlines()
    sections = []
    current: dict | None = None

    for i, line in enumerate(lines):
        if re.match(r'^#{1,3}\s', line):
            if current:
                current["end"] = i
                sections.append(current)
            current = {"name": line.strip("# ").strip(),
                       "header": line, "start": i, "end": len(lines)}
        elif i == 0 and line.startswith("---"):
            current = {"name": "frontmatter",
                       "header": "---", "start": 0, "end": len(lines)}

    if current:
        current["end"] = len(lines)
        sections.append(current)

    return sections


def run_refiner(judge_results: list, pass_num: int,
                round_dir: Path) -> dict:
    """Analyze judge results, generate targeted skill edits."""
    # Refine anything below perfect — goal is to maximize plugin quality
    weak = [
        j for j in judge_results
        if not j.get("error")
        and (j.get("plugin_score", 0) < 12 or j.get("value_add", 0) <= 0)
    ]

    if not weak:
        return {"pass": pass_num, "patterns_found": 0,
                "edits": [],
                "message": "All tests at 12/12 — nothing to refine."}

    # Group weaknesses by skill — include any dimension below 3
    by_skill: dict[str, list] = {}
    for j in weak:
        test_id = j["test_id"]
        test_def = next((t for t in TESTS if t["id"] == test_id), None)
        if not test_def:
            continue
        skill = test_def.get("expected_skill", "unknown")
        if skill not in by_skill:
            by_skill[skill] = []

        plugin_dims = j.get("plugin", {})
        for dim_name, dim_data in plugin_dims.items():
            if isinstance(dim_data, dict) and dim_data.get("score", 3) < 3:
                by_skill[skill].append({
                    "test_id": test_id,
                    "dimension": dim_name,
                    "score": dim_data.get("score", 0),
                    "reasoning": dim_data.get("reasoning", ""),
                })

    if not any(by_skill.values()):
        return {"pass": pass_num, "patterns_found": 0,
                "edits": [],
                "message": "No actionable weakness patterns found."}

    all_edits = []
    modified_skills: set[str] = set()  # Track which skills get edited

    for skill_name, weaknesses in by_skill.items():
        if not weaknesses:
            continue

        skill_path = PLUGIN_DIR / "skills" / skill_name / "SKILL.md"
        if not skill_path.exists():
            skill_path = (
                PLUGIN_DIR / "skills" / "using-leeroopedia" / "SKILL.md")
        if not skill_path.exists():
            continue

        current_content = skill_path.read_text(encoding="utf-8")
        sections = parse_sections(current_content)

        section_list = "\n".join(
            f"  - [{s['start']+1}-{s['end']}] {s['name']}"
            for s in sections
        )

        # Batch weaknesses to avoid prompt overflow (max 6 per batch)
        MAX_WEAKNESSES_PER_BATCH = 6
        batches = [weaknesses[i:i + MAX_WEAKNESSES_PER_BATCH]
                    for i in range(0, len(weaknesses), MAX_WEAKNESSES_PER_BATCH)]

        skill_applied_all = []
        skill_failed_all = []
        skill_skipped_all = []
        modified = current_content
        backup_path = None

        for batch_idx, batch in enumerate(batches):
            weakness_summary = "\n".join(
                f"- {w['test_id']}: {w['dimension']} scored {w['score']}/3 "
                f"— {w['reasoning']}"
                for w in batch
            )

            refiner_prompt = (
                f"You are improving an ML workflow skill file based on test "
                f"feedback.\n\n"
                f"## Skill file: {skill_name}\n\n"
                f"```\n{modified}\n```\n\n"
                f"## Sections in this file:\n{section_list}\n\n"
                f"## Weaknesses found by judge:\n\n{weakness_summary}\n\n"
                f"## Your task:\n"
                f"For each weakness, decide:\n"
                f"1. Is this fixable by editing the skill instructions? "
                f"(e.g., adding a warning to Anti-Patterns, adding a step "
                f"to a Phase, adding a tool call reminder)\n"
                f"2. Or is this a general code quality issue that skill "
                f"instructions can't fix? (e.g., syntax errors in generated "
                f"code, formatting issues) — if so, skip it.\n\n"
                f"For each fixable weakness, produce a targeted edit.\n\n"
                f"## Output format — return ONLY valid JSON:\n"
                f"```json\n"
                f'{{"edits": [\n'
                f'  {{\n'
                f'    "section": "section name",\n'
                f'    "action": "add_after | replace",\n'
                f'    "find": "exact line(s) to find in the file",\n'
                f'    "content": "new line(s) to add after find / '
                f'replace find with",\n'
                f'    "reason": "what this fixes"\n'
                f'  }}\n'
                f'],\n'
                f'"skipped": [\n'
                f'  {{"weakness": "...", '
                f'"reason": "not skill-fixable because..."}}\n'
                f']\n'
                f'}}\n'
                f"```\n\n"
                f"Rules:\n"
                f"- `find` must be an EXACT substring from the current "
                f"file\n"
                f"- `add_after`: inserts `content` on the line after "
                f"`find`\n"
                f"- `replace`: replaces `find` with `content`\n"
                f"- Keep edits small — 1-3 lines each\n"
                f"- Do NOT touch the YAML frontmatter (---)\n"
                f"- Do NOT delete Iron Laws or phase structure"
            )

            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            env["PATH"] = (
                f"{Path.home() / '.local/bin'}"
                f":{Path.home() / 'miniconda3/bin'}"
                f":{env.get('PATH', '')}"
            )

            try:
                result = subprocess.run(
                    ["claude", "--dangerously-skip-permissions",
                     "-p", refiner_prompt, "--max-turns", "1",
                     "--output-format", "json"],
                    capture_output=True, text=True, timeout=180,
                    env=env, cwd=str(PLUGIN_DIR),
                )
                output_data = json.loads(result.stdout)
                refiner_text = output_data.get("result", "")

                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', refiner_text)
                if not json_match:
                    if batch_idx == 0 and len(batches) == 1:
                        all_edits.append({
                            "file": str(
                                skill_path.relative_to(PLUGIN_DIR)),
                            "status": "error",
                            "reason": "refiner returned no JSON",
                            "raw": refiner_text[:300],
                            "weaknesses": weaknesses,
                        })
                    continue

                edit_plan = json.loads(json_match.group())
                edits = edit_plan.get("edits", [])
                skipped = edit_plan.get("skipped", [])
                skill_skipped_all.extend(skipped)

                for edit in edits:
                    find_str = edit.get("find", "")
                    content_str = edit.get("content", "")
                    action = edit.get("action", "add_after")
                    reason = edit.get("reason", "")

                    if not find_str or find_str not in modified:
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": "find string not found in file",
                        })
                        continue

                    if action == "replace":
                        new_modified = modified.replace(
                            find_str, content_str, 1)
                    elif action == "add_after":
                        new_modified = modified.replace(
                            find_str, find_str + "\n" + content_str, 1)
                    else:
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": f"unknown action: {action}",
                        })
                        continue

                    # Sanity: don't grow more than 20% over original
                    if (len(new_modified.splitlines())
                            > len(current_content.splitlines()) * 1.2):
                        skill_failed_all.append({
                            "edit": edit,
                            "reason": "would exceed 20% growth limit",
                        })
                        continue

                    modified = new_modified
                    skill_applied_all.append({
                        "action": action,
                        "section": edit.get("section", ""),
                        "reason": reason,
                        "lines_added": (
                            len(modified.splitlines())
                            - len(current_content.splitlines())),
                    })

            except Exception as e:
                if batch_idx == 0 and len(batches) == 1:
                    all_edits.append({
                        "file": str(
                            skill_path.relative_to(PLUGIN_DIR)),
                        "status": "error",
                        "reason": str(e),
                        "weaknesses": weaknesses,
                    })

        # After all batches for this skill — write once
        if skill_applied_all:
            backup_path = (
                round_dir / f"{skill_name}.pass{pass_num}.backup.md")
            backup_path.write_text(current_content, encoding="utf-8")
            skill_path.write_text(modified, encoding="utf-8")
            modified_skills.add(skill_name)

        all_edits.append({
            "file": str(skill_path.relative_to(PLUGIN_DIR)),
            "status": ("applied" if skill_applied_all
                        else "no_valid_edits"),
            "applied": skill_applied_all,
            "failed": skill_failed_all,
            "skipped": skill_skipped_all,
            "backup": str(backup_path) if skill_applied_all else None,
            "weaknesses": weaknesses,
        })

    # Only retest tests whose skill was actually modified
    retest_ids = list({
        w["test_id"] for skill_name, ws in by_skill.items()
        for w in ws
        if skill_name in modified_skills
    })

    refiner_result = {
        "pass": pass_num,
        "patterns_found": sum(len(ws) for ws in by_skill.values()),
        "edits": all_edits,
        "retest_needed": retest_ids,
        "modified_skills": sorted(modified_skills),
    }

    out_file = round_dir / f"refine_pass{pass_num}.json"
    out_file.write_text(json.dumps(refiner_result, indent=2), encoding="utf-8")
    return refiner_result


# ---------------------------------------------------------------------------
# Retest + regression check
# ---------------------------------------------------------------------------


def retest_weak(test_ids: list[str], retest_dir: Path,
                parent_dir: Path) -> list[dict]:
    """Re-run only the weak tests after a refiner pass."""
    tests = [t for t in TESTS if t["id"] in test_ids]
    results = []
    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]
        print(f"    Retesting {test_id}...")

        plugin = run_claude(
            prompt, test_id, "plugin",
            max_turns=PLUGIN_MAX_TURNS, timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=retest_dir,
        )

        # Read baseline from parent round dir
        baseline_file = parent_dir / f"{test_id}.baseline.json"
        if baseline_file.exists():
            baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        else:
            baseline = {"result": ""}

        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            judge = judge_responses(
                prompt, baseline_text, plugin_text,
                test_id, retest_dir,
                needs_gpu=test.get("needs_gpu", False),
            )
            if not judge.get("error"):
                delta = judge.get("value_add", 0)
                print(f"      Plugin: {judge.get('plugin_score', '?')}/12 | "
                      f"Delta: {delta:+d}")
            else:
                print(f"      Judge error: {judge.get('error')}")
        else:
            judge = {"test_id": test_id, "error": "missing result"}

        results.append(judge)
    return results


def check_regressions(old_judges: list, new_judges: list,
                      refine_result: dict):
    """Check for regressions on modified-skill tests only.

    Only revert the specific skill file that caused the regression.
    Requires a drop of >= 3 points to count as a regression (not noise).
    """
    REGRESSION_THRESHOLD = 3  # Must drop by this much to trigger revert

    # Build skill->file mapping from refine result
    skill_to_edit = {}
    for edit_group in refine_result.get("edits", []):
        if edit_group.get("status") == "applied" and edit_group.get("backup"):
            # Extract skill name from file path (e.g. "skills/ml-debug/SKILL.md" -> "ml-debug")
            parts = edit_group["file"].split("/")
            if len(parts) >= 2:
                skill_to_edit[parts[1]] = edit_group

    old_by_id = {j["test_id"]: j for j in old_judges if not j.get("error")}
    regressed_skills: set[str] = set()

    for new_j in new_judges:
        tid = new_j.get("test_id")
        if new_j.get("error") or tid not in old_by_id:
            continue

        # Find which skill this test maps to
        test_def = next((t for t in TESTS if t["id"] == tid), None)
        if not test_def:
            continue
        test_skill = test_def.get("expected_skill", "unknown")

        # Only check tests whose skill was actually modified
        if test_skill not in skill_to_edit:
            continue

        old_score = old_by_id[tid].get("plugin_score", 0)
        new_score = new_j.get("plugin_score", 0)
        drop = old_score - new_score

        if drop >= REGRESSION_THRESHOLD:
            print(f"    REGRESSION on {tid}: {old_score} -> {new_score} "
                  f"(drop={drop}). Reverting {test_skill}.")
            regressed_skills.add(test_skill)

    # Revert only the specific skills that regressed
    for skill_name in regressed_skills:
        edit_group = skill_to_edit.get(skill_name)
        if not edit_group:
            continue
        backup_path = Path(edit_group["backup"])
        target_path = PLUGIN_DIR / edit_group["file"]
        if backup_path.exists():
            target_path.write_text(
                backup_path.read_text(encoding="utf-8"),
                encoding="utf-8")
            print(f"      Reverted {edit_group['file']}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_summary(judge_results: list):
    """Print the A/B comparison summary table."""
    print(f"\n{'='*95}")
    print("A/B COMPARISON SUMMARY")
    print(f"{'='*95}")
    header = (f"{'Test':<30} {'Base':>5} {'Plugin':>7} {'Delta':>6} "
              f"{'Winner':<8} {'Eff':>5} {'Reason'}")
    print(header)
    print("-" * 95)

    total_base = 0
    total_plugin = 0
    wins = {"baseline": 0, "plugin": 0, "tie": 0}
    count = 0

    for j in judge_results:
        if j.get("error"):
            print(f"{j.get('test_id', '?'):<30} ERROR: {j['error'][:50]}")
            continue
        b = j.get("baseline_score", 0)
        p = j.get("plugin_score", 0)
        d = j.get("value_add", 0)
        w = j.get("winner", "?")
        e = j.get("efficiency", 0)
        reason = j.get("winner_reasoning", "")[:40]
        print(f"{j['test_id']:<30} {b:>4}/12 {p:>5}/12 {d:>+5} "
              f"{w:<8} {e:>5.2f} {reason}")
        total_base += b
        total_plugin += p
        wins[w] = wins.get(w, 0) + 1
        count += 1

    if count:
        print("-" * 95)
        print(f"{'AVERAGE':<30} {total_base/count:>5.1f} "
              f"{total_plugin/count:>7.1f} "
              f"{(total_plugin - total_base)/count:>+5.1f}")
        print(f"Plugin wins: {wins.get('plugin', 0)} | "
              f"Baseline wins: {wins.get('baseline', 0)} | "
              f"Ties: {wins.get('tie', 0)}")


def save_round_summary(judge_results: list, refine1: dict, refine2: dict,
                       round_dir: Path, round_num: int):
    """Save the complete round summary."""
    summary = {
        "round": round_num,
        "tests": len(judge_results),
        "judge_results": judge_results,
        "refine_pass1": refine1,
        "refine_pass2": refine2,
    }
    out_file = round_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRound {round_num} results saved to {round_dir}/")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(test_ids: list[str] | None = None):
    """Run the full A/B + judge + refiner pipeline."""
    tests = ([t for t in TESTS if t["id"] in test_ids]
             if test_ids else TESTS)

    # Create round directory
    existing = list(LOG_DIR.glob("round*"))
    round_num = max(
        (int(m.group())
         for d in existing
         if (m := re.search(r'\d+', d.name))),
        default=5,
    ) + 1
    round_dir = LOG_DIR / f"round{round_num}"
    round_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ROUND {round_num} — A/B Quality Evaluation")
    print(f"{'='*70}")

    all_judge_results = []

    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]
        gpu_note = " [GPU]" if test.get("needs_gpu") else " [E2E]"

        print(f"\n--- {test_id} [{test.get('category')}]{gpu_note} ---")

        # Phase 1: Baseline (no plugin)
        print(f"  [1/4] Baseline run (no plugin)...")
        baseline = run_claude(
            prompt, test_id, "baseline",
            max_turns=BASELINE_MAX_TURNS,
            timeout=BASELINE_TIMEOUT,
            use_plugin=False, round_dir=round_dir,
        )
        b_chars = len(baseline.get("result", ""))
        b_err = baseline.get("error", "")
        if b_err:
            print(f"        ERROR: {b_err}")
        else:
            print(f"        {b_chars} chars, "
                  f"{baseline.get('num_turns', '?')} turns, "
                  f"${baseline.get('cost_usd', 0):.2f}")

        # Phase 2: Plugin run
        print(f"  [2/4] Plugin run...")
        plugin = run_claude(
            prompt, test_id, "plugin",
            max_turns=PLUGIN_MAX_TURNS,
            timeout=PLUGIN_TIMEOUT,
            use_plugin=True, round_dir=round_dir,
        )
        p_chars = len(plugin.get("result", ""))
        p_err = plugin.get("error", "")
        if p_err:
            print(f"        ERROR: {p_err}")
        else:
            print(f"        {p_chars} chars, "
                  f"{plugin.get('num_turns', '?')} turns, "
                  f"${plugin.get('cost_usd', 0):.2f}")

        # Phase 3: Judge
        baseline_text = baseline.get("result", "")
        plugin_text = plugin.get("result", "")

        if baseline_text and plugin_text:
            print(f"  [3/4] Judge evaluation...")
            judge = judge_responses(
                prompt, baseline_text, plugin_text,
                test_id, round_dir,
                needs_gpu=test.get("needs_gpu", False),
            )
            if not judge.get("error"):
                print(
                    f"        Baseline: "
                    f"{judge.get('baseline_score', '?')}/12 | "
                    f"Plugin: {judge.get('plugin_score', '?')}/12 | "
                    f"Delta: {judge.get('value_add', '?'):+d} | "
                    f"Winner: {judge.get('winner', '?')}")
                reason = judge.get("winner_reasoning", "")
                if reason:
                    print(f"        {reason[:120]}")
            else:
                print(f"        Judge error: {judge.get('error')}")
        else:
            judge = {"test_id": test_id,
                     "error": "missing baseline or plugin result"}
            print(f"  [3/4] Judge skipped — missing response")

        # Phase 4: Efficiency
        if plugin_text:
            print(f"  [4/4] Efficiency analysis...")
            eff = parse_efficiency(plugin, test_id, round_dir)
            print(f"        {eff.get('citations_in_response', 0)} citations, "
                  f"eff={eff.get('efficiency_ratio', 0):.2f}, "
                  f"{eff.get('assessment', '')}")
            judge["efficiency"] = eff.get("efficiency_ratio", 0)
        else:
            print(f"  [4/4] Efficiency skipped — no plugin result")

        all_judge_results.append(judge)

    # Summary
    print_summary(all_judge_results)

    # Refiner pass 1
    print(f"\n{'='*70}")
    print("REFINER PASS 1")
    print(f"{'='*70}")
    refine1 = run_refiner(all_judge_results, 1, round_dir)
    applied1 = sum(
        len(e.get("applied", []))
        for e in refine1.get("edits", [])
    )
    print(f"  Patterns: {refine1.get('patterns_found', 0)} | "
          f"Edits applied: {applied1} | "
          f"Retest: {refine1.get('retest_needed', [])}")
    for e in refine1.get("edits", []):
        for a in e.get("applied", []):
            print(f"    + {e['file']} [{a['section']}]: {a['reason']}")
        for s in e.get("skipped", []):
            print(f"    ~ skipped: {s.get('reason', '')[:80]}")

    retest1_judges = []
    if refine1.get("retest_needed") and applied1 > 0:
        print("  Re-running weak tests after pass 1...")
        retest1_dir = round_dir / "refine_pass1_retest"
        retest1_dir.mkdir(exist_ok=True)
        retest1_judges = retest_weak(
            refine1["retest_needed"], retest1_dir, round_dir)
        check_regressions(all_judge_results, retest1_judges, refine1)

    # Merge retest results into full list for pass 2
    retested_ids = {
        j["test_id"] for j in retest1_judges if not j.get("error")}
    merged_judges = []
    for j in all_judge_results:
        retest_ver = next(
            (r for r in retest1_judges
             if r.get("test_id") == j.get("test_id")
             and not r.get("error")),
            None,
        )
        merged_judges.append(retest_ver if retest_ver else j)

    # Refiner pass 2 — sees ALL results with retests merged in
    print(f"\n{'='*70}")
    print("REFINER PASS 2")
    print(f"{'='*70}")
    refine2 = run_refiner(merged_judges, 2, round_dir)
    applied2 = sum(
        len(e.get("applied", []))
        for e in refine2.get("edits", [])
    )
    print(f"  Patterns: {refine2.get('patterns_found', 0)} | "
          f"Edits applied: {applied2} | "
          f"Retest: {refine2.get('retest_needed', [])}")
    for e in refine2.get("edits", []):
        for a in e.get("applied", []):
            print(f"    + {e['file']} [{a['section']}]: {a['reason']}")
        for s in e.get("skipped", []):
            print(f"    ~ skipped: {s.get('reason', '')[:80]}")

    if refine2.get("retest_needed") and applied2 > 0:
        print("  Re-running weak tests after pass 2...")
        retest2_dir = round_dir / "refine_pass2_retest"
        retest2_dir.mkdir(exist_ok=True)
        retest2_judges = retest_weak(
            refine2["retest_needed"], retest2_dir, round_dir)
        check_regressions(
            merged_judges, retest2_judges, refine2)

    # Save final summary
    save_round_summary(
        all_judge_results, refine1, refine2, round_dir, round_num)


def main():
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]
        valid = [t["id"] for t in TESTS]
        invalid = [t for t in test_ids if t not in valid]
        if invalid:
            print(f"Unknown test IDs: {invalid}")
            print(f"Available: {valid}")
            sys.exit(1)
        run_full_pipeline(test_ids)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
