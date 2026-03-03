#!/usr/bin/env python3
"""
Interactive test harness for leeroopedia plugin.
Runs Claude Code with --resume to simulate multi-turn conversations.
Captures full conversation logs for analysis.
"""

import subprocess
import json
import os
import sys
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PLUGIN_DIR / "test-logs"
LOG_DIR.mkdir(exist_ok=True)

# 10 diverse ML/AI engineering tasks
TESTS = [
    {
        "id": "t01_agent_orchestration",
        "prompt": "I'm building a customer support agent system using the OpenAI Agents SDK. I need a triage agent that routes to 3 specialists: billing, technical, and account. The triage agent should pass structured context (customer tier, issue category, sentiment score) via typed handoffs. Give me the full implementation with proper error handling.",
        "max_turns": 10,
        "timeout": 600,
        "category": "agent-building",
    },
    {
        "id": "t02_qlora_multimodal",
        "prompt": "I want to fine-tune LLaVA-1.5-7B for medical image analysis using QLoRA. My data is 50k image-text pairs of X-rays with radiologist reports. I have 1xA100 80GB. Walk me through the complete setup: data format, config, training script, and evaluation. What target_modules should I use for the vision encoder vs the language model?",
        "max_turns": 10,
        "timeout": 720,
        "category": "fine-tuning",
    },
    {
        "id": "t03_serving_optimization",
        "prompt": "I'm serving Llama-3-70B-Instruct with vLLM on 4xA100 80GB in production. Current config: tensor_parallel=4, max_model_len=4096, gpu_memory_utilization=0.9. I'm getting p99 latency of 8 seconds for 2048-token outputs at 50 concurrent users. I need to get this under 3 seconds. What specific optimizations should I make? Show me the exact vLLM launch command and any code changes.",
        "max_turns": 10,
        "timeout": 480,
        "category": "inference-optimization",
    },
    {
        "id": "t04_distributed_training",
        "prompt": "I need to pre-train a 3B parameter GPT model from scratch on a custom dataset (200B tokens). I have a cluster of 4 nodes, each with 8xH100 80GB connected via InfiniBand. Should I use Megatron-LM, DeepSpeed, or FSDP? Give me the complete training config including parallelism strategy (TP, PP, DP split), batch size, learning rate schedule, and the launch command.",
        "max_turns": 12,
        "timeout": 600,
        "category": "distributed-training",
    },
    {
        "id": "t05_rag_evaluation",
        "prompt": "My RAG system is in production but users complain answers are sometimes wrong or incomplete. I need to build an automated evaluation pipeline. The system uses ChromaDB + BGE-large embeddings + GPT-4o for generation. I want to measure: retrieval quality (recall, MRR), answer faithfulness, answer relevance, and hallucination rate. Show me how to set this up with RAGAS or a similar framework, including synthetic test set generation.",
        "max_turns": 10,
        "timeout": 480,
        "category": "evaluation",
    },
    {
        "id": "t06_dpo_alignment",
        "prompt": "I fine-tuned Mistral-7B-v0.3 on my custom instruction data and it works well, but it sometimes generates unsafe content. I want to align it using DPO. I have 5k preference pairs (chosen/rejected). My setup: 2xA100 40GB. Give me the complete TRL DPO training config, explain the key hyperparameters (beta, learning rate, loss type), and tell me what can go wrong. What should my data format look like?",
        "max_turns": 10,
        "timeout": 720,
        "category": "alignment",
    },
    {
        "id": "t07_embedding_pipeline",
        "prompt": "I'm building a code search engine that needs to index 10M code snippets across Python, JavaScript, and Rust. I need to: (1) choose the right embedding model for code, (2) build an efficient indexing pipeline that can process 10M items in under 4 hours on 2xA100, (3) serve similarity search with sub-100ms latency. Compare code embedding models (CodeBERT vs StarEncoder vs Voyage Code vs OpenAI) and give me the full architecture.",
        "max_turns": 10,
        "timeout": 480,
        "category": "embeddings",
    },
    {
        "id": "t08_model_merging",
        "prompt": "I have three Llama-3-8B fine-tunes: one for coding, one for math, and one for creative writing. I want to merge them into a single model that's good at all three. Compare merging methods: TIES, DARE, SLERP, linear. Which should I use? Show me the mergekit config and the evaluation strategy to verify the merge didn't degrade any capability.",
        "max_turns": 10,
        "timeout": 480,
        "category": "model-merging",
    },
    {
        "id": "t09_speculative_decoding",
        "prompt": "I want to speed up inference of Llama-3-70B using speculative decoding. I'm considering using Llama-3-8B as the draft model. How does speculative decoding work exactly? What's the expected speedup? How do I set this up with vLLM? Are there any gotchas with the draft model choice (tokenizer compatibility, acceptance rate)? Give me the complete setup.",
        "max_turns": 10,
        "timeout": 480,
        "category": "inference-optimization",
    },
    {
        "id": "t10_training_debug_complex",
        "prompt": """I'm training a mixture-of-experts model using Mixtral architecture with DeepSpeed on 8xA100. After 5000 steps, I see these symptoms:
1. Router load balancing loss keeps increasing (started at 0.01, now at 0.8)
2. Only 2 of 8 experts are being used (expert utilization histogram is extremely skewed)
3. Training loss plateaued at 2.1 and won't decrease
4. GPU memory usage is uneven across GPUs (GPU 0: 72GB, others: ~45GB)

Config: expert_parallel_size=4, num_experts=8, top_k=2, aux_loss_coeff=0.01, lr=3e-4.

What's happening and how do I fix each issue?""",
        "max_turns": 10,
        "timeout": 480,
        "category": "debugging",
    },
]


def run_test(test: dict) -> dict:
    """Run a single test scenario and capture output."""
    test_id = test["id"]
    prompt = test["prompt"]
    max_turns = test.get("max_turns", 10)
    timeout = test.get("timeout", 300)

    print(f"\n{'='*70}")
    print(f"RUNNING: {test_id} [{test.get('category', '?')}]")
    print(f"PROMPT: {prompt[:120]}...")
    print(f"{'='*70}")

    env = os.environ.copy()
    env["CLAUDECODE"] = ""
    env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}:{env.get('PATH', '')}"

    # Ensure API key is available
    if "LEEROOPEDIA_API_KEY" not in env:
        print("  WARNING: LEEROOPEDIA_API_KEY not set! MCP tools will fail.")
        print("  Set it: export LEEROOPEDIA_API_KEY=kpsk_your_key_here")

    log_file = LOG_DIR / f"{test_id}.json"
    text_log = LOG_DIR / f"{test_id}.txt"

    cmd = [
        "claude",
        "--plugin-dir", str(PLUGIN_DIR),
        "--dangerously-skip-permissions",
        "-p", prompt,
        "--max-turns", str(max_turns),
        "--output-format", "json",
        "--allowedTools", "mcp__leeroopedia__*,mcp__plugin_leeroopedia_leeroopedia__*,Read,Write,Edit,Bash,Glob,Grep,Agent,Skill",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(PLUGIN_DIR),
        )
        elapsed = time.time() - start

        # Save raw output
        text_log.write_text(
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nEXIT CODE: {result.returncode}\nTIME: {elapsed:.1f}s",
            encoding="utf-8",
        )

        # Parse JSON output
        output_data = None
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

        report = {
            "test_id": test_id,
            "category": test.get("category", "unknown"),
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_len": len(result.stdout),
        }

        if output_data and isinstance(output_data, dict):
            response = output_data.get("result", "")
            denials = output_data.get("permission_denials", [])
            report["num_turns"] = output_data.get("num_turns", 0)
            report["cost_usd"] = round(output_data.get("total_cost_usd", 0), 3)
            report["result_len"] = len(response)
            report["permission_denials"] = len(denials)

            # Check for citations
            has_explicit_citations = bool(
                # [Something/Something] pattern
                __import__("re").search(r"\[[\w/]+\]", response)
            )
            report["has_citations"] = has_explicit_citations

            # Check for code blocks
            report["has_code"] = "```" in response

            # Check if response is empty or just questions
            report["is_empty"] = len(response) < 50
            report["asks_questions"] = "?" in response[:500] and len(response) < 500

            report["response_preview"] = response[:600]

        log_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        status = "PASS" if report.get("result_len", 0) > 200 and not report.get("is_empty") else "FAIL"
        print(f"  Status: {status}")
        print(f"  Turns: {report.get('num_turns', '?')} | Time: {elapsed:.0f}s | Cost: ${report.get('cost_usd', '?')}")
        print(f"  Result: {report.get('result_len', 0)} chars | Citations: {report.get('has_citations', '?')} | Code: {report.get('has_code', '?')}")
        print(f"  Denials: {report.get('permission_denials', 0)}")
        if report.get("response_preview"):
            print(f"  Preview: {report['response_preview'][:200]}...")

        return report

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.0f}s")
        text_log.write_text(f"TIMEOUT after {elapsed:.1f}s", encoding="utf-8")
        return {"test_id": test_id, "category": test.get("category"), "error": "timeout", "elapsed_sec": round(elapsed, 1)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"test_id": test_id, "category": test.get("category"), "error": str(e)}


def main():
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]
        tests = [t for t in TESTS if t["id"] in test_ids]
        if not tests:
            print(f"Unknown test IDs: {test_ids}")
            print(f"Available: {[t['id'] for t in TESTS]}")
            sys.exit(1)
    else:
        tests = TESTS

    results = []
    for test in tests:
        report = run_test(test)
        results.append(report)

    # Summary
    print(f"\n\n{'='*70}")
    print("FULL TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':<30} {'Category':<22} {'Status':<8} {'Time':>6} {'Chars':>7} {'Cites':>6} {'Cost':>7}")
    print("-" * 90)
    for r in results:
        status = "PASS" if r.get("result_len", 0) > 200 else ("TIMEOUT" if r.get("error") == "timeout" else "FAIL")
        time_s = f"{r.get('elapsed_sec', 0):.0f}s"
        chars = str(r.get("result_len", "-"))
        cites = "Yes" if r.get("has_citations") else "No"
        cost = f"${r.get('cost_usd', 0):.2f}" if "cost_usd" in r else "-"
        print(f"{r['test_id']:<30} {r.get('category', '?'):<22} {status:<8} {time_s:>6} {chars:>7} {cites:>6} {cost:>7}")

    summary_file = LOG_DIR / "round_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {LOG_DIR}/")


if __name__ == "__main__":
    main()
