#!/usr/bin/env python3
"""
Production test harness for leeroopedia plugin.
Launches Claude Code with the plugin and runs each test scenario.
Captures full logs for analysis.
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

# Test scenarios from the plan
TESTS = [
    {
        "id": "test1_basic_setup",
        "prompt": "List all MCP tools available from the leeroopedia server. For each tool, show its name and a one-line description. Format as a markdown table.",
        "max_turns": 3,
        "expect_tools": ["search_knowledge", "build_plan", "review_plan", "verify_code_math", "diagnose_failure", "propose_hypothesis", "query_hyperparameter_priors", "get_page"],
    },
    {
        "id": "test2_rag_planning",
        "prompt": "Give me an implementation plan for a self-improving RAG system. Stack: FastAPI backend, ChromaDB for vectors, BM25 via rank_bm25 for keyword search, BGE-base-en for embeddings. It should: (1) hybrid retrieval with score fusion, (2) diagnose retrieval failures via recall metrics, (3) re-chunk documents adaptively when recall drops. Target: 100k technical docs, sub-2s latency. Don't ask me questions — just give me the plan.",
        "max_turns": 10,
        "timeout": 480,
        "expect_tools": ["build_plan", "review_plan"],
        "expect_skills": ["ml-plan"],
    },
    {
        "id": "test3_debug_oom",
        "prompt": "My QLoRA fine-tuning of Qwen2.5-7B is crashing with OOM on A100 40GB. Batch size 4, seq length 4096. Error: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "max_turns": 10,
        "expect_tools": ["diagnose_failure"],
        "expect_skills": ["ml-debug"],
    },
    {
        "id": "test4_verify_lora",
        "prompt": "Is this LoRA config correct for fine-tuning Llama-3 8B? lora_r=64, lora_alpha=16, lr=5e-5, target_modules=['q_proj','v_proj']",
        "max_turns": 10,
        "expect_tools": ["verify_code_math", "query_hyperparameter_priors"],
        "expect_skills": ["ml-verify"],
    },
    {
        "id": "test5_iterate",
        "prompt": "I've been building a RAG system. BM25-only gets recall@5=30%, vector-only gets 35%. I tried naive fusion (average scores) and got 36%. How can I improve further?",
        "max_turns": 10,
        "expect_tools": ["propose_hypothesis"],
        "expect_skills": ["ml-iterate"],
    },
    {
        "id": "test6_research",
        "prompt": "How does the OpenAI Agents SDK handle handoffs between specialist agents? What are the best practices for typed handoffs?",
        "max_turns": 10,
        "expect_tools": ["search_knowledge"],
        "expect_skills": ["ml-research"],
    },
    {
        "id": "test7_agent",
        "prompt": "Use the ml-expert agent to compare DeepSpeed ZeRO-3 vs FSDP for SFT of a 7B model on 4x A100 80GB with 4096 seq length. Which should I use? Include memory estimates and a concrete config.",
        "max_turns": 12,
        "timeout": 600,
        "expect_tools": ["search_knowledge"],
    },
    {
        "id": "test8_complex_debug",
        "prompt": "I'm running DPO training on Llama-3-8B-Instruct using TRL. After 200 steps, the chosen reward and rejected reward both diverge to -inf. Loss goes to 0. Config: beta=0.1, lr=5e-7, bf16, lora_r=16. What's happening and how do I fix it?",
        "max_turns": 10,
        "timeout": 300,
        "expect_tools": ["diagnose_failure", "query_hyperparameter_priors"],
    },
    {
        "id": "test9_multi_framework",
        "prompt": "I need to serve a Mixtral 8x7B model with 4-bit quantization for production. Compare vLLM vs SGLang vs TensorRT-LLM for this specific use case. I have 2xA100 80GB. Include config examples.",
        "max_turns": 10,
        "timeout": 480,
        "expect_tools": ["search_knowledge"],
    },
    {
        "id": "test10_verify_complex",
        "prompt": "Check this DeepSpeed ZeRO-3 config for distributed training of a 13B model on 8xH100:\n```json\n{\"train_micro_batch_size_per_gpu\": 4, \"gradient_accumulation_steps\": 8, \"zero_optimization\": {\"stage\": 3, \"offload_param\": {\"device\": \"cpu\"}, \"offload_optimizer\": {\"device\": \"cpu\"}, \"overlap_comm\": true, \"contiguous_gradients\": true, \"sub_group_size\": 1e9, \"reduce_bucket_size\": 5e8, \"stage3_prefetch_bucket_size\": 5e8, \"stage3_param_persistence_threshold\": 1e6}, \"bf16\": {\"enabled\": true}}\n```\nIs this optimal? What would you change?",
        "max_turns": 10,
        "timeout": 300,
        "expect_tools": ["verify_code_math", "query_hyperparameter_priors"],
    },
]


def run_test(test: dict) -> dict:
    """Run a single test scenario and capture output."""
    test_id = test["id"]
    prompt = test["prompt"]
    max_turns = test.get("max_turns", 10)

    print(f"\n{'='*70}")
    print(f"RUNNING: {test_id}")
    print(f"PROMPT: {prompt[:100]}...")
    print(f"{'='*70}")

    env = os.environ.copy()
    env["CLAUDECODE"] = ""  # Allow nested execution
    env["PATH"] = f"{Path.home() / '.local/bin'}:{Path.home() / 'miniconda3/bin'}:{env.get('PATH', '')}"

    log_file = LOG_DIR / f"{test_id}.json"
    text_log = LOG_DIR / f"{test_id}.txt"

    cmd = [
        "claude",
        "--plugin-dir", str(PLUGIN_DIR),
        "--dangerously-skip-permissions",
        "-p", prompt,
        "--max-turns", str(max_turns),
        "--output-format", "json",
        # Allow MCP tools and core tools, but block AskUserQuestion
        # (not supported in non-interactive -p mode)
        "--allowedTools", "mcp__leeroopedia__*,mcp__plugin_leeroopedia_leeroopedia__*,Read,Write,Edit,Bash,Glob,Grep,Agent,Skill",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test.get("timeout", 300),
            env=env,
            cwd=str(PLUGIN_DIR),
        )
        elapsed = time.time() - start

        # Save raw output
        text_log.write_text(
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nEXIT CODE: {result.returncode}\nTIME: {elapsed:.1f}s",
            encoding="utf-8",
        )

        # Try to parse JSON output
        output_data = None
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

        # Extract useful info
        report = {
            "test_id": test_id,
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_len": len(result.stdout),
            "stderr_len": len(result.stderr),
            "json_parsed": output_data is not None,
        }

        if output_data:
            # Extract tool calls and text from JSON output
            messages = output_data if isinstance(output_data, list) else [output_data]
            tool_calls = []
            skill_calls = []
            final_text = ""

            for msg in messages:
                if isinstance(msg, dict):
                    # Check for tool_use in content blocks
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_use":
                                    tool_name = block.get("name", "")
                                    tool_calls.append(tool_name)
                                    if tool_name == "Skill":
                                        skill_calls.append(block.get("input", {}).get("skill", ""))
                                elif block.get("type") == "text":
                                    final_text = block.get("text", "")
                    # Also check result field
                    if "result" in msg:
                        final_text = str(msg["result"])[:2000]

            report["tool_calls"] = tool_calls
            report["skill_calls"] = skill_calls
            report["mcp_tools_used"] = [t for t in tool_calls if t.startswith("mcp__leeroopedia__")]
            report["has_citations"] = "[" in final_text and "/" in final_text
            report["response_preview"] = final_text[:500]

        log_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Print summary
        print(f"  Exit code: {result.returncode}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output length: {len(result.stdout)} chars")
        if output_data:
            print(f"  MCP tools used: {report.get('mcp_tools_used', [])}")
            print(f"  Skills invoked: {report.get('skill_calls', [])}")
            print(f"  Has citations: {report.get('has_citations', False)}")
            print(f"  Response preview: {report.get('response_preview', '')[:200]}...")

        return report

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"test_id": test_id, "error": "timeout", "elapsed_sec": round(elapsed, 1)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"test_id": test_id, "error": str(e)}


def main():
    # Run specific test or all tests
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
    print("TEST SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r.get("exit_code") == 0 and not r.get("error") else "FAIL"
        print(f"  {r['test_id']}: {status} ({r.get('elapsed_sec', '?')}s)")

    # Save full results
    summary_file = LOG_DIR / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {LOG_DIR}/")


if __name__ == "__main__":
    main()
