#!/usr/bin/env python3
"""Generate the SVG dumbbell chart for TESTED_TASKS.md."""

TASKS = [
    # (short_name, category, plugin_score, baseline_score)
    # Fine-Tuning & Training
    ("Multimodal QLoRA", "Fine-Tuning", 15, 8),
    ("DPO Alignment", "Fine-Tuning", 15, 8),
    ("GRPO Alignment", "Fine-Tuning", 15, 8),
    ("Distributed Pretraining", "Fine-Tuning", 15, 6),
    ("Continual Pretraining", "Fine-Tuning", 14, 9),
    ("Vision Model Fine-tune", "Fine-Tuning", 14, 8),
    ("Embedding Fine-tune", "Fine-Tuning", 13, 10),
    ("Knowledge Distillation", "Fine-Tuning", 13, 8),
    ("Synthetic Data Pipeline", "Fine-Tuning", 13, 9),
    # Debugging & Verification
    ("Pre-launch Config Audit", "Debug / Verify", 15, 9),
    ("Post-training Iteration", "Debug / Verify", 15, 7),
    ("MoE Expert Collapse", "Debug / Verify", 14, 9),
    ("OOM on Multi-GPU", "Debug / Verify", 14, 7),
    ("Loss Spike Diagnosis", "Debug / Verify", 13, 7),
    # Inference & Serving
    ("Speculative Decoding", "Inference / Serving", 15, 8),
    ("FSDP vs DeepSpeed", "Inference / Serving", 15, 8),
    ("p99 Latency Tuning", "Inference / Serving", 14, 6),
    ("KV Cache / PagedAttn", "Inference / Serving", 14, 6),
    ("Quantization Shootout", "Inference / Serving", 13, 9),
    # RAG & Retrieval
    ("Multimodal RAG", "RAG / Retrieval", 15, 7),
    ("RAG Quality Evaluation", "RAG / Retrieval", 13, 6),
    ("Agentic RAG", "RAG / Retrieval", 13, 7),
    # Architecture & Systems
    ("Model Merging", "Architecture", 14, 8),
    ("Embedding Search at Scale", "Architecture", 14, 9),
    ("LLM A/B Testing", "Architecture", 12, 8),
    ("DSPy Prompt Tuning", "Architecture", 12, 9),
    ("Safety Guardrails", "Architecture", 12, 9),
    ("Structured Output", "Architecture", 11, 10),
    ("LLM-as-Judge Eval", "Architecture", 11, 10),
    # Agent Tasks
    ("Expert Agent Delegation", "Agent Tasks", 15, 7),
    ("Pipeline Audit", "Agent Tasks", 14, 10),
    ("Data Analysis Agent", "Agent Tasks", 10, 9),
    ("Multi-agent Routing", "Agent Tasks", 8, 9),
    # Negative Controls
    ("CI/CD Pipeline", "Negative Controls", 11, 9),
    ("Merge Sorted Lists", "Negative Controls", 7, 6),
    ("Trie Autocomplete", "Negative Controls", 6, 7),
    ("REST API (FastAPI)", "Negative Controls", 6, 9),
]

# Two-column layout constants
COL_LABEL_W = 160       # task label width per column
COL_CHART_W = 330       # chart area per column
COL_DELTA_W = 50        # delta label width
COL_GAP = 30            # gap between columns
COL_W = COL_LABEL_W + COL_CHART_W + COL_DELTA_W

TOTAL_WIDTH = 20 + COL_W + COL_GAP + COL_W + 20  # margins

ROW_HEIGHT = 22
CATEGORY_HEIGHT = 26
CATEGORY_GAP = 10
TOP_PADDING = 104
BOTTOM_PADDING = 20

# Colors
PLUGIN_COLOR = "#F9AE18"
BASELINE_COLOR = "#64748b"
WIN_COLOR = "#16a34a"
WIN_FILL = "#bbf7d0"
LOSS_COLOR = "#dc2626"
LOSS_FILL = "#fecaca"
NEUTRAL_COLOR = "#94a3b8"
GRID_STROKE = "#e2e8f0"
GRID_LIGHT = "#f1f5f9"
CATEGORY_BG = "#f8fafc"
CATEGORY_BORDER = "#e2e8f0"
TEXT_COLOR = "#334155"
BG_COLOR = "#ffffff"

MAX_SCORE = 15
MAX_PCT = 100
BAR_HEIGHT = 12


def generate_svg():
    # Split tasks into left and right columns by category
    categories = list(dict.fromkeys(t[1] for t in TASKS))

    # Build category groups
    cat_groups = []
    for cat in categories:
        items = [(n, c, p, b) for n, c, p, b in TASKS if c == cat]
        cat_groups.append((cat, items))

    # Split into two halves by total row count (categories + tasks)
    def col_height(groups):
        h = 0
        for i, (cat, items) in enumerate(groups):
            if i > 0:
                h += CATEGORY_GAP
            h += CATEGORY_HEIGHT + len(items) * ROW_HEIGHT
        return h

    # Find best split point
    total_groups = len(cat_groups)
    best_split = total_groups // 2
    best_diff = float('inf')
    for s in range(1, total_groups):
        left_h = col_height(cat_groups[:s])
        right_h = col_height(cat_groups[s:])
        diff = abs(left_h - right_h)
        if diff < best_diff:
            best_diff = diff
            best_split = s

    left_groups = cat_groups[:best_split]
    right_groups = cat_groups[best_split:]

    left_h = col_height(left_groups)
    right_h = col_height(right_groups)
    content_h = max(left_h, right_h)
    total_height = TOP_PADDING + content_h + BOTTOM_PADDING

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {TOTAL_WIDTH} {total_height}" '
        f'width="{TOTAL_WIDTH}" height="{total_height}">'
    )

    # Styles
    lines.append("""<style>
    text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .cat-label { font-size: 11px; font-weight: 600; fill: #1e293b; letter-spacing: 0.02em; }
    .task-label { font-size: 10px; fill: #334155; }
    .score-label { font-size: 8.5px; font-weight: 600; }
    .delta-label { font-size: 9.5px; font-weight: 700; }
    .axis-label { font-size: 8.5px; fill: #94a3b8; }
    .legend-text { font-size: 10px; fill: #475569; }
    .stat-value { font-size: 32px; font-weight: 700; }
    .stat-label { font-size: 11px; fill: #64748b; }
</style>""")

    # Background
    lines.append(
        f'<rect width="{TOTAL_WIDTH}" height="{total_height}" '
        f'fill="{BG_COLOR}" rx="10"/>'
    )

    # Header stats
    stats = [
        ("88%", "avg score", PLUGIN_COLOR, "with SuperML"),
        ("55%", "avg score", BASELINE_COLOR, "without SuperML"),
        ("91%", "win rate", WIN_COLOR, None),
    ]
    stat_width = TOTAL_WIDTH / len(stats)
    for i, (val, label, color, prefix) in enumerate(stats):
        cx = stat_width * i + stat_width / 2
        lines.append(
            f'<text x="{cx}" y="36" class="stat-value" '
            f'fill="{color}" text-anchor="middle">{val}</text>'
        )
        display_label = f"{prefix} {label}" if prefix else label
        lines.append(
            f'<text x="{cx}" y="52" class="stat-label" '
            f'text-anchor="middle">{display_label}</text>'
        )

    # Separator
    lines.append(
        f'<line x1="20" y1="62" x2="{TOTAL_WIDTH - 20}" y2="62" '
        f'stroke="{GRID_STROKE}" stroke-width="1"/>'
    )

    # Legend
    ly = 76
    lx = 20

    # With SuperML bar
    lines.append(f'<rect x="{lx}" y="{ly - 4}" width="30" height="8" rx="2" fill="{PLUGIN_COLOR}"/>')
    lines.append(f'<text x="{lx + 36}" y="{ly + 3}" class="legend-text">Cursor / Claude Code + SuperML</text>')

    # Without SuperML marker
    bx = lx + 230
    lines.append(f'<line x1="{bx}" y1="{ly - 3}" x2="{bx}" y2="{ly + 3}" stroke="{BASELINE_COLOR}" stroke-width="2.5" stroke-linecap="round"/>')
    lines.append(f'<circle cx="{bx}" cy="{ly}" r="2.5" fill="{BASELINE_COLOR}"/>')
    lines.append(f'<text x="{bx + 8}" y="{ly + 3}" class="legend-text">Cursor / Claude Code alone</text>')

    # Improvement gap
    gx = lx + 430
    lines.append(f'<rect x="{gx}" y="{ly - 4}" width="22" height="8" rx="2" fill="{WIN_FILL}" stroke="{WIN_COLOR}" stroke-width="0.5"/>')
    lines.append(f'<text x="{gx + 28}" y="{ly + 3}" class="legend-text">improvement</text>')

    # Loss gap
    rx = lx + 560
    lines.append(f'<rect x="{rx}" y="{ly - 4}" width="22" height="8" rx="2" fill="{LOSS_FILL}" stroke="{LOSS_COLOR}" stroke-width="0.5"/>')
    lines.append(f'<text x="{rx + 28}" y="{ly + 3}" class="legend-text">baseline wins</text>')

    def score_to_pct(score):
        """Convert raw score to percentage."""
        return round(score / MAX_SCORE * 100)

    def score_to_x(score, col_x):
        """Convert score to x position within a column's chart area."""
        return col_x + COL_LABEL_W + (score / MAX_SCORE) * COL_CHART_W

    def draw_column(groups, col_x, start_y):
        """Draw one column of the chart."""
        y = start_y

        # Grid lines for this column
        chart_left = col_x + COL_LABEL_W
        chart_right = chart_left + COL_CHART_W
        grid_bottom = start_y + col_height(groups)

        # Major grid at 0%, 25%, 50%, 75%, 100%
        for pct in (0, 25, 50, 75, 100):
            raw = pct / 100 * MAX_SCORE
            x = score_to_x(raw, col_x)
            lines.append(
                f'<line x1="{x}" y1="{start_y - 6}" x2="{x}" y2="{grid_bottom}" '
                f'stroke="{GRID_STROKE}" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x}" y="{start_y - 10}" class="axis-label" '
                f'text-anchor="middle">{pct}%</text>'
            )

        # Lighter grid at every 10% (skip major ones)
        for pct in range(10, 100, 10):
            if pct % 25 == 0:
                continue
            raw = pct / 100 * MAX_SCORE
            x = score_to_x(raw, col_x)
            lines.append(
                f'<line x1="{x}" y1="{start_y}" x2="{x}" y2="{grid_bottom}" '
                f'stroke="{GRID_LIGHT}" stroke-width="0.5"/>'
            )

        task_idx = 0
        for gi, (cat, items) in enumerate(groups):
            if gi > 0:
                y += CATEGORY_GAP

            # Category header
            lines.append(
                f'<rect x="{col_x}" y="{y}" '
                f'width="{COL_W}" height="{CATEGORY_HEIGHT - 2}" '
                f'fill="{CATEGORY_BG}" stroke="{CATEGORY_BORDER}" '
                f'stroke-width="0.5" rx="3"/>'
            )
            lines.append(
                f'<text x="{col_x + 8}" y="{y + 16}" '
                f'class="cat-label">{cat}</text>'
            )
            y += CATEGORY_HEIGHT

            for name, _, plugin, baseline in items:
                cy = y + ROW_HEIGHT / 2
                bar_top = cy - BAR_HEIGHT / 2
                half_bar = BAR_HEIGHT / 2

                # Alternating row bg
                if task_idx % 2 == 0:
                    lines.append(
                        f'<rect x="{col_x + COL_LABEL_W}" y="{y}" '
                        f'width="{COL_CHART_W}" height="{ROW_HEIGHT}" '
                        f'fill="#fafbfc"/>'
                    )

                # Task label
                lines.append(
                    f'<text x="{col_x + COL_LABEL_W - 8}" y="{cy + 3.5}" '
                    f'class="task-label" text-anchor="end">{name}</text>'
                )

                x_plugin = score_to_x(plugin, col_x)
                x_baseline = score_to_x(baseline, col_x)
                x_zero = score_to_x(0, col_x)
                delta = plugin - baseline

                if delta > 0:
                    gap_color, gap_stroke = WIN_FILL, WIN_COLOR
                elif delta < 0:
                    gap_color, gap_stroke = LOSS_FILL, LOSS_COLOR
                else:
                    gap_color, gap_stroke = GRID_LIGHT, NEUTRAL_COLOR

                x_left = min(x_plugin, x_baseline)
                x_right = max(x_plugin, x_baseline)

                # Gap fill
                if delta != 0:
                    lines.append(
                        f'<rect x="{x_left}" y="{bar_top + 1}" '
                        f'width="{x_right - x_left}" height="{BAR_HEIGHT - 2}" '
                        f'fill="{gap_color}" rx="2"/>'
                    )

                # Plugin bar (light fill from 0)
                bar_w = x_plugin - x_zero
                if bar_w > 0:
                    lines.append(
                        f'<rect x="{x_zero}" y="{bar_top + 2}" '
                        f'width="{bar_w}" height="{BAR_HEIGHT - 4}" '
                        f'fill="{PLUGIN_COLOR}" opacity="0.15" rx="2"/>'
                    )
                    # Solid cap
                    cap_w = min(5, bar_w)
                    lines.append(
                        f'<rect x="{x_plugin - cap_w}" y="{bar_top + 1}" '
                        f'width="{cap_w}" height="{BAR_HEIGHT - 2}" '
                        f'fill="{PLUGIN_COLOR}" rx="2"/>'
                    )

                # Baseline marker
                lines.append(
                    f'<line x1="{x_baseline}" y1="{cy - half_bar + 1}" '
                    f'x2="{x_baseline}" y2="{cy + half_bar - 1}" '
                    f'stroke="{BASELINE_COLOR}" stroke-width="2" '
                    f'stroke-linecap="round"/>'
                )
                lines.append(
                    f'<circle cx="{x_baseline}" cy="{cy}" r="2.5" '
                    f'fill="{BASELINE_COLOR}"/>'
                )

                # Score labels above markers (as percentages)
                plugin_pct = score_to_pct(plugin)
                baseline_pct = score_to_pct(baseline)
                delta_pct = plugin_pct - baseline_pct

                lines.append(
                    f'<text x="{x_plugin}" y="{bar_top - 1}" '
                    f'class="score-label" fill="{PLUGIN_COLOR}" '
                    f'text-anchor="middle">{plugin_pct}%</text>'
                )
                if abs(delta) >= 2:
                    lines.append(
                        f'<text x="{x_baseline}" y="{bar_top - 1}" '
                        f'class="score-label" fill="{BASELINE_COLOR}" '
                        f'text-anchor="middle">{baseline_pct}%</text>'
                    )

                # Delta label
                delta_color = WIN_COLOR if delta > 0 else LOSS_COLOR if delta < 0 else NEUTRAL_COLOR
                delta_str = f"+{delta_pct}%" if delta_pct > 0 else f"{delta_pct}%"
                lines.append(
                    f'<text x="{col_x + COL_LABEL_W + COL_CHART_W + 10}" y="{cy + 3.5}" '
                    f'class="delta-label" fill="{delta_color}">{delta_str}</text>'
                )

                y += ROW_HEIGHT
                task_idx += 1

    # Draw left column
    left_x = 20
    draw_column(left_groups, left_x, TOP_PADDING)

    # Draw right column
    right_x = 20 + COL_W + COL_GAP
    draw_column(right_groups, right_x, TOP_PADDING)

    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    svg = generate_svg()
    out = "assets/results-chart.svg"
    import os
    os.makedirs("assets", exist_ok=True)
    with open(out, "w") as f:
        f.write(svg)
    print(f"Written to {out}  ({TOTAL_WIDTH}x?)")
