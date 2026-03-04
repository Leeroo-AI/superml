---
name: ml-experiment
description: Use when starting, logging, or reviewing ML experiments — maintains a persistent experiment journal with hypotheses, results, and learnings across sessions
---

# Experiment Journal

Externalize your experimental reasoning. Every ML project is a sequence of hypotheses tested — this skill makes that sequence visible, persistent, and learnable.

## The Iron Law

```
NO NEW EXPERIMENT WITHOUT LOGGING THE HYPOTHESIS FIRST
```

If you're about to change a hyperparameter, swap a dataset, try a new architecture, or modify a training recipe — write down what you expect to happen and why BEFORE running it. This is how you learn from experiments instead of just running them.

## File Structure

Maintain these files in the project root (create if they don't exist):

```
experiments/
├── journal.md    — Running experiment log (append-only)
└── lessons.md    — Distilled patterns and rules (curated)
```

## Phases

### Phase 1: Before Any Experiment — Log the Hypothesis

Before changing anything or running anything new:

1. Read `experiments/journal.md` (if it exists) to see what's been tried
2. Write a new entry:

```markdown
### YYYY-MM-DD HH:MM — [Experiment Name]

**Status**: PLANNED

**Hypothesis**: [What you expect to happen and why]
**Change**: [Exactly what's being modified — one variable at a time]
**Config**:
- key_param_1: old_value → new_value
- key_param_2: value (unchanged)
**Expected outcome**: [Specific metric target or qualitative expectation]
**Baseline**: [Current best metric to beat]
```

**Gate**: Entry is written before any code runs. No exceptions.

### Phase 2: After the Experiment — Log the Result

Once results are in:

1. Update the journal entry:

```markdown
**Status**: COMPLETED
**Actual outcome**: [What actually happened — metrics, behavior]
**Delta**: [How this compared to expectation — better/worse/different than expected]
**Duration**: [Wall time, GPU hours]
**Learning**: [One sentence — what this taught you]
**Next**: [What to try based on this result]
```

**Gate**: Result is logged before starting the next experiment.

### Phase 3: Before the Next Iteration — Review History

Before proposing or starting the next experiment:

1. Read `experiments/journal.md` — scan recent entries
2. Check: Has this exact approach been tried before? What happened?
3. Read `experiments/lessons.md` — are there rules that apply?
4. Only then propose the next experiment

**Gate**: You can articulate why this experiment is different from previous attempts.

### Phase 4: Periodically — Distill Lessons

After every 3-5 experiments, or when a pattern emerges:

1. Review recent journal entries for patterns
2. Add rules to `experiments/lessons.md`:

```markdown
## Lessons

- [YYYY-MM-DD] [Context]: [Lesson]. Source: [user correction / experiment result / KB finding]
  Example: "2024-03-15 QLoRA: alpha/r ratio matters more than absolute rank for 7B models. Source: experiments showed r=32/alpha=64 outperformed r=64/alpha=64"

## Rules (hard-won)

- NEVER [thing that always fails] because [reason]. Learned: [date]
- ALWAYS [thing that always works] when [condition]. Learned: [date]
```

**Gate**: Lessons file has been updated before closing out a series of experiments.

## After This

- Starting a new experiment? Loop back to Phase 1.
- Need ideas for what to try next? Invoke **ml-iterate** — it reads your journal and proposes ranked options.
- Debugging a failed experiment? Invoke **ml-debug** — include the journal entry as context.
- Want to verify a config before running? Invoke **ml-verify** — catch mistakes before wasting GPU time.

## Anti-Patterns

| Mistake | Why it happens | What to do instead |
|---------|---------------|-------------------|
| Running without logging | "I'll just try this quick thing" | Even quick experiments get logged — they compound into knowledge |
| Changing multiple variables | "Let me also bump the LR while I'm at it" | One variable per experiment. Otherwise you can't attribute the result. |
| Not recording the baseline | "I'll remember what the old score was" | Write the baseline metric in the entry. Memory is unreliable. |
| Skipping the review step | "I know what I tried before" | Read the journal. You'll find experiments you forgot about. |
| Never distilling lessons | "The journal has everything" | A 200-entry journal is noise. Lessons are signal. Distill regularly. |

## Examples

**Starting a fine-tuning experiment:**
```
User: "Let's try QLoRA with rank 64 instead of 32"
Agent: [Reads experiments/journal.md]
Agent: [Writes new entry with hypothesis: "Higher rank captures more task-specific features, expecting +2% accuracy"]
Agent: [Proceeds with implementation]
```

**After getting results:**
```
User: "Training finished, eval accuracy went from 78% to 81%"
Agent: [Updates journal entry with actual outcome, delta (+3% vs expected +2%), learning]
Agent: [Suggests next experiment based on result]
```

**Before next iteration:**
```
User: "What should we try next?"
Agent: [Reads journal — sees rank 64 worked, rank 16 didn't, data augmentation untested]
Agent: [Invokes ml-iterate with full history context]
```
