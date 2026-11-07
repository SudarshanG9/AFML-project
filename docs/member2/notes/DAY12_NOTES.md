# Day 12 notes: figures and documentation

Files: `src/avfuse/member2_data/report.py` (figures), `scripts/m2_evaluate.py` (draws them),
`src/avfuse/member2_data/README.md`, `docs/member2/RUNBOOK.md`.

## The job in one line

Turn the numbers into **pictures** people can read at a glance, and write down **how to run
everything** so anyone on the team can reproduce it.

## The two figures (made automatically by `m2_evaluate.py evaluate`)

**1. `curves.png`: ROC and precision-recall curves, one line per config**
- **ROC (left):** x = false alarms on real clips, y = fakes caught. A curve closer to the top-left
  corner is better. The dotted diagonal is a coin toss.
- **Precision-recall (right):** how clean the flagged clips are as we try to catch more fakes.

If the `low_precision` line sits below `reference`, rounding hurt. If `restore_value_output`
climbs back towards `reference`, the restore helped.

**2. `recall_by_type.png`: bars per fake type**
For audio / video / both fakes: the share caught at the frozen threshold, one bar per config, with
error bars showing the 95% confidence range. This shows **which kind of fake** gets harder to catch.

Figures need `matplotlib`. If it isn't installed, the numbers are still written; only the pictures
are skipped.

## The documentation

| File | What it's for |
| --- | --- |
| `member2_data/README.md` | every Member 2 module, which plan day it belongs to, and the decisions to review |
| `docs/member2/RUNBOOK.md` | the exact commands, in order: raw LAV-DF -> clips -> manifest -> audit -> metrics -> paired comparison -> sealed test report |

The runbook also says what may be committed: code, tests, docs and small synthetic files only.
Real results go in as **totals**, never clip-level data or media.

## Status

Done. The figures appear once real predictions exist.

## Connection

This finishes the Member 2 code for the 12-day plan. What remains is running it on real outputs:
DiMoDif features and predictions from Member 1, and timings from Member 3.
