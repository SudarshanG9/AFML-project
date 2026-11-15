# Member 1: pretrained fusion and precision interventions

The working detector adapter is `dimodif_adapter.py`. It strictly loads the external DiMoDif DFD checkpoint, freezes parameters, aligns cached audio/video features, and returns one fake score per clip. `model_adapter.py` is the stable interface used by Member 3. `diagnostic_model.py` is a randomly initialized wiring fixture, not a trained forensic detector.

`precision.py` defines dynamic symmetric per-tensor fake INT8; `attention.py` applies its four modes through dense attention or SDPA. `dimodif_intervention.py` temporarily instruments all five pretrained joint self-attention layers while retaining the original projections, masks and output projection. Only selected Q/K/V and pre-projection attention-output activations are rounded; weights, softmax and arithmetic remain floating point. Higher precision at a boundary does not restore upstream hidden states to reference values.

## Historical module navigation

These original names remain because frozen source inventories, approval chains, package builders, imports and accepted experiment records refer to them. Renaming or wrapping the original source would change frozen bytes. Use this index rather than renaming the evidence.

| Retained module | Responsibility |
| --- | --- |
| `day2_probe.py` | inspect modality ablations and attention shapes |
| `dayc_guards.py` | frozen experiment approvals, access checks and file identities |
| `dayc_research.py` | validation damage/recovery and restoration decision logic |
| `dayc_validation.py` | matched validation execution and evidence packaging |
| `dayc_precision_audit.py` | audit precision boundaries and query restoration |
| `dayc_cache_mapping.py` | verify approved feature-cache path mappings |
| `dayc_freeze.py` | build/check the C9 scientific policy freeze |
| `dayc_test_contract.py` | canonical held-out cohort contract |
| `dayc_heldout.py` | authorized one-attempt held-out execution |
| `dayc_recovery.py` | failure-preserving recovery protocol |
| `dayc_additional_recovery.py` | separately authorized additional recovery |
| `dayc_progress.py` | heartbeat/progress and interruption observability |

The original Day C packages and candidate-source tests require private evidence absent from a public checkout. They skip with named missing inputs; the synthetic safety, math, schema and recovery tests still run. No cleanup authorizes reopening the accepted test cohort.

## Entry points and ownership

Use `python -m avfuse.member3_runtime.cli --help` for the current cached-feature runtime. Historical `scripts/dayB_inference.py` remains a validation-only runner. `scripts/dayC_member1/` preserves package preparation, approval, freeze, held-out and recovery tools; its filenames are scientific stage identifiers, not the recommended everyday workflow. Member 2 owns data/labels/statistics, and Member 3 owns orchestration/measurement. Do not replace those implementations inside this package.
