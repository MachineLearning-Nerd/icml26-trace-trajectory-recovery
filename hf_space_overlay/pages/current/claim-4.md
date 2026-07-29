# Claim 4 — UAVDT and CMU MoCap release audit

**Verdict: BLOCKED.** This page supersedes the deferred Claim 4 entry in the
**Historical rejected baseline**. Four distinct routes establish the missing
capabilities, but absence is not a numerical falsification.

## Exact claim

The paper displays UAVDT vehicle-turning trajectory correlation `0.960` for
TRACE versus `0.239` for NCTRL. For CMU MoCap it displays trial `127_37` at
`0.917` for TRACE versus `0.619` for NCTRL and reports aggregate TRACE
correlation `0.856 +/- 0.043`.

Paper protocol anchors include UAVDT `d=4`, lag 2, 500 pure-domain
trajectories per direction, and 150 epochs. The MoCap text names subjects
2, 7, 8, 9, 16, and 35, lag 3, hidden size 256, 100 epochs, and batch size
128.

## Four verification routes

1. **Released-code completeness:** no UAVDT implementation and no MoCap
   preprocessing/evaluation program are present.
2. **Protocol consistency:** released MoCap defaults are lag 2, hidden size
   128, 200 epochs, and batch size 64; all four differ from the paper.
3. **Data provenance:** no real-data archive, exact UAVDT frame manifest, or
   real-data checkpoint is included.
4. **Mandatory falsification:** no assumption-matched counterexample exists;
   a missing-data failure cannot contradict a finite empirical value.

The static verifier hashes all 46 official TRACE source files. A synthetic
complete-release control supplies the missing capabilities; the blocker audit
then fails and exits 1.

## Executable result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

Cumulative run `2f5c4b3b-d723-4423-9cd1-d5d67fdc6c9f`, Git
`73ae3790dc3b01d68eb2d715452e122643710e50`, ran locally in the 121.746-second
suite. Resource estimate was one core and under five minutes; the host exposed
eight logical CPUs.

**CPU/runtime:** estimated one core and under five minutes; local host exposed
eight logical CPUs; cumulative wall time 121.746 seconds. **Seeds:** none; the
46-file release audit and its injected-capability control are deterministic.

| Audit quantity | Result |
| --- | --- |
| released source files hashed | 46 |
| UAVDT code present | no |
| MoCap preprocessing/evaluation present | no |
| exact data/checkpoints present | no |
| paper/release defaults matching | 0/4 |
| valid falsification found | no |

## Evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_4/claim_contract.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_4/source_audit.md)
- [Verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_4/claim4_release_audit.py)
- [Raw primary output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_4/release_audit.json)
- [Raw complete-release control](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_4/release_audit_negative_control.json)

## Unblocking capability

Reproduction needs the precise UAVDT frame/label manifest and preprocessing,
the CMU `walk_run_data.npz` construction procedure, matching checkpoints, and
the exact NCTRL adaptations. Dataset availability alone would not resolve the
protocol and checkpoint gaps.
