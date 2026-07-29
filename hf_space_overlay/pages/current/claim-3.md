# Claim 3 — learned TRACE and NCTRL comparison

**Current verdict: BLOCKED.** This page supersedes the Claim 3 discussion in
the **Historical rejected baseline**. A paper-scale learned TRACE run is still
in progress; the exact NCTRL comparison cannot presently be regenerated from
the released artifacts.

## Exact claim and protocol

Table 1 reports three-mechanism synthetic trajectory correlation
`0.94 +/- 0.05` for TRACE, `0.67` for hard NCTRL, and `0.72` for soft NCTRL.
Figure 3 reports TRACE correlation up to `0.99` on calibrated transitions.
The released TRACE synthetic setup uses `d=8`, five pure domains, 40,000
examples per domain, a learned encoder, three active mechanisms, `T=50`, and
500 trajectory observations per time point.

Paper source: arXiv:2601.21135v2 HTML retrieved 2026-07-29; SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

## Four comparator routes

The audit retrieved and hash-verified five primary files from the official
NCTRL repository at Git `d2540bb5d0ebe7e75f68ebb490a94fe019a65c52`.

1. The TRACE release contains no NCTRL training, adaptation, checkpoint, or
   evaluation code.
2. Upstream NCTRL implements hard Viterbi assignments on a different
   length-four discrete ARHMM dataset and trains for 200 epochs.
3. No released soft-gating NCTRL variant, objective, temperature, checkpoint,
   or matching evaluation protocol was found.
4. Mandatory falsification found no valid counterexample: absent comparator
   assets cannot contradict finite reported values.

The complete-release negative control injects both missing comparator
capabilities, makes the blocker audit fail, and exits 1.

## Raw executable result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

Cumulative run `2f5c4b3b-d723-4423-9cd1-d5d67fdc6c9f`, Git
`73ae3790dc3b01d68eb2d715452e122643710e50`, local CPU, returned:

**CPU/runtime:** estimated one core and under five minutes; local host exposed
eight logical CPUs; cumulative wall time 121.746 seconds. **Seeds:** none for
this deterministic release audit. The learned TRACE route has a separate,
pre-registered five-seed protocol and is not used to remove the comparator
blocker.

| Released capability | Found |
| --- | --- |
| TRACE-side NCTRL protocol | no |
| upstream hard Viterbi implementation | yes |
| matching soft-gating implementation | no |
| all five upstream file hashes match | yes |
| valid numerical falsification | no |

## Evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim_contract.json)
- [NCTRL source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_source_audit.md)
- [Audit verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_nctrl_audit.py)
- [Independent checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_nctrl_independent.py)
- [Raw primary output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_release_audit.json)
- [Raw complete-release control](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_complete_release_control.json)
- [Raw independent output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_independent_checker.json)

## Limitation

Missing release assets are evidence of irreproducibility from the published
package, not evidence that the reported NCTRL numbers are false. The final
page will separately report the running learned TRACE result; it cannot repair
the missing comparator half of the compound claim.
