# Claim 2 — exact Theorem 4.3 counterexample

**Verdict: FALSIFIED.** This page supersedes the Claim 2 discussion in the
**Historical rejected baseline**. It falsifies the exact bound printed in
Theorem 4.3 / Equation (7); it does not dispute Theorem 4.2 or a possible
corrected theorem.

## Exact claim and source quantifiers

Theorem 4.3 of arXiv:2601.21135v2 considers every length-`T` coefficient path
with `TV(alpha*) <= V` and iid mean-zero sub-Gaussian noise `sigma`. For the
quadratic estimator

`argmin sum_t ||y_t - B alpha_t||^2 + lambda sum_t ||alpha_(t+1)-alpha_t||^2`

with `lambda` asymptotic to `T^(1/3)`, Equation (7) displays an MSE stochastic
term proportional to `(V/T)^(2/3)` plus
`delta_approx^2/sigma_min^2`. The statement does not exclude `V=0`.

Paper source: ar5iv HTML retrieved 2026-07-29 with an explicit browser
User-Agent; SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

## Assumption-satisfying instance

Set `d=1`, `K=2`, `B=[1]`, `sigma_min=1`, `alpha(t)=(0.5,0.5)` for every
`t`, `delta_approx=0`, and iid Gaussian noise with `sigma=0.5`.

- A1: `g(z)=z` is invertible.
- A2: component independence is vacuous for `d=1`.
- A3/A3′: the sole mechanism row difference is nonzero.
- A4: `B` has full column rank.
- A5: linear `f` and identity `h` are twice differentiable.
- A6/A.10: the constant simplex path has `TV=0`.
- A.11: Gaussian noise is sub-Gaussian.

For `H=(I+lambda D^T D)^(-1)`, the constant eigenvector of `D^T D` has
eigenvalue zero. The smoother therefore preserves its Gaussian noise
component, giving risk at least `sigma^2/T > 0`. The displayed Equation (7)
right-hand side is exactly zero because `V=delta_approx=0`.

## Raw result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

Run `0f6a8434-f54e-4397-a488-d68097e23024`, Git
`0acdcbd0141da4cf13c772a5ca492dae8dedd75f`, HF `cpu-upgrade`.
The scientific verifier was estimated at one core; the container exposed 64
CPUs. Job runtime was 37 seconds and cumulative checker runtime was 1.788
seconds.

| Quantity | Value |
| --- | ---: |
| `T` | 64 |
| `lambda=T^(1/3)` | 4 |
| exact expected MSE | 0.03404667009563923 |
| constant-mode lower bound | 0.00390625 |
| displayed Equation (7) RHS | 0 |
| primary verifier exit | 0 |

The independent implementation used seed `20260729` and 20,000 Gaussian
replications. It estimated the preserved constant-mode MSE as
`0.003966053385246882`, with Monte Carlo SE
`0.000040479978351390024` and a five-SE lower bound
`0.0037636534934899317 > 0`.

## Code, environment, and downloadable evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_2/claim_contract.json)
- [Exact verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_2/claim2_theorem43.py)
- [Independent checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_2/claim2_independent.py)
- [Raw JSON](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_2/raw_results.json)
- [Raw run log](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_2/baseline_run.log)
- [Pinned pyproject](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/environment/pyproject.toml)
- [uv.lock](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/environment/uv.lock)

## Negative controls

Both implementations were rerun with `sigma=0`. Each returned
`NOT_FALSIFIED` and exited `1`, which is the intended failure mode. The
cumulative runner itself exits nonzero if either control unexpectedly reports
a contradiction.

## Limitation

The counterexample shows that the exact displayed bound omits a nonzero noise
floor for constant paths. It does not show that a repaired statement with an
additive `sigma^2/T` term, another estimator, or different regularity class is
false.

