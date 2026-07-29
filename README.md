# TRACE reproduction campaign

Claim-by-claim CPU-only reproduction of “TRACE: Trajectory Recovery for
Continuous Mechanism Evolution in Causal Representation Learning”
(arXiv:2601.21135v2).

This branch is the frozen OpenResearch baseline. It vendors the authors'
released code at commit `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`, pins the
documented Python 3.8 environment with `uv`, and starts with an exact contract
audit of Theorem 4.3. The fixed command is:

```bash
uv run --frozen python -m trace_repro.run_all
```

Current evaluator-facing publication remains the historical Hugging Face
revision `DineshAI/xRN1Ym2hoa@8336cbc2a29260f27248e11b9c48f1bb0a7f2266`.
Nothing on this baseline branch should be read as a new judge result.
