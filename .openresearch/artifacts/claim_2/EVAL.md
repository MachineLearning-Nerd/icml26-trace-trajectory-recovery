# Evaluation status

Verdict after frozen baseline run
`0f6a8434-f54e-4397-a488-d68097e23024`: **FALSIFIED**.

The exact checker found risk `0.03404667009563923` and a constant-mode lower
bound `0.00390625` against a displayed Equation (7) right-hand side of zero.
The independent 20,000-repetition checker estimated the preserved
constant-mode risk as `0.003966053385246882` (five-standard-error lower bound
`0.0037636534934899317`). Both zero-noise negative controls returned
`NOT_FALSIFIED` and exited one, as required.

Theorem 4.2 is not contradicted by this construction. Because the judge groups
Theorems 4.2 and 4.3 into Claim 2, invalidity of the exact Theorem 4.3 statement
is sufficient to falsify the combined claim, subject to independent review of
the source interpretation.
