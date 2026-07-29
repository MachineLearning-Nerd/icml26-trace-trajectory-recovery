# Claim 6 evaluation status

Route 1 used unrelated Dirichlet weights. At `K_active=7`, complex trajectory,
the five-run means were alpha correlation `0.012641`, alpha MAE `0.125089`,
relative innovation error `0.897344`, released full-W correlation `0.997102`,
and centered-W correlation `0.593566`. The independent checker matched. Because
the pre-registered centered-correlation threshold was `<0.25`, Route 1 did not
pass and its contract was not changed after seeing the result.

Route 2 passed its pre-registered structural criterion. A constant mean-alpha
prediction had alpha MAE `0.093294`, alpha temporal-variation ratio below
`1e-15`, W temporal-variation ratio below `1.5e-14`, and nevertheless obtained
released full-W correlation `0.998742`. The independent manual-Pearson checker
matched. Exact-truth controls did not trigger the verifier and exited 1.

Route 3 measured exact K=10 training throughput on Hugging Face `cpu-upgrade`.
At 1.74--1.77 seconds per batch over 6,218 batches, the released 100-epoch
protocol projects to about 295 hours; the run was cancelled after 134 batches.

Route 4 restated the exact table claim and sought an assumption-matched
counterexample. The zero-temporal control defeats the metric interpretation,
but it is not a learned TRACE output and does not contradict the printed
numbers. Therefore it is not a valid falsification.

Final verdict: **BLOCKED**. All four required routes are complete. The missing
capability is the referenced 10-domain checkpoint or enough authorized CPU
time to complete the exact protocol. The complete-evidence negative control
removes this blocker, makes the verifier reject `BLOCKED`, and exits nonzero.
