# Claim 6 evaluation status

Route 1 used unrelated Dirichlet weights. At `K_active=7`, complex trajectory,
the five-run means were alpha correlation `0.012641`, alpha MAE `0.125089`,
relative innovation error `0.897344`, released full-W correlation `0.997102`,
and centered-W correlation `0.593566`. The independent checker matched. Because
the pre-registered centered-correlation threshold was `<0.25`, Route 1 did not
pass and its contract was not changed after seeing the result.

Route 2 is pending execution. It replaces the ambiguous centered-correlation
threshold with an exact structural control: a constant prediction has zero
temporal alpha and W variation by construction.

Even if the metric audit passes, it does not by itself reproduce or falsify
the learned-model table. It establishes whether the released W score is a
discriminative recovery metric. The exact learned-model result remains blocked
until the referenced 10-domain checkpoint is available or exact training can
be completed on authorized CPU compute.
