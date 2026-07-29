# Route 1 result: unrelated-alpha control

Experiment `df281054-faa5-4a58-9175-2ea613068182`, commit
`22ef5c4c3d432440a03bd0200045f2dec90648e5`, run
`1167bb83-69af-4b44-a3de-5dd0a874059d`.

Hugging Face `cpu-upgrade` allocated 64 logical CPUs; the suite estimated one
core and completed 9.18 seconds of scientific checks (42 seconds total job).

At `K_active=7`, complex trajectory, five repetitions:

| Metric | Mean |
|---|---:|
| alpha component correlation | 0.012641 |
| alpha MAE | 0.125089 |
| released global full-W correlation | 0.997102 |
| W-base-centered correlation | 0.593566 |
| relative innovation error | 0.897344 |

The independent implementation matched these values to numerical precision.
The truth-alpha controls returned alpha and W correlation 1.0 and exited 1 as
expected.

This route is honestly recorded as inconclusive under its pre-registered
criterion because centered-W correlation was not below 0.25. The high full-W
score despite 89.7% relative innovation error motivated the distinct
zero-temporal-signal route in the child node.
