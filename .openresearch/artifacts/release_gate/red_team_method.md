# Evaluator-blind red-team method

The red-team program receives only a freshly composed candidate Space
directory. It is not given repository paths, experiment IDs, run logs, or
locations of evidence.

It starts from `README.md`, `logbook.json`, and the logbook root page. It then
uses only logbook navigation and explicit Hugging Face blob links to locate
each current claim page, code, and raw data. For every claim it records every
file opened and checks that the page itself exposes:

- a terminal exact verdict and source/quantifier scope;
- assumptions or exact empirical protocol;
- the fixed command;
- inline numerical evidence and reachable raw data;
- code, checker, and negative control;
- limitations or unblocking capability;
- Git/revision, seed disposition, CPU allocation, and runtime.

Any inaccessible item is recorded as missing. The final campaign must run this
review on a fresh candidate composition, fix every failure, repeat the review,
and mirror the passing report into the candidate Space.
