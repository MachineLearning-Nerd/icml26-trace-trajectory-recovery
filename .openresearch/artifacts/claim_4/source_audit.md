# Claim 4 source audit

Paper source: arXiv `2601.21135v2`, ar5iv retrieval SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.
Released TRACE source: commit
`f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`.

## Exact reported results

- Section 6.4.1 and Figure 6: one 26-frame UAVDT left-turn sequence;
  TRACE correlation `0.960`, NCTRL `0.239`, centroid heuristic `0.911`,
  optical-flow heuristic `0.902`.
- Section 6.4.2 and Figure 7: trial `127_37`; TRACE absolute Pearson
  correlation `0.917`, NCTRL `0.619`, and five NCTRL transitions.
- The MoCap aggregate reported elsewhere is `0.856 +/- 0.043`.

The vehicle proxy is the velocity-direction ratio
`|v_y| / (|v_x| + |v_y|)`. The gait proxy is normalized hip speed. These are
projections rather than observed causal mixing weights.

## Exact paper protocol

The paper says MoCap uses subjects 2, 7, 8, 9, 16, and 35, 2,000 pure-walk
frames, 2,000 pure-run frames, `d=8`, lag 3, hidden dimension 256, 100 epochs,
and batch size 128. The release's only training script instead documents
subjects 35 and 127 and defaults to lag 2, hidden dimension 128, 200 epochs,
and batch size 64.
