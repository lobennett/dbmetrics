# Before a metric enters the website

1. Define the scientific question, measured quantity, units and sampling limits.
2. Inspect one video in a notebook; show the evidence and calculation.
3. Check contrasting examples and failures against manual judgments.
4. Freeze the method and evaluate a separate, untouched review set.
5. Add package tests and record model/code versions, thresholds and coverage.
6. Review the results with Logan before integrating into the website.

Set acceptance criteria before evaluation. Report sensitivity, specificity and uncertainty for classifications—not accuracy alone. Never turn failed or unmeasured values into negatives.

## Next experiments, one at a time

| Question | Small first deliverable |
| --- | --- |
| Channel representation | Compare uniform-video and uniform-channel sampling; report channel counts and missing identities. |
| Overlaid text/logos | Inspect reviewed OCR/Qwen predictions and failure examples; keep physical scene text separate. |
| First/second-half similarity | One-video embedding comparison, then manually ranked contrasting pairs. Compare semantic and visual-only representations. |
| Cut continuity | Plot adjacent-frame changes for real cuts, camera motion and continuously changing imagery. |
| Semantic scene changes | Compare embeddings around proposed cut boundaries; exploratory, not a replacement cut detector. |
| Valence/arousal | Define the rating rubric and manual labels before choosing an automated estimator. |
| Engagement | Clarify whether the target is content appeal or task engagement; YouTube popularity is only a proxy. |

Existing CPU measurements are reproducible computational proxies, not automatically validated scientific constructs.
The website retains its existing implementations until package equivalence and integration are reviewed.
No cluster jobs are switched by installing this package.
