# Evidence provenance

- `reported-results.json` records the selected published-in-manuscript aggregates transcribed from the formal PDF and VLDB LaTeX. `setup.tex`, `evaluation.tex`, and `reproducibility.tex` preserve the corresponding manuscript sections.
- `figures/formal-*.png` contains the nine graphics used in the VLDB manuscript. The formal source was the 12-page ICDE PDF with SHA-256 `b3397df826c259a178662f4ced8e9669232ce4bf07f8faf77be1581c01ea569a`. The figure PNGs preserve its task order and plotted outcomes.
- `saved-screening.json` is the saved synthetic-record timing file. At 1,000 records, the recorded component minima sum to 208.1 ms for recomputed source features and 40.1 ms for cached source features.
- The supplied workspace did not yield the original `formal/episodes.jsonl`, secondary or dense-only episode archives, full traces, compact facts, frozen source memory, or immutable executor and provider revisions. Exact replay or task-level reaggregation remains pending those files. The JSONL files under `structmem/benchmark_data/` contain benchmark workload definitions and labels. No model outcome records were reconstructed from the figures.

The portable configs preserve the current manuscript's task counts and limits. The historical configuration used a separate pair manifest for retrieval diagnostics. These proxy labels are outside the published primary metrics and the original manifests were unavailable. The historical ALFWorld results reflect a development slice with mixed retrieval policies, so the artifact treats them as archival diagnostics.
