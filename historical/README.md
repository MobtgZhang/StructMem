# Historical configurations

These three files are unmodified snapshots of the WSDM source configurations. They reference the original private API files, memory snapshot, and pair manifests. Use the portable configurations under `../configs/` for replay.

- `paper_300_per_method.yaml` specifies the primary 11-method matrix, with 100 tasks per benchmark. The VLDB main analysis uses its ScienceWorld and DBBench slices.
- `grok_4_6_100.yaml` specifies a broader historical secondary batch. Only the complete DBBench slice was selected for the VLDB comparison.
- `paper_300_memory.yaml` specifies the training task selection used by the memory collection workflow.
