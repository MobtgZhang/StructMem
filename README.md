# StructMem

Artifact for **StructMem: Procedure-Aware Retrieval and State-Conditioned Reuse of Agent Trajectories**. The package contains the inspected implementation, benchmark adapters, replay configurations, saved reporting assets, and analysis entry points. It was prepared from the WSDM implementation for the VLDB 2027 Regular Research Paper.

## Verify the artifact locally

From the `StructMem` directory, on Linux with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python scripts/check_evidence.py
```

After installation, these checks run offline with no model credentials, Java, or game data downloads. The unit tests use fixture model responses and fixed dense scores for the mapping and binding tests. `check_evidence.py` checks the reported totals against each other, enumerates DBBench IDs `dbbench-std:0` through `dbbench-std:99`, verifies their read-query labels and stratum sizes, and checks the saved screening totals. The reported model outcomes are stored as aggregates. See [evidence provenance](evidence/README.md) for their source and the inputs needed for task-level checks. Use `sha256sum -c SHA256SUMS` to check the distributed files.

## Code map

- `structmem/roles.py`, `graph.py`, `types.py`, and `memory.py` define trajectory records, role and graph construction, and ranking.
- `structmem/agents/structural.py` implements eligibility filtering, rank fusion, prompted correspondence, observation-conditioned binding, the reuse gate, and workload-specific execution routines. `structmem/agents/baselines.py` implements the comparison agents and dense-only retrieval.
- `structmem/environments/` connects ScienceWorld, DBBench, and ALFWorld. The DBBench adapter runs against one in-memory SQLite database per task with benchmark-provided labels. The packaged JSONL files are workload inputs.
- `structmem/runner.py`, `llm.py`, and `storage.py` implement evaluation, request caching, and the append-only episode format. `scripts/summarize_episodes.py` checks terminal record coverage and aggregates the selected paper workloads.
- `configs/` contains portable configurations. `historical/configs/` retains the corresponding original WSDM configurations for provenance. Older pilot, extension, and machine-specific supervisor scripts fall outside the paper protocol.

## Re-execute the study

This step makes model calls and requires the original frozen source memory to match the reported protocol. The original source memory and task logs were not located in the supplied project snapshot. A newly collected memory or another endpoint produces a new run whose outputs should be identified as such.

1. Install the runtime for the workloads being replayed. ScienceWorld uses a Java process and a local Py4J connection. The included ScienceWorld JAR is the one in the inspected source tree. DBBench uses the packaged workload and Python's SQLite. For ALFWorld, install the `alfworld` extra and set `ALFWORLD_DATA` to an official extraction with `logic/` and `json_2.1.1/`.

   ```bash
   python -m pip install -e '.[dense,scienceworld]'
   python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'
   ```

   The download command populates the local model cache. StructMem and the dense-only baseline subsequently load it offline. The historical model snapshot identifier is unavailable. Record the selected snapshot and environment versions for any new run.

2. Put the original frozen memory at `inputs/train_experiences.json` or set `STRUCTMEM_MEMORY_FILE` to its absolute path. Export `STRUCTMEM_API_BASE_URL` and `STRUCTMEM_API_KEY` in your shell. The base URL may end in `/v1`. Choose the endpoint serving the model label in each configuration. `STRUCTMEM_MODEL` can select a compatible replacement for a new run. Use a fresh `STRUCTMEM_OUTPUT_DIR` and `STRUCTMEM_LLM_CACHE_FILE` whenever the model, endpoint, memory, or protocol changes. Keep each endpoint batch in separate files.

3. Preflight and run the desired configuration. Preflight validates configuration, task capacity, the memory file, endpoint DNS, and dense-model availability where required. The run command repeats these checks and writes `preflight.json`.

   ```bash
   structmem preflight --config configs/primary.yaml
   structmem run --config configs/primary.yaml
   ```

   | Configuration | Workload and output |
   | --- | --- |
   | `configs/primary.yaml` | ScienceWorld test and DBBench standard, 100 targets per workload, 11 configurations, `results/primary/` |
   | `configs/secondary-dbbench.yaml` | DBBench standard, 100 targets, `grok-4.6`, `results/secondary-dbbench/` |
   | `configs/dense-only.yaml` | Separate dense-only comparator, 100 ScienceWorld and 100 DBBench targets, `results/dense-only/` |
   | `configs/memory-collection.yaml` | Training-split source collection, 100 tasks per benchmark, `results/memory-collection/` |

   The ScienceWorld configuration uses deterministic enumeration in the supplied simulator. The archived run's exact ScienceWorld task ID list was unavailable. If you have it, set `datasets.scienceworld.options.task_ids` to the archived list to lock the order and membership. The primary paper analysis excludes the historical ALFWorld mixed-policy development results. The dense-only configuration includes the development slice disabled by default. To run that slice, enable its `alfworld` dataset, install the `alfworld` extra, and run `STRUCTMEM_DATASETS=alfworld structmem run --config configs/dense-only.yaml`. The SQL workload covers standard indices 0–99 and the primary policy allows 50 ScienceWorld decisions or 30 DBBench decisions. The native engine/interface limits are 100 steps and 15 rounds, respectively.

If you intend to collect a new source memory, run the `memory-collection.yaml` pipeline after setting up all three runtimes and the endpoint. Freeze its terminal episodes with the command below. Training memory includes successful and failed records. Evaluation defaults to frozen state, and the structural agent's candidate pool uses successful same-benchmark records.

```bash
structmem run --config configs/memory-collection.yaml
structmem build-memory \
  --episodes results/memory-collection/episodes.jsonl \
  --output results/memory-collection/train_experiences.json \
  --include-failures
```

## Reaggregate and inspect results

Run the script below on one complete endpoint batch of original or newly generated logs. Use `--batch secondary` for the secondary DBBench records and `--batch dense` for the dense-only records. Add `--include-alfworld` to primary or dense reaggregation only when that file contains a complete ALFWorld slice. The script rejects duplicate terminal keys, missing or mismatched targets, and incorrect DBBench IDs. It keeps error attempts separate and checks source–target disjointness when given the corresponding frozen memory.

```bash
python scripts/summarize_episodes.py --batch primary \
  --episodes inputs/formal/episodes.jsonl \
  --memory inputs/train_experiences.json \
  --output results/primary/reaggregated.json
```

The output includes success counts, paired outcomes, declared SQL strata, binding and gate marginals, model usage, and ScienceWorld exact normalized goal overlap when the frozen memory is supplied. Each batch should be accompanied by its run manifest. Model tokens include recorded cached-response usage. The script does not estimate billing or recover omitted retries. The historical `structmem analyze` command remains available for earlier WSDM diagnostic outputs.

The CPU-only screening script uses synthetic trajectories, reports component minima for seven repeats, and writes to a separate output directory. The evidence folder contains the values used in the paper. Running the command below measures the local machine and leaves the saved values intact.

```bash
python scripts/measure_cost.py --output-dir results/screening
```

## Third-party material

- [ALFWorld](https://github.com/alfworld/alfworld), bundled text environment source from the inspected tree, version 0.5.0 and MIT license in `alfworld/LICENSE`.
- [ScienceWorld](https://github.com/allenai/ScienceWorld), bundled Python adapter and simulator JAR, version 1.2.3 and Apache 2.0 license in `scienceworld/LICENSE`.
- [AgentBench](https://github.com/THUDM/AgentBench), DBBench standard and training inputs copied from the inspected tree, with the upstream Apache 2.0 license in `structmem/benchmark_data/LICENSE`. The two files correspond to upstream `data/dbbench/standard.jsonl` and `data/dbbench/db_out_new.jsonl` as stored in that tree.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is acquired separately from its publisher for live execution. The local verification checks use fixture scores.

No project-specific license for the StructMem implementation was found in the supplied source. The authors can attach their chosen license when publishing.
