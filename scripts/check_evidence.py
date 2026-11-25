import json
from collections import Counter
from pathlib import Path

from structmem.environments.agentbench import AgentBenchEnvironment


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def require(condition, description):
    if not condition:
        raise SystemExit(description)


def query_type(entry):
    label, = entry["type"]
    return "aggregation" if label.startswith("aggregation-") else label


def main():
    report = json.loads((EVIDENCE / "reported-results.json").read_text())
    successes = report["successes_out_of_100"]
    for workload, pair in report["primary_pairing"].items():
        overall = successes[f"primary/{workload}"]
        require(sum(pair.values()) == 100, f"Pair denominator differs for {workload}")
        for method in ("no_memory", "ours"):
            require(pair["both"] + pair[f"{method}_only"] == overall[method],
                    f"Pair total differs for {workload}/{method}")

    for rows, workload, methods in (
        (report["scienceworld_overlap"], "scienceworld", ("no_memory", "voyager", "ours")),
        (report["dbbench_query_types"], "agentbench", ("no_memory", "ours")),
    ):
        require(sum(row["tasks"] for row in rows.values()) == 100, f"Invalid strata size for {workload}")
        for method in methods:
            require(sum(row[method] for row in rows.values()) == successes[f"primary/{workload}"][method],
                    f"Stratum total differs for {workload}/{method}")

    for workload, diagnostics in report["primary_ours_diagnostics"].items():
        for label, counts in diagnostics.items():
            require(sum(counts.values()) == 100, f"Invalid {workload}/{label} distribution")

    options = {"protocol": "local_dbbench", "suite": "dbbench-std"}
    environment = AgentBenchEnvironment("standard", options, 42)
    try:
        tasks = environment.tasks(100)
        require([task.task_id for task in tasks] == [f"dbbench-std:{index}" for index in range(100)],
                "DBBench task order differs")
        rows = [environment.dataset[task.native_id] for task in tasks]
        require(all(row["sql"]["query"].lstrip().upper().startswith("SELECT") for row in rows),
                "DBBench target slice contains a non-SELECT label")
        expected = {key: row["tasks"] for key, row in report["dbbench_query_types"].items()}
        require(Counter(map(query_type, rows)) == expected, "DBBench query-type counts differ")
    finally:
        environment.close()

    timings = json.loads((EVIDENCE / "saved-screening.json").read_text())
    row = next(row for row in timings["rows"] if row["pool_size"] == 1000)
    other_components = row["bm25_ms"] + row["fusion_ms"]
    require(round(other_components + row["structure_ms"], 1) == 208.1, "Screening total differs")
    require(round(other_components + row["structure_indexed_ms"], 1) == 40.1, "Cached total differs")
    print("Verified aggregate consistency, 100 DBBench read-query labels, and saved screening totals.")


if __name__ == "__main__":
    main()
