import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from check_evidence import query_type
from structmem.environments.agentbench import AgentBenchEnvironment
from structmem.memory import ExperienceStore
from structmem.storage import JSONLStore, write_json
from structmem.types import RunStatus


MAIN_METHODS = ("no_memory", "voyager", "reflexion", "expel", "memgpt", "mcma", "agentkb", "ours")
REMOVALS = ("w_o_roles", "w_o_structure", "w_o_binding")
TERMINAL = {RunStatus.COMPLETED.value, RunStatus.TASK_FAILED.value}


def select_terminal(records, required, workloads):
    terminal = defaultdict(lambda: defaultdict(dict))
    attempts = defaultdict(Counter)
    for record in records:
        task = record["task"]
        dataset, method, task_id = task["dataset"], record["method"], task["task_id"]
        status = record["status"]
        if dataset not in workloads:
            continue
        attempts[dataset][status] += 1
        if status not in TERMINAL:
            continue
        if task_id in terminal[dataset][method]:
            raise ValueError(f"Duplicate terminal result for {dataset}/{method}/{task_id}")
        terminal[dataset][method][task_id] = record

    for dataset in workloads:
        groups = terminal[dataset]
        reference = set(groups[required[0]])
        if len(reference) != 100:
            raise ValueError(f"Expected 100 terminal targets for {dataset}/{required[0]}, found {len(reference)}")
        if dataset == "agentbench" and reference != {f"dbbench-std:{index}" for index in range(100)}:
            raise ValueError("DBBench terminal targets differ from indices 0–99")
        for method in required:
            if set(groups[method]) != reference:
                raise ValueError(f"Missing or unmatched targets for {dataset}/{method}")
    return terminal, attempts


def success_counts(groups):
    return {method: sum(row["success"] for row in rows.values()) for method, rows in groups.items()}


def usage(rows):
    tokens = [sum(row["token_usage"][key] for key in ("input_tokens", "output_tokens")) for row in rows.values()]
    decisions = [len(row["steps"]) for row in rows.values()]
    return {"mean_tokens": mean(tokens), "median_tokens": median(tokens),
            "mean_decisions": mean(decisions), "median_decisions": median(decisions),
            "at_50_decisions": decisions.count(50)}


def strata(groups, labels, methods):
    by_label = defaultdict(list)
    for task_id, label in labels.items():
        by_label[label].append(task_id)
    return {
        label: {"tasks": len(ids), **{
            method: sum(groups[method][task_id]["success"] for task_id in ids) for method in methods
        }}
        for label, ids in by_label.items()
    }


def normalized_goal(goal):
    return goal.strip().lower()


def summarize(records, batch, memory, include_alfworld):
    methods = {"primary": MAIN_METHODS + REMOVALS, "secondary": MAIN_METHODS,
               "dense": ("embedding_retrieval",)}[batch]
    workloads = ["agentbench"] if batch == "secondary" else ["scienceworld", "agentbench"]
    if include_alfworld:
        workloads.append("alfworld")
    terminal, attempts = select_terminal(records, methods, workloads)
    output = {"batch": batch, "workloads": {}}
    for dataset in workloads:
        groups = {method: terminal[dataset][method] for method in methods}
        section = {"tasks_per_method": 100, "attempts_by_status": attempts[dataset],
                   "successes": success_counts(groups)}
        output["workloads"][dataset] = section
        if batch != "primary" or dataset == "alfworld":
            continue
        paired = ("no_memory", "ours")
        labels = {(True, True): "both", (False, True): "ours_only",
                  (True, False): "no_memory_only", (False, False): "neither"}
        section["paired"] = {label: 0 for label in labels.values()}
        for task_id in groups["no_memory"]:
            pair = tuple(groups[method][task_id]["success"] for method in paired)
            section["paired"][labels[pair]] += 1
        section["ours_diagnostics"] = {
            field: Counter(row.get(field) or "missing" for row in groups["ours"].values())
            for field in ("binding_outcome", "gate_verdict")
        }
        section["inference"] = {method: usage(groups[method]) for method in ("no_memory", "voyager", "ours")}
        if dataset == "agentbench":
            options = {"protocol": "local_dbbench", "suite": "dbbench-std"}
            environment = AgentBenchEnvironment("standard", options, 42)
            labels = {task.task_id: query_type(environment.dataset[task.native_id]) for task in environment.tasks(100)}
            section["declared_query_types"] = strata(groups, labels, paired)
            environment.close()
        if memory is not None:
            sources = [item for item in memory if item.dataset == dataset]
            if not sources or set(groups["no_memory"]) & {item.task_id for item in sources}:
                raise ValueError(f"Missing sources or source–target ID overlap for {dataset}")
            if dataset == "scienceworld":
                goals = {normalized_goal(item.goal) for item in sources}
                labels = {task_id: "matched" if normalized_goal(row["task"]["goal"]) in goals else "unmatched"
                          for task_id, row in groups["no_memory"].items()}
                section["goal_overlap"] = strata(groups, labels, ("no_memory", "voyager", "ours"))
    return output


def main():
    parser = argparse.ArgumentParser(description="Summarize selected StructMem terminal records for a single endpoint batch.")
    parser.add_argument("--batch", choices=("primary", "secondary", "dense"), required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--memory", type=Path, help="Frozen source memory for the overlap and disjointness checks")
    parser.add_argument("--include-alfworld", action="store_true", help="Include the diagnostic ALFWorld slice")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.include_alfworld and args.batch == "secondary":
        parser.error("the secondary paper batch covers DBBench only")
    for path in (args.episodes, args.memory):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.output.resolve() in {path.resolve() for path in (args.episodes, args.memory) if path is not None}:
        parser.error("output must have a different path from the inputs")
    try:
        records = JSONLStore(args.episodes).read()
        memory = ExperienceStore.load(args.memory).values() if args.memory else None
        result = summarize(records, args.batch, memory, args.include_alfworld)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
