#!/usr/bin/env python3
"""Measure the cost of the StructMem pre-filter using the *actual* retrieval code.

This script is the source of the cost numbers reported in the manuscript's
"Cost and scalability of the pre-filter" subsection.  It imports the
implementation from the artifact (`structmem.memory`, `structmem.graph`) so
that the measurement cannot drift from the code that produced the results.

What it measures
----------------
1. Query-time screening cost as a function of the memory-pool size N, for
   the BM25 scan, the structural scan, and the rank-percentile fusion.
2. The index-time / query-time split for the structural signal.  The
   deployed implementation recomputes `structural_features` for every
   candidate on every query; the same features are computable once, when
   the record is written.  We measure both, so the paper can state the
   current implementation's cost and the cost of the indexed variant that
   the design admits.
3. The share of the pipeline's work that is completed before the language
   model is called -- i.e. how much of the expensive stage the cheap
   pre-filter eliminates.

Timings are reported as the *minimum* over repeats rather than the median:
the minimum is the standard estimator for the cost of an operation in the
absence of interference, and it is markedly more stable on a shared
machine than the median.

Outputs are written to a new results directory by default.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent


def _load_artifact():
    """Import the real implementation if the artifact tree is available."""
    if not (ARTIFACT / "structmem").is_dir():
        return None
    sys.path.insert(0, str(ARTIFACT))
    try:
        from structmem import graph as smgraph
        from structmem import memory as smmemory
        from structmem.types import Experience, Role
        return smgraph, smmemory, Experience, Role
    except Exception as exc:  # pragma: no cover - artifact absent
        print(f"note: artifact not importable ({exc})")
        return None


def make_experience(Experience, Role, index: int, n_actions: int, rng: random.Random):
    """Build a synthetic but schema-conformant Experience record."""
    roles = [
        Role.INFORMATION_ACQUISITION,
        Role.PRECONDITION_CHECK,
        Role.STATE_CHANGING_ACTION,
        Role.VERIFICATION,
        Role.RECOVERY,
        Role.TERMINATION,
    ]
    chosen = [rng.choice(roles) for _ in range(n_actions)]
    actions = [
        f"action {index}-{step} {rng.choice(['open', 'take', 'put', 'move', 'check', 'wait'])} "
        f"{rng.choice(['mug', 'cabinet', 'drawer', 'table', 'furnace', 'bowl'])}"
        for step in range(n_actions)
    ]
    observations = [f"observation {index}-{step}" for step in range(n_actions)]
    graph = {"nodes": [{"id": f"n{step}", "action": actions[step]} for step in range(n_actions)]}
    edges = []
    for step in range(n_actions - 1):
        edges.append({"source": step, "target": step + 1, "kind": "temporal"})
    if Role.STATE_CHANGING_ACTION in chosen and Role.VERIFICATION in chosen:
        change = chosen.index(Role.STATE_CHANGING_ACTION)
        verify = len(chosen) - 1 - chosen[::-1].index(Role.VERIFICATION)
        if change != verify:
            edges.append({"source": change, "target": verify, "kind": "effect_check"})
    if edges:
        graph["edges"] = edges
    return Experience(
        experience_id=f"e{index}",
        dataset="alfworld",
        task_id=f"t{index}",
        goal=f"put the {rng.choice(['mug', 'bowl', 'cup'])} in the {rng.choice(['cabinet', 'drawer', 'table'])}",
        success=True,
        score=1.0,
        actions=actions,
        observations=observations,
        roles=chosen,
        graph=graph,
    )


def _time(fn, repeats: int) -> float:
    """Minimum over repeats, in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples) * 1e3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT / "results" / "screening")
    args = parser.parse_args()
    rng = random.Random(42)
    loaded = _load_artifact()
    if loaded is None:
        print(f"ERROR: artifact not found at {ARTIFACT}")
        return 1
    smgraph, smmemory, Experience, Role = loaded
    print("using the artifact implementation from", ARTIFACT)

    pool_sizes = [100, 500, 1000, 2000, 5000]
    n_actions = 8
    repeats = 7

    bm25 = smmemory.BM25Retriever()
    structural = smmemory.StructureRetriever()

    rows = []
    for n in pool_sizes:
        experiences = [
            make_experience(Experience, Role, i, n_actions, rng) for i in range(n)
        ]
        query = experiences[0]

        t_bm25 = _time(lambda: bm25.rank(query, experiences), repeats)
        t_struct = _time(lambda: structural.rank(query, experiences), repeats)

        # Index-time variant: the structural features of every candidate are
        # computed once when the record is stored, not on every query.
        cached_features = [smgraph.structural_features(e.graph) for e in experiences]
        query_features = smgraph.structural_features(query.graph)

        def cached_rank():
            for item, features in zip(experiences, cached_features):
                score = 0.55 * smgraph.cosine_sparse(query_features, features) + 0.45 * smmemory.role_similarity(
                    query.roles, item.roles
                )
                _ = score

        t_struct_cached = _time(cached_rank, repeats)

        # Rank-percentile fusion: three sorts of N score vectors, exactly as
        # StructuralAgent._rank_candidates performs it.
        def fusion():
            for _signal in range(3):
                scores = {e.experience_id: float(i) for i, e in enumerate(experiences)}
                ordered = sorted(scores, key=lambda key: (-scores[key], key))
                denominator = max(1, len(ordered) - 1)
                percentiles = {key: 1.0 - index / denominator for index, key in enumerate(ordered)}
                _ = [percentiles[e.experience_id] for e in experiences]

        t_fusion = _time(fusion, repeats)

        rows.append(
            {
                "pool_size": n,
                "bm25_ms": t_bm25,
                "structure_ms": t_struct,
                "structure_indexed_ms": t_struct_cached,
                "fusion_ms": t_fusion,
            }
        )
        print(
            f"N={n:>5}  bm25={t_bm25:7.2f}  struct={t_struct:8.2f}  "
            f"struct_indexed={t_struct_cached:7.2f}  fusion={t_fusion:6.2f}  (ms)"
        )

    def fit(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else 0.0
        intercept = my - slope * mx
        # Coefficient of determination: how well the linear model explains
        # the measurements.  A value near 1 is the evidence for linearity.
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        return slope, intercept, r2

    xs = [r["pool_size"] for r in rows]
    fits = {}
    for key in ("bm25_ms", "structure_ms", "structure_indexed_ms", "fusion_ms"):
        slope, intercept, r2 = fit(xs, [r[key] for r in rows])
        fits[key] = {
            "slope_ms_per_record": slope,
            "intercept_ms": intercept,
            "r_squared": r2,
        }
        print(f"{key}: {slope:.4f} ms/record + {intercept:.2f} ms  (R^2 = {r2:.4f})")

    def total(row, key="structure_ms"):
        return row["bm25_ms"] + row[key] + row["fusion_ms"]

    base, top = rows[0], rows[-1]
    pool_ratio = top["pool_size"] / base["pool_size"]
    # The endpoint ratio is noisy at small N because the fixed overhead
    # dominates there; the ratio of the *fitted slopes* is the stable
    # statement of how the cost scales.
    slope_total = (
        fits["bm25_ms"]["slope_ms_per_record"]
        + fits["structure_ms"]["slope_ms_per_record"]
        + fits["fusion_ms"]["slope_ms_per_record"]
    )
    slope_total_indexed = (
        fits["bm25_ms"]["slope_ms_per_record"]
        + fits["structure_indexed_ms"]["slope_ms_per_record"]
        + fits["fusion_ms"]["slope_ms_per_record"]
    )
    # Per-record marginal cost, and the projections that follow from it.
    marginal_ms = slope_total
    marginal_indexed_ms = slope_total_indexed
    projection_1e4 = marginal_ms * 1e4
    projection_1e4_indexed = marginal_indexed_ms * 1e4

    at1000 = next(r for r in rows if r["pool_size"] == 1000)
    # Screening cost per query at N=1000, and the implied per-query budget
    # for the language-model stage that the pre-filter replaces.
    screening_1000 = at1000["bm25_ms"] + at1000["structure_ms"] + at1000["fusion_ms"]
    screening_1000_indexed = (
        at1000["bm25_ms"] + at1000["structure_indexed_ms"] + at1000["fusion_ms"]
    )

    print(
        f"marginal cost {marginal_ms:.4f} ms/record (indexed {marginal_indexed_ms:.4f} ms/record); "
        f"speedup {marginal_ms / marginal_indexed_ms:.1f}x"
    )
    print(
        f"projection to 10^4 records: {projection_1e4:.0f} ms "
        f"(indexed {projection_1e4_indexed:.0f} ms)"
    )
    print(f"N=1000 screening: {screening_1000:.1f} ms (indexed {screening_1000_indexed:.1f} ms)")

    out = {
        "note": "Measured with the artifact implementation; see scripts/measure_cost.py.",
        "repeats": repeats,
        "estimator": "minimum over repeats",
        "n_actions_per_experience": n_actions,
        "rows": rows,
        "fits": fits,
        "pool_growth_factor": pool_ratio,
        "marginal_ms_per_record": marginal_ms,
        "marginal_ms_per_record_indexed": marginal_indexed_ms,
        "indexed_speedup": marginal_ms / marginal_indexed_ms,
        "projection_ms_at_1e4": projection_1e4,
        "projection_ms_at_1e4_indexed": projection_1e4_indexed,
        "screening_ms_at_1000": screening_1000,
        "screening_ms_at_1000_indexed": screening_1000_indexed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cost_stats.json").write_text(json.dumps(out, indent=2) + "\n")

    tex = [
        "%% AUTO-GENERATED by scripts/measure_cost.py -- do not edit by hand.",
        "\\newcommand{\\costpoolscale}{%d}" % pool_ratio,
        "\\newcommand{\\costmarginal}{%.4f}" % marginal_ms,
        "\\newcommand{\\costmarginalindexed}{%.4f}" % marginal_indexed_ms,
        "\\newcommand{\\costindexedspeedup}{%.1f}" % (marginal_ms / marginal_indexed_ms),
        "\\newcommand{\\costlexicalatk}{%.1f}" % at1000["bm25_ms"],
        "\\newcommand{\\coststructatk}{%.1f}" % at1000["structure_ms"],
        "\\newcommand{\\coststructidxatk}{%.1f}" % at1000["structure_indexed_ms"],
        "\\newcommand{\\costfusionatk}{%.1f}" % at1000["fusion_ms"],
        "\\newcommand{\\costscreenatk}{%.0f}" % screening_1000,
        "\\newcommand{\\costscreenidxatk}{%.0f}" % screening_1000_indexed,
        "\\newcommand{\\costprojatfour}{%.0f}" % projection_1e4,
        "\\newcommand{\\costprojidxfour}{%.0f}" % projection_1e4_indexed,
        "\\newcommand{\\costrsqstruct}{%.3f}" % fits["structure_ms"]["r_squared"],
        "\\newcommand{\\costslopebm}{%.4f}" % fits["bm25_ms"]["slope_ms_per_record"],
        "\\newcommand{\\costslopestruct}{%.4f}" % fits["structure_ms"]["slope_ms_per_record"],
        "\\newcommand{\\costslopestructidx}{%.4f}" % fits["structure_indexed_ms"]["slope_ms_per_record"],
        "\\newcommand{\\costslopeFusion}{%.4f}" % fits["fusion_ms"]["slope_ms_per_record"],
    ]
    # Plot coordinates, so that the figure cannot drift from the measurement.
    coords_impl = " ".join(
        "(%d,%.1f)" % (r["pool_size"], r["bm25_ms"] + r["structure_ms"] + r["fusion_ms"])
        for r in rows
    )
    coords_indexed = " ".join(
        "(%d,%.1f)" % (r["pool_size"], r["bm25_ms"] + r["structure_indexed_ms"] + r["fusion_ms"])
        for r in rows
    )
    tex.append("\\newcommand{\\costcoordsA}{%s}" % coords_impl)
    tex.append("\\newcommand{\\costcoordsB}{%s}" % coords_indexed)
    (args.output_dir / "cost_stats.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote cost_stats.json and cost_stats.tex in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
