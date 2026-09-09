"""Adjacent-residue side-chain objective and experimental penalty rules."""

import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import dimod


def load_model(energy_dir, num_res, num_rot):
    """Read complete tables; labels are residue-major, then rotamer-major."""
    if num_res < 2 or num_rot < 1:
        raise ValueError("Require num_res >= 2 and num_rot >= 1")
    prefix = Path(energy_dir) / f"{num_rot}rot_{num_res}res"
    with open(f"{prefix}_one_body_terms.csv", newline="") as stream:
        one = [(int(r["res i"]), int(r["rot A_i"]), float(r["E_ii"]))
               for r in csv.DictReader(stream)]
    with open(f"{prefix}_two_body_terms.csv", newline="") as stream:
        two = [(int(r["res i"]), int(r["res j"]), int(r["rot A_i"]),
                int(r["rot B_j"]), float(r["E_ij"])) for r in csv.DictReader(stream)]
    expected = {(i, r) for i in range(1, num_res + 1)
                for r in range(1, num_rot + 1)}
    if {(i, r) for i, r, _ in one} != expected or len(one) != len(expected):
        raise ValueError("One-body table must contain each 1-based residue/rotamer once")
    pairs = {(i, i + 1, r, s) for i in range(1, num_res)
             for r in range(1, num_rot + 1) for s in range(1, num_rot + 1)}
    if {row[:4] for row in two} != pairs or len(two) != len(pairs):
        raise ValueError("Two-body table must contain each forward adjacent pair once")
    if not all(math.isfinite(row[-1]) for row in one + two):
        raise ValueError("Energies must be finite")
    index = {key: v for v, key in enumerate(sorted(expected))}
    base = dimod.BinaryQuadraticModel(
        {index[i, r]: e for i, r, e in one},
        {(index[i, r], index[j, s]): e for i, j, r, s, e in two},
        0.0, dimod.BINARY,
    )
    blocks = [[index[i, r] for r in range(1, num_rot + 1)]
              for i in range(1, num_res + 1)]
    rules = {
        "UBP": sum(max(0.0, row[-1]) for row in one + two),
        "EA-MQCP": (
            sum(max(e for i, _, e in one if i == res)
                for res in range(1, num_res + 1))
            + sum(max(e for i, _, _, _, e in two if i == res)
                  for res in range(1, num_res))),
        "MQCP": max(row[-1] for row in one) + max(row[-1] for row in two),
    }
    digest = hashlib.sha256(json.dumps(
        [sorted(one), sorted(two)], separators=(",", ":")
    ).encode()).hexdigest()
    return base, blocks, rules, digest


def penalize(base, blocks, penalty, norm_target=None):
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Penalty must be finite and nonnegative; use --lambdas to override")
    bqm = base.copy()
    for block in blocks:
        bqm.offset += penalty
        for v in block:
            bqm.add_linear(v, -penalty)
        for u, v in combinations(block, 2):
            bqm.add_quadratic(u, v, 2 * penalty)
    scale = 1.0
    if norm_target is not None:
        if not math.isfinite(norm_target) or norm_target <= 0:
            raise ValueError("Normalization target must be finite and positive")
        maximum = max(abs(v) for v in [*bqm.linear.values(), *bqm.quadratic.values()])
        scale = (maximum or 1.0) / norm_target
        bqm.scale(1.0 / scale)
    return bqm, scale


def feasible(sample, blocks):
    return all(sum(sample[v] for v in block) == 1 for block in blocks)


def summarize(sampleset, base, blocks, reference=None):
    """Occurrence-weighted metrics; energies always exclude penalty and offsets."""
    records = []
    has_cbf = "chain_break_fraction" in sampleset.record.dtype.names
    cbf_sum = 0.0
    for row in sampleset.data():
        bits = "".join(str(int(row.sample[v])) for v in range(len(base)))
        count = int(row.num_occurrences)
        records.append({"bitstring": bits, "occurrences": count,
                        "feasible": feasible(row.sample, blocks),
                        "raw_qubo_energy": float(base.energy(row.sample))})
        if has_cbf:
            cbf_sum += float(row.chain_break_fraction) * count
    total = sum(r["occurrences"] for r in records)
    if not total:
        raise ValueError("Sampler returned no reads")
    valid = [r for r in records if r["feasible"]]
    return {
        "num_reads_returned": total,
        "feasible_rate": sum(r["occurrences"] for r in valid) / total,
        "ground_state_bitstring_success_rate": None if reference is None else
            sum(r["occurrences"] for r in records if r["bitstring"] == reference) / total,
        "chain_break_fraction": cbf_sum / total if has_cbf else None,
        "best_feasible_qubo_energy": min((r["raw_qubo_energy"] for r in valid), default=None),
        "samples": records,
    }


def input_arguments(parser):
    parser.add_argument("--energy-dir", type=Path, required=True)
    parser.add_argument("--num-res", type=int, required=True)
    parser.add_argument("--num-rot", type=int, required=True)


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
