"""Run penalty and chain-strength sweeps on a D-Wave QPU or a tiny exact check."""

import argparse
import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import dimod

from model import feasible, input_arguments, load_model, penalize, summarize, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.add_argument("--backend", choices=["qpu", "exact"], default="exact")
    parser.add_argument("--rules", nargs="+", choices=["UBP", "EA-MQCP", "MQCP"],
                        default=["UBP", "EA-MQCP", "MQCP"])
    parser.add_argument("--lambdas", nargs="+", type=float, help="Override rules with explicit values")
    parser.add_argument("--norm-target", type=float, default=None)
    parser.add_argument("--chain-strengths", nargs="+", type=float, default=None,
                        help="Omit for Ocean uniform torque compensation")
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--anneal-time", type=float, default=100.0, help="Microseconds")
    parser.add_argument("--solver", help="QPU solver name available to your account")
    parser.add_argument("--embedding-seed", type=int, default=42)
    parser.add_argument("--reference", type=Path, help="JSON from solve_reference.py")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.num_reads <= 0 or not math.isfinite(args.anneal_time) or args.anneal_time <= 0:
        parser.error("Reads and annealing time must be positive and finite")
    if args.chain_strengths and any(not math.isfinite(c) or c <= 0 for c in args.chain_strengths):
        parser.error("Chain strengths must be finite and positive")
    base, blocks, rules, digest = load_model(args.energy_dir, args.num_res, args.num_rot)
    reference = None
    if args.reference:
        data = json.loads(args.reference.read_text())
        reference = data.get("bitstring")
        if (data.get("optimal") is not True or data.get("input_sha256") != digest
                or not isinstance(reference, str) or len(reference) != len(base)
                or any(b not in "01" for b in reference)
                or not feasible([int(b) for b in reference], blocks)):
            parser.error("Reference must be optimal, feasible, and match the input tables")
        if not math.isclose(base.energy([int(b) for b in reference]),
                            data["raw_qubo_energy"], abs_tol=1e-6, rel_tol=0):
            parser.error("Reference energy does not match its bitstring")
    settings = ([("explicit", lam) for lam in args.lambdas] if args.lambdas
                else [(name, rules[name]) for name in args.rules])
    models = [(name, lam, *penalize(base, blocks, lam, args.norm_target))
              for name, lam in settings]
    if args.backend == "exact" and len(base) > 20:
        parser.error("Exact enumeration is limited to 20 variables; use --backend qpu")
    qpu = None
    embedding = None
    versions = {"dimod": version("dimod")}
    try:
        if args.backend == "qpu":
            from dwave.system import DWaveSampler, FixedEmbeddingComposite
            from dwave.embedding.chain_breaks import majority_vote
            from dwave.embedding.chain_strength import uniform_torque_compensation
            import minorminer

            qpu = DWaveSampler(**({"solver": args.solver} if args.solver else {}))
            # Reuse one embedding across every lambda and CS in this invocation.
            source = list(models[0][2].quadratic) + [(v, v) for v in base.variables]
            embedding = minorminer.find_embedding(source, qpu.edgelist,
                                                  random_seed=args.embedding_seed)
            if set(embedding) != set(base.variables):
                raise RuntimeError("No complete embedding found")
            sampler = FixedEmbeddingComposite(qpu, embedding)
            versions.update({p: version(p) for p in ("dwave-system", "minorminer")})
        for name, lam, bqm, scale in models:
            strengths = (args.chain_strengths or [None]) if qpu is not None else [None]
            for requested_cs in strengths:
                applied_cs = None
                if qpu is None:
                    response = dimod.ExactSolver().sample(bqm)
                else:
                    applied_cs = (requested_cs if requested_cs is not None
                                  else float(uniform_torque_compensation(bqm, embedding)))
                    response = sampler.sample(
                        bqm, num_reads=args.num_reads, annealing_time=args.anneal_time,
                        chain_strength=applied_cs, chain_break_method=majority_vote,
                        chain_break_fraction=True, auto_scale=True,
                    )
                result = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "input_sha256": digest, "num_res": args.num_res, "num_rot": args.num_rot,
                    "backend": args.backend, "versions": versions,
                    "method": name, "lambda_value": lam, "norm_target": args.norm_target,
                    "scale_factor": scale, "requested_chain_strength": requested_cs,
                    "applied_chain_strength": applied_cs, "embedding": embedding,
                    "embedding_seed": args.embedding_seed if qpu is not None else None,
                    "solver": qpu.solver.id if qpu is not None else None,
                    "auto_scale": True if qpu is not None else None,
                    "num_reads_requested": args.num_reads if qpu is not None else None,
                    "anneal_time_us": args.anneal_time if qpu is not None else None,
                    "reference_bitstring": reference,
                    "timing": response.info.get("timing", {}),
                    **summarize(response, base, blocks, reference),
                }
                output = args.output_dir / f"run_{uuid4().hex}.json"
                write_result(output, result)
                print(f"Saved {name}, lambda={lam:g}, CS={applied_cs}: {output}")
    finally:
        if qpu is not None:
            qpu.close()


if __name__ == "__main__":
    main()
