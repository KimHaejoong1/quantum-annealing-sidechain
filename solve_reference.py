"""Solve the raw objective with hard one-hot constraints using Gurobi."""

import argparse
import math

from model import feasible, input_arguments, load_model, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--output", default="results/reference.json")
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive")
    base, blocks, _, digest = load_model(args.energy_dir, args.num_res, args.num_rot)
    import gurobipy as gp

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            model.Params.TimeLimit = args.time_limit
            model.Params.MIPGap = 1e-10
            model.Params.PoolSearchMode = 0
            x = model.addVars(range(len(base)), vtype=gp.GRB.BINARY)
            for block in blocks:
                model.addConstr(gp.quicksum(x[v] for v in block) == 1)
            model.setObjective(
                gp.quicksum(e * x[v] for v, e in base.linear.items())
                + gp.quicksum(e * x[u] * x[v] for (u, v), e in base.quadratic.items()),
                gp.GRB.MINIMIZE,
            )
            model.optimize()
            result = {"input_sha256": digest, "num_res": args.num_res,
                      "num_rot": args.num_rot, "gurobi_version": gp.gurobi.version(),
                      "status": int(model.Status),
                      "optimal": model.Status == gp.GRB.OPTIMAL,
                      "runtime_s": model.Runtime, "mip_gap": None,
                      "bitstring": None, "raw_qubo_energy": None}
            if model.SolCount:
                sample = {v: int(round(x[v].X)) for v in range(len(base))}
                if not feasible(sample, blocks):
                    raise RuntimeError("Returned solution violates one-hot constraints")
                result.update(bitstring="".join(str(sample[v]) for v in range(len(base))),
                              raw_qubo_energy=float(base.energy(sample)), mip_gap=model.MIPGap)
            write_result(args.output, result)
    print(f"Saved reference (optimal={result['optimal']}) to {args.output}")


if __name__ == "__main__":
    main()
