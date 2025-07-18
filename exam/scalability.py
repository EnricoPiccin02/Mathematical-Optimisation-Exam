import networkx as nx
import random
import numpy as np
import pandas as pd
from gurobipy import GRB
from collections import defaultdict
from itertools import product
from test import GraphData, ILPModel, MILPModel, Matheuristic, SUBTOUR_NONE, SUBTOUR_GCS

SEED = 3175351982
NUMBER_TUNING_INSTANCES = 72
RND_TEST_CLASS = "RND"
SWN_TEST_CLASS = "SWN"

# -----------------------------
# Configuration Parameters
# -----------------------------

# General parameters
node_numbers = [100, 200, 300, 400, 500]
conflict_densities = {
    100: [1e-3, 2e-3, 3e-3],
    200: [1e-4, 2e-4, 3e-4],
    300: [1e-5, 2e-5, 3e-5],
    400: [1e-5, 2e-5, 3e-5],
    500: [1e-5, 2e-5, 3e-5],
}
arc_cost_range = (1, 20)

# Random instances parameters
rnd_arc_density_values = [0.1, 0.2, 0.3]
rnd_penalty_ranges = {
    "p1": (25, 125),
    "p2": (50, 150),
    "p3": (75, 175),
    "p4": (100, 200),
}

# Small-World Network instances parameters
swn_penalty_range = (25, 200)
swn_beta = 0.5
swn_neighbours_frac = [0.15, 0.3, 0.45]

# -----------------------------
# Instance Generation
# -----------------------------

def generate_conflict_pairs_and_penalties(A, r, p_range):
    num_conflicts = int(r * len(A) * (len(A) - 1) / 2)
    conflict_pairs = set()
    while len(conflict_pairs) < num_conflicts:
        e, f = random.sample(A, 2)
        if e != f:
            pair = (e, f) 
            conflict_pairs.add(pair)

    conflict_pairs = list(conflict_pairs)
    penalties = [random.randint(*p_range) for _ in conflict_pairs]
    return conflict_pairs, penalties

def generate_random_topology_instance(n, d, r, p_range, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    m = int(d * n * (n - 1))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # Generate s–t path covering all nodes (random order)
    nodes = list(G.nodes())
    random.shuffle(nodes)
    source, target = nodes[0], nodes[-1]
    path = list(zip(nodes[:-1], nodes[1:]))
    G.add_edges_from(path)

    # Add random edges to reach the required m number of edges
    while G.number_of_edges() < m:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    A = list(G.edges())
    arc_weights = {a: random.randint(*arc_cost_range) for a in A}

    # Conflict generation
    conflict_pairs, penalties = generate_conflict_pairs_and_penalties(A, r, p_range)

    return GraphData(
        V=list(G.nodes()),
        A=A,
        source=source,
        target=target,
        arc_weights=arc_weights,
        conflict_pairs=conflict_pairs,
        penalties=penalties
    )

def generate_small_world_instance(n, k_frac, beta, r, p_range, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    # Generate undirected Watts–Strogatz ring and rewire
    k = int(k_frac * n)
    G = nx.watts_strogatz_graph(n, k, beta, seed=seed)

    # Convert to directed and use Newman–Watts augmentation
    G = G.to_directed()
    current_arcs = G.number_of_edges()
    max_arcs = n * (n - 1)
    target_arcs = int((G.number_of_edges() + r * max_arcs))
    m_extra = max(0, target_arcs - current_arcs)

    # Compute all possible directed non-self loops
    possible = set(product(range(n), repeat=2)) - set(G.edges()) - {(i, i) for i in range(n)}
    if m_extra > 0 and len(possible) >= m_extra:
        extra = random.sample(sorted(possible), m_extra)
        G.add_edges_from(extra)

    A = list(G.edges())
    arc_weights = {a: random.randint(*arc_cost_range) for a in A}

    # Conflict generation
    conflict_pairs, penalties = generate_conflict_pairs_and_penalties(A, r, p_range)

    # Randomly choose source and target
    source, target = random.sample(list(G.nodes()), 2)

    # Compute empirical arc density
    arc_density = len(A) / max_arcs

    # Pack into GraphData
    return GraphData(
        V=list(G.nodes()),
        A=A,
        source=source,
        target=target,
        arc_weights=arc_weights,
        conflict_pairs=conflict_pairs,
        penalties=penalties,
    ), arc_density

def generate_all_random_instances():
    all_rnd_instances = []
    for n in node_numbers:
        for d in rnd_arc_density_values:
            for r in conflict_densities[n]:
                for p_key, p_range in rnd_penalty_ranges.items():
                    for i in range(3):  # 3 instances per configuration
                        seed_base = (n, d, r, p_key, i)
                        seed = (hash(seed_base) ^ SEED) % (2**32)
                        instance = generate_random_topology_instance(n, d, r, p_range, seed)
                        entry = {
                            "n": n,
                            "d": d,
                            "r": r,
                            "p": p_range,
                            "i": i,
                            "seed": seed,
                            "data": instance
                        }
                        all_rnd_instances.append(entry)
    return all_rnd_instances

def generate_all_small_world_instances():
    all_swn_instances = []
    for n in node_numbers:
        for k_frac in swn_neighbours_frac:
            for r in conflict_densities[n]:
                for i in range(3):  # 3 instances per configuration
                    seed_base = (n, k_frac, swn_beta, r, i)
                    seed = (hash(seed_base) ^ SEED) % (2**32)
                    instance, arc_density = generate_small_world_instance(n, k_frac, swn_beta, r, swn_penalty_range, seed)
                    entry = {
                        "n": n,
                        "d": arc_density,
                        "r": r,
                        "p": swn_penalty_range,
                        "i": i,
                        "seed": seed,
                        "data": instance
                    }
                    all_swn_instances.append(entry)
    
    # Sort by arc density and classify into 3 arc density classes
    all_swn_instances.sort(key=lambda x: x["d"])
    for idx, entry in enumerate(all_swn_instances):
        if idx < 45:
            entry["d"] = "AD_1"
        elif idx < 90:
            entry["d"] = "AD_2"
        else:
            entry["d"] = "AD_3"

    return all_swn_instances


# -----------------------------
# Sample & Parameter Tuning
# -----------------------------

def sample_tuning_instances(all_instances):
    grouped = defaultdict(list)
    for instance in all_instances:
        key = (instance["n"], instance["d"], instance["r"], instance["p"])
        grouped[key].append(instance)
    
    keys = list(grouped.keys())
    random.seed(SEED)
    sampled_keys = random.sample(keys, NUMBER_TUNING_INSTANCES)

    tuning_instances = [random.choice(grouped[key]) for key in sampled_keys]
    return tuning_instances

def parameter_tuning(tuning_instances):
    results = []

    for data in tuning_instances:
        n, d, r, p, i, seed = data["n"], data["d"], data["r"], data["p"], data["i"], data["seed"]

        # ILP and MILP tuning over epsilon
        for epsilon in [1, 5, 10, 20, 1000]:
            for model_type, ModelClass in [("ILP", ILPModel), ("MILP", MILPModel)]:
                model = ModelClass(data["data"], subtour_method=SUBTOUR_GCS, epsilon=epsilon, time_limit=3600)
                model.build_model()
                model.solve()
                
                status = model.model.Status
                is_opt = status == GRB.OPTIMAL
                runtime = model.model.Runtime
                cuts = model.num_cuts

                results.append({
                    "model": model_type,
                    "n": n, "d": d, "r": r, "p": p, "i": i, "seed": seed,
                    "epsilon": epsilon,
                    "tau": None,
                    "status": status,
                    "optimal": is_opt,
                    "cuts": cuts,
                    "runtime": runtime,
                    "gap": None
                })

                print(f"[{model_type}] ε={epsilon}, Opt={is_opt}, Cuts={cuts}, Time={runtime:.2f}s")

        # Relaxed MILP tuning over tau
        for tau in [10, 20, 30, 40, 50, 60]:
            relaxed_model = MILPModel(data["data"], subtour_method=SUBTOUR_NONE, time_limit=tau)
            relaxed_model.build_model()
            relaxed_model.solve()

            status = relaxed_model.model.Status
            is_opt = status == GRB.OPTIMAL
            runtime = relaxed_model.model.Runtime
            gap = relaxed_model.model.MIPGap if relaxed_model.model.SolCount > 0 else None

            results.append({
                "model": "RelaxedMILP",
                "n": n, "d": d, "r": r, "p": p, "i": i, "seed": seed,
                "epsilon": None,
                "tau": tau,
                "status": status,
                "optimal": is_opt,
                "cuts": None,
                "runtime": runtime,
                "gap": gap
            })

            gap_str = f"{gap:.2%}" if gap is not None else "N/A"
            print(f"[Relaxed MILP] τ={tau}, Opt={is_opt}, Gap={gap_str}, Time={runtime:.2f}s")

    df = pd.DataFrame(results)

    # Aggregated summary for ILP and MILP
    print("\n=== Tuning Results Summary: ILP and MILP ===")
    for model_type in ["ILP", "MILP"]:
        for epsilon in [1, 5, 10, 20, 1000]:
            subset = df[(df["model"] == model_type) & (df["epsilon"] == epsilon)]
            opt_count = subset["optimal"].sum()
            avg_cuts = subset["cuts"].mean()
            avg_time = subset["runtime"].mean()
            print(f"{model_type} | ε={epsilon}: #Opt={opt_count}, #Cuts={avg_cuts:.2f}, Time={avg_time:.2f}s")

    # Aggregated summary for the relaxed MILP
    print("\n=== Tuning Results Summary: Relaxed MILP ===")
    for tau in [10, 20, 30, 40, 50, 60]:
        subset = df[(df["model"] == "RelaxedMILP") & (df["tau"] == tau)]
        opt_count = subset["optimal"].sum()
        avg_gap = subset["gap"].mean() * 100 if subset["gap"].notnull().any() else None
        avg_time = subset["runtime"].mean()
        print(f"τ={tau}: #Opt={opt_count}, GapLB={avg_gap:.3f}%, Time={avg_time:.3f}s")

    df.to_csv("tuning_results.csv", index=False)

    return df


# -----------------------------
# Run & Summarize Experiments
# -----------------------------

def run_full_experiments(all_instances, test_class):
    results = []

    for data in all_instances:
        n, d, r, p, i = data["n"], data["d"], data["r"], data["p"], data["i"]
        instance_id = f"{test_class}N{n}D{d}R{r}P{p}I{i}"

        # Run ILP (ε = 1)
        ilp = ILPModel(data["data"], subtour_method=SUBTOUR_GCS, epsilon=1, time_limit=3600)
        ilp.build_model()
        ilp.solve()
        results.append({
            "method": "ILP",
            "n": n, "d": d,
            "instance_id": instance_id,
            "LB": ilp.model.ObjBound,
            "UB": ilp.model.ObjVal,
            "Cuts": ilp.num_cuts,
            "TotalTime": ilp.model.Runtime,
            "Gap%": 100 * ilp.model.MIPGap if ilp.model.MIPGap is not None else 0,
        })

        # Run MILP (ε = 1)
        milp = MILPModel(data["data"], subtour_method=SUBTOUR_GCS, epsilon=1, time_limit=3600)
        milp.build_model()
        milp.solve()
        results.append({
            "method": "MILP",
            "n": n, "d": d,
            "instance_id": instance_id,
            "LB": milp.model.ObjBound,
            "UB": milp.model.ObjVal,
            "Cuts": milp.num_cuts,
            "TotalTime": milp.model.Runtime,
            "Gap%": 100 * milp.model.MIPGap if milp.model.MIPGap is not None else 0,
        })

        # Run Matheuristic (τ = 60s)
        matheuristic = Matheuristic(data["data"], time_limit=60)
        matheuristic.run()
        results.append({
            "method": "Matheuristic",
            "n": n, "d": d,
            "instance_id": instance_id,
            "Cost": matheuristic.best_arc_cost + matheuristic.best_penalty_cost,
            "FirstStageTime": matheuristic.first_stage_time,
            "SecondStageTime": matheuristic.second_stage_time,
            "TotalTime": matheuristic.first_stage_time + matheuristic.second_stage_time,
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{test_class}_experimental_results.csv", index=False)
    return df

def summarize_exact_results(df):
    table = df[df["method"].isin(["ILP", "MILP"])]
    grouped = table.groupby(["method", "n", "d"])

    return grouped.agg({
        "LB": "mean",
        "UB": "mean",
        "Cuts": "mean",
        "TotalTime": "mean",
        "Gap%": "mean"
    }).reset_index()

def summarize_matheuristic(df):
    mat = df[df["method"] == "Matheuristic"]
    grouped = mat.groupby(["n", "d"])
    return grouped.agg({
        "Cost": "mean",
        "FirstStageTime": "mean",
        "SecondStageTime": "mean",
        "TotalTime": "mean"
    }).reset_index()


# -----------------------------
# Main Testing Execution
# -----------------------------

if __name__ == "__main__":
    all_rnd_instances = generate_all_random_instances()

    # Parameter Tuning
    parameter_tuning(sample_tuning_instances(all_rnd_instances))

    # Run full experiments for random instances
    rnd_results_df = run_full_experiments(all_rnd_instances, RND_TEST_CLASS)
    print("=== Random Instances Results ===")
    print(summarize_exact_results(rnd_results_df))
    print(summarize_matheuristic(rnd_results_df))

    # Run full experiments for small-world network instances
    all_swn_instances = generate_all_small_world_instances()
    swn_results_df = run_full_experiments(all_swn_instances, SWN_TEST_CLASS)
    print("=== Small-World Network Instances Results ===")
    print(summarize_exact_results(swn_results_df))
    print(summarize_matheuristic(swn_results_df))