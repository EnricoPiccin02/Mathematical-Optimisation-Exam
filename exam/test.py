import gurobipy as gp
import networkx as nx
import time
from gurobipy import GRB

SUBTOUR_NONE = "none"
SUBTOUR_GCS = "gcs"

# -----------------------------
# Data Models & Utilities
# -----------------------------

class GraphData:
    def __init__(self, V, A, arc_weights, conflict_pairs, penalties, source, target):
        self.V = V                           # List of vertices
        self.A = A                           # List of arcs (tuples of vertices)
        self.arc_weights = arc_weights
        self.conflict_pairs = conflict_pairs # List of conflict pairs (tuples of arcs)
        self.penalties = penalties           # List of penalties for conflict pairs
        self.source = source
        self.target = target

    def get_outgoing_arcs(self, i):
        return [a for a in self.A if a[0] == i]

    def get_incoming_arcs(self, i):
        return [a for a in self.A if a[1] == i]


# -----------------------------
# Base Optimization Model
# -----------------------------

class BaseSPEDACModel:
    def __init__(self, graph_data: GraphData, subtour_method: str=SUBTOUR_GCS, epsilon=1, time_limit=None):
        self.data = graph_data
        self.subtour_method = subtour_method
        self.epsilon = epsilon
        self.time_limit = time_limit
        self.model = gp.Model()
        self.model.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            self.model.setParam('TimeLimit', self.time_limit)
        self.x = {}
        self.num_cuts = 0 # Number of added violated cuts

    def add_flow_constraints(self):
        for i in self.data.V:
            expr = gp.quicksum(self.x[a] for a in self.data.get_outgoing_arcs(i)) - \
                   gp.quicksum(self.x[a] for a in self.data.get_incoming_arcs(i))
            if i == self.data.source:
                self.model.addConstr(expr == 1)
            elif i == self.data.target:
                self.model.addConstr(expr == -1)
            else:
                self.model.addConstr(expr == 0)

    def add_single_outgoing_arc_constraints(self):
        for i in self.data.V:
            self.model.addConstr(gp.quicksum(self.x[a] for a in self.data.get_outgoing_arcs(i)) <= 1)

    def extract_path(self):
        selected_arcs = [a for a in self.data.A if self.x[a].X > 0.5]
        path = [self.data.source]
        current = self.data.source
        while current != self.data.target:
            for (u, v) in selected_arcs:
                if u == current:
                    path.append(v)
                    current = v
                    break
        return path

    def print_solution(self):
        print("Optimal path from the source", self.data.source, "to the target", self.data.target, "is:", " → ".join(map(str, self.extract_path())))
        print(f"Total time taken: {self.model.Runtime:.2f} seconds")
        if self.subtour_method != SUBTOUR_NONE:
            print(f"Subtour elimination method: {self.subtour_method}")
            print(f"Number of cuts: {self.num_cuts}")

        total_arc_cost = sum(self.data.arc_weights[a] * self.x[a].X for a in self.data.A)
        print(f"Objective value: {self.model.ObjVal}")
        print(f"  Arc cost: {total_arc_cost}")
        print(f"  Penalty cost: {self.model.ObjVal - total_arc_cost}")

    def _GCS_subtour_elimination_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(self.x)
            selected_arcs = [a for a in self.data.A if vals[a] > 0.5]

            # Build the solution graph from selected arcs
            G_sol = nx.DiGraph()
            G_sol.add_edges_from(selected_arcs)

            # Collect violated constraints
            violated_constraints = []

            # Loop through strongly connected components (SCCs)
            for component in nx.strongly_connected_components(G_sol):
                if (len(component) <= 1 or self.data.source in component or self.data.target in component):
                    continue  # Skip trivial or terminal SCCs

                S = set(component)

                for k in S:
                    Sk = S - {k}

                    # x(δ⁺(k)): outgoing arcs from node k
                    delta_plus_k = self.data.get_outgoing_arcs(k)
                    x_delta_plus_k = sum(vals[a] for a in delta_plus_k)

                    # x(δ⁺(S_k)): outgoing arcs from all nodes in Sk
                    delta_plus_Sk = [a for i in Sk for a in self.data.get_outgoing_arcs(i)]
                    x_delta_plus_Sk = sum(vals[a] for a in delta_plus_Sk)

                    # Violation amount
                    violation_amount = x_delta_plus_k - x_delta_plus_Sk

                    violated_constraints.append({
                        'lhs': delta_plus_k,
                        'rhs': delta_plus_Sk,
                        'violation': violation_amount
                    })

            # Sort and add up to epsilon most violated constraints
            for vc in sorted(violated_constraints, key=lambda x: -x['violation'])[:self.epsilon]:
                lhs_expr = gp.quicksum(self.x[a] for a in vc['lhs'])
                rhs_expr = gp.quicksum(self.x[a] for a in vc['rhs'])
                self.num_cuts += 1
                model.cbLazy(lhs_expr <= rhs_expr)

    def solve(self):
        if self.subtour_method not in {SUBTOUR_NONE, SUBTOUR_GCS}:
            raise ValueError(f"Invalid subtour_method: {self.subtour_method}. Choose from {SUBTOUR_NONE} or {SUBTOUR_GCS}.")

        if self.subtour_method == SUBTOUR_NONE:
            self.model.optimize()
        else:
            self.num_cuts = 0
            self.model.setParam("Cuts", 0)            # Disable all Gurobi cuts in order to prevent Lazy cuts from being redundant
            self.model.setParam("GomoryPasses", 0)
            self.model.setParam("CliqueCuts", 0)
            self.model.setParam("CoverCuts", 0)
            self.model.setParam("FlowCoverCuts", 0)
            self.model.setParam("FlowPathCuts", 0)
            self.model.setParam("GUBCoverCuts", 0)
            self.model.setParam("ImpliedCuts", 0)
            self.model.setParam("MIPSepCuts", 0)
            self.model.setParam("MIRCuts", 0)
            self.model.setParam("StrongCGCuts", 0)
            self.model.setParam("ZeroHalfCuts", 0)
            self.model.setParam("PreCrush", 1)        # Allows lazy constraints to work properly
            self.model.setParam("LazyConstraints", 1) # Activates subtour_elimination_callback
            self.model.optimize(callback=self._GCS_subtour_elimination_callback)

        return {a: var.X for a, var in self.x.items()}


# -----------------------------
# ILP Model (Formulation 1)
# -----------------------------

class ILPModel(BaseSPEDACModel):
    def build_model(self):
        self.x = {a: self.model.addVar(vtype=GRB.BINARY, name=f"x_{a}") for a in self.data.A}
        self.y = {k: self.model.addVar(vtype=GRB.BINARY, name=f"y_{k}") for k in range(len(self.data.conflict_pairs))}
        self.z = {k: self.model.addVar(vtype=GRB.BINARY, name=f"z_{k}") for k in range(len(self.data.conflict_pairs))}

        self.model.setObjective(
            gp.quicksum(self.data.arc_weights[a] * self.x[a] for a in self.data.A) +
            gp.quicksum(self.data.penalties[k] * (self.z[k] + self.y[k]) for k in range(len(self.data.conflict_pairs))),
            GRB.MINIMIZE
        )

        self.add_flow_constraints()
        self.add_single_outgoing_arc_constraints()

        # Add conflict constraints
        for k, (e_k, f_k) in enumerate(self.data.conflict_pairs):
            self.model.addConstr(self.x[e_k] + self.x[f_k] <= self.y[k] + 1)
            self.model.addConstr(self.z[k] >= 1 - self.x[e_k] - self.x[f_k])

    def print_solution(self, show_conflicts=True):
        super().print_solution()

        if show_conflicts:
            print("Conflict Variables (ILP):")
            for k, (e_k, f_k) in enumerate(self.data.conflict_pairs):
                print(f"  Conflict {k}: Arcs {e_k}, {f_k} -> y_{k}={int(round(self.y[k].X))}, z_{k}={int(round(self.z[k].X))}")


# -----------------------------
# MILP Model (Formulation 2)
# -----------------------------

class MILPModel(BaseSPEDACModel):
    def build_model(self):
        self.x = {a: self.model.addVar(vtype=GRB.BINARY, name=f"x_{a}") for a in self.data.A}
        self.omega = {k: self.model.addVar(lb=0.0, name=f"omega_{k}") for k in range(len(self.data.conflict_pairs))}

        self.model.setObjective(
            gp.quicksum(self.data.arc_weights[a] * self.x[a] for a in self.data.A) +
            gp.quicksum(
                (2 * self.omega[k] - self.x[self.data.conflict_pairs[k][0]] - self.x[self.data.conflict_pairs[k][1]]) * self.data.penalties[k]
                for k in range(len(self.data.conflict_pairs))
            ) +
            gp.quicksum(self.data.penalties[k] for k in range(len(self.data.conflict_pairs))),
            GRB.MINIMIZE
        )

        self.add_flow_constraints()
        self.add_single_outgoing_arc_constraints()

        # Add conflict constraints
        for k, (e_k, f_k) in enumerate(self.data.conflict_pairs):
            self.model.addConstr(self.omega[k] <= self.x[e_k])
            self.model.addConstr(self.omega[k] <= self.x[f_k])
            self.model.addConstr(self.omega[k] >= self.x[e_k] + self.x[f_k] - 1)

    def print_solution(self, show_conflicts=True):
        super().print_solution()

        if show_conflicts:
            print("Conflict Variables (MILP):")
            for k, (e_k, f_k) in enumerate(self.data.conflict_pairs):
                is_penalty_paid = 2 * self.omega[k].X - self.x[self.data.conflict_pairs[k][0]].X - self.x[self.data.conflict_pairs[k][1]].X + 1
                print(f"  Conflict {k}: Arcs {e_k}, {f_k} -> Penalty paid={is_penalty_paid}")


# -----------------------------
# Matheuristic First Stage
# -----------------------------

class FirstStageHeuristic:
    def __init__(self, graph_data: GraphData, time_limit: float):
        self.data = graph_data
        self.time_limit = time_limit

    def run(self):
        # Solve the MILP relaxation without subtour elimination (relax constraint)
        model = MILPModel(self.data, subtour_method=SUBTOUR_NONE)
        model.model.setParam('TimeLimit', self.time_limit)
        model.build_model()
        solution = model.solve()

        # Build the subgraph G_bar using arcs where x_a = 1
        arcs_used = [a for a, v in solution.items() if v > 0.5]
        G_bar = nx.DiGraph()
        G_bar.add_nodes_from(self.data.V)
        G_bar.add_edges_from(arcs_used)

        # Extract the unique s–t path in G_bar
        path_arcs = self.extract_path(G_bar)

        # Find vertex-disjoint cycles in G_bar
        cycles = self.detect_cycles(G_bar, path_arcs)

        # Assign priorities ρ
        A_P = set(path_arcs)
        A_C = set(a for cycle in cycles for a in cycle)

        rho = {}
        for a in self.data.A:
            if a in A_C:
                rho[a] = 2
            elif a in A_P:
                rho[a] = 1
            else:
                rho[a] = 0

        return path_arcs, rho

    # Extract the s-t path from G_bar
    def extract_path(self, G_bar):
        try:
            path_nodes = nx.shortest_path(G_bar, source=self.data.source, target=self.data.target)
            return [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
        except nx.NetworkXNoPath:
            return []

    # Detect vertex-disjoint cycles by removing path nodes from G_bar
    def detect_cycles(self, G_bar, path_arcs):
        path_nodes = set(u for u, v in path_arcs) | {path_arcs[-1][1]} if path_arcs else set()
        G_cycles = G_bar.copy()
        G_cycles.remove_nodes_from(path_nodes)
        cycles = list(nx.simple_cycles(G_cycles))
        return [[(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))] for cycle in cycles]


# -----------------------------
# Matheuristic Second Stage
# -----------------------------

class SecondStageHeuristic:
    def __init__(self, graph_data, path, rho):
        self.data = graph_data
        self.P = path
        self.rho = rho

    def run(self):
        best_path = None
        best_cost = float('inf')

        # Extract ordered list of vertices from the path
        A_P = list(self.P)

        for idx, (i, j) in enumerate(A_P):
            # Get prefix path up to (i,j) (excluding it)
            prefix_arcs = A_P[:idx]
            visited = set()
            partial_path = []

            for u, v in prefix_arcs:
                partial_path.append((u, v))
                visited.add(u)
            if prefix_arcs:
                visited.add(prefix_arcs[-1][1])

            # Temporarily negate priority of (i,j)
            original_rho = self.rho.get((i, j), 0)
            self.rho[(i, j)] = -original_rho

            # Start greedy search from tail = i
            tail = i
            while tail != self.data.target:
                neighbors = [h for _, h in self.data.get_outgoing_arcs(tail) if h not in visited]

                if not neighbors:
                    break

                # Check for direct connection to target
                if self.data.target in neighbors:
                    final_arc = (tail, self.data.target)
                    candidate_path = partial_path + [final_arc]
                    cost = sum(self.data.arc_weights.get(a, 0) for a in candidate_path)
                    if cost < best_cost:
                        best_path = list(candidate_path)
                        best_cost = cost

                # Choose next arc by highest priority
                best_head = max(neighbors, key=lambda h: self.rho.get((tail, h), 0))
                next_arc = (tail, best_head)
                partial_path.append(next_arc)
                visited.add(tail)
                tail = best_head

            # Restore priority
            self.rho[(i, j)] = original_rho

        return best_path, best_cost
    

# -----------------------------
# Matheuristic
# -----------------------------

class Matheuristic:
    def __init__(self, graph_data: GraphData, time_limit: float = 10):
        self.data = graph_data
        self.time_limit = time_limit
        self.first_stage_time = 0
        self.second_stage_time = 0
        self.best_path = None
        self.best_arc_cost = float('inf')
        self.best_penalty_cost = 0

    def run(self):
        # Run first stage and compute run-time
        self.first_stage_time = time.time()
        first_stage = FirstStageHeuristic(self.data, self.time_limit)
        best_st_path, rho = first_stage.run()
        self.first_stage_time = time.time() - self.first_stage_time
        
        # Run second stage and compute run-time
        self.second_stage_time = time.time()
        second_stage = SecondStageHeuristic(self.data, best_st_path, rho)
        self.best_path, self.best_arc_cost = second_stage.run()
        self.second_stage_time = time.time() - self.second_stage_time
        
        path_set = set(self.best_path)
        self.best_penalty_cost = sum(
            p_k for (e_k, f_k), p_k in zip(self.data.conflict_pairs, self.data.penalties)
            if (e_k in path_set and f_k in path_set) or (e_k not in path_set and f_k not in path_set)
        )

    def print_solution(self):
        print("Optimal path from the source", self.data.source, "to the target", self.data.target, "is:", 
              " → ".join(map(str, [self.best_path[0][0]] + [v for _, v in self.best_path])))
        print(f"Total time taken: {self.first_stage_time + self.second_stage_time:.2f} seconds")
        print(f"Objective value: {self.best_arc_cost + self.best_penalty_cost}")
        print(f"  Arc cost: {self.best_arc_cost}")
        print(f"  Penalty cost: {self.best_penalty_cost}")


# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    V = list(range(1, 11))
    A = [
        (1, 2), (1, 3), (2, 4), (3, 4), (4, 5),
        (2, 6), (6, 7), (7, 8), (8, 5), (5, 9),
        (9, 10), (3, 7), (7, 10), (6, 9), (1, 6)
    ]
    arc_weights = {a: (i % 5 + 1) for i, a in enumerate(A)}
    conflict_pairs = [((1, 2), (1, 3)), ((2, 4), (3, 4)), ((6, 7), (7, 8))]
    penalties = [3, 2, 5]
    source, target = 1, 10

    data = GraphData(V, A, arc_weights, conflict_pairs, penalties, source, target)

    print("\n--- ILP ---")
    ilp = ILPModel(data, subtour_method=SUBTOUR_GCS, epsilon=1)
    ilp.build_model()
    ilp.solve()
    ilp.print_solution()

    print("\n--- MILP ---")
    milp = MILPModel(data, subtour_method=SUBTOUR_GCS, epsilon=10)
    milp.build_model()
    milp.solve()
    milp.print_solution()

    print("\n--- Matheuristic ---")
    matheuristic = Matheuristic(data, time_limit=20)
    matheuristic.run()
    matheuristic.print_solution()