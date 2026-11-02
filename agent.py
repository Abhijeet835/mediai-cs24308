"""
MediAI agent: AI modules for the CS24308 project.

Module II  : Search (BFS, Greedy, A*) for ambulance routing
Module III : Knowledge representation + forward chaining for triage
Module IV  : Bayesian reasoning + goal-stack planning
Module V   : Learning (neural network, loaded in app.py)
"""
import heapq
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =====================================================================
# MODULE II: SEARCH  (city grid + BFS / Greedy / A*)
# =====================================================================
N = 20
FREE, BLOCKED, TRAFFIC = 0, 1, 2
COST = {FREE: 1, TRAFFIC: 3}          # minutes to drive through one cell
BASE = (1, 1)                         # ambulance base (row, col)
HOSPITAL = (18, 18)                   # hospital (row, col)

# named places where the patient can be (row, col)
PATIENT_SPOTS = {
    "Near the hospital (south-east)": (15, 16),
    "City centre (heavy traffic)": (9, 9),
    "West side": (14, 3),
    "North-east": (3, 17),
    "School (near the base)": (3, 3),
    "College campus (north-west)": (5, 3),
    "Bus stand (north)": (2, 10),
    "Highway toll (far north)": (1, 13),
    "Market (traffic zone)": (11, 10),
    "Industrial area (east)": (10, 17),
    "Railway station (south-west)": (17, 5),
    "Stadium (south)": (18, 10),
}


def make_city(seed=8):
    """Random city: roads (0), blocked cells (1), traffic jams (2)."""
    rng = np.random.RandomState(seed)
    g = np.zeros((N, N), dtype=int)
    g[rng.rand(N, N) < 0.10] = BLOCKED
    for r0, r1, c0, c1 in [(5, 14, 8, 12), (13, 18, 2, 8), (8, 13, 6, 9)]:
        block = g[r0:r1, c0:c1]
        block[block == FREE] = TRAFFIC
    g[BASE] = FREE
    g[HOSPITAL] = FREE
    return g


def nearest_free(grid, cell):
    """If the chosen cell is blocked, move to the closest open road cell."""
    if grid[cell] != BLOCKED:
        return cell
    best, best_d = None, 10 ** 9
    for r in range(N):
        for c in range(N):
            if grid[r, c] != BLOCKED:
                d = abs(r - cell[0]) + abs(c - cell[1])
                if d < best_d:
                    best, best_d = (r, c), d
    return best


def _neighbours(grid, cell):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] != BLOCKED:
            yield (nr, nc)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _rebuild(parent, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def path_time(grid, path):
    """Total travel time in minutes (entering each cell after the start)."""
    return sum(COST[grid[cell]] for cell in path[1:])


def bfs(grid, start, goal):
    """Uninformed: fewest cells, ignores traffic."""
    parent, q, explored = {start: None}, deque([start]), set()
    while q:
        cur = q.popleft()
        explored.add(cur)
        if cur == goal:
            p = _rebuild(parent, goal)
            return p, explored, path_time(grid, p)
        for nb in _neighbours(grid, cur):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    return None, explored, None


def greedy(grid, start, goal):
    """Heuristic only: always heads toward the goal, ignores cost so far."""
    parent, explored = {start: None}, set()
    pq = [(_manhattan(start, goal), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in explored:
            continue
        explored.add(cur)
        if cur == goal:
            p = _rebuild(parent, goal)
            return p, explored, path_time(grid, p)
        for nb in _neighbours(grid, cur):
            if nb not in parent:
                parent[nb] = cur
                heapq.heappush(pq, (_manhattan(nb, goal), nb))
    return None, explored, None


def astar(grid, start, goal):
    """A*: f = g (time so far) + h (Manhattan distance). Optimal here."""
    parent, g, explored = {start: None}, {start: 0}, set()
    pq = [(_manhattan(start, goal), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in explored:
            continue
        explored.add(cur)
        if cur == goal:
            p = _rebuild(parent, goal)
            return p, explored, g[goal]
        for nb in _neighbours(grid, cur):
            ng = g[cur] + COST[grid[nb]]
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(pq, (ng + _manhattan(nb, goal), nb))
    return None, explored, None


ALGOS = {"BFS": bfs, "Greedy": greedy, "A*": astar}


def draw_city(ax, grid, paths=(), explored=None, start=BASE, hospital=HOSPITAL,
              patient=None, title=""):
    cmap = ListedColormap(["#f5f5f5", "#2b2b2b", "#f4a261"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
    if explored:
        ys, xs = zip(*explored)
        ax.scatter(xs, ys, s=14, marker="s", color="#4cc9f0", alpha=0.55)
    for p in paths:
        if p:
            ys, xs = zip(*p)
            ax.plot(xs, ys, color="crimson", lw=2.5)
    ax.scatter([start[1]], [start[0]], s=140, color="green", zorder=5)
    ax.scatter([hospital[1]], [hospital[0]], s=200, color="red", marker="P", zorder=5)
    if patient is not None:
        ax.scatter([patient[1]], [patient[0]], s=150, color="royalblue",
                   marker="*", zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def compare_searches(grid):
    """Run all three algorithms base -> hospital. Returns figure + numbers."""
    results = {}
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    for i, (name, fn) in enumerate(ALGOS.items()):
        path, explored, minutes = fn(grid, BASE, HOSPITAL)
        results[name] = {"minutes": minutes, "explored": len(explored)}
        ax = fig.add_subplot(gs[0, i])
        draw_city(ax, grid, [path], explored,
                  title="%s\n%s min | %d cells explored" % (name, minutes, len(explored)))
    names = list(ALGOS)
    colors = ["#4cc9f0", "#f4a261", "#2a9d8f"]
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar(names, [results[n]["explored"] for n in names], color=colors)
    ax1.set_title("Cells explored (lower = faster search)", fontsize=10)
    ax2 = fig.add_subplot(gs[1, 1:3])
    ax2.bar(names, [results[n]["minutes"] for n in names], color=colors)
    ax2.set_title("Ambulance travel time, minutes (lower = better)", fontsize=10)
    fig.suptitle("Ambulance routing: green = base, red + = hospital, orange = traffic, "
                 "black = blocked, blue = explored", fontsize=11)
    fig.tight_layout()
    return fig, results


# ---- NEW: compare the three algorithms on every patient location ----
def analyse_locations(grid):
    """Run BFS, Greedy and A* from the base to every patient location,
    plus A* from the patient to the hospital. Returns a table (DataFrame)."""
    nz = lambda v: np.nan if v is None else v
    rows = []
    for i, (name, cell) in enumerate(PATIENT_SPOTS.items(), 1):
        p = nearest_free(grid, cell)
        rec = {"No.": i, "Location": name}
        for alg, fn in ALGOS.items():
            _, explored, minutes = fn(grid, BASE, p)
            rec[alg + " min"] = nz(minutes)
            rec[alg + " cells"] = len(explored)
        _, _, to_h = astar(grid, p, HOSPITAL)
        rec["To hospital min"] = nz(to_h)
        rec["Total ETA (A*)"] = rec["A* min"] + rec["To hospital min"]
        rows.append(rec)
    return pd.DataFrame(rows)


def locations_map(grid):
    """City map with every possible patient location numbered."""
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_city(ax, grid, title="Possible patient locations (numbers match the table)")
    for i, (name, cell) in enumerate(PATIENT_SPOTS.items(), 1):
        r, c = nearest_free(grid, cell)
        ax.scatter([c], [r], s=230, color="royalblue", edgecolor="white", zorder=6)
        ax.text(c, r, str(i), color="white", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=7)
    fig.tight_layout()
    return fig


def locations_charts(df):
    """Three charts comparing the algorithms across all locations."""
    x = np.arange(len(df))
    w = 0.27
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(x - w, df["BFS min"], w, label="BFS", color="#4cc9f0")
    ax1.bar(x, df["Greedy min"], w, label="Greedy", color="#f4a261")
    ax1.bar(x + w, df["A* min"], w, label="A*", color="#2a9d8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["No."])
    ax1.set_xlabel("Patient location number")
    ax1.set_ylabel("Minutes, base to patient")
    ax1.set_title("Travel time to each patient: A* is never slower than BFS or Greedy")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    d = df.sort_values("Total ETA (A*)")
    ax2.barh(d["Location"], d["A* min"], color="#2a9d8f", label="Base to patient")
    ax2.barh(d["Location"], d["To hospital min"], left=d["A* min"], color="#e63946",
             label="Patient to hospital")
    ax2.set_xlabel("Minutes (A* route)")
    ax2.set_title("Total ambulance time per location")
    ax2.tick_params(axis="y", labelsize=8)
    ax2.legend(fontsize=8, loc="lower right")

    ax3 = fig.add_subplot(gs[1, 1])
    for alg, col in (("BFS", "#4cc9f0"), ("Greedy", "#f4a261"), ("A*", "#2a9d8f")):
        ax3.plot(df["No."], df[alg + " cells"], "o-", label=alg, color=col)
    ax3.set_xlabel("Patient location number")
    ax3.set_ylabel("Cells explored")
    ax3.set_title("Search effort (fewer = less work)")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# MODULE III: KNOWLEDGE REPRESENTATION + FORWARD CHAINING
# =====================================================================
# Each rule: (id, [conditions], conclusion)
RULES = [
    ("R1",  ["chest_pain", "sweating"], "suspect_heart_attack"),
    ("R2",  ["age_over_50", "suspect_heart_attack"], "cardiac_emergency"),
    ("R3",  ["face_droop", "slurred_speech"], "suspect_stroke"),
    ("R4",  ["arm_weakness", "suspect_stroke"], "stroke_emergency"),
    ("R5",  ["breathing_difficulty", "chest_pain"], "cardiac_emergency"),
    ("R6",  ["unconscious"], "life_threatening"),
    ("R7",  ["frequent_urination", "high_glucose", "thirst"], "suspect_diabetes"),
    ("R8",  ["high_bp", "suspect_diabetes"], "needs_urgent_checkup"),
    ("R9",  ["fever", "joint_pain", "rash"], "suspect_dengue"),
    ("R10", ["suspect_dengue"], "needs_urgent_checkup"),
    ("R11", ["breathing_difficulty"], "needs_urgent_checkup"),
    ("R12", ["cardiac_emergency"], "CRITICAL"),
    ("R13", ["stroke_emergency"], "CRITICAL"),
    ("R14", ["life_threatening"], "CRITICAL"),
    ("R15", ["needs_urgent_checkup"], "URGENT"),
    ("R16", ["fever"], "ROUTINE"),
    ("R17", ["cough"], "ROUTINE"),
]

SYMPTOMS = {
    "chest_pain": "Chest pain", "sweating": "Heavy sweating",
    "breathing_difficulty": "Difficulty breathing", "unconscious": "Unconscious",
    "face_droop": "Face drooping", "slurred_speech": "Slurred speech",
    "arm_weakness": "Arm weakness", "frequent_urination": "Frequent urination",
    "high_glucose": "High blood glucose reading", "thirst": "Excess thirst",
    "high_bp": "High blood pressure", "fever": "Fever",
    "joint_pain": "Joint pain", "rash": "Skin rash", "cough": "Cough",
    "headache": "Headache",
}

TRIAGE_MAP = {
    "CRITICAL": ("RED", "Call an ambulance now", "#e63946"),
    "URGENT": ("ORANGE", "See a doctor today", "#f4a261"),
    "ROUTINE": ("YELLOW", "Book a routine visit", "#e9c46a"),
    None: ("GREEN", "Rest at home and monitor", "#2a9d8f"),
}


def forward_chain(facts):
    """Repeatedly fire rules whose conditions are all known until nothing new.
    Returns (all facts, list of fired rules, triage level or None)."""
    known = set(facts)
    fired = []
    changed = True
    while changed:
        changed = False
        for rid, conds, concl in RULES:
            if concl not in known and all(c in known for c in conds):
                known.add(concl)
                fired.append((rid, conds, concl))
                changed = True
    level = None
    for lv in ("CRITICAL", "URGENT", "ROUTINE"):
        if lv in known:
            level = lv
            break
    return known, fired, level


# =====================================================================
# MODULE IV (a): BAYESIAN REASONING
# =====================================================================
# P(evidence | diabetic), P(evidence | healthy), measured on the CDC data
EVIDENCE = {
    "High blood pressure":      (0.753, 0.377),
    "High cholesterol":         (0.670, 0.384),
    "BMI 30 or more (obese)":   (0.582, 0.308),
    "Age 60 or older":          (0.680, 0.450),
    "Poor general health":      (0.406, 0.134),
    "Difficulty walking":       (0.371, 0.135),
}


def bayes_update(prior, findings):
    """findings: dict name -> True (present) / False (absent).
    Returns list of (label, posterior) starting with the prior."""
    odds = prior / (1 - prior)
    steps = [("Start (prior)", prior)]
    for name, present in findings.items():
        p_d, p_h = EVIDENCE[name]
        lr = p_d / p_h if present else (1 - p_d) / (1 - p_h)
        odds *= lr
        steps.append(("%s %s" % ("+" if present else "-", name), odds / (1 + odds)))
    return steps


# =====================================================================
# MODULE IV (b): GOAL-STACK PLANNING
# =====================================================================
# action name -> (preconditions, effect)
ACTIONS = {
    "Triage patient":        ([], "triaged"),
    "Call ambulance":        (["triaged"], "ambulance_called"),
    "Route ambulance (A*)":  (["ambulance_called"], "ambulance_at_patient"),
    "Transport to hospital": (["ambulance_at_patient"], "at_hospital"),
    "Stabilize patient":     (["at_hospital"], "stabilized"),
    "Run blood tests":       (["triaged"], "tests_done"),
    "Start treatment":       (["stabilized", "tests_done"], "under_treatment"),
}
ACHIEVER = {eff: name for name, (_, eff) in ACTIONS.items()}


def goal_stack_plan(goal="under_treatment"):
    """Classic goal-stack planner. Returns (plan, trace)."""
    state, plan, trace = set(), [], []
    stack = [("goal", goal)]
    while stack:
        kind, item = stack.pop()
        if kind == "goal":
            if item in state:
                continue
            action = ACHIEVER[item]
            trace.append("need '%s' -> push action '%s' and its preconditions" % (item, action))
            stack.append(("do", action))
            for pre in reversed(ACTIONS[action][0]):
                stack.append(("goal", pre))
        else:
            plan.append(item)
            state.add(ACTIONS[item][1])
    return plan, trace


# =====================================================================
# THE AGENT: perceive -> reason -> act
# =====================================================================
def run_mission(name, age_over_50, symptoms, bayes_findings, nn_risk, base_rate,
                patient_spot, grid):
    """Run the full MediAI agent. patient_spot is either the name of a place
    in PATIENT_SPOTS or a (row, col) tuple. Returns a dict to display."""
    out = {"name": name}

    # 1. perceive + reason: triage by forward chaining
    facts = set(symptoms)
    if age_over_50:
        facts.add("age_over_50")
    known, fired, level = forward_chain(facts)
    colour_name, advice, colour = TRIAGE_MAP[level]
    out.update(facts=sorted(facts), fired=fired, level=level,
               triage=colour_name, advice=advice, colour=colour)

    # 2. reason: risk estimates
    steps = bayes_update(base_rate, bayes_findings)
    out["bayes_steps"] = steps
    out["bayes_risk"] = steps[-1][1]
    out["nn_risk"] = nn_risk
    out["base_rate"] = base_rate

    # 3. act: dispatch ambulance only for RED
    out["dispatch"] = colour_name == "RED"
    out["route"] = None
    spot_cell = PATIENT_SPOTS[patient_spot] if isinstance(patient_spot, str) else tuple(patient_spot)
    patient = nearest_free(grid, spot_cell)
    out["patient"] = patient
    if out["dispatch"]:
        p1, e1, t1 = astar(grid, BASE, patient)
        p2, e2, t2 = astar(grid, patient, HOSPITAL)
        out["route"] = {"legs": [p1, p2], "explored": e1 | e2,
                        "minutes": t1 + t2, "to_patient": t1, "to_hospital": t2}
        plan, trace = goal_stack_plan("under_treatment")
    elif colour_name == "ORANGE":
        plan, trace = ["Triage patient", "Run blood tests", "See doctor today"], []
    else:
        plan, trace = ["Triage patient", "Rest and monitor"], []
    out["plan"], out["plan_trace"] = plan, trace
    return out


def mission_figure(res, grid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    if res["dispatch"]:
        r = res["route"]
        draw_city(ax1, grid, r["legs"], r["explored"], patient=res["patient"],
                  title="A* ambulance route: %d min (blue star = patient)" % r["minutes"])
    else:
        draw_city(ax1, grid, [], None, patient=res["patient"],
                  title="No ambulance dispatched")
    labels = ["Average\nperson", "Neural\nnetwork", "Bayes"]
    vals = [res["base_rate"] * 100, res["nn_risk"] * 100, res["bayes_risk"] * 100]
    ax2.bar(labels, vals, color=["grey", "teal", "purple"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Estimated diabetes risk (%)")
    ax2.set_title("Risk estimates | Triage: %s" % res["triage"])
    for i, v in enumerate(vals):
        ax2.text(i, v + 1.5, "%.0f%%" % v, ha="center", fontweight="bold")
    fig.suptitle("MediAI: %s" % res["name"], fontweight="bold")
    fig.tight_layout()
    return fig
