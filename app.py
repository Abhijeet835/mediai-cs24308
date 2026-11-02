import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import agent

st.set_page_config(page_title="MediAI", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #141C2E; border: 1px solid #243049;
        border-radius: 12px; padding: 12px 16px;
    }
    .block-container { padding-top: 2.5rem; }
    .card { padding: 18px 22px; border-radius: 12px; margin: 8px 0 14px 0; }
    .card h2 { margin: 0; }
    .card p { margin: 2px 0 0 0; opacity: 0.9; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    return joblib.load("mediai_model.joblib")


@st.cache_resource
def get_city():
    return agent.make_city()


@st.cache_resource
def get_comparison():
    """Cache the finished PNG image (not the figure) so threads never share a figure."""
    fig, results = agent.compare_searches(get_city())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), results


@st.cache_resource
def get_location_analysis():
    """Run the three search algorithms on every patient location (cached)."""
    df = agent.analyse_locations(get_city())
    images = []
    for f in (agent.locations_map(get_city()), agent.locations_charts(df)):
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(f)
        images.append(buf.getvalue())
    return images[0], images[1], df


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
base_rate = bundle["base_rate"]
grid = get_city()

AGE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
       "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
HEALTH = ["1 - Excellent", "2 - Very good", "3 - Good", "4 - Fair", "5 - Poor"]
CUSTOM = "Custom (choose row and column)"


def risk_many(rows):
    """Neural-network risk for many patients at once, corrected from the 50/50
    training sample back to the real-world rate with Bayes' theorem."""
    X = pd.DataFrame(rows)[features]
    p = model.predict_proba(scaler.transform(X))[:, 1]
    return (p * base_rate) / (p * base_rate + (1 - p) * (1 - base_rate))


def real_risk(vals):
    """Raw neural-network score and the corrected real-world risk for one patient."""
    row = pd.DataFrame([vals])[features]
    p_model = model.predict_proba(scaler.transform(row))[0, 1]
    p_real = (p_model * base_rate) / (p_model * base_rate + (1 - p_model) * (1 - base_rate))
    return p_model, p_real


def base_vals(**kw):
    """A default patient: every answer set to a neutral value, then overridden."""
    v = {"HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 25, "Smoker": 0, "Stroke": 0,
         "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
         "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
         "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 8,
         "Education": 4, "Income": 5}
    v.update(kw)
    return v


def risk_sweeps(age_idx, bmi, gen_idx, highbp):
    """Three charts showing how the neural network's risk responds to its inputs."""
    ref = base_vals(Age=age_idx + 1, BMI=bmi, GenHlth=gen_idx, HighBP=int(highbp))
    bmis = list(range(15, 51))
    factors = {"High blood pressure": {"HighBP": 1}, "High cholesterol": {"HighChol": 1},
               "Difficulty walking": {"DiffWalk": 1}, "Heart disease": {"HeartDiseaseorAttack": 1},
               "Stroke": {"Stroke": 1}, "Smoker": {"Smoker": 1},
               "No physical activity": {"PhysActivity": 0}}
    rows = ([{**ref, "BMI": b} for b in bmis]
            + [{**ref, "Age": i + 1} for i in range(len(AGE))]
            + [{**ref, **v} for v in factors.values()]
            + [ref])
    risk = risk_many(rows) * 100            # one prediction call for all scenarios
    r_bmi = risk[:len(bmis)]
    r_age = risk[len(bmis):len(bmis) + len(AGE)]
    r_fac = risk[len(bmis) + len(AGE):-1]
    here = risk[-1]
    delta = dict(zip(factors, r_fac - here))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a1.plot(bmis, r_bmi, color="crimson", lw=2.5)
    a1.axhline(base_rate * 100, color="grey", ls="--", label="Average person")
    a1.axvline(bmi, color="teal", ls=":", label="This patient")
    a1.set_xlabel("BMI")
    a1.set_ylabel("Estimated diabetes risk (%)")
    a1.set_title("Risk vs BMI")
    a1.legend(fontsize=8)

    a2.bar(range(len(AGE)), r_age,
           color=["teal" if i == age_idx else "#9bb" for i in range(len(AGE))])
    a2.axhline(base_rate * 100, color="grey", ls="--")
    a2.set_xticks(range(len(AGE)))
    a2.set_xticklabels(AGE, rotation=60, ha="right", fontsize=8)
    a2.set_ylabel("Estimated diabetes risk (%)")
    a2.set_title("Risk vs age group")

    order = sorted(delta, key=delta.get)
    a3.barh(order, [delta[k] for k in order],
            color=["crimson" if delta[k] > 0 else "teal" for k in order])
    a3.axvline(0, color="black", lw=0.8)
    a3.set_xlabel("Change in risk (percentage points)")
    a3.set_title("What the model reacts to (same patient)")
    fig.tight_layout()
    return fig


def card(title, text, colour):
    st.markdown(
        "<div class='card' style='background:%s22;border-left:6px solid %s'>"
        "<h2>%s</h2><p>%s</p></div>" % (colour, colour, title, text),
        unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("MediAI")
    st.markdown("**CS24308 Artificial Intelligence Lab**  \nSession MO-26")
    st.markdown("**Modules used**")
    st.markdown("- I: Intelligent agent\n"
                "- II: A*, Greedy, BFS search\n"
                "- III: Forward-chaining rules\n"
                "- IV: Bayes, goal-stack planning\n"
                "- V: Neural network")
    st.divider()
    st.caption("Abhijeet Gorai, Nilesh Kumar, Yash Raj  \n"
               "Guide: Dr. Satish Chander  \n"
               "Educational project, not medical advice.")

st.title("MediAI: Intelligent Medical Emergency Agent")
st.caption("CS24308 Artificial Intelligence Lab | Educational project, not medical advice")

tab_agent, tab_route, tab_bayes, tab_nn = st.tabs([
    "Agent mission", "Search: ambulance routing", "Bayes reasoning", "Neural network"])

# ================================================================ TAB: agent mission
with tab_agent:
    st.subheader("Run the MediAI agent on a patient")
    st.write("The agent **perceives** the symptoms, **reasons** (rules, neural network, Bayes), "
             "then **acts**: it dispatches an ambulance along an A* route and builds a "
             "rescue plan if the patient is RED.")

    PRESETS = {
        "Patient A (58 yr man, chest pain)": dict(
            m_name="Patient A, 58-year-old man", m_age="55-59", m_sex="Male", m_bmi=31,
            m_gen="4 - Fair", m_highbp=True, m_highchol=False, m_diffwalk=False,
            m_sym=["chest_pain", "sweating"]),
        "Patient B (35 yr woman, diabetes signs)": dict(
            m_name="Patient B, 35-year-old woman", m_age="35-39", m_sex="Female", m_bmi=33,
            m_gen="3 - Good", m_highbp=True, m_highchol=False, m_diffwalk=False,
            m_sym=["frequent_urination", "high_glucose", "thirst"]),
        "Patient C (22 yr student, fever)": dict(
            m_name="Patient C, 22-year-old student", m_age="18-24", m_sex="Male", m_bmi=22,
            m_gen="2 - Very good", m_highbp=False, m_highchol=False, m_diffwalk=False,
            m_sym=["cough", "fever", "joint_pain", "rash"]),
        "Patient D (30 yr man, headache)": dict(
            m_name="Patient D, 30-year-old man", m_age="30-34", m_sex="Male", m_bmi=23,
            m_gen="2 - Very good", m_highbp=False, m_highchol=False, m_diffwalk=False,
            m_sym=["headache"]),
    }

    def load_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[k] = v

    # defaults for first run
    for k, v in PRESETS["Patient A (58 yr man, chest pain)"].items():
        st.session_state.setdefault(k, v)

    st.write("**Quick demo patients** (click one to fill the form):")
    pcols = st.columns(4)
    for col, pname in zip(pcols, PRESETS):
        col.button(pname, key="btn_" + pname, on_click=load_preset, args=(pname,))

    f1, f2, f3 = st.columns(3)
    with f1:
        with st.container(border=True):
            st.markdown("**Patient**")
            st.text_input("Name / label", key="m_name")
            st.selectbox("Age group", AGE, key="m_age")
            st.radio("Sex", ["Female", "Male"], horizontal=True, key="m_sex")
            st.slider("BMI", 15, 50, key="m_bmi")
            st.selectbox("General health", HEALTH, key="m_gen")
    with f2:
        with st.container(border=True):
            st.markdown("**Symptoms and history**")
            sym_options = [s for s in agent.SYMPTOMS if s != "high_bp"]
            st.multiselect("Symptoms", sym_options, key="m_sym",
                           format_func=lambda s: agent.SYMPTOMS[s])
            st.checkbox("High blood pressure", key="m_highbp")
            st.checkbox("High cholesterol", key="m_highchol")
            st.checkbox("Serious difficulty walking", key="m_diffwalk")
    with f3:
        with st.container(border=True):
            st.markdown("**Where is the patient?**")
            st.selectbox("Patient location on the city map",
                         list(agent.PATIENT_SPOTS) + [CUSTOM], key="m_spot")
            if st.session_state.m_spot == CUSTOM:
                st.slider("Row (0 = top)", 0, 19, 10, key="m_row")
                st.slider("Column (0 = left)", 0, 19, 10, key="m_col")
            st.caption("Green dot = ambulance base, red + = hospital, "
                       "blue star = patient. Orange = traffic jam.")

    st.caption("Only the fields above are asked. The other 14 health-form answers "
               "(smoking, exercise, diet and so on) use fixed default values.")

    if st.button("Run MediAI agent", type="primary"):
        age_idx = AGE.index(st.session_state.m_age)
        gen_idx = HEALTH.index(st.session_state.m_gen) + 1
        bmi_v = st.session_state.m_bmi

        # perceive: build the 21 features the neural network needs
        vals = base_vals(
            HighBP=int(st.session_state.m_highbp), HighChol=int(st.session_state.m_highchol),
            BMI=bmi_v, GenHlth=gen_idx, DiffWalk=int(st.session_state.m_diffwalk),
            Sex=1 if st.session_state.m_sex == "Male" else 0, Age=age_idx + 1)
        _, p_real = real_risk(vals)

        syms = list(st.session_state.m_sym)
        if st.session_state.m_highbp:
            syms.append("high_bp")
        findings = {
            "High blood pressure": bool(st.session_state.m_highbp),
            "High cholesterol": bool(st.session_state.m_highchol),
            "BMI 30 or more (obese)": bmi_v >= 30,
            "Age 60 or older": age_idx >= 8,
            "Poor general health": gen_idx >= 4,
            "Difficulty walking": bool(st.session_state.m_diffwalk),
        }
        if st.session_state.m_spot == CUSTOM:
            spot = (st.session_state.m_row, st.session_state.m_col)
        else:
            spot = st.session_state.m_spot
        st.session_state["mission"] = agent.run_mission(
            st.session_state.m_name, age_idx >= 6, syms, findings, p_real, base_rate,
            spot, grid)

    res = st.session_state.get("mission")
    if res:
        st.divider()
        card("Triage: %s" % res["triage"], res["advice"], res["colour"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Triage", res["triage"])
        m2.metric("Neural network risk", "%.0f%%" % (res["nn_risk"] * 100),
                  "%+.0f pts vs average" % ((res["nn_risk"] - res["base_rate"]) * 100),
                  delta_color="inverse")
        m3.metric("Bayes risk", "%.0f%%" % (res["bayes_risk"] * 100),
                  "%+.0f pts vs average" % ((res["bayes_risk"] - res["base_rate"]) * 100),
                  delta_color="inverse")
        if res["dispatch"]:
            m4.metric("Ambulance ETA", "%d min" % res["route"]["minutes"])
        else:
            m4.metric("Ambulance", "Not needed")

        fig = agent.mission_figure(res, grid)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rules that fired (forward chaining)**")
            if res["fired"]:
                for rid, conds, concl in res["fired"]:
                    st.code("%s: IF %s THEN %s" % (rid, " AND ".join(conds), concl),
                            language=None)
            else:
                st.info("No rule fired, so no warning sign was found.")
        with c2:
            st.markdown("**Rescue plan (goal-stack planning)**")
            for i, step in enumerate(res["plan"], 1):
                st.write("%d. %s" % (i, step))
            if res["dispatch"]:
                r = res["route"]
                st.caption("Route: %d min to the patient + %d min to the hospital."
                           % (r["to_patient"], r["to_hospital"]))

        with st.expander("Goal-stack trace (how the plan was built)"):
            if res["plan_trace"]:
                st.code("\n".join(res["plan_trace"]), language=None)
            else:
                st.write("No ambulance plan was needed for this patient.")

# ================================================================ TAB: routing
with tab_route:
    st.subheader("Ambulance routing: which search algorithm is best?")
    st.write("The city is a 20x20 grid. Black cells are blocked roads, orange cells are traffic "
             "(3 minutes per cell instead of 1). The ambulance leaves the green base and must "
             "reach the red hospital. Three search algorithms solve the same problem:")
    png, results = get_comparison()
    a, b, c = st.columns(3)
    a.metric("BFS (uninformed)", "%d min" % results["BFS"]["minutes"],
             "%d cells explored" % results["BFS"]["explored"], delta_color="off")
    b.metric("Greedy best-first", "%d min" % results["Greedy"]["minutes"],
             "%d cells explored" % results["Greedy"]["explored"], delta_color="off")
    c.metric("A* (best)", "%d min" % results["A*"]["minutes"],
             "%d cells explored" % results["A*"]["explored"], delta_color="off")
    st.image(png)
    st.info(
        "**BFS** ignores traffic, so it is slow on the road and explores almost everything. "
        "**Greedy** searches very little but can pick a slower road because it only looks at the "
        "distance left. **A\\*** adds the time already spent to the distance left "
        "(f = g + h), so it finds the fastest route while exploring far fewer cells than BFS. "
        "This is why the agent uses A* to dispatch the ambulance.")

    st.divider()
    st.subheader("Every patient location")
    st.write("The same three algorithms, run from the ambulance base to each of the "
             "possible patient locations.")
    png_map, png_charts, loc_df = get_location_analysis()
    best = int((loc_df["A* min"] <= loc_df[["BFS min", "Greedy min"]].min(axis=1)).sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Locations tested", len(loc_df))
    q2.metric("A* fastest or tied", "%d of %d" % (best, len(loc_df)))
    q3.metric("Longest total trip", "%d min" % loc_df["Total ETA (A*)"].max())
    left, right = st.columns([1, 1.2])
    left.image(png_map)
    right.dataframe(loc_df[["No.", "Location", "BFS min", "Greedy min", "A* min",
                            "Total ETA (A*)"]], hide_index=True)
    st.image(png_charts)

# ================================================================ TAB: Bayes
with tab_bayes:
    st.subheader("Bayesian reasoning: belief updates as findings arrive")
    st.write("Start from the population rate (%.1f%% diabetic). Choose what you know about a "
             "patient. Each finding multiplies the odds by its likelihood ratio "
             "P(finding | diabetic) / P(finding | healthy)." % (base_rate * 100))
    DEFAULT_BAYES = {"High blood pressure": "Present", "BMI 30 or more (obese)": "Present",
                     "Age 60 or older": "Present", "Poor general health": "Present",
                     "Difficulty walking": "Absent"}
    OPTIONS = ["Unknown", "Present", "Absent"]
    cols = st.columns(3)
    findings = {}
    for i, name in enumerate(agent.EVIDENCE):
        with cols[i % 3]:
            choice = st.radio(name, OPTIONS, horizontal=True, key="bayes_" + name,
                              index=OPTIONS.index(DEFAULT_BAYES.get(name, "Unknown")))
        if choice != "Unknown":
            findings[name] = choice == "Present"

    if not findings:
        st.info("Choose Present or Absent for at least one finding to see the belief change.")
    steps = agent.bayes_update(base_rate, findings)
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(steps)))
    ys = [p * 100 for _, p in steps]
    ax.plot(xs, ys, "o-", color="crimson", lw=2.5)
    for x, y in zip(xs, ys):
        ax.annotate("%.0f%%" % y, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontweight="bold")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0].replace(" (", "\n(").replace(" ", "\n", 1) if len(s[0]) > 14
                        else s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Probability of diabetes (%)")
    ax.set_ylim(0, max(100, max(ys) + 10))
    ax.grid(alpha=0.3)
    ax.set_title("Bayesian reasoning: belief updates as each finding arrives")
    st.pyplot(fig)
    plt.close(fig)
    st.metric("Final probability", "%.1f%%" % (steps[-1][1] * 100),
              "%+.1f pts vs population" % ((steps[-1][1] - base_rate) * 100),
              delta_color="inverse")

    # how strongly does each finding move the estimate?
    names = list(agent.EVIDENCE)
    yy = np.arange(len(names))
    lr_p = [agent.EVIDENCE[n][0] / agent.EVIDENCE[n][1] for n in names]
    lr_a = [(1 - agent.EVIDENCE[n][0]) / (1 - agent.EVIDENCE[n][1]) for n in names]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(yy + 0.2, lr_p, 0.4, color="crimson", label="Finding present (raises odds)")
    ax.barh(yy - 0.2, lr_a, 0.4, color="teal", label="Finding absent (lowers odds)")
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.set_xlabel("Likelihood ratio (1 = no effect)")
    ax.set_title("How strongly each finding moves the Bayes estimate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Likelihood table (measured on the CDC data)"):
        st.table(pd.DataFrame(
            [(k, v[0], v[1], round(v[0] / v[1], 2)) for k, v in agent.EVIDENCE.items()],
            columns=["Finding", "P(E | diabetic)", "P(E | healthy)", "Likelihood ratio"]))

# ================================================================ TAB: neural network
with tab_nn:
    st.subheader("Neural network: results on 50,736 unseen patients")
    a, b, c, d = st.columns(4)
    a.metric("AUC", "0.824")
    b.metric("Accuracy", "70.8%")
    c.metric("Recall (diabetic)", "81%")
    d.metric("Precision (diabetic)", "30%")
    st.caption("Network: 21 inputs, then 64 neurons, then 32 neurons, then 1 output. "
               "Trained on a balanced sample; early stopping used.")
    st.write("Of 7,069 diabetic patients in the test set, the model caught **5,712** and missed **1,357**. "
             "For a screening tool, missing fewer sick patients matters more than avoiding false alarms.")
    cm = pd.DataFrame([[30211, 13456], [1357, 5712]],
                      index=["Actually healthy", "Actually diabetic"],
                      columns=["Predicted healthy", "Predicted diabetic"])
    st.table(cm)
    # optional charts: upload these PNGs from Colab to the repo to show them here
    for fname, caption in [("roc_curve.png", "ROC curve and confusion matrix"),
                           ("feature_importance.png", "Which health factors drive the prediction"),
                           ("overfitting.png", "Underfitting, good fit and overfitting")]:
        if os.path.exists(fname):
            st.image(fname, caption=caption)

    st.divider()
    st.subheader("How the neural network's risk changes")
    k1, k2, k3, k4 = st.columns(4)
    sw_age = k1.selectbox("Age group", AGE, index=7, key="sw_age")
    sw_bmi = k2.slider("BMI", 15, 50, 30, key="sw_bmi")
    sw_gen = k3.selectbox("General health", HEALTH, index=2, key="sw_gen")
    sw_bp = k4.checkbox("High blood pressure", key="sw_bp")
    fig = risk_sweeps(AGE.index(sw_age), sw_bmi, HEALTH.index(sw_gen) + 1, sw_bp)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("All other answers use default values. These are patterns in 2015 US survey "
               "data, not proof of cause.")
