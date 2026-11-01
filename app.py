import os

import joblib
import matplotlib.pyplot as plt
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
    return agent.compare_searches(get_city())


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
base_rate = bundle["base_rate"]
grid = get_city()

AGE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
       "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
HEALTH = ["1 - Excellent", "2 - Very good", "3 - Good", "4 - Fair", "5 - Poor"]
EDU = ["Never attended / kindergarten", "Grades 1-8", "Grades 9-11",
       "Grade 12 / GED", "College 1-3 years", "College 4+ years"]
INCOME = ["Below 10k", "10k-15k", "15k-20k", "20k-25k",
          "25k-35k", "35k-50k", "50k-75k", "75k or more"]


def real_risk(vals):
    """Neural network score, then Bayes correction from 50/50 training data
    back to the real-world prevalence."""
    row = pd.DataFrame([vals])[features]
    p_model = model.predict_proba(scaler.transform(row))[0, 1]
    p_real = (p_model * base_rate) / (p_model * base_rate + (1 - p_model) * (1 - base_rate))
    return p_model, p_real


def card(title, text, colour):
    st.markdown(
        "<div class='card' style='background:%s22;border-left:6px solid %s'>"
        "<h2>%s</h2><p>%s</p></div>" % (colour, colour, title, text),
        unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🩺 MediAI")
    st.write("An intelligent medical-emergency agent that combines search, logic, "
             "probabilistic reasoning, planning and learning.")
    st.markdown("**CS24308 Artificial Intelligence Lab**  \nSession MO-26")
    st.divider()
    st.markdown("**Syllabus modules used**")
    st.markdown(
        "- Module I: Intelligent agent (perceive, reason, act)\n"
        "- Module II: A*, Greedy, BFS search\n"
        "- Module III: Forward chaining rules\n"
        "- Module IV: Bayes + goal-stack planning\n"
        "- Module V: Neural network learning")
    st.divider()
    st.caption("Educational project. Not medical advice.")

st.title("🩺 MediAI: Intelligent Medical Emergency Agent")
st.caption("CS24308 Artificial Intelligence Lab | Educational project, not medical advice")

tab_agent, tab1, tab_route, tab_bayes, tab2, tab3 = st.tabs([
    "🚑 Agent mission", "Risk checker", "🗺️ Ambulance routing",
    "🧮 Bayes reasoning", "Model performance", "About the team"])

# ================================================================ AGENT TAB
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
        col.button(pname, key="btn_" + pname, on_click=load_preset, args=(pname,),
                   use_container_width=True)

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
                         list(agent.PATIENT_SPOTS), key="m_spot")
            st.caption("Green dot = ambulance base, red + = hospital, "
                       "blue star = patient. Orange = traffic jam.")

    if st.button("🚑 Run MediAI agent", type="primary", use_container_width=True):
        age_idx = AGE.index(st.session_state.m_age)
        gen_idx = HEALTH.index(st.session_state.m_gen) + 1
        bmi_v = st.session_state.m_bmi

        # perceive: build the 21 features the neural network needs
        vals = {
            "HighBP": int(st.session_state.m_highbp), "HighChol": int(st.session_state.m_highchol),
            "CholCheck": 1, "BMI": bmi_v, "Smoker": 0, "Stroke": 0,
            "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
            "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": gen_idx, "MentHlth": 0, "PhysHlth": 0,
            "DiffWalk": int(st.session_state.m_diffwalk),
            "Sex": 1 if st.session_state.m_sex == "Male" else 0,
            "Age": age_idx + 1, "Education": 4, "Income": 5,
        }
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
        st.session_state["mission"] = agent.run_mission(
            st.session_state.m_name, age_idx >= 6, syms, findings, p_real, base_rate,
            st.session_state.m_spot, grid)

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

        st.download_button("⬇️ Download mission report", agent.mission_report(res),
                           file_name="mediai_mission_report.txt")

# ================================================================ TAB: risk checker
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.subheader("Basic details")
            age = st.selectbox("Age group", AGE, index=6)
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            height = st.number_input("Height (cm)", 120, 220, 168)
            weight = st.number_input("Weight (kg)", 30, 250, 70)
            bmi = weight / ((height / 100) ** 2)
            st.write("Calculated BMI: **%.1f**" % bmi)
            genhlth = st.selectbox("General health", HEALTH, index=2)
            edu = st.selectbox("Education level", EDU, index=3)
            income = st.selectbox("Yearly income band (US dollars, as in the survey)",
                                  INCOME, index=4)

    with c2:
        with st.container(border=True):
            st.subheader("Medical history")
            highbp = st.checkbox("High blood pressure")
            highchol = st.checkbox("High cholesterol")
            cholcheck = st.checkbox("Cholesterol checked in last 5 years", value=True)
            stroke = st.checkbox("Ever had a stroke")
            heart = st.checkbox("Heart disease or heart attack")
            diffwalk = st.checkbox("Serious difficulty walking or climbing stairs")
            ment = st.slider("Days of poor mental health (last 30)", 0, 30, 0)
            phys = st.slider("Days of poor physical health (last 30)", 0, 30, 0)

    with c3:
        with st.container(border=True):
            st.subheader("Lifestyle and access")
            smoker = st.checkbox("Smoked 100+ cigarettes in lifetime")
            physact = st.checkbox("Physical activity in last 30 days", value=True)
            fruits = st.checkbox("Eats fruit daily", value=True)
            veggies = st.checkbox("Eats vegetables daily", value=True)
            alcohol = st.checkbox("Heavy alcohol consumption")
            healthcare = st.checkbox("Has health coverage", value=True)
            nodoc = st.checkbox("Could not see a doctor due to cost")

    vals = {
        "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
        "BMI": bmi, "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart), "PhysActivity": int(physact),
        "Fruits": int(fruits), "Veggies": int(veggies),
        "HvyAlcoholConsump": int(alcohol), "AnyHealthcare": int(healthcare),
        "NoDocbcCost": int(nodoc), "GenHlth": HEALTH.index(genhlth) + 1,
        "MentHlth": ment, "PhysHlth": phys, "DiffWalk": int(diffwalk),
        "Sex": 1 if sex == "Male" else 0, "Age": AGE.index(age) + 1,
        "Education": EDU.index(edu) + 1, "Income": INCOME.index(income) + 1,
    }

    if st.button("Check my risk", type="primary"):
        missing = [f for f in features if f not in vals]
        if missing:
            st.error("Feature names do not match the model: " + ", ".join(missing))
            st.stop()

        p_model, p_real = real_risk(vals)

        if p_real < base_rate:
            card("%.0f%% estimated risk" % (p_real * 100),
                 "Below the population average. The model would not flag this person.",
                 "#2A9D8F")
        elif p_real < 0.30:
            card("%.0f%% estimated risk" % (p_real * 100),
                 "Above average. A blood sugar test (fasting glucose or HbA1c) is advisable.",
                 "#F4A261")
        else:
            card("%.0f%% estimated risk" % (p_real * 100),
                 "High risk. Please consult a doctor and get tested soon.", "#E63946")

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw model score", "%.0f%%" % (p_model * 100))
        m2.metric("Estimated real-world risk", "%.0f%%" % (p_real * 100))
        m3.metric("Population average", "%.1f%%" % (base_rate * 100))

        # what-if: healthier weight
        if bmi > 25:
            alt = dict(vals)
            alt["BMI"] = 24.9
            _, p_alt = real_risk(alt)
            st.metric("What if BMI were 24.9 (healthy range)?", "%.0f%%" % (p_alt * 100),
                      "%+.1f pts" % ((p_alt - p_real) * 100), delta_color="inverse")

        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.barh(["Average person", "This patient"], [base_rate * 100, p_real * 100],
                color=["grey", "crimson" if p_real >= 0.30 else "teal"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Estimated diabetes risk (%)")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("How is this calculated?"):
        st.write(
            "A neural network (2 hidden layers: 64 and 32 neurons) was trained on 253,680 "
            "real survey records from the CDC BRFSS 2015 survey. It was trained on a balanced "
            "50/50 sample, so its raw score overestimates real risk. We apply Bayes' theorem "
            "with the real prior (%.1f%% of people are diabetic) to correct it. "
            "This is a screening aid for learning purposes, not a diagnosis." % (base_rate * 100))

# ================================================================ TAB: routing
with tab_route:
    st.subheader("Ambulance routing: which search algorithm is best?")
    st.write("The city is a 20x20 grid. Black cells are blocked roads, orange cells are traffic "
             "(3 minutes per cell instead of 1). The ambulance leaves the green base and must "
             "reach the red hospital. Three search algorithms solve the same problem:")
    fig, results = get_comparison()
    a, b, c = st.columns(3)
    a.metric("BFS (uninformed)", "%d min" % results["BFS"]["minutes"],
             "%d cells explored" % results["BFS"]["explored"], delta_color="off")
    b.metric("Greedy best-first", "%d min" % results["Greedy"]["minutes"],
             "%d cells explored" % results["Greedy"]["explored"], delta_color="off")
    c.metric("A* (best)", "%d min" % results["A*"]["minutes"],
             "%d cells explored" % results["A*"]["explored"], delta_color="off")
    st.pyplot(fig)
    st.info(
        "**BFS** ignores traffic, so it is slow on the road and explores almost everything. "
        "**Greedy** searches very little but can pick a slower road because it only looks at the "
        "distance left. **A\\*** adds the time already spent to the distance left "
        "(f = g + h), so it finds the fastest route while exploring far fewer cells than BFS. "
        "This is why the agent uses A* to dispatch the ambulance.")

# ================================================================ TAB: Bayes
with tab_bayes:
    st.subheader("Bayesian reasoning: belief updates as findings arrive")
    st.write("Start from the population rate (%.1f%% diabetic). Choose what you know about a "
             "patient. Each finding multiplies the odds by its likelihood ratio "
             "P(finding | diabetic) / P(finding | healthy)." % (base_rate * 100))
    cols = st.columns(3)
    findings = {}
    for i, name in enumerate(agent.EVIDENCE):
        with cols[i % 3]:
            choice = st.radio(name, ["Unknown", "Present", "Absent"], horizontal=True,
                              key="bayes_" + name)
        if choice != "Unknown":
            findings[name] = choice == "Present"

    steps = agent.bayes_update(base_rate, findings)
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(steps)))
    ys = [p * 100 for _, p in steps]
    ax.plot(xs, ys, "o-", color="crimson", lw=2.5)
    for x, y in zip(xs, ys):
        ax.annotate("%.0f%%" % y, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontweight="bold")
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
    with st.expander("Likelihood table (measured on the CDC data)"):
        st.table(pd.DataFrame(
            [(k, v[0], v[1], round(v[0] / v[1], 2)) for k, v in agent.EVIDENCE.items()],
            columns=["Finding", "P(E | diabetic)", "P(E | healthy)", "Likelihood ratio"]))

# ================================================================ TAB: performance
with tab2:
    st.subheader("Results on 50,736 unseen patients")
    a, b, c, d = st.columns(4)
    a.metric("AUC", "0.824")
    b.metric("Accuracy", "70.8%")
    c.metric("Recall (diabetic)", "81%")
    d.metric("Precision (diabetic)", "30%")
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

# ================================================================ TAB: team
with tab3:
    st.subheader("Project team")
    st.table(pd.DataFrame({
        "Name": ["Abhijeet Gorai", "Nilesh Kumar", "Yash Raj"],
        "Roll number": ["BTECH/10024/24", "BTECH/10054/24", "BTECH/10047/24"],
    }))
    st.write("**Program:** B.Tech CSE")
    st.write("**Subject:** CS24308 Artificial Intelligence Lab | **Session:** MO-26")
    st.write("**Submitted to:** Dr. Satish Chander")
    st.write("**Dataset:** CDC Diabetes Health Indicators (UCI repository, 253,680 patients)")
