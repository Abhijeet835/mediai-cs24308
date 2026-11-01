import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MediAI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("mediai_model.joblib")

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
base_rate = bundle["base_rate"]

AGE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
       "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
HEALTH = ["1 - Excellent", "2 - Very good", "3 - Good", "4 - Fair", "5 - Poor"]
EDU = ["Never attended / kindergarten", "Grades 1-8", "Grades 9-11",
       "Grade 12 / GED", "College 1-3 years", "College 4+ years"]
INCOME = ["Below 10k", "10k-15k", "15k-20k", "20k-25k",
          "25k-35k", "35k-50k", "50k-75k", "75k or more"]

st.title("🩺 MediAI: Diabetes Risk Screening Assistant")
st.caption("CS24308 Artificial Intelligence Lab | Educational project, not medical advice")

tab1, tab2, tab3 = st.tabs(["Risk checker", "Model performance", "About the team"])

# ---------------- TAB 1: risk checker ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Basic details")
        age = st.selectbox("Age group", AGE, index=6)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        height = st.number_input("Height (cm)", 120, 220, 168)
        weight = st.number_input("Weight (kg)", 30, 250, 70)
        bmi = weight / ((height / 100) ** 2)
        st.write("Calculated BMI: **%.1f**" % bmi)
        genhlth = st.selectbox("General health", HEALTH, index=2)
        edu = st.selectbox("Education level", EDU, index=3)
        income = st.selectbox("Yearly income band (US dollars, as in the survey)", INCOME, index=4)
    with c2:
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

        row = pd.DataFrame([vals])[features]
        p_model = model.predict_proba(scaler.transform(row))[0, 1]
        # Bayes correction: the model was trained on 50/50 data, reality is ~14% diabetic
        p_real = (p_model * base_rate) / (p_model * base_rate + (1 - p_model) * (1 - base_rate))

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw model score", "%.0f%%" % (p_model * 100))
        m2.metric("Estimated real-world risk", "%.0f%%" % (p_real * 100))
        m3.metric("Population average", "%.1f%%" % (base_rate * 100))

        if p_real < base_rate:
            st.success("Below the population average. The model would not flag this person.")
        elif p_real < 0.30:
            st.warning("Above average. A blood sugar test (fasting glucose or HbA1c) is advisable.")
        else:
            st.error("High risk. Please consult a doctor and get tested soon.")

        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.barh(["Average person", "This patient"], [base_rate * 100, p_real * 100],
                color=["grey", "crimson" if p_real >= 0.30 else "teal"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Estimated diabetes risk (%)")
        st.pyplot(fig)

        with st.expander("How is this calculated?"):
            st.write(
                "A neural network (2 hidden layers: 64 and 32 neurons) was trained on 253,680 "
                "real survey records from the CDC BRFSS 2015 survey. It was trained on a balanced "
                "50/50 sample, so its raw score overestimates real risk. We apply Bayes' theorem "
                "with the real prior (%.1f%% of people are diabetic) to correct it. "
                "This is a screening aid for learning purposes, not a diagnosis." % (base_rate * 100))

# ---------------- TAB 2: performance ----------------
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

# ---------------- TAB 3: team ----------------
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
