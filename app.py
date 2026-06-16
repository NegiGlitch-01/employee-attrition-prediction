import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

#

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

# Header

st.markdown("""
<div style="
background: linear-gradient(135deg,#0F172A,#2563EB);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
">
<h1>🏢 Employee Attrition Intelligence Platform</h1>
<h4>AI Powered HR Analytics System</h4>
<p>Predict Employee Turnover Risk Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Sidebar Inputs

st.sidebar.header("👨‍💼 Employee Information")

age = st.sidebar.number_input("Age",18,65,30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

gender_value = 1 if gender == "Male" else 0

marital_option = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

marital_mapping = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

marital_status = marital_mapping[marital_option]

department_name = st.sidebar.selectbox(
    "Department",
    ["HR","IT","Sales","Finance","Marketing"]
)

department_map = {
    "HR":0,
    "IT":1,
    "Sales":2,
    "Finance":3,
    "Marketing":4
}

department = department_map[department_name]


job_role_name = st.sidebar.selectbox(
    "Job Role",
    [
        "Manager" , 
        "Executive" , 
        "Analyst" , 
        "Assistant"
    ]
)

job_role_map = {
    "Manager" : 0, 
    "Executive" : 1, 
    "Analyst" : 2, 
    "Assistant" : 3
}

job_role = job_role_map[job_role_name]

job_level = st.sidebar.slider(
    "Job Level",
    1,5,1
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    value=10000
)

hourly_rate = st.sidebar.number_input(
    "Hourly Rate",
    min_value=1,
    value=30
)

years_at_company = st.sidebar.number_input(
    "Years At Company",
    min_value=0,
    value=5
)

years_in_current_role = st.sidebar.number_input(
    "Years In Current Role",
    min_value=0,
    value=3
)

years_since_last_promotion = st.sidebar.number_input(
    "Years Since Last Promotion",
    min_value=0,
    value=1
)

work_life_balance = st.sidebar.slider(
    "Work Life Balance",
    1,4,3
)

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,4,3
)

performance_rating = st.sidebar.slider(
    "Performance Rating",
    1,4,3
)

training_hours = st.sidebar.number_input(
    "Training Hours",
    min_value=0,
    value=20
)

overtime = st.sidebar.selectbox(
    "Overtime",
    ["No","Yes"]
)

overtime = 1 if overtime == "Yes" else 0

project_count = st.sidebar.number_input(
    "Project Count",
    min_value=0,
    value=5
)

average_hours = st.sidebar.number_input(
    "Average Hours Worked",
    min_value=20,
    value=40
)

absenteeism = st.sidebar.number_input(
    "Absenteeism",
    min_value=0,
    value=2
)

work_environment = st.sidebar.slider(
    "Work Environment",
    1,4,3
)

relationship_manager = st.sidebar.slider(
    "Relationship With Manager",
    1,4,3
)

job_involvement = st.sidebar.slider(
    "Job Involvement",
    1,4,3
)

distance_from_home = st.sidebar.number_input(
    "Distance From Home",
    min_value=0,
    value=10
)

companies_worked = st.sidebar.number_input(
    "Companies Worked",
    min_value=0,
    value=2
)

# Dashboard Metrics

c1,c2,c3,c4 = st.columns(4)

c1.metric("👤 Age", age)
c2.metric("💰 Income", f"₹{monthly_income:,}")
c3.metric("📂 Projects", project_count)
c4.metric("🏢 Experience", years_at_company)

st.divider()

# Predict Button

predict = st.button(
    "🚀 Predict Attrition",
    use_container_width=True
)


# Prediction Logic

if predict:

    with st.spinner("🤖 Running AI Analysis..."):
        time.sleep(2)

        st.toast(
            "🤖 AI Analysis Completed Successfully!",
            icon="✅"
)

    features = np.array([[
        age,
        gender_value,
        marital_status,
        department,
        job_role,
        job_level,
        monthly_income,
        hourly_rate,
        years_at_company,
        years_in_current_role,
        years_since_last_promotion,
        work_life_balance,
        job_satisfaction,
        performance_rating,
        training_hours,
        overtime,
        project_count,
        average_hours,
        absenteeism,
        work_environment,
        relationship_manager,
        job_involvement,
        distance_from_home,
        companies_worked
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]


    feature_names = [
        "Age",
        "gender_value",
        "Marital Status",
        "Department",
        "Job Role",
        "Job Level",
        "Monthly Income",
        "Hourly Rate",
        "Years At Company",
        "Years In Current Role",
        "Promotion Gap",
        "Work Life Balance",
        "Job Satisfaction",
        "Performance",
        "Training Hours",
        "Overtime",
        "Projects",
        "Working Hours",
        "Absenteeism",
        "Work Environment",
        "Manager Relation",
        "Job Involvement",
        "Distance From Home",
        "Companies Worked"
    ]

# Probability Graph

    st.subheader("📊 Attrition Probability Analysis")

    graph_data = pd.DataFrame({
        "Category": ["Stay", "Attrition"],
        "Probability": [1 - probability, probability]
    })

    fig, ax = plt.subplots(figsize=(4,2.5))

    ax.bar(
        graph_data["Category"],
        graph_data["Probability"]
)

    st.pyplot(fig)

    ax.set_ylabel("Probability")
    ax.set_title("Employee Attrition Prediction")

    st.pyplot(fig)

# Result

    if prediction[0] == 1:

        st.error("🚨 High Attrition Risk Employee")
        st.snow()

    else:

        st.success("✅ Employee Likely To Stay")
        st.balloons()

    st.subheader("🎯 Prediction Result")

    st.success(
    f"Model Confidence : {(max(probability,1-probability))*100:.2f}%"
)
    st.write("Prediction:", prediction[0])
    st.write("Probability:", probability)

# Risk Score

    st.subheader("📈 Attrition Risk Score")

    st.metric(
        "Risk Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("📈 Employee Metrics")

    chart_data = pd.DataFrame({
        "Feature": [
            "Job Satisfaction",
            "Work Life Balance",
            "Performance",
            "Job Involvement",
            "Work Environment"
        ],
        "Score": [
            job_satisfaction,
            work_life_balance,
            performance_rating,
            job_involvement,
            work_environment
        ]
})

    fig, ax = plt.subplots(figsize=(2,1.5))

    ax.bar(
        chart_data["Feature"],
        chart_data["Score"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("🥧 Attrition Distribution")

    fig2, ax2 = plt.subplots(figsize=(3.5,3.5))

    ax2.pie(
        [1-probability, probability],
        labels=["Stay", "Attrition"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)
        
    

# Risk Level

    if probability < 0.30:
        st.success("🟢 LOW RISK")

    elif probability < 0.60:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")

    # Employee Summary

    st.subheader("📋 Employee Profile")

    col1,col2 = st.columns(2)

    with col1:
        st.info(f"""
        👤 Age : {age}

        🏢 Department : {department_name}

        💼 Job Role : {job_role_name}

        📂 Projects : {project_count}
        """)

    with col2:
        st.info(f"""
        💰 Income : ₹{monthly_income:,}

        ⏳ Experience : {years_at_company} Years

        🚗 Distance : {distance_from_home} KM

        📈 Job Level : {job_level}
        """)

        st.write("Prediction:", prediction)
        st.write("Probability:", probability)

# HR Recommendation

    st.subheader("🎯 HR Recommendations")

    if prediction[0] == 1:

        st.warning("""
✅ Conduct feedback session

✅ Review workload

✅ Discuss promotion opportunities

✅ Improve engagement

✅ Monitor employee satisfaction
""")

    else:

        st.success("""
✅ Employee appears stable

✅ Continue engagement

✅ Maintain work-life balance

✅ Provide training opportunities
""")
        
# FOOTER 

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:20px;
background:#F1F5F9;
border-radius:10px;
">

<h3>🚀 Employee Attrition Intelligence Platform</h3>

<p>
SVM Classifier |
Streamlit |
Scikit-Learn |
Python
</p>

<p>
Developed By <b>Karan Negi 👽 </b>
</p>

</div>
""", unsafe_allow_html=True)
