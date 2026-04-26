import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

st.title("🎓 Student Placement Predictor")
st.write("Predict whether a student is likely to be placed based on academic and skill parameters.")

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df.drop("placed", axis=1)
y = df["placed"]

# Split + Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy: {round(accuracy * 100, 2)}%")

# User inputs
st.subheader("Enter Student Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
internships = st.number_input("Internships", 0, 5, 1)
projects = st.number_input("Projects", 0, 5, 1)
tech_skills = st.slider("Tech Skills", 1, 10, 5)
comm_skills = st.slider("Communication Skills", 1, 10, 5)
backlogs = st.selectbox("Backlogs", [0, 1])

# Prediction
if st.button("Predict Placement"):
    sample = pd.DataFrame([{
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'tech_skills': tech_skills,
        'comm_skills': comm_skills,
        'backlogs': backlogs
    }])

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.success("🎉 High chances of placement!")
    else:
        st.warning("⚠️ Low chances of placement. Improve skills!")

# Show dataset
st.subheader("Sample Dataset")
st.dataframe(df)

import matplotlib.pyplot as plt

st.subheader("📊 Data Insights")

# Graph 1: Average CGPA by Placement
avg_cgpa = df.groupby("placed")["cgpa"].mean()

fig1, ax1 = plt.subplots()
ax1.bar(["Not Placed", "Placed"], avg_cgpa)
ax1.set_title("Average CGPA by Placement")
ax1.set_ylabel("CGPA")
st.pyplot(fig1)

# Graph 2: Average Tech Skills by Placement
avg_skills = df.groupby("placed")["tech_skills"].mean()

fig2, ax2 = plt.subplots()
ax2.bar(["Not Placed", "Placed"], avg_skills)
ax2.set_title("Average Tech Skills by Placement")
ax2.set_ylabel("Skill Level")
st.pyplot(fig2)