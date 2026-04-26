import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df.drop("placed", axis=1)
y = df["placed"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Take user input
cgpa = float(input("Enter CGPA: "))
internships = int(input("Enter number of internships: "))
projects = int(input("Enter number of projects: "))
tech_skills = int(input("Tech skills (1-10): "))
comm_skills = int(input("Communication skills (1-10): "))
backlogs = int(input("Backlogs (0 or 1): "))

# Create DataFrame with proper column names
sample_df = pd.DataFrame([{
    'cgpa': cgpa,
    'internships': internships,
    'projects': projects,
    'tech_skills': tech_skills,
    'comm_skills': comm_skills,
    'backlogs': backlogs
}])

prediction = model.predict(sample_df)

print("\nPrediction:", "Placed" if prediction[0] == 1 else "Not Placed")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

