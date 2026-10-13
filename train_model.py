"""
Train a simple demonstration classifier and save it as a Pickle (.pkl) file.
Run: python train_model.py
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Demonstration data only; not real student records.
data = [
    # gpa, attendance, study_hours, difficult_courses, failed_courses, support_label
    [3.7, 95, 18, 1, 0, "Strong"],
    [3.4, 90, 15, 1, 0, "Strong"],
    [3.8, 92, 20, 0, 0, "Strong"],
    [3.2, 85, 12, 2, 0, "Strong"],
    [2.8, 80, 10, 2, 0, "Moderate"],
    [2.6, 75, 8, 3, 1, "Moderate"],
    [2.9, 82, 7, 2, 0, "Moderate"],
    [2.3, 70, 8, 3, 1, "Moderate"],
    [2.0, 60, 4, 5, 2, "Priority"],
    [1.7, 55, 3, 6, 3, "Priority"],
    [1.9, 65, 5, 4, 2, "Priority"],
    [1.2, 45, 2, 7, 4, "Priority"],
    [3.5, 88, 14, 1, 0, "Strong"],
    [2.4, 72, 9, 3, 0, "Moderate"],
    [2.1, 68, 6, 4, 1, "Moderate"],
    [1.5, 40, 2, 8, 5, "Priority"],
    [3.0, 84, 11, 1, 0, "Strong"],
    [2.7, 78, 9, 2, 1, "Moderate"],
    [1.8, 58, 4, 5, 3, "Priority"],
    [3.6, 96, 16, 0, 0, "Strong"],
    [2.5, 73, 7, 3, 1, "Moderate"],
    [1.6, 50, 3, 6, 4, "Priority"],
    [3.3, 89, 13, 1, 0, "Strong"],
    [2.2, 66, 6, 4, 2, "Priority"],
    [2.9, 86, 10, 2, 0, "Moderate"],
    [1.9, 62, 5, 5, 2, "Priority"],
    [3.1, 83, 12, 2, 0, "Strong"],
    [2.6, 77, 8, 3, 1, "Moderate"],
    [1.4, 48, 2, 7, 4, "Priority"],
    [3.9, 98, 22, 0, 0, "Strong"],
]
columns = ["gpa", "attendance", "study_hours", "difficult_courses", "failed_courses", "support_label"]
df = pd.DataFrame(data, columns=columns)
X = df.drop(columns=["support_label"])
y = df["support_label"]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

output_path = Path(__file__).parent / "academic_advisor_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained on {len(df)} demonstration records.")
print(f"Saved model to: {output_path}")
print("Labels:", sorted(y.unique()))
