# CSC 303 Assignment: AI Academic Advisor System

## 1. Introduction
The AI Academic Advisor System is a prototype designed to help a student reflect on academic performance and identify areas that may need attention. It combines software engineering practices with a basic machine-learning technique.

## 2. Problem statement
Students may struggle to decide which academic habits to improve first. The system gathers a few indicators and provides a simple support category and actionable suggestions.

## 3. Objectives
- Build a usable academic-support web application.
- Train a machine-learning classifier.
- Save the trained model in a Pickle (`.pkl`) file.
- integrate the model with a web framework.
- Use Git and GitHub for source-code management.
- Train the model in Google Colab and test the web application.

## 4. System inputs and outputs
**Inputs:** GPA on a 4.0 scale, attendance percentage, weekly independent study hours, number of difficult courses, and number of failed courses to retake.

**Outputs:** Strong, Moderate, or Priority support category, plus suggested actions.

## 5. AI technique
A Decision Tree classifier is trained using a small synthetic dataset. The classifier learns patterns linking academic indicators to the three support categories. The model is serialized using Pickle and loaded by the Streamlit application.

## 6. Software engineering practices
The project separates model training from the user interface, records dependencies, uses version control, documents setup instructions, limits input ranges, and includes a testing checklist.

## 7. Testing
Test the web application with different combinations of GPA, attendance, study hours, difficult courses, and failed courses. Confirm that a recommendation appears after submission. The model's outputs are illustrative and should not be treated as a validated academic assessment.

## 8. Limitations and future improvements
The training dataset is small and synthetic. Future versions could use ethically collected, anonymized data with institutional approval, evaluate model performance on a held-out test set, add course-specific advice, and allow an academic adviser to review recommendations.

## 9. Conclusion
The project demonstrates a basic end-to-end AI application workflow: data preparation, model training, model serialization, web deployment, and source-code version control.
