import gradio as gr
import joblib

model = joblib.load("loan_model.joblib")

def predict_loan(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_term, credit_history, property_area):
    row = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": 1.0 if credit_history == "Good (1)" else 0.0,
        "Property_Area": property_area,
    }])
    p = float(model.predict_proba(row)[0, 1])
    verdict = "✅ Loan likely APPROVED" if p >= 0.5 else "❌ Loan likely REJECTED"
    return verdict, {"Approved": p, "Rejected": 1 - p}

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(["Male", "Female"], value="Male", label="Gender"),
        gr.Dropdown(["Yes", "No"], value="Yes", label="Married"),
        gr.Dropdown(["0", "1", "2", "3+"], value="0", label="Dependents"),
        gr.Dropdown(["Graduate", "Not Graduate"], value="Graduate", label="Education"),
        gr.Dropdown(["Yes", "No"], value="No", label="Self Employed"),
        gr.Number(value=5000, label="Applicant Income (monthly)"),
        gr.Number(value=0, label="Coapplicant Income (monthly)"),
        gr.Number(value=130, label="Loan Amount (in thousands)"),
        gr.Dropdown([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], value=360, label="Loan Term (months)"),
        gr.Radio(["Good (1)", "Bad (0)"], value="Good (1)", label="Credit History"),
        gr.Dropdown(["Urban", "Semiurban", "Rural"], value="Urban", label="Property Area"),
    ],
    outputs=[gr.Textbox(label="Prediction"), gr.Label(label="Approval probability")],
    title="🏦 Loan Approval Predictor",
    description="Enter the applicant's details to predict whether the loan will be approved.",
    examples=[["Male", "Yes", "0", "Graduate", "No", 5849, 0, 130, 360, "Good (1)", "Urban"],
              ["Female", "No", "1", "Not Graduate", "Yes", 2000, 0, 200, 360, "Bad (0)", "Rural"]],
)

# Non-blocking in Jupyter: opens the interface inline / at http://127.0.0.1:7860
demo.launch(share=True)
