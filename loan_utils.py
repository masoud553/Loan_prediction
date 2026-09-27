from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Clean Dependents
        if "Dependents" in X.columns:
            X["Dependents"] = X["Dependents"].replace("3+", "3")

        # Total income
        if "ApplicantIncome" in X.columns and "CoapplicantIncome" in X.columns:
            X["TotalIncome"] = (
                X["ApplicantIncome"] +
                X["CoapplicantIncome"]
            )

        # EMI
        if "LoanAmount" in X.columns and "Loan_Amount_Term" in X.columns:
            X["EMI"] = (
                X["LoanAmount"] * 1000
            ) / (
                X["Loan_Amount_Term"] + 1e-6
            )

        # Income to EMI
        if "TotalIncome" in X.columns and "EMI" in X.columns:
            X["Income_to_EMI"] = (
                X["TotalIncome"] / (X["EMI"] + 1e-6)
            )

        # Log transformations
        if "ApplicantIncome" in X.columns:
            X["ApplicantIncome"] = np.log1p(
                X["ApplicantIncome"].clip(lower=0)
            )

        if "CoapplicantIncome" in X.columns:
            X["CoapplicantIncome"] = np.log1p(
                X["CoapplicantIncome"].clip(lower=0)
            )

        if "TotalIncome" in X.columns:
            X["TotalIncome"] = np.log1p(
                X["TotalIncome"].clip(lower=0)
            )

        return X