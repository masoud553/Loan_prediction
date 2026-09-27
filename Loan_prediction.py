import warnings, zipfile, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

from loan_utils import FeatureEngineer      # custom transformer (see loan_utils.py)

RANDOM_STATE = 42

# Extract the CSV from the uploaded zip (skip this if you already have the csv)
CSV_PATH = "loan_data.csv"

df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print(df.head())

df.info()
print("\nTarget distribution:\n", df["Loan_Status"].value_counts())

# --- Step 1: drop the ID column (a unique identifier carries no predictive signal)
df = df.drop(columns=["Loan_ID"])

# --- Step 2: encode target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Missing value report (motivates step 6)
missing = df.isna().sum()
print("Missing values per column:\n", missing[missing > 0])

# --- Step 5 (detection part): IQR outlier detection on numeric money columns
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
rows = []
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    rows.append([c, round(df[c].skew(), 2), n_out, round(100 * n_out / len(df), 1)])
print(pd.DataFrame(rows, columns=["column", "skewness", "IQR outliers", "% outliers"]))

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
fe_demo = FeatureEngineer().transform(df.drop(columns="Loan_Status"))
for i, c in enumerate(num_cols):
    sns.boxplot(x=df[c], ax=axes[0, i]);     axes[0, i].set_title(f"{c} (raw)")
    sns.boxplot(x=fe_demo[c], ax=axes[1, i]); axes[1, i].set_title(f"{c} (log1p)")
plt.tight_layout(); plt.show()

# Steps 3, 4 and 5 (treatment) in action: what FeatureEngineer produces
fe_demo.head()

# Quick look at how the key features relate to the target
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(data=df, x="Credit_History", y="Loan_Status", ax=axes[0]); axes[0].set_title("Approval rate by Credit_History")
sns.barplot(data=df, x="Property_Area", y="Loan_Status", ax=axes[1]);  axes[1].set_title("Approval rate by Property_Area")
sns.barplot(data=df, x="Married", y="Loan_Status", ax=axes[2]);        axes[2].set_title("Approval rate by Married")
plt.tight_layout(); plt.show()

X = df.drop(columns="Loan_Status")
y = df["Loan_Status"]

# Column groups AFTER FeatureEngineer has run
numeric_features = ["Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                    "Loan_Amount_Term", "Credit_History", "TotalIncome", "EMI", "Income_to_EMI"]
categorical_features = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),      # step 6
    ("scale",  StandardScaler()),                      # step 7
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),                 # step 6
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),# step 7
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

def build_pipeline(model):
    return Pipeline([("features", FeatureEngineer()),
                     ("prep", preprocessor),
                     ("model", model)])

primary_model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)
pipe = build_pipeline(primary_model)
pipe

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train approval rate: %.3f | Test approval rate: %.3f" % (y_train.mean(), y_test.mean()))

pipe.fit(X_train, y_train)
print("\nTraining accuracy: %.4f" % pipe.score(X_train, y_train))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_res = cross_validate(build_pipeline(primary_model), X_train, y_train, cv=cv,
                        scoring=scoring, return_train_score=True, n_jobs=-1)

print("Per-fold accuracy:", np.round(cv_res["test_accuracy"], 4))
print()
baseline_summary = {}
for m in scoring:
    s = cv_res[f"test_{m}"]
    baseline_summary[m] = (s.mean(), s.std())
    print(f"{m:10s}: {s.mean():.4f} ± {s.std():.4f}")
print(f"\nMean train accuracy: {cv_res['train_accuracy'].mean():.4f}  "
      f"(gap vs CV accuracy shows how much the untuned forest overfits)")
baseline_cv_auc = cv_res["test_roc_auc"].mean()

# ---------- Randomized search (wide space)
rand_space = {
    "model__n_estimators":      [100, 200, 300],
    "model__max_depth":         [2, 3, 4, 5, 6, 8, 10, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 8, 12],
    "model__max_features":      ["sqrt", "log2", 0.5],
    "model__class_weight":      ["balanced", "balanced_subsample", None],
}
print("Randomized-search space:")
for k, v in rand_space.items(): print(f"  {k:26s}: {v}")

rand_search = RandomizedSearchCV(
    build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
    rand_space, n_iter=30, cv=cv, scoring="roc_auc",
    random_state=RANDOM_STATE, n_jobs=-1, refit=True)
rand_search.fit(X_train, y_train)

print("\nRandomized search -> best CV ROC-AUC: %.4f" % rand_search.best_score_)
print("Best params:", rand_search.best_params_)

# ---------- Grid search (focused grid built around the randomized-search findings)
grid_space = {
    "model__n_estimators":      [200],
    "model__max_depth":         [3, 4, 5, 6],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf":  [2, 4, 8],
    "model__max_features":      ["sqrt", 0.5],
    "model__class_weight":      ["balanced", None],
}
n_combos = int(np.prod([len(v) for v in grid_space.values()]))
print(f"Grid-search space ({n_combos} combinations x 5 folds = {n_combos*5} fits):")
for k, v in grid_space.items(): print(f"  {k:26s}: {v}")

grid_search = GridSearchCV(
    build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
    grid_space, cv=cv, scoring={"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"},
    refit="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nGrid search -> best CV ROC-AUC: %.4f" % grid_search.best_score_)
print("Best params:", grid_search.best_params_)

# Top-10 combinations tested by the grid search
res = pd.DataFrame(grid_search.cv_results_)
cols = ["rank_test_roc_auc", "mean_test_roc_auc", "std_test_roc_auc",
        "mean_test_accuracy", "mean_test_f1"] + [c for c in res.columns if c.startswith("param_")]
top = (res[cols].sort_values("rank_test_roc_auc").head(10)
       .rename(columns=lambda c: c.replace("param_model__", "")))
top.round(4).reset_index(drop=True)

candidates = {
    "Baseline Random Forest (untuned)": (None, baseline_cv_auc),
    "RandomizedSearchCV best":          (rand_search, rand_search.best_score_),
    "GridSearchCV best":                (grid_search, grid_search.best_score_),
}
comp = pd.DataFrame({k: {"CV ROC-AUC": v[1]} for k, v in candidates.items()}).T
print(comp.round(4))

tuned = {k: v for k, v in candidates.items() if v[0] is not None}
best_name = max(tuned, key=lambda k: tuned[k][1])
best_model = tuned[best_name][0].best_estimator_
print("\nSelected final model:", best_name)
print("Final hyper-parameters:")
for k, v in best_model.named_steps["model"].get_params().items():
    if k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
             "max_features", "class_weight"]:
        print(f"   {k:18s} = {v}")

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy":            accuracy_score(y_test, y_pred),
    "Precision":           precision_score(y_test, y_pred),
    "Recall":              recall_score(y_test, y_pred),
    "F1-score":            f1_score(y_test, y_pred),
    "ROC-AUC":             roc_auc_score(y_test, y_proba),
}
for k, v in metrics.items(): print(f"{k:10s}: {v:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nConfusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"]))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Rejected", "Approved"],
                                        cmap="Blues", ax=axes[0]); axes[0].set_title("Confusion matrix (test)")
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="Random Forest")
axes[1].plot([0, 1], [0, 1], "k--", label="chance"); axes[1].legend(); axes[1].set_title("ROC curve (test)")

names = [n.split("__")[-1] for n in best_model.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(best_model.named_steps["model"].feature_importances_, index=names).sort_values()
imp.tail(10).plot.barh(ax=axes[2], color="steelblue"); axes[2].set_title("Top-10 feature importances")
plt.tight_layout(); plt.savefig("evaluation_plots.png", dpi=130); plt.show()

# Save the trained pipeline for the Gradio app
joblib.dump(best_model, "loan_model.joblib")
print("Saved -> loan_model.joblib")