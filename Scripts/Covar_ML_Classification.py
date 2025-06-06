# ----------------------------------
# 1. Set-Up
# ----------------------------------

import pandas as pd
import numpy as np 
import sklearn as sk
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")

# ----------------------------------
# 2. Data Prep
# ----------------------------------

covar_final = covar_final.sort_values(by=["date"])
covar_final["date"] = pd.to_datetime(covar_final["date"]).dt.to_period("Q")
covar_final = covar_final.drop(columns=["conm"])

# Create classification target
covar_final["diff_delta_covar_class"] = (covar_final["diff_delta_covar"] > 0).astype(int)

# Define date splits
train_end = pd.Period("2013Q4", freq="Q")
val_end   = pd.Period("2017Q4", freq="Q")

print(covar_final["date"].unique())

# Split
train = covar_final[covar_final["date"] <= train_end]
val = covar_final[(covar_final["date"] > train_end) & (covar_final["date"] <= val_end)]
test = covar_final[covar_final["date"] > val_end]

print(len(train))
print(len(val))
print(len(test))

print(train["diff_delta_covar_class"].mean())


# Isolate features and labels
feature_cols = [col for col in covar_final.columns if col not in ["date", "gvkey", "diff_delta_covar", "diff_delta_covar_class"]]

x_train = train[feature_cols]
y_train = train["diff_delta_covar_class"]
x_val   = val[feature_cols]
y_val   = val["diff_delta_covar_class"]
x_test  = test[feature_cols]
y_test  = test["diff_delta_covar_class"]


print(y_train.mean(), y_val.mean(), y_test.mean())

# Combine training and validation for tuning
X_tv = pd.concat([x_train, x_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)
test_fold = np.r_[np.full(len(x_train), -1), np.zeros(len(x_val))]
ps = sk.model_selection.PredefinedSplit(test_fold)

# ----------------------------------
# 3. Define Models
# ----------------------------------

logreg_model = sk.linear_model.LogisticRegression(max_iter=1000)
rf_model = sk.ensemble.RandomForestClassifier(random_state=187)
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", sk.linear_model.LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=187))
])

# ----------------------------------
# 4. Define Tuning Grids
# ----------------------------------

logreg_grid = {"C": np.logspace(-3, 3, 10)}
rf_grid = {"n_estimators": [250], 
           "max_features": list(range(1, len(X_tv.columns))),
            "bootstrap":[True],
           "max_depth": [5, 10, 20]}
lasso_grid = {"model__C": np.logspace(-3, 3, 100)}

# ----------------------------------
# 5. Grid Search
# ----------------------------------

logreg_wf = sk.model_selection.GridSearchCV(
    estimator=logreg_model,
    param_grid=logreg_grid,
    cv=ps,
    scoring="roc_auc",
    refit=True
)
logreg_wf.fit(X_tv, y_tv)

rf_wf = sk.model_selection.GridSearchCV(
    estimator=rf_model,
    param_grid=rf_grid,
    cv=ps,
    scoring="roc_auc",
    refit=True,
    n_jobs=-1
)
rf_wf.fit(X_tv, y_tv)

lasso_wf = sk.model_selection.GridSearchCV(
    estimator=lasso_model,
    param_grid=lasso_grid,
    cv=ps,
    scoring="roc_auc",
    refit=True
)
lasso_wf.fit(X_tv, y_tv)

# ----------------------------------
# 6. Evaluate Test Set
# ----------------------------------

def evaluate_classifier(model, x_data, y_data, name="Model", split="Test"):
    y_prob = model.predict_proba(x_data)[:, 1]
    y_pred = model.predict(x_data)
    accuracy = accuracy_score(y_data, y_pred)
    roc_auc = roc_auc_score(y_data, y_prob)
    print(f"\n{name} {split} Classification Report:")
    print(classification_report(y_data, y_pred))
    print(f"{name} {split} Confusion Matrix:\n{confusion_matrix(y_data, y_pred)}")
    print(f"{name} {split} Accuracy: {accuracy:.4f}")
    print(f"{name} {split} ROC AUC: {roc_auc:.4f}")
    return {"Model": name, "Split": split, "Accuracy": accuracy, "ROC AUC": roc_auc}

# Evaluate models on both validation and test sets
results = []
for model_name, model in {
    "Logistic Regression": logreg_wf.best_estimator_,
    "Random Forest": rf_wf.best_estimator_,
    "Lasso Logistic Regression": lasso_wf.best_estimator_
}.items():
    for split_name, (X, y) in [("Validation", (x_val, y_val)), ("Test", (x_test, y_test))]:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        report = classification_report(y, y_pred, output_dict=True)
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        results.append({
            "Model": model_name,
            "Split": split_name,
            "Accuracy": accuracy,
            "ROC AUC": roc_auc,
            "Precision": precision,
            "Recall": recall
        })

# Summarize results in a table
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)

# Save the summary table to a CSV file
results_df.to_csv(output + "Result_Models/covar_classification_summary.csv", index=False)

# ----------------------------------
# 7. Save Results
# ----------------------------------

# Predict and store results
models = {
    "Logistic Regression": logreg_wf.best_estimator_,
    "Random Forest": rf_wf.best_estimator_,
    "Lasso Logistic Regression": lasso_wf.best_estimator_
}

splits = {"train": (x_train, y_train, train), "val": (x_val, y_val, val), "test": (x_test, y_test, test)}
prediction_list = []

for model_name, model in models.items():
    for split_name, (X, y_true, df_meta) in splits.items():
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        preds = pd.DataFrame({
            "Model": model_name,
            "Split": split_name,
            "Actual": y_true.values,
            "Predicted": y_pred,
            "Probability": y_prob,
            "gvkey": df_meta["gvkey"].values,
            "date": df_meta["date"].astype(str).values
        })
        prediction_list.append(preds)

predictions_df = pd.concat(prediction_list, ignore_index=True)
predictions_df.to_csv(output + "Result_Models/covar_classification_predictions.csv", index=False)



# %%

### ----------------------------------

# Testing area
print(y_val.value_counts(normalize=True))


print(len(train["date"]))
print(len(val["date"]))
print(len(test["date"]))


val["diff_delta_covar_class"].mean()
test["diff_delta_covar_class"].mean()


val["date"].value_counts()
val.groupby("date")["diff_delta_covar_class"].mean().plot(title="Positive Class Share by Quarter (Validation)")


quarter_stats = covar_final.groupby("date")["diff_delta_covar_class"].mean()






quarter_stats = covar_final.groupby("date")["diff_delta_covar_class"].mean().sort_index()
print(quarter_stats)

# Step 2: Select a train/val/test split
# For example, this is arbitrary — adjust based on your real class balance
train_end = pd.Period("2013Q4", freq="Q")
val_end   = pd.Period("2016Q4", freq="Q")

# Step 3: Assign splits (still temporal!)
train = covar_final[covar_final["date"] <= train_end]
val   = covar_final[(covar_final["date"] > train_end) & (covar_final["date"] <= val_end)]
test  = covar_final[covar_final["date"] > val_end]

# Step 4: Check balance across splits
print("Train:", train["diff_delta_covar_class"].mean())
print("Val:", val["diff_delta_covar_class"].mean())
print("Test:", test["diff_delta_covar_class"].mean())



import pandas as pd

# Ensure your 'date' column is in Period format
covar_final["date"] = pd.to_datetime(covar_final["date"]).dt.to_period("Q")

# Calculate class proportions per quarter
quarter_stats = covar_final.groupby("date")["diff_delta_covar_class"].mean().reset_index()
quarter_stats.columns = ["date", "positive_rate"]
print(quarter_stats)

# Plot to visualize
import matplotlib.pyplot as plt
quarter_stats.set_index("date")["positive_rate"].plot(title="Positive Class Share by Quarter")
plt.axhline(0.38, color='green', linestyle='--', label='Train/Test Target Level')
plt.legend()
plt.show()


# For val set:
preds_val = predictions_df[(predictions_df["Split"] == "val") & (predictions_df["Model"] == "Logistic Regression")]
grouped = preds_val.groupby("date")
for date, group in grouped:
    print(f"{date}: AUC = {roc_auc_score(group['Actual'], group['Probability']):.3f}")

    
mean_auc = np.mean([all the AUCs above])
median_auc = np.median([...])

label_var_by_q = covar_final.groupby("date")["diff_delta_covar_class"].var()

covar_final.groupby("date")["diff_delta_covar_class"].describe()

print(label_var_by_q.describe())