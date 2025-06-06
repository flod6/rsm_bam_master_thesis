#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import torch

# Define paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
results_ML_Regression_absolute = pd.read_csv(output + "Result_Models/ML_CoVar_Regression_absolute_results.csv")
results_GNN_Regression_absolute = pd.read_csv(output + "Result_Models/covar_results_GNN_Regression_absolute.csv")
results_FCNN_Regression_absolute = pd.read_csv(output + "Result_Models/covar_results_FCNN_Regression_absolute.csv")
results_ML_Regression_change = pd.read_csv(output + "Result_Models/ML_CoVar_Regression_change_results.csv")
results_GNN_Regression_change = pd.read_csv(output + "Result_Models/covar_results_GNN_Regression_change.csv")
results_FCNN_Regression_change = pd.read_csv(output + "Result_Models/covar_results_FCNN_Regression_change.csv")
predictions_ML_Regression_absolute = pd.read_csv(output + "Predictions_Models/ML_CoVar_Regression_absolute_predictions.csv")
predictions_GNN_Regression_absolute = pd.read_csv(output + "Predictions_Models/covar_predictions_GNN_Regression_absolute.csv")
predictions_FCNN_Regression_absolute = pd.read_csv(output + "Predictions_Models/covar_predictions_FCNN_Regression_absolute.csv")
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")
bank_data_q_clean = pd.read_parquet(input + "Clean/bank_data_q_clean.parquet")
feature_importance = pd.read_csv(output + "Feature_Importance/features_CoVar_regression_absolute.csv")


#----------------------------------
# 2. Assemble Final Results
#----------------------------------

#------------
# a. ML Regression Aboslute
#------------

# Drop R2
results_ML_Regression_absolute = results_ML_Regression_absolute.drop(columns=["Train R2", "Test R2"])

# Split up the different sets
ML_val = results_ML_Regression_absolute.iloc[:, :5]
ML_test = results_ML_Regression_absolute.drop(columns=["Train RMSE", "Train MSE", "Train MAE", "Train MPE"])
GNN_val = results_GNN_Regression_absolute.iloc[0:1, 2:6]
GNN_test = results_GNN_Regression_absolute.iloc[1:2, 2:6]
FCNN_val = results_FCNN_Regression_absolute.iloc[0:1, 2:6]
FCNN_test = results_FCNN_Regression_absolute.iloc[1:2, 2:6]

# Arrange columns
ML_val = ML_val[["Model", "Train RMSE", "Train MSE", "Train MAE", "Train MPE"]]
ML_test = ML_test[["Model", "Test RMSE", "Test MSE", "Test MAE", "Test MPE"]]

# Add columns with name for GNN and FCNN
GNN_val.insert(0, "Model", ["GNN"])
GNN_test.insert(0, "Model", ["GNN"])
FCNN_val.insert(0, "Model", ["NN"])
FCNN_test.insert(0, "Model", ["NN"])

ML_val.columns = GNN_val.columns
ML_test.columns = GNN_test.columns

# Combine the results
final_results_val = pd.concat([ML_val, FCNN_val, GNN_val], axis=0)
final_results_test = pd.concat([ML_test, FCNN_test, GNN_test], axis=0)

# Save the results
final_results_val.to_csv(output + "Result_Models/final_errors_Regression_abolute_val.csv", index=False)
final_results_test.to_csv(output + "Result_Models/final_errors_Regression_abolute_test.csv", index=False)


#------------
# a. ML Regression Change
#------------

# Drop R2
results_ML_Regression_change = results_ML_Regression_change.drop(columns=["Train R2", "Test R2"])

# Split up the different sets
ML_val = results_ML_Regression_change.iloc[:, :5]
ML_test = results_ML_Regression_change.drop(columns=["Train RMSE", "Train MSE", "Train MAE", "Train MPE"])
GNN_val = results_GNN_Regression_change.iloc[0:1, 2:6]
GNN_test = results_GNN_Regression_change.iloc[1:2, 2:6]
FCNN_val = results_FCNN_Regression_change.iloc[0:1, 2:6]
FCNN_test = results_FCNN_Regression_change.iloc[1:2, 2:6]

# Arrange columns
ML_val = ML_val[["Model", "Train RMSE", "Train MSE", "Train MAE", "Train MPE"]]
ML_test = ML_test[["Model", "Test RMSE", "Test MSE", "Test MAE", "Test MPE"]]

# Add columns with name for GNN and FCNN
GNN_val.insert(0, "Model", ["GNN"])
GNN_test.insert(0, "Model", ["GNN"])
FCNN_val.insert(0, "Model", ["NN"])
FCNN_test.insert(0, "Model", ["NN"])

ML_val.columns = GNN_val.columns
ML_test.columns = GNN_test.columns

# Combine the results
final_results_val = pd.concat([ML_val, FCNN_val, GNN_val], axis=0)
final_results_test = pd.concat([ML_test, FCNN_test, GNN_test], axis=0)

# Save the results
final_results_val.to_csv(output + "Result_Models/final_errors_Regression_change_val.csv", index=False)
final_results_test.to_csv(output + "Result_Models/final_errors_Regression_change_test.csv", index=False)



#----------------------------------
# 2. Assemble Final Predictions
#----------------------------------

# Filter out test sets
predictions_ML = predictions_ML_Regression_absolute
predictions_GNN = predictions_GNN_Regression_absolute[predictions_GNN_Regression_absolute["split"] == "test"]
predictions_FCNN = predictions_FCNN_Regression_absolute[predictions_FCNN_Regression_absolute["split"] == "test"]


# Rename columns
predictions_GNN = predictions_GNN.rename(columns={
    "prediction": "GNN"
})

predictions_FCNN = predictions_FCNN.rename(columns={
    "prediction": "FCNN"
})

# Join different sets 
predictions_test = pd.merge(predictions_ML, predictions_GNN, 
                             left_on = ["gvkey", "date"],
                             right_on = ["firm", "quarter"],
                             how = "left")
predictions_test = pd.merge(predictions_test, predictions_FCNN,
                            left_on = ["gvkey", "date"],
                            right_on = ["firm", "quarter"],
                            how = "left")

# Drop unnecessary columns
predictions_test = predictions_test[["gvkey", "date", "Actual",
                                     "Linear Regression", "Lasso", 
                                     "Random Forest", "GNN", "FCNN"]]


# Define systemic quarters
systemic_quarters = ["2020Q1", "2020Q2", "2022Q1", "2023Q1"]

# Assign 1 for systemic events and 0 otherwise
predictions_test["systemic_events"] = predictions_test["date"].apply(
    lambda x: 1 if x in systemic_quarters else 0
)


# Squared Error (SE)
predictions_test["Reg_SE"] = (predictions_test["Actual"] - predictions_test["Linear Regression"]) ** 2 
predictions_test["Lasso_SE"] = (predictions_test["Actual"] - predictions_test["Lasso"]) ** 2 
predictions_test["RF_SE"] = (predictions_test["Actual"] - predictions_test["Random Forest"]) ** 2 
predictions_test["GNN_SE"] = (predictions_test["Actual"] - predictions_test["GNN"]) ** 2
predictions_test["FCNN_SE"] = (predictions_test["FCNN"] - predictions_test["Actual"]) ** 2

# Absolute Error (AE)
predictions_test["Reg_AE"] = abs(predictions_test["Actual"] - predictions_test["Linear Regression"])
predictions_test["Lasso_AE"] = abs(predictions_test["Actual"] - predictions_test["Lasso"])
predictions_test["RF_AE"] = abs(predictions_test["Actual"] - predictions_test["Random Forest"])
predictions_test["GNN_AE"] = abs(predictions_test["Actual"] - predictions_test["GNN"])
predictions_test["FCNN_AE"] = abs(predictions_test["Actual"] - predictions_test["FCNN"])

# Percentage Error (PE)
predictions_test["Reg_PE"] = (predictions_test["Actual"] - predictions_test["Linear Regression"]) / predictions_test["Actual"]
predictions_test["Lasso_PE"] = (predictions_test["Actual"] - predictions_test["Lasso"]) / predictions_test["Actual"]
predictions_test["RF_PE"] = (predictions_test["Actual"] - predictions_test["Random Forest"]) / predictions_test["Actual"]
predictions_test["GNN_PE"] = (predictions_test["Actual"] - predictions_test["GNN"]) / predictions_test["Actual"]
predictions_test["FCNN_PE"] = (predictions_test["Actual"] - predictions_test["FCNN"]) / predictions_test["Actual"]


np.sqrt(predictions_test["FCNN_SE"].mean())
np.sqrt(predictions_test["GNN_SE"].mean())
np.sqrt(predictions_test["RF_SE"].mean())
np.sqrt(predictions_test["Lasso_SE"].mean())
predictions_test["Reg_SE"].mean()
predictions_test["FCNN_SE"].mean()

# Group the results
error_group = predictions_test.groupby("systemic_events").agg({
    "Reg_SE": "mean",
    "Lasso_SE": "mean",
    "RF_SE": "mean",
    "GNN_SE": "mean",
    "FCNN_SE": "mean",
    "Reg_AE": lambda x: np.mean(np.abs(x)),
    "Lasso_AE": lambda x: np.mean(np.abs(x)),
    "RF_AE": lambda x: np.mean(np.abs(x)),
    "GNN_AE": lambda x: np.mean(np.abs(x)),
    "FCNN_AE": lambda x: np.mean(np.abs(x)),
    "Reg_PE": "mean",
    "Lasso_PE": "mean",
    "RF_PE": "mean",
    "GNN_PE": "mean",
    "FCNN_PE": "mean",
}).rename(columns={
    "Reg_SE": "Reg_MSE", "Lasso_SE": "Lasso_MSE", "RF_SE": "RF_MSE", 
    "GNN_SE": "GNN_MSE", "FCNN_SE": "FCNN_MSE",
    "Reg_AE": "Reg_MAE", "Lasso_AE": "Lasso_MAE", "RF_AE": "RF_MAE",
    "GNN_AE": "GNN_MAE", "FCNN_AE": "FCNN_MAE",
    "Reg_PE": "Reg_MPE", "Lasso_PE": "Lasso_MPE", "RF_PE": "RF_MPE",
    "GNN_PE": "GNN_MPE", "FCNN_PE": "FCNN_MPE"
})

# Step 2: Add RMSE as sqrt(MSE)
for model in ["Reg", "Lasso", "RF", "GNN", "FCNN"]:
    error_group[f"{model}_RMSE"] = np.sqrt(error_group[f"{model}_MSE"])



# Finalize the table 
rmse_group = error_group[["Reg_RMSE", "Lasso_RMSE", "RF_RMSE", "FCNN_RMSE", "GNN_RMSE"]].transpose()
mpe_group = error_group[["Reg_MPE", "Lasso_MPE", "RF_MPE", "FCNN_MPE", "GNN_MPE"]].transpose()



# Split up index name to extract the model
rmse_group["Model"] = rmse_group.index.str.split('_').str[0]
mpe_group["Model"] = mpe_group.index.str.split('_').str[0]

group = pd.merge(rmse_group, mpe_group, on="Model")
group.reset_index(drop=True, inplace=True)


# Set colnames
group.columns = ["Non-Crisis RMSE", "Crisis RMSE", "Model", "Non-Crisis MPE", "Crisis MPE"]
group = group[["Model", "Non-Crisis RMSE", "Crisis RMSE", "Non-Crisis MPE", "Crisis MPE"]]

group["Model"] = ["Linear Regression", "Lasso", "Random Forest", "FCNN", "GNN"]

# Save the results
group.to_csv(output + "Result_Models/final_performance_crisis.csv", index=False)






# ----------------------------------
# 3. Size Comparison
# ----------------------------------

covar_final_comb = pd.merge(covar_final, bank_data_q_clean[["gvkey", "datacqtr", "atq"]],
                            left_on = ["gvkey", "date"],
                            right_on = ["gvkey", "datacqtr"],
                            how = "left")


# Filter observations in the test set 
covar_final_comb = covar_final_comb[covar_final_comb["date"] >= "2018Q1"]
covar_final_comb = covar_final_comb.sort_values(['date', 'atq'], ascending=[True, False])


# Create a new column to store the top 5% flag
covar_final_comb['top_5_percent'] = 0  # default: not top 5%

# Step 3: Loop through each quarter and flag the top 5%
unique_quarters = covar_final_comb['date'].unique()

for quarter in unique_quarters:
    # Filter that quarter
    mask = covar_final_comb['date'] == quarter
    n_banks = mask.sum()
    top_n = max(1, int(n_banks * 0.05))  # ensure at least one bank
    
    # Get indices of top 5%
    top_indices = covar_final_comb[mask].nlargest(top_n, "atq").index
    
    # Mark them
    covar_final_comb.loc[top_indices, "top_5_percent"] = 1


predictions_test["gvkey"] = predictions_test["gvkey"].astype(str).str.zfill(6) 

# Join Data on predictions: 
predictions_test["gvkey"] = predictions_test["gvkey"].astype(str)
predictions_test = pd.merge(predictions_test, covar_final_comb[["gvkey", "date", "top_5_percent"]],
                            left_on = ["gvkey", "date"],
                            right_on= ["gvkey", "date"],
                            how = "left")



## Okay there are many missing values inside



# Groupy by top 5% and calculate the mean error for each type of firms:
error_group = predictions_test.groupby("top_5_percent").agg({
    "Reg_SE": "mean",
    "Lasso_SE": "mean",
    "RF_SE": "mean",
    "GNN_SE": "mean",
    "FCNN_SE": "mean",
    "Reg_AE": lambda x: np.mean(np.abs(x)),
    "Lasso_AE": lambda x: np.mean(np.abs(x)),
    "RF_AE": lambda x: np.mean(np.abs(x)),
    "GNN_AE": lambda x: np.mean(np.abs(x)),
    "FCNN_AE": lambda x: np.mean(np.abs(x)),
    "Reg_PE": "mean",
    "Lasso_PE": "mean",
    "RF_PE": "mean",
    "GNN_PE": "mean",
    "FCNN_PE": "mean",
}).rename(columns={
    "Reg_SE": "Reg_MSE", "Lasso_SE": "Lasso_MSE", "RF_SE": "RF_MSE", 
    "GNN_SE": "GNN_MSE", "FCNN_SE": "FCNN_MSE",
    "Reg_AE": "Reg_MAE", "Lasso_AE": "Lasso_MAE", "RF_AE": "RF_MAE",
    "GNN_AE": "GNN_MAE", "FCNN_AE": "FCNN_MAE",
    "Reg_PE": "Reg_MPE", "Lasso_PE": "Lasso_MPE", "RF_PE": "RF_MPE",
    "GNN_PE": "GNN_MPE", "FCNN_PE": "FCNN_MPE"
})

# Step 2: Add RMSE as sqrt(MSE)
# Step 2: Add RMSE as sqrt(MSE)
for model in ["Reg", "Lasso", "RF", "GNN", "FCNN"]:
    error_group[f"{model}_RMSE"] = np.sqrt(error_group[f"{model}_MSE"])



# Finalize the table 
rmse_group = error_group[["Reg_RMSE", "Lasso_RMSE", "RF_RMSE", "FCNN_RMSE", "GNN_RMSE"]].transpose()
mpe_group = error_group[["Reg_MPE", "Lasso_MPE", "RF_MPE", "FCNN_MPE", "GNN_MPE"]].transpose()



# Split up index name to extract the model
rmse_group["Model"] = rmse_group.index.str.split('_').str[0]
mpe_group["Model"] = mpe_group.index.str.split('_').str[0]

group = pd.merge(rmse_group, mpe_group, on="Model")
group.reset_index(drop=True, inplace=True)


# Set colnames
group.columns = ["Small Banks RMSE", "Large Banks RMSE", "Model", "Small Banks MPE", "Large Banks MPE"]
group = group[["Model", "Small Banks RMSE", "Large Banks RMSE", "Small Banks MPE", "Large Banks MPE"]]

group["Model"] = ["Linear Regression", "Lasso", "Random Forest", "FCNN", "GNN"]

# Save the results
group.to_csv(output + "Result_Models/final_performance_size.csv", index=False)




# ----------------------------------
# 3. Model Feature Importance
# ----------------------------------

feature_importance = feature_importance.pivot(index='Feature', columns='Model', values='Importance')
feature_importance = feature_importance.drop(columns=["Linear Regression"])  
feature_importance.index.name = None

feature_importance.to_csv(output + "Feature_Importance/features_CoVar_regression_absolute.csv", index=True)




# %%
# ----------------------------------
# 4. Get Quarterly Returns
# ----------------------------------

# Just to confirm that the banking system data suffered during this period

# Get quarterly returns:
banking_system = pd.read_csv(input + "Clean/bank_returns_system.csv")

# Ensure date is datetime and assign quarters
banking_system['date'] = pd.to_datetime(banking_system['date'])
banking_system['quarter'] = banking_system['date'].dt.to_period('Q')

# Compounded return: (1 + r1)(1 + r2)... - 1
quarterly_returns = banking_system.groupby('quarter')['ret'].apply(
    lambda x: (1 + x).prod() - 1
).reset_index(name='quarterly_return')

# Convert quarter to string
quarterly_returns['quarter'] = quarterly_returns['quarter'].astype(str)

print(quarterly_returns.tail())





