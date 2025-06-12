#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import sklearn as sk
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np 
import sklearn as sk
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import sklearn as sk
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import seaborn as sns
import joblib

# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")

#----------------------------------
# 1. Prepare Data
#----------------------------------

# Arrange data frame by date 
covar_final = covar_final.sort_values(by=["date"])
covar_final["date"] = pd.to_datetime(covar_final["date"]).dt.to_period("Q")#.astype(str)
covar_final = covar_final.drop(columns=["conm"])

# Define data split dates
train_end = pd.Period("2013Q4", freq="Q")
val_end   = pd.Period("2017Q4", freq="Q")

# Split data set according to data
train = covar_final[covar_final["date"] <= train_end]
val = covar_final[(covar_final["date"] > train_end) & (covar_final["date"] <= val_end)]
test = covar_final[covar_final["date"] > val_end]

# Isolate variables
y_train = train["covar"]
x_train = train.drop(columns= ["date", "gvkey", "diff_delta_covar" , "covar", "lag_diff_delta_covar"
                               ])
y_val = val["covar"]
x_val = val.drop(columns= ["date", "gvkey", "diff_delta_covar" , "covar", "lag_diff_delta_covar"
                           ])
y_test = test["covar"]
x_test = test.drop(columns= ["date", "gvkey", "diff_delta_covar", "covar", "lag_diff_delta_covar"
                             ])


## Create train test split
# Merge data sets again
X_tv = pd.concat([x_train, x_val], ignore_index=True)
y_tv = pd.concat([y_train, y_val], ignore_index=True)

# Build index for folds
test_fold = np.r_[np.full(len(x_train), -1),   
                  np.zeros(len(x_val))]

# Define split
ps = sk.model_selection.PredefinedSplit(test_fold)  


# Define function to get RMSE
def rmse_score(y, y_hat):
  return np.sqrt(sk.metrics.mean_squared_error(y, y_hat))

# Set up metrics
metrics_ml = {
  "rmse": "neg_root_mean_squared_error",
  "r2":   "r2",
  "neg_mean_absolute_error": "neg_mean_absolute_error",
  "mse":  "neg_mean_squared_error",
  "mpe":  sk.metrics.make_scorer(lambda y, y_hat: np.mean((y - y_hat) / y), greater_is_better=False)
}

#----------------------------------
# 2. Define Models
#----------------------------------

# Linear Reg. 
linreg_model = sk.pipeline.Pipeline([
        ("scaler", sk.preprocessing.StandardScaler()),
        ("model",  sk.linear_model.LinearRegression())])

# linear_model = sk.linear_model.LinearRegression()



# Lasso
lasso_model  = sk.pipeline.Pipeline([
        ("scaler", sk.preprocessing.StandardScaler()),
        ("model",  sk.linear_model.Lasso(random_state = 187))])

# Random Forest 
rf_model = sk.ensemble.RandomForestRegressor(random_state = 187)

# Linear Regression
# None

#----------------------------------
# 3. Set-Up Tuning Grids
#----------------------------------

# Lasso
lasso_tuning_grid = {"model__alpha": np.logspace(-3, 3, 30)}

# Random Forest
rf_tuning_grid = {"n_estimators": [500],
  "max_features": list(range(1, len(X_tv.columns))),
  "max_depth": [5,10, 15, 20],
  "bootstrap":[True]}


#----------------------------------
# 4. Perform Grid Search
#----------------------------------

# Linear Regression
linreg_wf = sk.model_selection.GridSearchCV(estimator = linreg_model,
                                            param_grid = {}, 
                                            cv = ps, 
                                            scoring = metrics_ml,
                                            refit = "rmse")
# Fit the model
linreg_wf.fit(X_tv, y_tv)

# Lasso
lasso_wf = sk.model_selection.GridSearchCV(estimator = lasso_model, 
                                           param_grid = lasso_tuning_grid,
                                           cv = ps, 
                                           scoring = metrics_ml,
                                           refit = "rmse")
# Fit the model
lasso_wf.fit(X_tv, y_tv)

# Random Forest
rf_wf = sk.model_selection.GridSearchCV(estimator = rf_model,
                                        param_grid = rf_tuning_grid,
                                        cv = ps,
                                        scoring = metrics_ml,
                                        refit = "rmse",
                                        n_jobs = -1)
# Fit the model
rf_wf.fit(X_tv, y_tv)


# Get best models
best_lasso_model = lasso_wf.best_estimator_
best_linreg_model = linreg_wf.best_estimator_
best_rf_model = rf_wf.best_estimator_


#----------------------------------
# 5. Evaluate Validation Split
#----------------------------------

# Evaluate Performance
# Linear Regression
print(f"Linear Regression: Best RMSE: {linreg_wf.best_score_}")
linreg_cv_results = linreg_wf.cv_results_
linreg_best_index = linreg_wf.best_index_
print(f"Linear Regression: Best R2: {linreg_cv_results['mean_test_r2'][linreg_best_index]:.4f}")
print(f"Linear Regression: Best MAE: {-linreg_cv_results['mean_test_neg_mean_absolute_error'][linreg_best_index]:.4f}\n")

# Lasso
print(f"Lasso: Best alpha: {lasso_wf.best_params_['model__alpha']}")
print(f"Lasso: Best RMSE: {lasso_wf.best_score_}")
lasso_cv_results = lasso_wf.cv_results_
lasso_best_index = lasso_wf.best_index_
print(f"Lasso: Best R2: {lasso_cv_results['mean_test_r2'][lasso_best_index]:.4f}")
print(f"Lasso: Best MAE: {-lasso_cv_results['mean_test_neg_mean_absolute_error'][lasso_best_index]:.4f}\n")

# Random Forrest 
print(f"RF: Best MF: {rf_wf.best_params_['max_features']}")
print(f"RF: Best MD: {rf_wf.best_params_['max_depth']}")
print(f"RF: Best RMSE: {rf_wf.best_score_}")
rf_cv_results = rf_wf.cv_results_
rf_best_index = rf_wf.best_index_
print(f"RF: Best R2: {rf_cv_results['mean_test_r2'][rf_best_index]:.4f}")
print(f"RF: Best MAE: {-rf_cv_results['mean_test_neg_mean_absolute_error'][rf_best_index]:.4f}\n")


#----------------------------------
# 5. Evaluate Test Split
#----------------------------------

# Linear Regression
linreg_y_hat = best_linreg_model.predict(x_test)
linreg_test_mse = sk.metrics.mean_squared_error(y_test, linreg_y_hat)
linreg_test_rmse = np.sqrt(linreg_test_mse)
linreg_r_squared = sk.metrics.r2_score(y_test, linreg_y_hat)
linreg_test_mae = sk.metrics.mean_absolute_error(y_test, linreg_y_hat)
print(f"Linear Regression: Test RMSE: {linreg_test_rmse}")
print(f"Linear Regression: R-Squared: {linreg_r_squared:.4f}")
print(f"Linear Regression: Test MAE: {linreg_test_mae}:.4f")

# Lasso
lasso_y_hat = best_lasso_model.predict(x_test)
lasso_test_mse = sk.metrics.mean_squared_error(y_test, lasso_y_hat)
lasso_test_rmse = np.sqrt(lasso_test_mse)
lasso_r_squared = sk.metrics.r2_score(y_test, lasso_y_hat)
lasso_test_mae = sk.metrics.mean_absolute_error(y_test, lasso_y_hat)
print(f"Lasso: Test RMSE: {lasso_test_rmse}")
print(f"Lasso: Test R-Squared: {lasso_r_squared:.4f}")
print(f"Lasso: Test MAE: {lasso_test_mae}:.4f")

# Random Forrest 
rf_y_hat = best_rf_model.predict(x_test)
rf_test_mse = sk.metrics.mean_squared_error(y_test, rf_y_hat)
rf_test_rmse = np.sqrt(rf_test_mse)
rf_r_squared = sk.metrics.r2_score(y_test, rf_y_hat)
rf_test_mae = sk.metrics.mean_absolute_error(y_test, rf_y_hat)
print(f"Random Forest: Test RMSE: {rf_test_rmse}")
print(f"Random Forest: Test R-Squared: {rf_r_squared:.4f}")
print(f"Random Forest: Test MAE: {rf_test_mae}:.4f")


# Summarize the results
summary = {
  "Model": ["Linear Regression", "Lasso", "Random Forest"],
  "Train RMSE": [-linreg_wf.best_score_, -lasso_wf.best_score_, -rf_wf.best_score_],
  "Train R2": [
    linreg_cv_results['mean_test_r2'][linreg_best_index],
    lasso_cv_results['mean_test_r2'][lasso_best_index],
    rf_cv_results['mean_test_r2'][rf_best_index]
  ],
  "Train MAE": [
    -linreg_cv_results['mean_test_neg_mean_absolute_error'][linreg_best_index],
    -lasso_cv_results['mean_test_neg_mean_absolute_error'][lasso_best_index],
    -rf_cv_results['mean_test_neg_mean_absolute_error'][rf_best_index]
  ],
  "Train MSE": [
    -linreg_cv_results['mean_test_mse'][linreg_best_index],
    -lasso_cv_results['mean_test_mse'][lasso_best_index],
    -rf_cv_results['mean_test_mse'][rf_best_index]
  ],
  "Train MPE": [
    linreg_cv_results['mean_test_mpe'][linreg_best_index],
    lasso_cv_results['mean_test_mpe'][lasso_best_index],
    rf_cv_results['mean_test_mpe'][rf_best_index]
  ],
  "Test RMSE": [linreg_test_rmse, lasso_test_rmse, rf_test_rmse],
  "Test R2": [linreg_r_squared, lasso_r_squared, rf_r_squared],
  "Test MAE": [linreg_test_mae, lasso_test_mae, rf_test_mae],
  "Test MSE": [linreg_test_mse, lasso_test_mse, rf_test_mse],
  "Test MPE": [
    np.mean((y_test - linreg_y_hat) / y_test),
    np.mean((y_test - lasso_y_hat) / y_test),
    np.mean((y_test - rf_y_hat) / y_test)
  ]
}

summary_df = pd.DataFrame(summary)
print(summary_df)

# Save the results on the splits
summary_df.to_csv(output + "Result_Models/ML_CoVar_Regression_absolute_results.csv", index=False)

# Save the models locally
joblib.dump(best_linreg_model, output + "Models/linreg_CoVar_regression_absolute.pkl")
joblib.dump(best_lasso_model, output + "Models/lasso_CoVar_regression_absolute.pkl")
joblib.dump(best_rf_model, output + "Models/rf_CoVar_regression_absolute.pkl")


# Combine feature importances into one DataFrame
feature_importances = pd.DataFrame()

# For Linear Regression (coefficients)
linreg_features = pd.DataFrame({
    "Feature": x_test.columns,
    "Importance": best_linreg_model.named_steps["model"].coef_,
    "Model": "Linear Regression"
})
feature_importances = pd.concat([feature_importances, linreg_features], ignore_index=True)

# For Lasso (coefficients)
lasso_features = pd.DataFrame({
    "Feature": x_test.columns,
    "Importance": best_lasso_model.named_steps["model"].coef_,
    "Model": "Lasso"
})
feature_importances = pd.concat([feature_importances, lasso_features], ignore_index=True)

# For Random Forest (feature importances)
rf_features = pd.DataFrame({
    "Feature": x_test.columns,
    "Importance": best_rf_model.feature_importances_,
    "Model": "Random Forest"
})
feature_importances = pd.concat([feature_importances, rf_features], ignore_index=True)

# Save the combined feature importances
feature_importances.to_csv(output + "Feature_Importance/features_CoVar_regression_absolute"".csv", index=False)


# Combine predictions into one DataFrame
predictions = pd.DataFrame({
    "gvkey": test["gvkey"].values,
    "date": test["date"].values,
    "Actual": y_test.values,
    "Linear Regression": linreg_y_hat,
    "Lasso": lasso_y_hat,
    "Random Forest": rf_y_hat
})

# Save the predictions to a CSV file
predictions.to_csv(output + "Predictions_Models/ML_CoVar_Regression_absolute_predictions.csv", index=False)

