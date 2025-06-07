#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import statsmodels.api as sm

# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
stock_data_d_clean = pd.read_parquet(input + "Clean/stock_data_d_clean.parquet")
mapping_table = pd.read_parquet(input + "mapping_table.parquet")
covar_macro_variables = pd.read_parquet(input + "Clean/covar_macro_variables_clean.parquet")
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")
bank_data_q_clean = pd.read_parquet(input + "Clean/bank_data_q_clean.parquet")
real_estate_returns_d_raw = pd.read_parquet(input + "Raw/real_estate_returns_d_raw.parquet")




#----------------------------------
# 2. Define Variables & Data Prep
#----------------------------------

# Set the parameter for quantiles 
quantile = 0.01
median = 0.5

# Select important Data Columns
stock_data_d_clean = stock_data_d_clean[["permno", "gvkey", "date", "ret"]]

# Extract the quarter for each day
stock_data_d_clean["quarter"] = pd.to_datetime(stock_data_d_clean["date"]).dt.to_period("Q")

# Add an Id to the quarters
stock_data_d_clean["quarter_id"] = stock_data_d_clean.groupby("quarter").ngroup() + 1

# Lag the macro-economic variables
lagged_covar_macro_variables = covar_macro_variables.copy()
for col in lagged_covar_macro_variables.columns:
    if col != "date":
        lagged_covar_macro_variables[col] = lagged_covar_macro_variables[col].shift(1)
        lagged_covar_macro_variables.rename(columns={col: f"lag_{col}"}, inplace=True)
        

# Check for enough data for Covar Derivation!
# Aggregate the number of observations of daily returns for each bank
bank_observations = stock_data_d_clean.groupby("gvkey")["ret"].count()

# Filter all observations below a threshold
bank_observations = bank_observations[bank_observations >= 260]

# Extract the bank keys
bank_keys = bank_observations.index


## Define System Returns
# Filter the bank returns to include only banks in the unique_banks_final list
unique_banks_final = covar_final["gvkey"].unique()
bank_returns = stock_data_d_clean[stock_data_d_clean["gvkey"].isin(unique_banks_final)]
bank_returns = bank_returns[bank_returns["gvkey"].isin(bank_keys)]

# Aggregate system returns by taking the mean return
bank_returns = bank_returns.groupby("date")["ret"].mean().reset_index()

# Store the aggregated bank returns:
bank_returns.to_csv(input + "Clean/bank_returns_system.csv")



# Derive the real-estate returns
# Filter real estate returns to start from a specific date
real_estate_returns_d_raw = real_estate_returns_d_raw[real_estate_returns_d_raw["date"] >= "1990-01-02"]

# Align dates between real estate returns and bank returns
# Ensure 'date' columns are of the same type
real_estate_returns_d_raw["date"] = pd.to_datetime(real_estate_returns_d_raw["date"])
bank_returns["date"] = pd.to_datetime(bank_returns["date"])

aligned_data = pd.merge(real_estate_returns_d_raw, bank_returns, on="date", how="inner")

# Calculate the real estate excess return
covar_macro_variables["real_estate_excess_return"] = aligned_data["real_estate_return"] - aligned_data["ret"]



#----------------------------------
# 3. For-Loop to derive CoVar
#----------------------------------

# Create an empty data frames to store the results
results_var = pd.DataFrame(np.nan,
    index=stock_data_d_clean["quarter"].unique(),
    columns=bank_keys, dtype = float)
results_delta_covar = pd.DataFrame(np.nan,
    index=stock_data_d_clean["quarter"].unique(),
    columns=bank_keys, dtype = float)

# Define Function for the q-regression Estimation
def q_regression(y, X, q):
    y = y.astype(float)
    X = X.astype(float)
    model = sm.QuantReg(y, X)
    results = model.fit(q=q, max_iter=5000)
    return results


# For loop to iterate over the number of quarters 
for i in bank_keys:
    
    # Select the relevant stock returns for this firm
    y = stock_data_d_clean[stock_data_d_clean["gvkey"] == i]
    
    # Handle multiple permno per gvkey by keeping the one with the longest sequence
    if y["permno"].nunique() > 1:
        longest_permno = y.groupby("permno")["date"].count().idxmax()
        y = y[y["permno"] == longest_permno]
    
    # Filter the dates for the macro variables that are in the stock data as well
    #tmp_M = lagged_covar_macro_variables[pd.to_datetime(lagged_covar_macro_variables["date"]).isin(pd.to_datetime(tmp["date"]))]
    X = lagged_covar_macro_variables
    
    # Align and index X and y
    y.set_index(pd.to_datetime(y["date"]), inplace=True)
    X.set_index(pd.to_datetime(X["date"]), inplace=True)
    
    # Drop unimportant columns
    X = X.drop(columns=["date"])
    y = y[["ret"]]
    
    # Remove firms with less than 260 obs --> similar to paper
    if len(y) < 260:
        continue
    
    # Retrieve common intersection
    intersection = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[intersection]
    X = sm.add_constant(X)
    y = y.loc[intersection]
    
    # Retrive the system returns
    y_sys = bank_returns
    y_sys.set_index(pd.to_datetime(y_sys["date"]), inplace = True)
    y_sys = y_sys.drop(columns = ["date"])
    y_sys = y_sys.loc[intersection]
    
    # Create Banks VAR at q
    var_q = q_regression(y, X, quantile)
    var_m = q_regression(y, X, median)
    fitted_var_q = var_q.fittedvalues
    fitted_var_m = var_m.fittedvalues
    
    # Adjust the regressors data frame for the VaR
    X2 = pd.concat([X, y], axis=1)
    
    # Create Banks CoVaR
    covar_q = q_regression(y_sys, X2, quantile)
    covar_m = q_regression(y_sys, X2, median)
    
    # Use the fitted VAR values and the covar models to 
    # get the CoVar estimations    
    # Adjust the regressors data frame for prediction
    X_pred_q = pd.concat([X, fitted_var_q.rename("fitted_var_q")], axis=1)
    X_pred_m = pd.concat([X, fitted_var_m.rename("fitted_var_m")], axis=1)

    # Predict CoVaR using the covar_q model
    covar_pred_q = covar_q.predict(X_pred_q)
    covar_pred_m = covar_q.predict(X_pred_m)
    
    # Derive Delta Covar
    delta_covar = covar_pred_q - covar_pred_m
    
    # Multiply by -1 and 100 to get the return in %
    delta_covar = delta_covar * (-1) * 100
    
    # Aggregate delta CoVaR to quarterly estimations
    delta_covar_quarterly = delta_covar.resample("QE").mean()

    # Stop

    # Multiply by -1 and 100 to get the return in %
    var_daily = fitted_var_q * (-1) * 100

    # Aggregate the VaR to quarterly estimations
    var_q_quarterly = var_daily.resample("QE").mean()

    # Store the aggregated delta CoVaR values in the results data frame
    # Ensure the index aligns with results_delta_covar
    for quarter, value in delta_covar_quarterly.items():
        if quarter in results_delta_covar.index:
            results_delta_covar.loc[quarter, i] = value
    print(i)

    # Store the quarterly VaR estimates
    for quarter, value in var_q_quarterly.items():
        if quarter in results_var.index:
            results_var.loc[quarter, i] = value
    
# Save all the results as parquet files! 
results_delta_covar.to_parquet(input + "Clean/results_delta_covar.parquet")
results_var.to_parquet(input + "Clean/results_var.parquet")

