#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns


# Define paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
bank_data_q_clean = pd.read_parquet(input + "Clean/bank_data_q_clean.parquet")
market_returns_d_clean = pd.read_parquet(input + "Clean/market_returns_d_clean.parquet")
stock_data_q_clean = pd.read_parquet(input + "Clean/stock_data_q_clean.parquet")
results_covar = pd.read_parquet(input + "Clean/results_delta_covar.parquet")
macro_variables = pd.read_parquet(input + "Clean/macro_variables_clean.parquet")
results_var = pd.read_parquet(input + "Clean/results_var.parquet")


# Turn warnings off
warnings.filterwarnings("ignore")

#----------------------------------
# 2. Prepare Macro Variables
#----------------------------------

# Lag all variables by one quarter except for date
lagged_macro_variables = pd.concat([
    macro_variables[['date']],
    macro_variables.drop('date', axis=1).shift(1).rename(columns=lambda x: f'lag_{x}')
], axis=1)

# Remove all observations before 2000
lagged_macro_variables = lagged_macro_variables[lagged_macro_variables["date"] >= "1999Q1"]


#----------------------------------
# 3. Prepare Firm Features  
#----------------------------------


# Adjust name of the date column in bank_data_q_clean
bank_data_q_clean = bank_data_q_clean.rename(columns={"datacqtr": "date"})


# Remove banks with negative equity
bank_data_q_clean = bank_data_q_clean[bank_data_q_clean["ceqq"] >= 1]
bank_data_q_clean = bank_data_q_clean[bank_data_q_clean["atq"] > 0]


# Create the firm specific features
bank_data_q_clean["size"] = np.log1p(bank_data_q_clean["atq"])
bank_data_q_clean["roa"] =  (bank_data_q_clean["niq"] / bank_data_q_clean["atq"])
bank_data_q_clean["leverage"] = bank_data_q_clean["atq"] / bank_data_q_clean["ceqq"]
bank_data_q_clean["interbank_exposure"] = bank_data_q_clean["ffsspq"] / bank_data_q_clean["atq"]

# Take the logerithm of the interbank borrowing


count_per_quarter = bank_data_q_clean.groupby("date").size()
print(count_per_quarter) # Just to check

count_per_quarter = stock_data_q_clean.groupby("date").size()
print(count_per_quarter) # Just to check

# Select only the columns important for the model
bank_features = bank_data_q_clean[[
    "date",
    "gvkey",
    "conm",
    "size",
    "roa",
    "leverage",
    "interbank_exposure"
]]

# Remove Obersavtions with incomplete data
bank_features = bank_features.dropna()

count_per_quarter = bank_features.groupby("date").size()
print(count_per_quarter) # Just to check


#-------------
# Lag the data
#-------------

# Data is lagged in a way that if there are no 
# subsequent quarter in the data frame then the row is skipped 
# and it is continued with the next row
def lag_with_gap(df):
    # Ensure proper ordering and create a quarter period variable.
    df = df.sort_values(["gvkey", "date"]).copy()
    df["period"] = pd.PeriodIndex(df["date"], freq="Q")
    # Determine which columns to lag.
    cols_to_lag = [col for col in df.columns if col not in ["gvkey", "conm", "date", "period"]]

    def lag_group(g):
        # Sort each group by date.
        g = g.sort_values("date").copy()
        # Shift the selected columns.
        shifted = g[cols_to_lag].shift(1)
        # Calculate the quarter difference between the current and previous rows.
        period_diff = g["period"].astype(int) - g["period"].shift(1).astype(int)
        # Where the current quarter is exactly 1 period after the previous one, the lag is valid.
        valid = period_diff.eq(1)
        # Use shifted values if valid; otherwise assign missing values.
        g.loc[:, cols_to_lag] = shifted.where(valid)
        return g

    # Apply the custom lag function to each institution.
    df = df.groupby("gvkey", group_keys=False).apply(lag_group, include_groups = True)
    return df.drop(columns=["period"])

bank_features_lagged = lag_with_gap(bank_features)

# Change the names of the variables to have the prefix lag_
bank_features_lagged = bank_features_lagged.rename(columns={col: f'lag_{col}' for col in bank_features_lagged.columns if col not in ['date', 'gvkey', 'conm']})

# Drop missing values
bank_features_lagged = bank_features_lagged.dropna()

# Print the number of banks per quarter
count_per_quarter = bank_features_lagged.groupby("date").size()
print(count_per_quarter) # Just to check


#----------------------------------
# 4. Assemble Everything
#----------------------------------

# Join the lagged macro features on the lagged bank data
all_features_lagged = pd.merge(bank_features_lagged, lagged_macro_variables,
                               left_on = "date", right_on = "date",
                               how = "left")

count_per_quarter = all_features_lagged.groupby("date").size()
print(count_per_quarter) # Just to check



#-------------
# VaR
#-------------

results_var = results_var.reset_index().melt(
    id_vars="index",
    var_name="gvkey",
    value_name="var"
).rename(columns={"index": "date"}).sort_values(
    ["date", "gvkey"]
).reset_index(drop=True)

results_var = results_var.sort_values(by=["gvkey", "date"]).reset_index(drop=True)

# Lag the variable
tmp = lag_with_gap(results_var)
results_var["lag_var"] = tmp["var"]
results_var = results_var.drop(columns=["var"])

# Join the remaining features
results_var["date"] = results_var["date"].astype(str)
all_features_lagged["date"] = all_features_lagged["date"].astype(str)
all_features_lagged = pd.merge(all_features_lagged, results_var,
                       left_on = ["date", "gvkey"],
                       right_on = ["date", "gvkey"],
                       how = "left")

#-------------
# CoVar
#-------------

# Transform the data frame to join later
results_covar = results_covar.reset_index().melt(
    id_vars="index",
    var_name="gvkey",
    value_name="covar"
).rename(columns={"index": "date"}).sort_values(
    ["date", "gvkey"]
).reset_index(drop=True)


results_covar = results_covar.sort_values(by=["gvkey", "date"]).reset_index(drop=True)

# Calculate the change in CoVar between quarter
#results_covar["diff_delta_covar"] = results_covar.groupby("gvkey")["covar"].diff()

# Lag Covar to have another predictor
tmp = lag_with_gap(results_covar)
results_covar["lag_covar"] = tmp["covar"]

# Get the difference
results_covar["diff_delta_covar"] = results_covar["covar"] - results_covar["lag_covar"]
tmp = lag_with_gap(results_covar)
results_covar["lag_diff_delta_covar"] = tmp["diff_delta_covar"]


# Arrange the data frame by gvkey and date
results_covar = results_covar.sort_values(by=["gvkey", "date"]).reset_index(drop=True)

# Join the features on the Covar Data
results_covar["date"] = results_covar["date"].astype(str)
all_features_lagged["date"] = all_features_lagged["date"].astype(str)
covar_final = pd.merge(all_features_lagged, results_covar,
                       left_on = ["date", "gvkey"],
                       right_on = ["date", "gvkey"],
                       how = "left")

# Drop missing values
covar_final = covar_final.dropna()
#covar_final = covar_final.drop(columns=["covar"])

count_per_quarter = covar_final.groupby("date").size()
print(count_per_quarter) # Just to check


count_per_quarter.mean()


# Join on ffsspq and ffsspq to generate the adjected matrix and store in seperate dataframe
# to generate the interbank exposures
interbank_vars = pd.merge(covar_final, bank_data_q_clean[["date", "gvkey", "conm", "ffsspq", "ffpssq"]],
                       left_on = ["date", "gvkey", "conm"],
                       right_on = ["date", "gvkey", "conm"],
                       how = "left")

interbank_vars = interbank_vars[["date", "gvkey", "ffsspq", "ffpssq"]].copy()


# Check for duplicates in gvkey and date in interbank_vars
duplicates = bank_data_q_clean.duplicated(subset=["gvkey", "date"], keep=False)

# Print the duplicates if any
if duplicates.any():
    print("Duplicates found in interbank_vars:")
    print(interbank_vars[duplicates])
else:
    print("No duplicates found in interbank_vars.")


# Filter all observations before 2001
covar_final = covar_final[covar_final["date"] >= "2000Q1"]
covar_final = covar_final.drop(columns=["lag_interbank_exposure"])


# Store the results
covar_final.to_parquet(input + "Clean/covar_final.parquet")
interbank_vars.to_parquet(input + "Clean/interbank_vars.parquet")


# Just to check
covar_final_arranged = covar_final.sort_values(by=["gvkey", "date"]).reset_index(drop=True)


## Make some descriptive statistics for for the result section

# Arrange the oder of the variable for the descriptive statistics
covar_final = covar_final[["covar", "lag_size", "lag_roa", "lag_leverage", "lag_var", "lag_covar",
                           "lag_ted_spread", "lag_gdp_growth", "lag_market_return", "lag_vix", "lag_t_bill_delta",
                           "gvkey", "conm", "date", "diff_delta_covar", "lag_diff_delta_covar"]].copy()

# Create subset of the predictors
covar_predictors = covar_final.drop(columns=["diff_delta_covar", "date", 
                                             "gvkey", "conm", "lag_diff_delta_covar"])
covar_predictors.describe().transpose()


# Store the table 
covar_predictors.describe().transpose().to_csv(output + "Descriptives/covar_predictors.csv", index=True)

# Make descriptives for the robustness part
covar_change = covar_final[["diff_delta_covar", "lag_diff_delta_covar"]].copy()

covar_change.describe().transpose()
covar_change.describe().transpose().to_csv(output + "Descriptives/covar_change.csv", index=True)



# Some further explanatory analysis: 


# Generate a correlation matrix for covar_predictors
correlation_matrix = covar_predictors.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Covar Predictors")
plt.show()


# Calculate the average of diff_delta_covar for each quarter
average_diff_delta_covar = covar_final.groupby("date")["diff_delta_covar"].mean().reset_index()

# Plot the average diff_delta_covar over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=average_diff_delta_covar, x="date", y="diff_delta_covar", marker="o", color="blue")
plt.xlabel("Date")
plt.ylabel("Average diff_delta_covar")
plt.title("Average diff_delta_covar per Quarter")
plt.xticks(ticks=range(0, len(average_diff_delta_covar), 4), labels=average_diff_delta_covar["date"][::4], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Calculate the average and standard deviation of "covar" for each quarter
covar_stats = covar_final.groupby("date")["covar"].agg(["mean", "std"]).reset_index()
covar_stats["upper_bound"] = covar_stats["mean"] + covar_stats["std"]
covar_stats["lower_bound"] = covar_stats["mean"] - covar_stats["std"]

# Plot the average covar with 1SD bounds over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=covar_stats, x="date", y="mean", label="Average Covar", color="blue"#, marker="o"
             )
plt.fill_between(covar_stats["date"], covar_stats["lower_bound"], covar_stats["upper_bound"], 
                 color="blue", alpha=0.2, label="1 SD Range")
plt.xlabel("Date")
plt.ylabel("Covar")
#plt.title("Average Covar with 1 SD Bounds per Quarter")
plt.xticks(ticks=range(0, len(covar_stats), 4), labels=covar_stats["date"][::4], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the last figure to a file
plt.savefig(output + "Final_Figures/average_covar_with_bounds.png", dpi=300, bbox_inches="tight")

plt.show()

