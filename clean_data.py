#----------------------------------
# 1. Set-Up
#----------------------------------



# Load Libraries
import pandas as pd
import os

# Define Paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data Sets
bank_data_q_raw = pd.read_parquet(input + "Raw/bank_quarterly_data_raw.parquet")
stock_returns_raw = pd.read_parquet(input + "Raw/bank_daily_returns_raw.parquet")
mapping_table = pd.read_parquet(input + "mapping_table.parquet")
market_returns_raw = pd.read_parquet(input + "Raw/market_returns_raw.parquet")
ted_spread_q_raw = pd.read_parquet(input + "Raw/ted_spread_q_raw.parquet")
ted_spread_d_raw = pd.read_parquet(input + "Raw/ted_spread_d_raw.parquet")
gdp_q_raw = pd.read_parquet(input + "Raw/gdp_q_raw.parquet")
vix_data_q_raw = pd.read_parquet(input + "Raw/vix_data_q_raw.parquet")
vix_data_d_raw = pd.read_parquet(input + "Raw/vix_data_d_raw.parquet")
exchange_rate_q_raw = pd.read_parquet(input + "Raw/exchange_rate_q_raw.parquet")
federal_funds_rate_q_raw = pd.read_parquet(input + "Raw/federal_funds_rate_q_raw.parquet")
t_bill_q_raw_delta = pd.read_parquet(input + "Raw/t_bill_q_delta.parquet")
t_bill_d_raw_delta = pd.read_parquet(input + "Raw/t_bill_d_delta.parquet")
liquidity_spread_q_raw = pd.read_parquet(input + "Raw/liquidity_spread_q_raw.parquet")
slope_yield_curve_d_raw = pd.read_parquet(input + "Raw/yield_curve_slope_change_d_raw.parquet")
credit_spreads_d_raw = pd.read_parquet(input + "Raw/credit_spread_d_raw.parquet")
real_estate_excess_returns_d_raw = pd.read_parquet(input + "Raw/real_estate_excess_return_d_raw.parquet")


#----------------------------------
# 2. Clean Bank Data Quaterly
#---------------------------------- 

# Select all the relevant columns 
bank_data_q_clean = bank_data_q_raw[["gvkey", "conm", 
                                     "datadate", "fyearq", "fyr", "datacqtr", "fic", "naics",
                                     "atq", "niq", 
                                     "ceqq", # Common/Ordinary Equity 
                                     "ffsspq", 
                                     "ffpssq"]] 

# Convert columns to string   
bank_data_q_clean.loc[:, "naics"] = bank_data_q_clean["naics"].astype(str)

#------
## Filter the Data Set
#------

# Filter out all the irrelevant industries not starting with 52 (NAICS) (BANKS)
bank_data_q_clean = bank_data_q_clean[bank_data_q_clean["naics"].str.startswith("522")]

# Filter out all non-American Banks 
bank_data_q_clean =  bank_data_q_clean[bank_data_q_clean["fic"] == "USA"]

# Filter out all observations later than 1986 and after 2024
bank_data_q_clean = bank_data_q_clean[bank_data_q_clean["fyearq"] >= 1999]
bank_data_q_clean = bank_data_q_clean[bank_data_q_clean["fyearq"] <= 2024]


# Remove all the observations with missing values
bank_data_q_clean = bank_data_q_clean.dropna() 

# Save the file locally as csv
bank_data_q_clean.to_parquet(input + "Clean/bank_data_q_clean.parquet", 
                             index=False)

# Extract the gvkeys to filter the data
unique_gvkeys = bank_data_q_clean["gvkey"].astype(str).unique()

# Get count per quarter
count_per_quarter = bank_data_q_clean.groupby("datacqtr").size()
print(count_per_quarter)

print(bank_data_q_clean["gvkey"].unique())


#----------------------------------
# 3. Clean Stock Return Data
#----------------------------------

#-------
# a. Clean Bank Returns
#-------

# Select the relevant columns
stock_data_d_clean = stock_returns_raw[["permno", "date", "ret"]]

# Ensure the mapping table has unique permno values
mapping_table = mapping_table.drop_duplicates(subset=["permno"])

# Join mapping table on the data frame
stock_data_d_clean = pd.merge(stock_data_d_clean, mapping_table, 
                              left_on = "permno", right_on = "permno",
                              how = "left")

## Filter

# Filter out all stock where no quarterly data is availible
stock_data_d_clean = stock_data_d_clean[stock_data_d_clean["gvkey"].astype(str).isin(unique_gvkeys)]

# Turn date in to a time variable
stock_data_d_clean["date"] = pd.to_datetime(stock_data_d_clean["date"]).dt.date

# Filter out all days before 1986 and after 2024
stock_data_d_clean = stock_data_d_clean[stock_data_d_clean["date"] >= pd.to_datetime("1990-01-01").date()]
stock_data_d_clean = stock_data_d_clean[stock_data_d_clean["date"] <= pd.to_datetime("2023-12-31").date()]

# Drop observations with missing values
stock_data_d_clean = stock_data_d_clean.dropna()

# Store the results locally 
stock_data_d_clean.to_parquet(input + "Clean/stock_data_d_clean.parquet")


### Transform the stock return to quarterly data 

# Ensure the date column is datetime and sort the data
stock_data_d_clean["date"] = pd.to_datetime(stock_data_d_clean["date"])
stock_data_d_clean = stock_data_d_clean.sort_values(["gvkey", "date"])

# Compute the daily return factor (assumes a column 'return' with daily return values)
stock_data_d_clean["daily_factor"] = 1 + stock_data_d_clean["ret"]

# Group by bank identifier and resample to quarterly frequency by multiplying daily factors
stock_data_q_clean = stock_data_d_clean.groupby(["gvkey", pd.Grouper(key="date", freq="QE")])["daily_factor"].prod().reset_index()

# Calculate the quarterly return
stock_data_q_clean["quarterly_return"] = stock_data_q_clean["daily_factor"] - 1

# Optionally, drop the intermediate factor column if not needed
stock_data_q_clean = stock_data_q_clean.drop(columns=["daily_factor"])

# Convert date variable 
stock_data_q_clean["date"] = pd.to_datetime(stock_data_q_clean["date"]).dt.to_period("Q").astype(str)

# Safe the variable
stock_data_q_clean.to_parquet(input + "Clean/stock_data_q_clean.parquet")


# Calculate quarterly volatility for each stock
# Ensure the 'ret' column is numeric
stock_data_d_clean["ret"] = pd.to_numeric(stock_data_d_clean["ret"], errors="coerce")

# Group by stock and quarter, then calculate the standard deviation of daily returns
stock_volatility_q = stock_data_d_clean.groupby(
    ["gvkey", pd.Grouper(key="date", freq="Q")]
)["ret"].std().reset_index()

# Rename the columns for clarity
stock_volatility_q = stock_volatility_q.rename(columns={"ret": "quarterly_volatility"})

# Convert the date column to quarterly period format
stock_volatility_q["date"] = stock_volatility_q["date"].dt.to_period("Q").astype(str)

# Save the results to a parquet file
stock_volatility_q.to_parquet(input + "Clean/stock_volatility_q.parquet", index=False)


#-------
# b. Clean Market Returns (S&P500)
#-------

# Copy the data frame 
market_returns_d_clean = market_returns_raw

# Create a time variable 
market_returns_d_clean["caldt"] = pd.to_datetime(market_returns_d_clean["caldt"]).dt.date

# Filter out all days before 1986
market_returns_d_clean = market_returns_d_clean[market_returns_d_clean["caldt"] >= pd.to_datetime("1986-01-01").date()]

# Rename the variable 
market_returns_d_clean = market_returns_d_clean.rename(columns={"caldt": "date"})

# Save the data frame
market_returns_d_clean.to_parquet(input + "Clean/market_returns_d_clean.parquet")


#----------------------------------
# 4. Clean Macro Data for Features
#----------------------------------

#-------
# a. Clean TED Spread
#-------

# Format the date variable to be suitable for later
ted_spread_q_raw["date"] = pd.to_datetime(ted_spread_q_raw["date"]).dt.to_period("Q").astype(str)

#-------
# b. Clean GDP
#-------

# Create the growth rate of GDP
gdp_q_raw["gdp_growth"] = gdp_q_raw["gdp"] / gdp_q_raw["gdp"].shift(1) - 1
gdp_q_raw = gdp_q_raw.drop(columns=["gdp"])

# Format the date variable to fit other data frames 
gdp_q_raw["date"] = pd.to_datetime(gdp_q_raw["date"]).dt.to_period("Q").astype(str)

#-------
# c. Clean Market Return
#-------

# Ensure the 'date' column is in datetime format
market_returns_d_clean["date"] = pd.to_datetime(market_returns_d_clean["date"])

# Create a quarterly period column
market_returns_d_clean["quarter"] = market_returns_d_clean["date"].dt.to_period("Q")

# Compound the daily returns within each quarter
market_returns_q_clean = market_returns_d_clean.groupby("quarter")["vwretd"].apply(
    lambda x: (1 + x).prod() - 1
).reset_index()

# Rename column
market_returns_q_clean = market_returns_q_clean.rename(columns={"vwretd": "market_return"})

# Format 'quarter' to string for merging consistency
market_returns_q_clean["quarter"] = market_returns_q_clean["quarter"].astype(str)

#-------
# d. Clean VIX
#-------

# Change the format of the date 
vix_data_q_raw["date"] = pd.to_datetime(vix_data_q_raw["date"]).dt.to_period("Q").astype(str)
vix_data_q_raw["vix"] = vix_data_q_raw["vix"] / 100

#-------
# e. Clean Exchange Rate
#-------

# Change the format of the date 
exchange_rate_q_raw["date"] = pd.to_datetime(exchange_rate_q_raw["date"]).dt.to_period("Q").astype(str)

#-------
# f. Clean Interest Rate
#-------

# Change the format of the date 
federal_funds_rate_q_raw["date"] = pd.to_datetime(federal_funds_rate_q_raw["date"]).dt.to_period("Q").astype(str)

#-------
# g. Clean T-Bill Delta
#-------

# Change the format of the date
t_bill_q_raw_delta["date"] = pd.to_datetime(t_bill_q_raw_delta["date"]).dt.to_period("Q").astype(str)

#-------
# h. Clean Liquidity Spread
#-------

# Change the format of the date
liquidity_spread_q_raw["date"] = pd.to_datetime(liquidity_spread_q_raw["date"]).dt.to_period("Q").astype(str)

#-------
# x.  Macro Features: Putting Everything together 
#-------

# Merge all the datasets together for the Macro Features
macro_variables = pd.merge(
    ted_spread_q_raw,
    gdp_q_raw,
    left_on = "date",
    right_on = "date",
    how="left"
)
macro_variables = pd.merge(
    macro_variables,
    market_returns_q_clean,
    left_on = "date",
    right_on = "quarter",
    how="left"
)
macro_variables = pd.merge(
    macro_variables,
    vix_data_q_raw,
    left_on = "date",
    right_on = "date",
    how="left"
)
macro_variables = pd.merge(
    macro_variables,
    t_bill_q_raw_delta,
    left_on = "date",
    right_on = "date",
    how="left"
)


# Safe the file 
macro_variables.to_parquet(input + "Clean/macro_variables_clean.parquet")


#-------
# x.  CoVar Features: Putting Everything together for the covar estimation 
#-------

# Merge everything together
covar_macro_variables = pd.merge(
    ted_spread_d_raw,
    vix_data_d_raw,
    left_on="date",
    right_on="date",
    how="inner"
)
covar_macro_variables = pd.merge(
    covar_macro_variables,
    t_bill_d_raw_delta,
    left_on="date",
    right_on="date",
    how="inner"
)
covar_macro_variables = pd.merge(
    covar_macro_variables,
    market_returns_d_clean,
    left_on="date",
    right_on="date",
    how="inner"
)
covar_macro_variables = pd.merge(
    covar_macro_variables,
    slope_yield_curve_d_raw,
    left_on="date",
    right_on="date",
    how="inner"
)
covar_macro_variables = pd.merge(
    covar_macro_variables,
    credit_spreads_d_raw,
    left_on="date",
    right_on="date",
    how="inner"
)
covar_macro_variables = pd.merge(
    covar_macro_variables,
    real_estate_excess_returns_d_raw,
    left_on="date",
    right_on="date",
    how="inner"
)

# Safe the file 
covar_macro_variables.to_parquet(input + "Clean/covar_macro_variables_clean.parquet")
