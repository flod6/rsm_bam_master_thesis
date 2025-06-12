#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import wrds
import os
import fredapi

# Define paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"


# Retrieve credentials from environment variables
username = os.getenv("WRDS_USERNAME") or # Please insert your WRDS username here
password = os.getenv("WRDS_PASSWORD") or # Please insert your WRDS password here

conn = wrds.Connection(wrds_username=username, wrds_password=password)


#----------------------------------
# 1. Download Banking Data
#----------------------------------

# Download quarterly data for banks 
bank_quarterly_data = conn.raw_sql("""
    SELECT * 
    FROM comp.bank_fundq 
        WHERE datafmt = 'STD'
    """)

# Save quarterly data to CSV
bank_quarterly_data.to_parquet(input + "Raw/bank_quarterly_data_raw.parquet", index=False)

# Extract unique GVKEYs for banks
bank_gvkeys = bank_quarterly_data["gvkey"].unique() #2403

#----------------------------------
# 2. Download Stock Return Data
#----------------------------------

#-------
# a. Download Bank Returns
#-------

# Download linking table to convert Compustat GVKEYs into CRSP PERMNOs
link_data = conn.raw_sql(f"""
    SELECT gvkey, lpermno as permno
    FROM crsp.ccmxpf_linktable
    WHERE gvkey IN ({','.join("'" + str(gvkey) + "'" for gvkey in bank_gvkeys)})
      AND usedflag = 1
      AND linktype IN ('LU', 'LC') 
""") # Remove link type probably it helps 

# Save the mapping table for later steps
link_data.to_parquet(input + "mapping_table.parquet")

# Extract unique PERMNOs
bank_permnos = link_data["permno"].unique()

# Download daily stock returns from the CRSP daily file for the identified banks
bank_daily_returns = conn.raw_sql(f"""
    SELECT *
    FROM crsp.dsf
    WHERE permno IN ({','.join(str(permno) for permno in bank_permnos)}) 
""") # 2104

# Unique returns
unique_returns = bank_daily_returns["permno"].unique()

# Save daily returns data to CSV
bank_daily_returns.to_parquet(input + "Raw/bank_daily_returns_raw.parquet", index=False)


#-------
# b. Download Market Returns (S&P500)
#-------

# Download S&P 500 index returns from the CRSP daily index file (dsi)
market_returns = conn.raw_sql("""
    SELECT date as caldt, vwretd
    FROM crsp.dsi
""")

# Save S&P 500 returns data to CSV
market_returns.to_parquet(input + "Raw/market_returns_raw.parquet", index=False)


# Close the connection
conn.close()

#----------------------------------
# 3. Download Macro Data
#----------------------------------

#-------
# a. Download TED Spread
#-------

# Connect with API key
fred_api_key = os.getenv("FRED_API_KEY") or # Please insert your FRED API key here
fred = fredapi.Fred(api_key=fred_api_key)

# 1. Download TED Spread data from FRED
ted_spread_data = fred.get_series("TEDRATE", observation_start="1986-01-01")

# Convert TED Spread data to DataFrame
ted_spread_data = ted_spread_data.reset_index()
ted_spread_data.columns = ["date", "ted_spread"]

# 2. Extract the last date where TED Spread data is available
last_ted_date = ted_spread_data["date"].max()

# 3. Download T-Bill and SOFR data for the period after the last TED Spread date
t_bill_data = fred.get_series("DTB3", observation_start=last_ted_date + pd.Timedelta(days=1))
sofr_data = fred.get_series("SOFR", observation_start=last_ted_date + pd.Timedelta(days=1))

# Convert T-Bill and SOFR data to DataFrames
t_bill_data = t_bill_data.reset_index()
t_bill_data.columns = ["date", "t_bill_rate"]
sofr_data = sofr_data.reset_index()
sofr_data.columns = ["date", "sofr_rate"]

# Merge T-Bill and SOFR data
approx_ted_spread_data = pd.merge(t_bill_data, sofr_data, on="date", how="inner")

# 4. Derive the approximation of the TED Spread
approx_ted_spread_data["ted_spread"] = approx_ted_spread_data["sofr_rate"] - approx_ted_spread_data["t_bill_rate"]

# 5. Merge the original TED Spread data with the approximation
full_ted_spread_data = pd.concat([ted_spread_data, approx_ted_spread_data[["date", "ted_spread"]]])

# 6. Save the daily TED Spread data as a parquet file
full_ted_spread_data.to_parquet(input + "Raw/ted_spread_d_raw.parquet", index=False)

# 7. Aggregate the TED Spread data to quarterly frequency
ted_spread_quarterly = full_ted_spread_data.set_index("date")["ted_spread"].resample("QE").mean()

# Convert the quarterly data to a DataFrame
ted_spread_quarterly_df = ted_spread_quarterly.reset_index()
ted_spread_quarterly_df.columns = ["date", "ted_spread"]

# Save the quarterly TED Spread data as a parquet file
ted_spread_quarterly_df.to_parquet(input + "Raw/ted_spread_q_raw.parquet", index=False)
#-------
# b. GDP Data 
#-------

# Download quarterly GDP data from FRED
gdp_data = fred.get_series("GDP", observation_start="1986-01-01", observation_end="2024-12-31")

# Convert the GDP data to a DataFrame
gdp_data_q_raw = gdp_data.reset_index()
gdp_data_q_raw.columns = ["date", "gdp"]

# Save quarterly GDP data to CSV
gdp_data_q_raw.to_parquet(input + "Raw/gdp_q_raw.parquet", index=False)


#-------
# c. VIX Index 
#-------

# Download daily VIX data from FRED
vix_data = fred.get_series("VIXCLS", observation_start="1986-01-01")

# Convert the VIX data to a DataFrame
vix_data_df = vix_data.reset_index()
vix_data_df.columns = ["date", "vix"]

# Save the daily VIX data to a parquet file
vix_data_df.to_parquet(input + "Raw/vix_data_d_raw.parquet", index=False)

# Resample VIX data to quarterly frequency by taking the mean
vix_data_q_raw = vix_data.resample("QE").mean()

# Convert the quarterly data to a DataFrame
vix_data_q_raw = vix_data_q_raw.reset_index()
vix_data_q_raw.columns = ["date", "vix"]

# Save quarterly VIX data to CSV
vix_data_q_raw.to_parquet(input + "Raw/vix_data_q_raw.parquet", index=False)


#-------
# d. Exchange Rate Dollar / Euro
#-------

# Download daily Dollar/Euro exchange rate data from FRED
exchange_rate_data = fred.get_series("DEXUSEU", observation_start="1986-01-01")

# Resample exchange rate data to quarterly frequency by taking the mean
exchange_rate_q_raw = exchange_rate_data.resample("QE").mean()

# Convert the quarterly data to a DataFrame
exchange_rate_q_raw = exchange_rate_q_raw.reset_index()
exchange_rate_q_raw.columns = ["date", "exchange_rate"]

# Save quarterly exchange rate data to CSV
exchange_rate_q_raw.to_parquet(input + "Raw/exchange_rate_q_raw.parquet", index=False)


#-------
# d. Federal Funding Rate
#-------

# Download daily Federal Funds Rate data from FRED
federal_funds_rate_data = fred.get_series("FEDFUNDS", observation_start="1986-01-01")

# Resample Federal Funds Rate data to quarterly frequency by taking the mean
federal_funds_rate_q_raw = federal_funds_rate_data.resample("QE").mean()

# Convert the quarterly data to a DataFrame
federal_funds_rate_q_raw = federal_funds_rate_q_raw.reset_index()
federal_funds_rate_q_raw.columns = ["date", "federal_funds_rate"]

# Save quarterly Federal Funds Rate data to CSV
federal_funds_rate_q_raw.to_parquet(input + "Raw/federal_funds_rate_q_raw.parquet", index=False)


#-------
# e. T-Bill Delta 
#-------

# Download daily 3-Month Treasury Bill data from FRED
t_bill_data = fred.get_series("DTB3", observation_start="1986-01-01")

# Convert data to df
t_bill_data_df = t_bill_data.reset_index()
t_bill_data_df.columns = ["date", "t_bill_delta"]

# Diff the delta
t_bill_data_df["t_bill_delta"] = t_bill_data_df["t_bill_delta"].diff()

# Store the daily data
t_bill_data_df.to_parquet(input + "Raw/t_bill_d_delta.parquet", index = False)

# Resample T-Bill data to quarterly frequency by taking the mean
t_bill_q_raw = t_bill_data.resample("QE").mean()

# Calculate the quarterly delta (difference) of the T-Bill rate
t_bill_q_raw_delta = t_bill_q_raw.diff()

# Convert the quarterly delta data to a DataFrame
t_bill_q_raw_delta = t_bill_q_raw_delta.reset_index()
t_bill_q_raw_delta.columns = ["date", "t_bill_delta"]

# Save quarterly T-Bill delta data to CSV
t_bill_q_raw_delta.to_parquet(input + "Raw/t_bill_q_delta.parquet", index=False)


#-------
# f. Litquidity Spread
#-------

# Download daily 3-Month Repo Rate data from FRED
repo_rate_data = fred.get_series("RIFSPPFAAD90NB", observation_start="1986-01-01")

# Resample Repo Rate data to quarterly frequency by taking the mean
repo_rate_q_raw = repo_rate_data.resample("QE").mean()

# Calculate the liquidity spread (Repo Rate - T-Bill Rate)
liquidity_spread_q_raw = repo_rate_q_raw - t_bill_q_raw

# Convert the liquidity spread data to a DataFrame
liquidity_spread_q_raw = liquidity_spread_q_raw.reset_index()
liquidity_spread_q_raw.columns = ["date", "liquidity_spread"]

# Save quarterly liquidity spread data to CSV
liquidity_spread_q_raw.to_parquet(input + "Raw/liquidity_spread_q_raw.parquet", index=False)


#-------
# g. Term Spread
#-------

# Download daily 10-Year Treasury Constant Maturity Rate (long-term bond yield) from FRED
long_term_yield_data = fred.get_series("DGS10", observation_start="1986-01-01")

# Download daily 3-Month Treasury Bill data from FRED
short_term_yield_data = fred.get_series("DTB3", observation_start="1986-01-01")

# Calculate the daily slope of the yield curve (Long-Term Yield - Short-Term Yield)
yield_curve_slope = long_term_yield_data - short_term_yield_data

# Calculate the daily change in the slope of the yield curve
yield_curve_slope_change = yield_curve_slope.diff()

# Resample the daily change in the slope of the yield curve to quarterly frequency by taking the mean
#yield_curve_slope_change_q = yield_curve_slope_change.resample("QE").mean()

# Convert the data to a DataFrame
yield_curve_slope_change_df = yield_curve_slope_change.reset_index()
yield_curve_slope_change_df.columns = ["date", "yield_curve_slope_change"]

# Save quarterly change in the slope of the yield curve data to CSV
yield_curve_slope_change_df.to_parquet(input + "Raw/yield_curve_slope_change_d_raw.parquet", index=False)

#-------
# h. Credit Spread
#-------

# Download daily Moody's Baa-rated bond yield data from FRED
baa_yield_data = fred.get_series("DBAA", observation_start="1986-01-01")

# Download daily 10-Year Treasury Constant Maturity Rate data from FRED
ten_year_treasury_yield_data = fred.get_series("DGS10", observation_start="1986-01-01")

# Align indices and drop missing values before calculating the daily credit spread
credit_spread = baa_yield_data.align(ten_year_treasury_yield_data, join='inner')[0] - \
                baa_yield_data.align(ten_year_treasury_yield_data, join='inner')[1]

# Calculate the daily change in the credit spread
credit_spread_change = credit_spread.diff()

# Convert to data frame 
credit_spread_change_df = credit_spread_change.reset_index()
credit_spread_change_df.columns = ["date", "credit_spread_change"]

# Store the data
credit_spread_change_df.to_parquet(input + "Raw/credit_spread_d_raw.parquet", index = False)

# Resample the daily change in the credit spread to quarterly frequency by taking the mean
#credit_spread_change_q = credit_spread_change.resample("QE").mean()

# Convert the quarterly data to a DataFrame
#credit_spread_change_q_df = credit_spread_change_q.reset_index()
#credit_spread_change_q_df.columns = ["date", "credit_spread_change"]

# Save quarterly change in credit spread data to CSV
#credit_spread_change_q_df.to_parquet(input + "credit_spread_change_q.parquet", index=False)

#-------
# h. Real Estate Returns
#-------

# Re-open WRDS connection for real estate data
conn = wrds.Connection(wrds_username=username, wrds_password=password)

# Retrieve GVKEYs for real estate companies (SIC starting with 65 or 66)
re_names = pd.read_sql_query(
    """
    SELECT DISTINCT gvkey
    FROM   comp.names
    WHERE  sic Like '65%' OR sic LIKE '66%'
    """,
    conn.connection.connection           # or after fixing packages: conn
)
re_gvkeys = re_names['gvkey'].astype(str).unique()

# Map GVKEYs to CRSP PERMNOs using the linking table
re_link_data = conn.raw_sql(f"""
    SELECT gvkey, lpermno AS permno
    FROM crsp.ccmxpf_linktable
    WHERE gvkey IN ({','.join("'" + gvkey + "'" for gvkey in re_gvkeys)})
      AND usedflag = 1
      AND linktype IN ('LU', 'LC')
""")
re_permnos = re_link_data['permno'].unique()

# Download daily returns for the identified real estate companies
re_daily_returns = conn.raw_sql(f"""
    SELECT date, permno, ret
    FROM crsp.dsf
    WHERE permno IN ({','.join(str(permno) for permno in re_permnos)})
""")

conn.close()


# Convert date column and compute the equally weighted daily return for the real estate sector
re_daily_returns["date"] = pd.to_datetime(re_daily_returns["date"])
re_sector = re_daily_returns.groupby("date")["ret"].mean().reset_index()
re_sector.rename(columns={"ret": "real_estate_return"}, inplace=True)

# Store the returns
re_sector.to_parquet(input + "Raw/real_estate_returns_d_raw.parquet")


# Aggregate the bank returns to get the return of the financial 
# system 
bank_daily_returns["date"] = pd.to_datetime(bank_daily_returns["date"])
bank_sector = bank_daily_returns.groupby("date")["ret"].mean().reset_index()
bank_sector.rename(columns={"ret": "bank_return"}, inplace=True)

# Store the data frame
bank_sector.to_parquet(input + "Raw/bank_sector_returns_d_raw.parquet", index = False)

# Merge real estate and market returns on date and calculate the excess return 
# (real estate return minus market return)
merged_returns = pd.merge(re_sector, bank_sector, on= "date", how = "inner")
merged_returns["real_estate_excess_return"] = merged_returns['real_estate_return'] - merged_returns['bank_return']

# Save the excess return data to disk
merged_returns[["date", "real_estate_excess_return"]].to_parquet(input + "Raw/real_estate_excess_return_d_raw.parquet", index=False)

