#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
library(NetworkRiskMeasures)
library(tidyverse)
library(arrow)

# Define Paths
input  <-  "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output  <-  "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
bank_data_q_clean  <- read_parquet(paste0(input, "Clean/interbank_vars.parquet"))

#----------------------------------
# 2. Define Variables & Data Prep for CoVar
#----------------------------------

# Extract the different Quarter that exist
quarters <- unique(as.character(bank_data_q_clean$date))

# Order the quarter in chronological order
key <- as.integer(substr(quarters, 1, 4)) * 4 +
  as.integer(substr(quarters, 6, 6))

quarters <- quarters[order(key)]


######## ADJUST HERE WHEN FINAL ########

#quarters <- quarters[1:10]


#----------------------------------
# 2. Clean Folder 
#----------------------------------

# Delete the folder where the adjacency matrices are stored
# that for the case that the period changes there are no problems 
# with old matrices 
#unlink(file.path(paste0(input, "Adjacency_Matrices/")), recursive = TRUE)


# Create a new folder for the results 
#dir.create(paste0(input, "Adjacency_Matrices/"))
out_dir <- paste0(input, "Adjacency_Matrices/")

#----------------------------------
# 3. For Loop To Create the Graphs
#----------------------------------

# Set Seed for reproduceability
set.seed(187)

# For Loop to iterate over the quarters 
for (i in seq_len(length(quarters) - 1L)) {
  
    # Define the quarter names to store than later correctly
    this_q <- quarters[i]          
    next_q <- quarters[i + 1L]     

    # Filter the relevant quarters 
    tmp_bank_data_q_clean  <- bank_data_q_clean %>% 
        filter(date == this_q)
    
    # Skip empty quarters
    if (nrow(tmp_bank_data_q_clean) == 0L) next 
    
    # Extract the loans and deposit values 
    loans <- tmp_bank_data_q_clean$ffsspq
    deposits  <- tmp_bank_data_q_clean$ffpssq
    
    # Estimate the matrix of banking relations
    adj_tmp  <- matrix_estimation(loans, deposits, method = "md", verbose = F)
    
    # Extract the gvkeys and firm names
    firm_keys <- as.vector(tmp_bank_data_q_clean$gvkey)
    firm_names <- as.vector(tmp_bank_data_q_clean$conm)
    
    # Adjust the rownames and colnames of the matrix in two 
    # separate matrices one for the gvkey and one for the firmname 
    adj_tmp_gvkeys <- adj_tmp
    colnames(adj_tmp_gvkeys) <- firm_keys
    rownames(adj_tmp_gvkeys) <- firm_keys
    adj_tmp_firm_names <- adj_tmp
    colnames(adj_tmp_firm_names) <- firm_names
    rownames(adj_tmp_firm_names) <- firm_names
    
    # Create helper function
    to_table <- function(mat) Table$create(as.data.frame(mat))
    
    
    # Safe the files 
    write_parquet(to_table(adj_tmp_gvkeys),
                  paste0(out_dir,
                            sprintf("adjacency_matrix_md_gvkey_%s.parquet", next_q)))
    
    write_parquet(to_table(adj_tmp_gvkeys),
                  paste0(out_dir,
                            sprintf("adjacency_matrix_md_firmname_%s.parquet", next_q)))
    
    # Print message when executed
    message("finished ", this_q, " → stored as ", next_q)
    
}

