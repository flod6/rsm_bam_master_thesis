#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Packages
library(tidyverse)
library(stargazer)
library(xtable)

# Define Paths
input  <-  "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output  <-  "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data


#----------------------------------
# 1. Create Tables Appendix 
#----------------------------------

#----------------------------------
# 1. Create Covar Descriptives
#----------------------------------

table <- read.csv(paste0(output, "Descriptives/covar_predictors.csv"))

table <- table %>%
  select(-X, -X25., -X75.) %>%
  select(count, mean, std, X50., min, max)

colnames(table) <- c("N", "Mean", "SD", "Median", "Min", "Max")

rownames(table) <- c("$\\Delta$-CoVaR$_{i,t}$", "Size$_{i,t-1}$", "ROA$_{i,t-1}$", "Leverage$_{i,t-1}$", 
                     "VaR$_{i,t-1}$", "$\\Delta$-CoVaR$_{i,t-1}$",
                     "TED Spread$_{t-1}$", "GDP Growth$_{t-1}$", "Market Return$_{t-1}$",
                     "VIX$_{t-1}$", "$\\Delta$ T-Bill$_{t-1}$")


table[, -1] <- lapply(table[, -1], function(x) formatC(x, digits = 5, format = "f"))


table$N <- formatC(table$N, digits = 0, format = "f", big.mark = ",")


print(xtable(table,
             caption = "",#"Descriptive Information of the Features",
             label = "",#"tab:predictors_descriptives",
            align = c("l", rep("r", ncol(table)))),
      include.rownames = TRUE,
      sanitize.rownames.function = identity,
      caption.placement = "top",
      comment = FALSE,
      floating = FALSE,
      file = paste0(output, "Final_Tables/covar_predictors.tex"))



#----------------------------------
# 1. Results Regression Absolute
#----------------------------------

table <- read.csv(paste0(output, "Result_Models/final_errors_Regression_abolute_val.csv"))


colnames(table) <- c("Model", "Val. RMSE", "Val. MSE", "Val. MAE", "Val. MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "",#"tab:results_regression_absolute_validation",
               title = "",#"Validation Set Results",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_regression_absolute_validation.txt"))
writeLines(t, t2)
close(t2)



table <- read.csv(paste0(output, "Result_Models/final_errors_Regression_abolute_test.csv"))


colnames(table) <- c("Model", "Test. RMSE", "Test. MSE", "Test. MAE", "Test. MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "",#"tab:results_regression_absolute_test",
               title = "",#"Test Set Results",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_regression_absolute_test.txt"))
writeLines(t, t2)
close(t2)



#----------------------------------
# 2. Results Regression Change 
#----------------------------------


table <- read.csv(paste0(output, "Result_Models/final_errors_Regression_change_val.csv"))


colnames(table) <- c("Model", "Val. RMSE", "Val. MSE", "Val. MAE", "Val. MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "",#"tab:results_regression_change_validation",
               title = "",#"Validation Set Results",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_regression_change_validation.txt"))
writeLines(t, t2)
close(t2)



table <- read.csv(paste0(output, "Result_Models/final_errors_Regression_change_test.csv"))


colnames(table) <- c("Model", "Test. RMSE", "Test. MSE", "Test. MAE", "Test. MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "", #"tab:results_regression_change_test",
               title = "", #"Test Set Results",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_regression_change_test.txt"))
writeLines(t, t2)
close(t2)



#----------------------------------
# 2. Crisis Performance Absolute
#----------------------------------

table <- read.csv(paste0(output, "Result_Models/final_performance_crisis.csv"))

colnames(table) <- c("Model", "Non-Crisis RMSE", "Crisis RMSE", "Non-Crisis MPE", "Crisis MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "",#"tab:results_crisis_performance_Regression_absolute",
               title = "",#"Performance Results Non-Crisis versus Crisis Periods",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_crisis_performance_Regression_absolute.txt"))
writeLines(t, t2)
close(t2)


#----------------------------------
# 2. Size Performance Absolute
#----------------------------------


table <- read.csv(paste0(output, "Result_Models/final_performance_size.csv"))

colnames(table) <- c("Model", "Small RMSE", "Large RMSE", "Small MPE", "Large MPE")
rownames(table) <- NULL
table <- as.matrix(table)

# Export the Table
t <- stargazer(table,
               label = "", #tab:results_size_performance_Regression_absolute",
               title = "", #"Performance Results Small Banks versus Large Banks",
               summary = FALSE,
               float = FALSE)

t2 = file(paste0(output, "/Final_Tables/results_size_performance_Regression_absolute.txt"))
writeLines(t, t2)
close(t2)


#----------------------------------
# 2. Descriptives Robustness Test
#----------------------------------

table <- read.csv(paste0(output, "Descriptives/covar_change.csv"))

table <- table %>%
  select(-X, -X25., -X75.) %>%
  select(count, mean, std, X50., min, max)

colnames(table) <- c("N", "Mean", "SD", "Median", "Min", "Max")

rownames(table) <- c("$\\Delta$-$\\Delta$-CoVaR$_{i,t}$", 
                     "$\\Delta$-$\\Delta$-CoVaR$_{i,t-1}$")


table[, -1] <- lapply(table[, -1], function(x) formatC(x, digits = 5, format = "f"))


table$N <- formatC(table$N, digits = 0, format = "f", big.mark = ",")


print(xtable(table,
             caption = "",#NULL,#"Descriptive Information of Change Features",
             label = "", #tab:predictors_change",
             align = c("l", rep("r", ncol(table)))),
      include.rownames = TRUE,
      sanitize.rownames.function = identity,
      caption.placement = "top",
      comment = FALSE,
      floating = FALSE,
      file = paste0(output, "Final_Tables/covar_change.tex"))



#----------------------------------
# 2. Feature Importance
#----------------------------------

table <- read.csv(paste0(output, "Feature_Importance/features_CoVar_regression_absolute.csv"))

colnames(table) <- c("Features", "Lasso", "Random Forest")

rownames(table) <- c("$\\Delta$-CoVaR$_{i,t-1}$", "GDP Growth$_{t-1}$",
                     "Leverage$_{i,t-1}$", "Market Return$_{t-1}$", 
                     "ROA$_{i,t-1}$", "Size$_{i,t-1}$", "$\\Delta$ T-Bill$_{t-1}$",
                     "TED Spread$_{t-1}$", "VaR$_{i,t-1}$", "VIX$_{t-1}$")


table <- table %>% select(-Features)

#table[, -1] <- lapply(table[, -1], function(x) formatC(x, digits = 5, format = "f"))


print(xtable(table,
             caption = "",
             label = "",
             align = c("l", rep("r", ncol(table))),
             digits = c(0, rep(6, ncol(table)))),  # ← This line controls decimal precision
      include.rownames = TRUE,
      sanitize.rownames.function = identity,
      caption.placement = "top",
      comment = FALSE,
      floating = FALSE,
      file = paste0(output, "Final_Tables/feature_importance.tex"))
