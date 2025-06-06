#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna 
import math
from tqdm.notebook import tqdm
from itertools import product 
from sklearn.metrics import r2_score
from functools import partial
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)


# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")


# Set torch seed
torch.manual_seed(187)

# Load GNN Objects
from FCNN_objects_Classification_change import prepare_FCNN
from FCNN_objects_Classification_change import FCNN
from FCNN_objects_Classification_change import run_epoch
from FCNN_objects_Classification_change import objective
from FCNN_objects_Classification_change import suggest_from_grid
from FCNN_objects_Classification_change import optimizer_fun
from FCNN_objects_Classification_change import check_data_loader
from FCNN_objects_Classification_change import make_table
from FCNN_objects_Classification_change import extract_predictions


#----------------------------------
# 2. Data Preperation CoVar
#----------------------------------

# Drop not needed columns
covar_final = covar_final.drop(columns=["covar", "lag_covar"])

# Rename dependent variable
covar_final = covar_final.rename(columns={"diff_delta_covar": "y"})

# Extract the different types of data sets 
(train_loader_covar, val_loader_covar, test_loader_covar) = prepare_FCNN(covar_final)

# Check the data
#check_data_loader(train_loader_covar)
#check_data_loader(val_loader_covar)


#----------------------------------
# 4. Run Study CoVar
#----------------------------------

# Define Vars
# Adjust to access the first batch and its shape correctly
first_batch = next(iter(train_loader_covar))
in_channels = first_batch[0].shape[1]
device      = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 250

# Define tuning Grid 
grid_covar = {
    "hidden_channels": [64, 128, 256],
    "num_layers":      [2,3],
    "dropout":         [0.1],
    "lr":              [1e-3],
    "weight_decay":    [1e-5],
}

# Define Sampler and the number of trials 
sampler_covar = optuna.samplers.GridSampler(search_space=grid_covar)
n_trials_covar = len(list(product(*grid_covar.values())))


# Define partical function
objective_covar = partial(
    objective,
    train_loader=train_loader_covar,
    val_loader=val_loader_covar,
    num_epochs=num_epochs,
    in_channels=in_channels,
    output=output,
    device = device,
    covar = True,
    grid = grid_covar
)

# Conduct study using the grid 
study_covar = optuna.create_study(direction="minimize", sampler=sampler_covar)
study_covar.optimize(objective_covar, n_trials=n_trials_covar, timeout=None)

# Extract the hyper parameter from the best model
best_state  = study_covar.best_trial.user_attrs["state_dict"]
best_params = study_covar.best_params
print("Best hyper‑parameters:\n", best_params,
      "\nValidation loss:", (study_covar.best_value))


# Rebuild a fresh model with the best params retrived from the study
best_model =   FCNN(
    in_channels       = in_channels,
    hidden_channels   = best_params["hidden_channels"],
    num_layers        = best_params["num_layers"],
    dropout           = best_params["dropout"],
).to(device)


best_model.load_state_dict(best_state)
best_model.eval()

# Print the test results 
optimizer = optimizer_fun(best_model, best_params["lr"], best_params["weight_decay"])
val_loss_covar = run_epoch(val_loader_covar, best_model, device, optimizer,  nn.BCELoss(),
                             training=False)
test_loss_covar = run_epoch(test_loader_covar, best_model, device, optimizer, nn.BCELoss(),
                             training=False)
print("Test Loss:", (test_loss_covar))

# Save the model
torch.save(best_model.state_dict(), output + "/Models/covar_best_FCNN_Classification_change.pt")


# Create evaluation Table
results_covar, all_preds_val, all_preds_test = make_table(val_loader_covar, test_loader_covar, best_model, device)
print(results_covar)



# Store the results 
results_covar.to_csv(output + "/Result_Models/covar_results_FCNN_Classification_change.csv")

# Extract and store the predictions
predictions_covar = extract_predictions(train_loader_covar, val_loader_covar, test_loader_covar, best_model, device)
predictions_covar.to_csv(output + "/Result_Models/covar_predictions_FCNN_Classification_change.csv")
