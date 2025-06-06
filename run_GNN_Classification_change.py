#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.data as data
import optuna 
import math
from tqdm.notebook import tqdm
from itertools import product 
from functools import partial
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

# Load GNN Objects
from GNN_objects_Classification_change import GraphDataset
from GNN_objects_Classification_change import print_graphs
from GNN_objects_Classification_change import PyTorchGraphLoader
from GNN_objects_Classification_change import batch_check
from GNN_objects_Classification_change import GraphSage
from GNN_objects_Classification_change import run_epoch
from GNN_objects_Classification_change import objective
from GNN_objects_Classification_change import optimizer_fun
from GNN_objects_Classification_change import make_table
from GNN_objects_Classification_change import extract_predictions


# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
# CoVar
with open(input + "Clean/Clean/covar_graphs_Regression_change.pkl", "rb") as file:
    covar_graphs = pickle.load(file)
    
# Set torch seed
torch.manual_seed(187)
np.random.seed(187)

# 

#----------------------------------
# 2. Data Preperation CoVar
#----------------------------------

# Turn Graph into PyTorch Object
covar_dataset = GraphDataset(covar_graphs)

# Turn the graphs into batches
covar_batch = covar_dataset.full_batch

# Check if everything is correct:
#print_graphs(covar_batch)

# Create Train / Val / Test Split
train_loader_covar, val_loader_covar, test_loader_covar = PyTorchGraphLoader(covar_dataset)

# Check if the splitting is correct 
#batch_check(train_loader_covar)


#----------------------------------
# 3. Run Study CoVar
#----------------------------------

# Define in channels
in_channels = covar_dataset[0].x.size(1)
device      = "cuda" if torch.cuda.is_available() else "cpu"

# Define tuning Grid 
grid_covar = {
    "hidden_channels": [64],
    "num_layers":      [1],
    "dropout":         [0.1],
    "lr":              [1e-3],
    "weight_decay":    [1e-5],
    "aggregator_type": ["attentional"],
   # aggregator_type: ["mean", "pooling"], 
}
# Define Sampler and the number of trials 
sampler_covar = optuna.samplers.GridSampler(search_space=grid_covar)
n_trials_covar = len(list(product(*grid_covar.values())))

# Define other variables
num_epochs = 250
in_channels = covar_dataset[0].x.size(1)

# Add partial function call 
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
      "\nValidation loss:", study_covar.best_value)

# Rebuild a fresh model with the best params retrived from the study
best_model = GraphSage(
    in_channels       = in_channels,
    hidden_channels   = best_params["hidden_channels"],
    aggregator_type   = best_params["aggregator_type"],
    num_layers          = best_params["num_layers"],
    dropout           = best_params["dropout"]
).to(device)
best_model.load_state_dict(best_state)

# Print the test results 
optimizer = optimizer_fun(best_model, best_params["lr"], best_params["weight_decay"])
test_loss_covar = run_epoch(test_loader_covar, best_model, device, optimizer,  nn.BCELoss(),
                             training=False)
print("Test Loss:", test_loss_covar)


# Store the best model
torch.save(best_state, output + "Models/covar_best_graphsage_Classification_change.pt")

best_model.eval()


# Make the table with metrics 
results_df_covar, all_preds_val, all_target_val = make_table(val_loader_covar, test_loader_covar, best_model, device)
print(results_df_covar)

# Save the results
results_df_covar.to_csv(output + "Result_Models/covar_results_GNN.csv")

# You can then call the function and, for example, convert the results to a DataFrame:
pred_results = extract_predictions(best_model, train_loader_covar, val_loader_covar, test_loader_covar)
pred_df = pd.DataFrame(pred_results)

# Save the predictions
pred_df.to_csv(output + "Result_Models/Result_Models/covar_predictions_GNN_Classification_change.csv", index=False)



