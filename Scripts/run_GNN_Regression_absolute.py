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



# Load GNN Objects
from GNN_objects_Regression import GraphDataset
from GNN_objects_Regression import print_graphs
from GNN_objects_Regression import PyTorchGraphLoader
from GNN_objects_Regression import batch_check
from GNN_objects_Regression import GraphSage
from GNN_objects_Regression import run_epoch
from GNN_objects_Regression import objective
from GNN_objects_Regression import optimizer_fun
from GNN_objects_Regression import make_table
from GNN_objects_Regression import extract_predictions
from GNN_objects_Regression import verify_split_quarters


# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
# CoVar
with open(input + "Clean/covar_graphs_Regression_absolute.pkl", "rb") as file:
    covar_graphs = pickle.load(file)

# Set torch seed
torch.manual_seed(187)
np.random.seed(187)

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
# batch_check(train_loader_covar)
verify_split_quarters(covar_dataset, train_loader_covar, val_loader_covar, test_loader_covar)

#----------------------------------
# 3. Run Study CoVar
#----------------------------------

# Define in channels
in_channels = 10
device      = "cuda" if torch.cuda.is_available() else "cpu"

# Define tuning Grid 
grid_covar = {
    "hidden_channels": [10,20,30],
    "num_layers":      [2,3,4],
    "dropout":         [0, 0.1, 0.2],
    "lr":              [0.01, 0.001, 0.0001],
    "weight_decay":    [0],
    "aggregator_type": ["mean", "lstm"]
}


# Define Sampler and the number of trials 
sampler_covar = optuna.samplers.GridSampler(search_space=grid_covar)
n_trials_covar = len(list(product(*grid_covar.values())))

# Define other variables
num_epochs = 150

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

print("Number of trials:", n_trials_covar)

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
    num_layers        = best_params["num_layers"],
    dropout           = best_params["dropout"]
).to(device)
best_model.load_state_dict(best_state)

# Print the test results 
optimizer = optimizer_fun(best_model, best_params["lr"], best_params["weight_decay"])
test_loss_covar = run_epoch(test_loader_covar, best_model, device, optimizer,  nn.MSELoss(),
                             training=False)
print("Test Loss:", test_loss_covar)


# Store the best model
torch.save(best_state, output + "Models/covar_best_graphsage_Regression_absolute.pt")
torch.save(best_model, output + "/Models/covar_best_graphsage_Regression_abolute_full.pt")

best_model.eval()


# Make the table with metrics 
results_df_covar, all_preds_val, all_target_val = make_table(val_loader_covar, test_loader_covar, best_model, device)
print(results_df_covar)




# Save the results
results_df_covar.to_csv(output + "Result_Models/covar_results_GNN_Regression_absolute.csv")

# Make predictions
predictions_covar = extract_predictions(
    best_model, train_loader_covar, val_loader_covar, test_loader_covar, device
)

# Save the predictions
predictions_covar.to_csv(output + "Predictions_Models/covar_predictions_GNN_Regression_absolute.csv")


# Permutation Invariance Test
batch = next(iter(train_loader_covar)).to(device)
best_model.eval()
with torch.no_grad():
    out1 = best_model(batch)
    out2 = best_model(batch)
diff = (out1 - out2).abs().sum().item()
print("Sum of abs differences (should be > 0):", diff)



# def check_batches_dgl(loader, dataset_name="Unknown", expected_node_dim=None):
#     print(f"\n🔎 Checking batches for: {dataset_name}")
    
#     for batch_idx, batch in enumerate(loader):
#         print(f"\nBatch {batch_idx + 1}:")
#         print(f"- Number of graphs in batch: {batch.batch_size}")
#         print(f"- Total number of nodes: {batch.num_nodes()}")
#         print(f"- Total number of edges: {batch.num_edges()}")

#         # Node features
#         x = batch.ndata['x']
#         y = batch.ndata['y']
#         print(f"- Node feature shape: {x.shape}")
#         print(f"- Target (y) shape:    {y.shape}")

#         # Optional: Check node dim
#         if expected_node_dim is not None and x.shape[1] != expected_node_dim:
#             print(f"❌ Unexpected node feature dimension! Expected {expected_node_dim}, got {x.shape[1]}")

#         # Metadata if available
#         if 'metadata' in batch.ndata:
#             metadata = batch.ndata['metadata']
#             print(f"- Metadata shape: {metadata.shape}")
#             quarters = torch.unique(metadata[:, 0])
#             print(f"- Quarters in batch: {quarters.tolist()}")

#         # Edge weights
#         if 'weight' in batch.edata:
#             weights = batch.edata['weight']
#             print(f"- Edge weight shape: {weights.shape}")
#             print(f"- Weight sample: {weights[:5].cpu().numpy().round(3)}")

#         # Sanity checks
#         assert x.shape[0] == batch.num_nodes(), "❌ Node count mismatch"
#         assert y.shape[0] == batch.num_nodes(), "❌ Target count mismatch"
#         if 'metadata' in batch.ndata:
#             assert metadata.shape[0] == batch.num_nodes(), "❌ Metadata count mismatch"


# check_batches_dgl(train_loader_covar, dataset_name="Train Set", expected_node_dim=10)
# check_batches_dgl(val_loader_covar, dataset_name="Validation Set", expected_node_dim=10)
# check_batches_dgl(test_loader_covar, dataset_name="Test Set", expected_node_dim=10)


