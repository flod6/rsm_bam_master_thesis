#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import torch
import shap
import dgl
from tqdm import tqdm
from itertools import islice

# Import other objects
from FCNN_objects_Regression import prepare_FCNN
from GNN_objects_Regression import GraphDataset, PyTorchGraphLoader
from dgl.nn.pytorch.explain import GNNExplainer

# Define paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"


# Load Model and data
fcnn_model = torch.load(output + "Models/covar_best_FCNN_Regression_abolute_full.pt")
gnn_model = torch.load(output + "Models/covar_best_graphsage_Regression_abolute_full.pt")
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")
covar_graphs = pd.read_pickle(input + "Clean/covar_graphs_Regression_absolute.pkl")



#----------------------------------
# 1. Features FCNN
#----------------------------------


# Drop not needed columns
covar_final = covar_final.drop(columns=["diff_delta_covar", "lag_diff_delta_covar"])

# Rename dependent variable
covar_final = covar_final.rename(columns={"covar": "y"})

# Extract the different types of data sets 
(train_loader_covar, val_loader_covar, test_loader_covar) = prepare_FCNN(covar_final)

# Re-collect full validation data
X_val_all = []
for batch in val_loader_covar:
    X_val_all.append(batch[0])
X_val_all = torch.cat(X_val_all)



# Use a small sample for background (e.g., 100 samples)
background = X_val_all[:100]

# Use a sample for SHAP evaluation (e.g., 200)
explain_data = X_val_all[:200]


# Create a wrapped model to ensure the output shape is correct for SHAP
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)  # make it (batch_size, 1)
        return out
    

# Use the wrapped model for SHAP
wrapped_model = WrappedModel(fcnn_model)


# SHAP DeepExplainer
explainer = shap.DeepExplainer(wrapped_model, background)
shap_values = explainer.shap_values(explain_data)


shap_vals = shap_values


feature_names = covar_final.drop(columns=["y", "date", "gvkey", "conm", "date"]).columns.tolist()


mean_abs_shap = np.abs(shap_vals).mean(axis=0).flatten()
importance_df = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
print(importance_df)

importance_df.to_csv(output + "Feature_Importance/FCNN_feature_importance.csv", index=True)


# Probably delete later
# %%




#----------------------------------
# 3. Features GNN
#----------------------------------

# Turn Graph into PyTorch Object
covar_dataset = GraphDataset(covar_graphs)

# Turn the graphs into batches
covar_batch = covar_dataset.full_batch

# Check if everything is correct:
#print_graphs(covar_batch)

# Create Train / Val / Test Split
train_loader_covar, val_loader_covar, test_loader_covar = PyTorchGraphLoader(covar_dataset)



#feature_names = covar_final.drop(columns=["y", "date", "gvkey", "conm", "date"]).columns.tolist()
feature_dim = len(feature_names)
device = torch.device("cpu")  # or "cuda" if you're using GPU

# List to collect gradients
gradients = []

gnn_model.eval()

num_graphs_to_use = 16
nodes_per_graph = 12

for g in tqdm(islice(val_loader_covar, num_graphs_to_use)):
    g = g.to(device)
    x = g.ndata['x']
    x.requires_grad_()  # allow gradient tracking
    g.ndata['x'] = x

    num_nodes = g.num_nodes()
    sampled_nodes = torch.randperm(num_nodes)[:min(nodes_per_graph, num_nodes)]

    for nid in sampled_nodes:
        try:
            # Forward pass
            out = gnn_model(g)

            # Get output for this node
            node_output = out[nid]

            # Backward pass to get gradients w.r.t input features
            gnn_model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            node_output.backward(retain_graph=True)

            # Get gradients for that node
            grad = x.grad[nid].detach().cpu().numpy()
            gradients.append(np.abs(grad))
        except Exception as e:
            print(f"Skipping node {nid.item()} due to error: {e}")

# Aggregate
if gradients:
    gradients = np.array(gradients)
    avg_importance = gradients.mean(axis=0)
    importance_df = pd.Series(avg_importance, index=feature_names).sort_values(ascending=False)

    print(f"\n✅ GNN Feature Importance (based on gradients, {len(gradients)} nodes):")
    print(importance_df)
else:
    print("\n❌ No gradients collected — likely all nodes failed.")


# %%

# Turn Graph into PyTorch Object
covar_dataset = GraphDataset(covar_graphs)

# Turn the graphs into batches
covar_batch = covar_dataset.full_batch

# Create Train / Val / Test Split
train_loader_covar, val_loader_covar, test_loader_covar = PyTorchGraphLoader(covar_dataset)

feature_dim = len(feature_names)
device = torch.device("cpu")  # or "cuda" if you're using GPU

# Lists to collect gradients and corresponding input values
gradients = []
input_values = []

gnn_model.eval()

num_graphs_to_use = 16
nodes_per_graph = 12

for g in tqdm(islice(val_loader_covar, num_graphs_to_use)):
    g = g.to(device)
    x = g.ndata['x']
    x.requires_grad_()  # allow gradient tracking
    g.ndata['x'] = x

    num_nodes = g.num_nodes()
    sampled_nodes = torch.randperm(num_nodes)[:min(nodes_per_graph, num_nodes)]

    for nid in sampled_nodes:
        try:
            # Forward pass
            out = gnn_model(g)

            # Get output for this node
            node_output = out[nid]

            # Backward pass to get gradients w.r.t input features
            gnn_model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            node_output.backward(retain_graph=True)

            # Get gradients and inputs for that node
            grad = x.grad[nid].detach().cpu().numpy()
            input_val = x[nid].detach().cpu().numpy()

            gradients.append(np.abs(grad))
            input_values.append(input_val)
        except Exception as e:
            print(f"Skipping node {nid.item()} due to error: {e}")

# Aggregate and compute adjusted importance
gradients = np.array(gradients)
input_values = np.array(input_values)

avg_grad = gradients.mean(axis=0)
mean_inputs = input_values.mean(axis=0)

# Scale-adjusted importance
adjusted_importance = avg_grad * mean_inputs

importance_df = pd.Series(adjusted_importance, index=feature_names).sort_values(ascending=False)

print(importance_df)

importance_df.to_csv(output + "Feature_Importance/GNN_feature_importance.csv", index=True)
