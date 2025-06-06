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
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data as data
from torch_geometric.nn import SAGEConv, BatchNorm
import optuna 
import math
from tqdm.notebook import tqdm
from itertools import product 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)


# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data
# CoVar
with open(input + "Clean/covar_graphs.pkl", "rb") as file:
    covar_graphs = pickle.load(file)
    
# Set torch seed
torch.manual_seed(187)



#----------------------------------
# 2. Create Torch Class for GNN 
#----------------------------------


# Wrap it in a pytorch class
class GraphDataset(data.InMemoryDataset):
    def __init__(self, graphs, root: str = "/tmp", 
                 keep_full_batch: bool = True, 
                 transform = None, pre_transform = None):
        
        super().__init__(root, transform, pre_transform)
        
        # Arrange graphs by data for the split later
        # Arrange graphs by quarters
        self.graphs_by_quarter = {}
        for quarter, graph in graphs.items():
            self.graphs_by_quarter[quarter] = graph
            
        data_list = [torch_geometric.utils.from_networkx(i) for i in graphs.values()]
        
        # Collate in big tensors
        self.data, self.slices = self.collate(data_list)
        
        # Create a large batch of the data
        self.full_batch = data.Batch.from_data_list(data_list) if keep_full_batch else None


# Define Function to retrive and validate correct graph structure
def print_graphs(batch):
    print("Number of Graphs:", batch.num_graphs)
    print("Graph at index 1:", batch[1])
 #   print("Print adjecency tensor:", batch[1].weight)
  #  print("Print Features:", batch[1].x)
  #  print("Print CoVar:", batch[1].y)
    print("Print agg. Information:", batch)


# Define Function for the Data Loader and Splits 
def PyTorchGraphLoader(dataset):
    
    # Define splits
    train_idx = list(range(0, 52)) # 2000Q1 – 2013Q4
    val_idx = list(range(52, 68)) # 2014Q1 – 2017Q4
    test_idx = list(range(68, 92)) # 2018Q1 – 2023Q4

    # Create training, validation, and training set:
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    # Define batch size
    batch_size = 4 # --> set to 1 to keep the order and avoid any leakage of data --> apparently no leakage --> check again before submission

    # Define Data loaders
    train_loader = torch_geometric.loader.DataLoader(train_ds, 
                                                     batch_size = batch_size, 
                                                     shuffle = True)
    val_loader = torch_geometric.loader.DataLoader(val_ds, 
                                                   batch_size = batch_size, 
                                                   shuffle = False)
    test_loader = torch_geometric.loader.DataLoader(test_ds, 
                                                    batch_size = batch_size, 
                                                    shuffle = False)
    
    return train_loader, val_loader, test_loader
    

# Define Funtion to assess the batches of the splits created
def batch_check(x):
    first_train_batch = next(iter(x))
    print("Whole Summary:", first_train_batch)
    print("Node Features:", first_train_batch.x[:3])
    print("Node Attributes:", first_train_batch.y[:3])
    print("Edge Features:", first_train_batch.edge_index[:, :5])
    print("Unique Nodes:", first_train_batch.batch.unique())
    
    
    
#----------------------------------
# 4. Create GNN Model
#----------------------------------

class GraphSage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        aggregator_type: str,
        num_layers: int,
        dropout: float,
    ):
        super(GraphSage, self).__init__()

        self.aggregator_type = aggregator_type

        # A list of SAGEConvs and a list of BNs for after each conv
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Dropout attribute 
        self.dropout = nn.Dropout(dropout)

        # Aggregator attribure
        self.aggregator_type = aggregator_type

        # Input layer
        #self.bns.append(BatchNorm(in_channels))
        #self.convs.append(
        #    SAGEConv(in_channels, hidden_channels, aggr=aggregator_type)
        #)

        # Add Layers 
        if aggregator_type == "attentional":
            # Input Layer 
            self.bns.append(BatchNorm(in_channels))
            self.convs.append(CustomAttentionLayer(in_channels, hidden_channels))

            # Hidden Layers
            for _ in range(num_layers - 1):
                self.bns.append(BatchNorm(hidden_channels))
                self.convs.append(CustomAttentionLayer(hidden_channels, hidden_channels))
        else:
            # Input Layers
            self.bns.append(BatchNorm(in_channels))
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator_type))

            # Hidden Layers
            for _ in range(num_layers - 1):
                self.bns.append(BatchNorm(hidden_channels))
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator_type))

        # Output Layer
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(x)
            if self.aggregator_type == "attentional":
                x = conv(x, edge_index, batch)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return torch.sigmoid(self.lin(x).squeeze(-1))

    

# Create a custom layer for attentional aggregation
class CustomAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.gate_nn = nn.Linear(in_channels, 1)
        self.attn_aggr = torch_geometric.nn.aggr.AttentionalAggregation(gate_nn=self.gate_nn, nn=self.lin)
        self.out = nn.Linear(out_channels, out_channels)  

    def forward(self, x, edge_index, batch):
        # Step 1: expand edge_index to neighbor list
        row, col = edge_index
        messages = x[col]  # neighbor features
        out = self.attn_aggr(messages, row, dim_size=x.size(0))
        return self.out(out)


# Define Training Function for the trining loop
def run_epoch(loader, model, device, optimizer, criterion, training=True, epoch_desc=""):
    
    # Define whether or not the function trains or evaluates
    model.train() if training else model.eval()
    
    # Define the running loss
    running_loss, n = 0.0, 0


    # Iterate over the number of epochs
    #for batch in tqdm(loader, desc = epoch_desc, leave = False, ncols = 90):
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none = True)

        with torch.set_grad_enabled(training):
            preds = model(batch.x, batch.edge_index)
            labels = batch.y.float().squeeze()
            loss = criterion(preds, labels)
        
            if training:
                loss.backward()
                optimizer.step()

        # Update Loss
        running_loss += loss.item() * batch.num_nodes
        n += batch.num_nodes 
        avg_loss = running_loss / n

    return avg_loss


# Define Optuna Training Algorithm 
# Define tuning grid function 
def objective(trial,
              train_loader,
              val_loader,
              num_epochs,
              in_channels,
              output,
              device,
              covar,
              grid):
    
    # Define hyperparameter layer
    hidden_channels = suggest_from_grid(trial, "hidden_channels", grid)
    num_layers = suggest_from_grid(trial, "num_layers", grid)
    dropout = suggest_from_grid(trial, "dropout", grid)
    lr = suggest_from_grid(trial, "lr", grid)
    weight_decay = suggest_from_grid(trial, "weight_decay", grid)
    aggregator = suggest_from_grid(trial, "aggregator_type", grid)
    
    
    # Load the model
    model = GraphSage(in_channels = in_channels,
                        hidden_channels = hidden_channels,
                        num_layers = num_layers,
                        dropout = dropout,
                        aggregator_type= aggregator).to(device := "cuda" if torch.cuda.is_available() else "cpu")


    #----------------------------------
    #  Define Training Variables
    #----------------------------------

    # Define error
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optimizer_fun(model, lr, weight_decay)

    # Define scheduler to alter learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min",
                                                       factor = 0.5, patience = 5,
                                                       verbose = False)
    
 
    #----------------------------------
    # Initiate Training Loop
    #----------------------------------

    # Define Parameters for the training loop
    best_val = math.inf # Validation set
    patience, patience_ctr = 25, 0 # Set patience for early quiting 

    # Define tqdm object to investigate the training progress
    #outer_bar = tqdm(range(1, num_epochs), desc = "Epochs", ncols = 90)

    # Start training loop
    for epoch in range(1, num_epochs + 1):
    
        # Train the model
        train_loss = run_epoch(train_loader, model, device, optimizer, criterion,
                               training = True, epoch_desc = f"train {epoch:03d}")
        
        # Evaluate on validation set
        val_loss = run_epoch(val_loader, model, device, optimizer, criterion,
                             training = False, epoch_desc = f"val {epoch:03d}")
        
        scheduler.step(val_loss)

        # print(val_loss)
        
        # Print Progress
        #outer_bar.set_postfix(
        #    train=f"{train_loss:.4f}",
        #    val=f"{val_loss:.4f}",
        #    lr=optimizer.param_groups[0]['lr'])
        
        # Continue if loss can be reduced significatly 
        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Increase patience counter
        else:
            patience_ctr += 1
            
            # Trigger early exist
            if patience_ctr >= patience:
                #outer_bar.write("Early stopping triggered.")
                break
    model.load_state_dict(best_state)
    val_auc = compute_auc(val_loader, model, device)
    trial.set_user_attr("state_dict", best_state)
    return val_auc


# Create helper function to like param grid with the objective function
def suggest_from_grid(trial, name, grid):
    value = trial.suggest_categorical(name, grid[name])
    return value

# Define Optimizer
def optimizer_fun(model, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer



# Create function for evaluation table
def make_table(val_loader, test_loader, model, device, threshold = 0.5):
    def evaluate(loader, split_name):
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for graph in loader:
                graph = graph.to(device)
                preds = model(graph.x, graph.edge_index)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(graph.y.cpu().numpy())

        # Flatten predictions and labels
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets).reshape(-1)

        # Convert probabilities to class predictions
        pred_labels = (all_preds > threshold).astype(int)

        # Compute metrics
        acc = accuracy_score(all_targets, pred_labels)
        roc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else np.nan
        f1 = f1_score(all_targets, pred_labels)
        precision = precision_score(all_targets, pred_labels)
        recall = recall_score(all_targets, pred_labels)
        cm = confusion_matrix(all_targets, pred_labels)

        return {
            "Accuracy": acc,
            "ROC AUC": roc,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": cm,
            "Probabilities": all_preds,
            "True Labels": all_targets,
            "Pred Labels": pred_labels,
        }

    val_results = evaluate(val_loader, "Validation")
    test_results = evaluate(test_loader, "Test")

    results_df = pd.DataFrame({
        "Set": ["Validation", "Test"],
        "Accuracy": [val_results["Accuracy"], test_results["Accuracy"]],
        "ROC AUC": [val_results["ROC AUC"], test_results["ROC AUC"]],
        "F1 Score": [val_results["F1 Score"], test_results["F1 Score"]],
        "Precision": [val_results["Precision"], test_results["Precision"]],
        "Recall": [val_results["Recall"], test_results["Recall"]],
    })

    # Also return raw predictions for plotting if needed
    return results_df, val_results, test_results


# Make function to extract the predictions
    
def extract_predictions(best_model, train_loader, val_loader, test_loader, device="cpu"):

    best_model.eval()
    best_model.to(device)

    prediction_results = []
    loaders = {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }

    with torch.no_grad():
        for split, loader in loaders.items():
            for batch in loader:
                batch = batch.to(device)

                preds = best_model(batch.x, batch.edge_index, batch.batch)
                preds = preds.squeeze().cpu().numpy()
                y_true = batch.y.squeeze().cpu().numpy()

                # Convert metadata tensor to numpy for easier handling
                metadata_np = batch.metadata.cpu().numpy()

                for i in range(batch.num_nodes):
                    encoded_date = int(metadata_np[i][0])
                    gvkey = int(metadata_np[i][1])

                    # Reconstruct string quarter from encoded date (e.g. 20201 → "2020Q1")
                    date_str = f"{str(encoded_date)[:-1]}Q{str(encoded_date)[-1]}"

                    prediction_results.append({
                        "split": split,
                        "firm": gvkey,
                        "quarter": date_str,
                        "target": y_true[i],
                        "prediction": preds[i],
                    })

    return pd.DataFrame(prediction_results) 



def compute_auc(loader, model, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds.append(out.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return roc_auc_score(labels, preds)

