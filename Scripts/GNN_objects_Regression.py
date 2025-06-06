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
import torch_geometric.utils
from tqdm.notebook import tqdm
from itertools import product 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
import dgl
from dgl.dataloading import GraphDataLoader



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
    def __init__(self, graphs: dict, keep_full_batch: bool = True):

        # Pass graph to the self
        self.graphs_by_quarter = graphs
        self.graph_list = []
        
        # Arrange graphs by data for the split later
        for quarter, nx_g in sorted(graphs.items()):

           # print(quarter)
            # Convert to DGLGraph
            g = dgl.from_networkx(
                nx_g, 
                node_attrs=['x', 'y', 'metadata'], 
                edge_attrs=['weight'] if 'weight' in next(iter(nx_g.edges(data=True)))[2] else []
            )

            # Convert edge data types
            g.edata['weight'] = g.edata['weight'].float()

            #print("Node dtype:", g.ndata['x'].dtype)
            #print("Edge dtype:", g.edata['weight'].dtype)

            self.graph_list.append(g)
        
        self.full_batch = dgl.batch(self.graph_list) if keep_full_batch else None
        
    def __getitem__(self, idx):
        return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)

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
    train_idx = list(range(0, 56)) # 2000Q1 – 2013Q4
    val_idx = list(range(56, 72)) # 2014Q1 – 2017Q4
    test_idx = list(range(72, 96)) # 2018Q1 – 2023Q4


    # Create training, validation, and training set:
    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in test_idx]

    # Define batch size
    batch_size = 3 # 

    # Define Data loaders
    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = GraphDataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)
    
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

        # A list of SAGEConvs and a list of BNs for after each conv
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Aggregator attribure
        self.aggregator_type = aggregator_type

        # Add Layers 
        # Input Layers
        self.bns.append(BatchNorm(in_channels))
        self.convs.append(dgl.nn.SAGEConv(in_channels, hidden_channels, aggregator_type=aggregator_type,
                                          feat_drop = dropout))

        # Hidden Layers
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm(hidden_channels))
            self.convs.append(dgl.nn.SAGEConv(hidden_channels, hidden_channels, aggregator_type=aggregator_type,
                                              feat_drop = dropout))

        # Output Layer
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, g):
        h = g.ndata['x']
        for conv, bn in zip(self.convs, self.bns):
            h = bn(h)
            edge_weight = g.edata.get('weight')
            if edge_weight is not None:
                edge_weight = torch.log1p(edge_weight)
            h = conv(g, h, edge_weight=edge_weight)
            h = F.relu(h)
        return self.lin(h).squeeze(-1)


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
        # optimizer.zero_grad(set_to_none = True)

        with torch.set_grad_enabled(training):
            preds = model(batch)
            loss = criterion(preds.view(-1), batch.ndata['y'].view(-1))
        
            if training:
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()

        # Update Loss
        running_loss += loss.item() * batch.num_nodes()
        n += batch.num_nodes()
        rmse = math.sqrt(running_loss / n)

    return rmse



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


    # Total validation loss per split
    total_val_loss = []
    
    
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
    criterion = nn.MSELoss()

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
        
        #print(train_loss)
        
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
                print("Early stopping triggered. At Epoch:", epoch)
                break
    model.load_state_dict(best_state)
    trial.set_user_attr("state_dict", best_state)
    return(best_val)


# Create helper function to like param grid with the objective function
def suggest_from_grid(trial, name, grid):
    value = trial.suggest_categorical(name, grid[name])
    return value

# Define Optimizer
def optimizer_fun(model, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def make_table(val_loader, test_loader, model, device):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error

    results = {
        "Set": ["Validation", "Test"],
        "RMSE": [],
        "MSE": [],
        "MAE": [],
        "MPE": [],
        "R2": [],
    }

    model.eval()

    def evaluate(loader):
        all_preds, all_targets = [], []
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(device)
                preds = model(graph)
                all_preds.append(preds.cpu().numpy().astype(np.float64))
                all_targets.append(graph.ndata["y"].cpu().numpy().astype(np.float64))
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets).reshape(-1)

        rmse = np.sqrt(((all_preds - all_targets) ** 2).mean())
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        mse = ((all_preds - all_targets) ** 2).mean()
        mpe = ((all_preds - all_targets) / all_targets).mean()
        
        return rmse, mse, mae, mpe, r2, all_preds, all_targets

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    for metrics in [val_metrics, test_metrics]:
        rmse, mse, mae, mpe, r2, _, _ = metrics
        results["RMSE"].append(rmse)
        results["MSE"].append(mse)
        results["MAE"].append(mae)
        results["MPE"].append(mpe)
        results["R2"].append(r2)

    results_df = pd.DataFrame(results)
    return results_df, val_metrics[5], val_metrics[6]



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
            for graph in loader:
                graph = graph.to(device)
                preds = best_model(graph).squeeze().cpu().numpy()
                y_true = graph.ndata["y"].squeeze().cpu().numpy()
                metadata_np = graph.ndata["metadata"].cpu().numpy()

                for i in range(graph.num_nodes()):
                    encoded_date = int(metadata_np[i][0])
                    gvkey = int(metadata_np[i][1])
                    date_str = f"{str(encoded_date)[:-1]}Q{str(encoded_date)[-1]}"
                    prediction_results.append({
                        "split": split,
                        "firm": gvkey,
                        "quarter": date_str,
                        "target": y_true[i],
                        "prediction": preds[i],
                    })

    return pd.DataFrame(prediction_results)




def verify_split_quarters(dataset, train_loader, val_loader, test_loader):
    # Define expected split ranges
    train_range = list(range(0, 56))     # 2000Q1 – 2013Q4
    val_range   = list(range(56, 72))    # 2014Q1 – 2017Q4
    test_range  = list(range(72, 96))    # 2018Q1 – 2023Q4

    def get_quarters_from_loader(loader):
        quarters = []
        for batch in loader:
            # Subset returns indices into the original dataset
            for idx in batch.batch.unique().tolist():
                quarters.append(dataset.graphs_by_quarter_keys[idx])
        return quarters

    # Recover ordered keys from dataset
    dataset.graphs_by_quarter_keys = list(sorted(dataset.graphs_by_quarter.keys()))

    # Manually get the actual quarters per split
    train_quarters = [dataset.graphs_by_quarter_keys[i] for i in train_range]
    val_quarters   = [dataset.graphs_by_quarter_keys[i] for i in val_range]
    test_quarters  = [dataset.graphs_by_quarter_keys[i] for i in test_range]

    # Print or return results
    print("Train Quarters:", train_quarters)
    print("Val Quarters:  ", val_quarters)
    print("Test Quarters: ", test_quarters)

