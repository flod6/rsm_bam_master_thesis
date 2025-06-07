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
import optuna 
import math
from tqdm.notebook import tqdm
from itertools import product 
from sklearn.metrics import r2_score, mean_absolute_error

# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")

# Set torch seed
torch.manual_seed(187)


#----------------------------------
# 2. Prepare Data for FCNN 
#----------------------------------

# Define Funtion that prepares the data for the input
def prepare_FCNN(data, batch_size = 1300): 

    # Arrange data frame by date 
    data = data.sort_values(by=["date"])
    data["date"] = pd.PeriodIndex(data["date"], freq="Q")

    # Define data split dates
    train_end = pd.Period("2013Q4", freq="Q")
    val_end   = pd.Period("2017Q4", freq="Q")
    
    # Split data set according to data
    train = data[data["date"] <= train_end]
    val = data[(data["date"] > train_end) & (data["date"] <= val_end)]
    test = data[data["date"] > val_end]


    # Adjust the split data
    splits = {}

    # Iterate over the splits
    for i, j in [("train", train), ("val", val), ("test", test)]:
        df = data.loc[j.index]
        
        X = df.drop(columns=["date", "gvkey", "conm", "y"]).values.astype(np.float32)
        y = df["y"].values.astype(np.float32).reshape(-1, 1)
        
        # Metadata
        firm_ids = df["gvkey"].values
        quarters = df["date"].astype(str).values  # keep as string for easy logging

        # Create tuple with metadata
        splits[i] = (
            torch.from_numpy(X),
            torch.from_numpy(y),
            firm_ids,
            quarters
        )

    # Create the loaders
    loaders = {}
    for name, (X_t, y_t, firm_ids, quarters) in splits.items():
        ds = torch.utils.data.TensorDataset(X_t, y_t)

        # Wrap dataset to include metadata during iteration
        dataset = list(zip(X_t, y_t, firm_ids, quarters))

        shuffle = True if name == "train" else False
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    return loaders["train"], loaders["val"], loaders["test"]


# Create helper function to asses if everthing in the data loader is correct
def check_data_loader(data_loader):
    i = 0
    for batch_features, batch_targets, firm_ids, quarters in data_loader:
        print("Batch Features Shape:", batch_features.shape)
        print("Batch Targets Shape:", batch_targets.shape)
        print("Quarter Data (First Batch):", batch_features[:5])  # Display first 5 rows of features
        print("Quarter Targets (First Batch):", batch_targets[:5])
        i = i + 1  # Display first 5 rows of targets
        if i > 1:
            break



#----------------------------------
# 3. Create Model Class
#----------------------------------

# Create a Fully Connected Neural Network (FCNN) class for regression tasks
class FCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        # object to store the layer 
        self.blocks = nn.ModuleList()
        # Input Layer
        self.blocks.append(nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        ))
        
        # Hidden Layer
        for _ in range(num_layers - 1):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        # Output Layer
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # Pass through each block
        for block in self.blocks:
            x = block(x)

        return self.lin(x).squeeze(-1)
    


# Define Training Function for the trining loop
def run_epoch(loader, model, device, optimizer, criterion, training=True, epoch_desc=""):
    
    # Define whether or not the function trains or evaluates
    model.train() if training else model.eval()
    
    # Define the running loss
    running_loss, n = 0.0, 0


    # Iterate over the number of epochs
    #for batch in tqdm(loader, desc = epoch_desc, leave = False, ncols = 90):
    for batch in loader:
        # Extract the variables form each
        features, targets, firm_ids, quarters = batch
        features = features.to(device)
        targets =  targets.float().squeeze()

        with torch.set_grad_enabled(training):
            preds = model(features)
            loss = criterion(preds, targets)
        
            if training:
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()

        # Update Loss
        running_loss += loss.item() * targets.size(0)
        n += targets.size(0)
        rmse = math.sqrt(running_loss / n)

    return rmse


#----------------------------------
# 3. Define Optuna Algorithm
#----------------------------------

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
    
    
    # Load the model
    model = FCNN(in_channels = in_channels,
                        hidden_channels = hidden_channels,
                        num_layers = num_layers,
                        dropout = dropout).to(device := "cuda" if torch.cuda.is_available() else "cpu")
    
    # Define error
    criterion = nn.MSELoss()

    # Define optimizer --> Adam is used 
    optimizer = optimizer_fun(model, lr, weight_decay)

    # Define scheduler to alter learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min",
                                                       factor = 0.5, patience = 5)
    
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

        # Update the learning rate
        scheduler.step(val_loss)
        
        # Print Progress
        #outer_bar.set_postfix(
        #    train=f"{train_loss:.4f}",
        #    val=f"{val_loss:.4f}",
        #    lr=optimizer.param_groups[0]['lr'])
        
        # Continue if loss can be reduced significatly 
        if val_loss < best_val:

            best_val = val_loss
            patience_ctr =  0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
           # trial.set_user_attr("state_dict", best_state)
        # Increase patience counter
        else:
            patience_ctr += 1
            
            # Trigger early exist
            if patience_ctr >= patience:
                #outer_bar.write("Early stopping triggered.")
                print("Early stopping triggered. At Epoch:", epoch)
                break
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


# Create function for evaluation table
def make_table(val_loader, test_loader, model, device):

    # Initialize a dictionary to store the results
    results = {
        "Set": ["Validation", "Test"],
        "RMSE": [],
        "MSE": [],
        "MAE": [],
        "MPE": [],
        "R2": [],
        }

    # Set model to evaluation mode
    model.eval()

    # Iterate over the validation set
    all_preds_val, all_targets_val = [], []

    with torch.no_grad():
        for features, targets, _, _ in val_loader:
            features = features.to(device)
            targets = targets.to(device).squeeze()
            preds = model(features)
            all_preds_val.append(preds.cpu().numpy().astype(np.float64))
            all_targets_val.append(targets.cpu().numpy().astype(np.float64))

    # Flatten the lists for the validation set
    all_preds_val = np.concatenate(all_preds_val)
    all_targets_val = np.concatenate(all_targets_val).reshape(-1)

    # Compute metrics for the validation set
    val_rmse = np.sqrt(((all_preds_val - all_targets_val) ** 2).mean())
    val_r2 = r2_score(all_targets_val, all_preds_val)
    val_mae = mean_absolute_error(all_targets_val, all_preds_val)
    val_mse = ((all_preds_val - all_targets_val) ** 2).mean()
    val_mpe = ((all_preds_val - all_targets_val) / all_targets_val).mean()

    # Append validation set metrics to the results
    results["RMSE"].append(val_rmse)
    results["MSE"].append(val_mse)
    results["R2"].append(val_r2)
    results["MAE"].append(val_mae)
    results["MPE"].append(val_mpe)

    # Iterate over the test set
    all_preds_test, all_targets_test = [], []
    with torch.no_grad():
        for features, targets, _, _ in test_loader:
            features = features.to(device)
            targets = targets.to(device).squeeze()
            preds = model(features)
            all_preds_test.append(preds.cpu().numpy().astype(np.float64))
            all_targets_test.append(targets.cpu().numpy().astype(np.float64))
        
    # Flatten the lists for the test set
    all_preds_test = np.concatenate(all_preds_test)
    all_targets_test = np.concatenate(all_targets_test).reshape(-1)

    # Compute metrics for the test set
    test_rmse = np.sqrt(((all_preds_test - all_targets_test) ** 2).mean())
    test_r2 = r2_score(all_targets_test, all_preds_test)
    test_mae = mean_absolute_error(all_targets_test, all_preds_test)
    test_mse = ((all_preds_test - all_targets_test) ** 2).mean()
    test_mpe = ((all_preds_test - all_targets_test) / all_targets_test).mean()

    # Append test set metrics to the results
    results["RMSE"].append(test_rmse)
    results["MSE"].append(test_mse)
    results["R2"].append(test_r2)
    results["MAE"].append(test_mae)
    results["MPE"].append(test_mpe)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df, all_preds_val, all_preds_test


# Function to extract predictions for each observation
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
                features, targets, firm_ids, quarters = batch
                features = features.to(device)
                preds = best_model(features).squeeze().cpu().numpy()
                targets = targets.squeeze().cpu().numpy()

                for i in range(len(firm_ids)):
                    prediction_results.append({
                        "split": split,
                        "firm": firm_ids[i],
                        "quarter": quarters[i],
                        "target": targets[i],
                        "prediction": preds[i],
                        })

    return pd.DataFrame(prediction_results)