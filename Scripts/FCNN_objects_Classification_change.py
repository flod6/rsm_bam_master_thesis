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


#----------------------------------
# 2. Prepare Data for FCNN 
#----------------------------------

# Define Funtion that prepares the data for the input
def prepare_FCNN(data, batch_size = 1024): 
    
    # Rename risk variables to y
    #if 'diff_delta_covar' in data.columns:
    #    data = data.rename(columns={"diff_delta_covar": "y"})
    #elif 'diff_SRISK' in data.columns:
    #    data = data.rename(columns={"diff_SRISK": "y"})

    data["y"] = (data["y"] > 0).astype(np.float32)

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





class FCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        # Input Layer
        self.blocks = nn.ModuleList()
        # first hidden layer
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
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # Pass through each block
        for block in self.blocks:
            x = block(x)

        # 3) final regression output (no activation)
        return torch.sigmoid(self.out(x).squeeze(-1))



#----------------------------------
# 3. Define Epoch Trainer
#----------------------------------

# Define function to run epochs later
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
        
        optimizer.zero_grad(set_to_none = True)

        with torch.set_grad_enabled(training):
            preds = model(features)
            loss = criterion(preds, targets)
        
            if training:
                loss.backward()
                optimizer.step()

        # Update Loss
        running_loss += loss.item() * targets.size(0) 
        n += targets.size(0) 

    avg_loss = running_loss / n
    return avg_loss


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
    criterion = nn.BCELoss()

    # Define optimizer --> Adam is used 
    optimizer = optimizer_fun(model, lr, weight_decay)

    # Define scheduler to alter learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min",
                                                       factor = 0.1, patience = 5)
    
    #----------------------------------
    # Initiate Training Loop
    #----------------------------------

    # Define Parameters for the training loop
    best_val = math.inf # Validation set
    patience, patience_ctr = 50, 0 # Set patience for early quiting 

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


# Create table for evaluation:

def make_table(val_loader, test_loader, best_model, device, threshold=0.5):
    def evaluate(loader, split_name):
        best_model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets, firm_ids, quarters in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = best_model(inputs)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        pred_labels = (all_preds > threshold).astype(int)

        return {
            "Accuracy": accuracy_score(all_targets, pred_labels),
            "ROC AUC": roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else np.nan,
            "F1 Score": f1_score(all_targets, pred_labels),
            "Precision": precision_score(all_targets, pred_labels),
            "Recall": recall_score(all_targets, pred_labels),
            "Confusion Matrix": confusion_matrix(all_targets, pred_labels),
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

    return results_df, val_results, test_results


# Create function to extract the individual predictions
def extract_predictions(train_loader, val_loader, test_loader, model, device="cpu"):
  model.eval()
  model.to(device)

  prediction_results = []
  loaders = {
  "train": train_loader,
  "validation": val_loader,
  "test": test_loader,
  }

  with torch.no_grad():
    for split, loader in loaders.items():
      for batch in loader:
        # Unpack the batch as 4 elements
          inputs, targets, firm_ids, quarters = batch

          inputs = inputs.to(device)
          targets = targets.to(device)

          # Make predictions 
          preds = model(inputs).squeeze().cpu().numpy()
          targets = targets.squeeze().cpu().numpy()

          # Convert meta data for tensors 
          firm_ids = [fid for fid in firm_ids]
          quarters = [q for q in quarters]

          for i in range(len(preds)):
            prediction_results.append({
              "split": split,
              "gvkey": firm_ids[i],
              "quarter": quarters[i],
              "target": targets[i],
              "prediction": preds[i],
              })
            
  prediction_results = pd.DataFrame(prediction_results)

  return prediction_results