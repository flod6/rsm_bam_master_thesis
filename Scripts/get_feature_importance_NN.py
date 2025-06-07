#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import torch
import shap

# Import other objects
from FCNN_objects_Regression import prepare_FCNN

# Define paths 
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"


# Load Model and data
fcnn_model = torch.load(output + "Models/covar_best_FCNN_Regression_abolute_full.pt")

covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")



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




#----------------------------------
# 3. Features GNN
#----------------------------------