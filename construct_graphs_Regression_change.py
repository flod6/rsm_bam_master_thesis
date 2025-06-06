#----------------------------------
# 1. Set-Up
#----------------------------------

# Load Libraries
import pandas as pd
import numpy as np 
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import torch
from matplotlib.colors import LogNorm

# Define Paths
input = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Input/"
output = "/Users/floriandahlbender/Documents/Erasmus University Rotterdam/Courses/BAM Master Thesis/03 Code/Output/"

# Load Data 
covar_final = pd.read_parquet(input + "Clean/covar_final.parquet")

# Load the different adjacency matrices
file_names = [f for f in os.listdir(input + "Adjacency_Matrices/") if os.path.isfile(os.path.join(input + "Adjacency_Matrices/", f))]

# Load all datasets into a dictionary
adjacency_matrices = {}

# Iterate over the different files in the folder 
for file_name in file_names:
    # Extract complete file name 
    file_path = os.path.join(input, "Adjacency_Matrices", file_name)
    
    # Only load data stored in parquet files 
    if file_name.endswith('.parquet'):
        # Only select the files with gvkey in the name and append them 
        # to the dictionary
        if "gvkey" in file_name:
            adjacency_matrices[file_name] = pd.read_parquet(file_path)


# Prepare the data frame and drop columns that are not needed
covar_final = covar_final.drop(columns=["covar", "lag_covar"])


#----------------------------------
# 2. Create Graphs for CoVar
#----------------------------------

# Create an empty dictionary for store the graphs
covar_graphs = {}


# Iterate over the different periods / i.e. adjacency matrices
for i, j in adjacency_matrices.items():
    
    # Extract the bank keys 
    bank_keys = j.columns
    
    # Add rownames for later
    j.index = bank_keys
    
    # Extract the period name
    period_name = i.rsplit("_", 1)[-1].split(".")[0]
    
    # Stop the create a graph for the period when there is no data
    if period_name not in covar_final["date"].values:
        continue
    
    # Filter the period 
    tmp_sys_risk = covar_final[covar_final["date"] == period_name]
    
    #-------
    
    # Create an empty graph object 
    G = nx.DiGraph()
    
    # Select columns not included as features
    skip_cols = ["gvkey", "date", "conm", "diff_delta_covar"] 

    # Extract metadata columns
    metadata = tmp_sys_risk[["date", "gvkey"]]
    
    # Select all columns that are float or numeric 
    feature_cols = (
            tmp_sys_risk.drop(columns=skip_cols)
              .select_dtypes(include=[np.number])
              .columns.tolist())

    
    # Create copy 
    tmp_sys_risk = tmp_sys_risk.copy() 
    
    # Skip any values that are not stores as int or float 
    tmp_sys_risk[feature_cols] = (
            tmp_sys_risk[feature_cols]
            .apply(pd.to_numeric, errors="coerce")   # convert strings → numbers / NaN
            .astype(np.float32))
    

    # Add nodes and node features
    for _, row in tmp_sys_risk.iterrows():
        node_id = row["gvkey"]
        y_value  = torch.tensor([row["diff_delta_covar"]], dtype=torch.float32)  
        x_features = torch.tensor(
            row[feature_cols].to_numpy(dtype=np.float32, copy=False),
            dtype=torch.float32)
        date_encoded = int(str(row["date"]).replace("Q", ""))
        gvkey_encoded = int(row["gvkey"])
        metadata_tensor = torch.tensor([date_encoded, gvkey_encoded], dtype=torch.float32)
        G.add_node(node_id, x=x_features, y=y_value, metadata=metadata_tensor)
    
    #-------
    
    # Add the edges:
    # Iterate over the columns in the adjacency matrix 
    for k in j.columns:
        # Iterate over the rows
        for l in j.index:
            # Check that there are no edges between the same nodes 
            if k != l:
                # Only add the edge if both nodes exist in the graph
                if G.has_node(k) and G.has_node(l):
                    # Retrieve the edge weight from the adjacency matrix
                    weight = j.at[l, k]
                    # Add the edge if weight is not zero and not NaN
                    if pd.notna(weight) and weight != 0:
                        G.add_edge(k, l, weight=weight)
    
    # Append the graphs                 
    covar_graphs[period_name] = G
    
    #-----
    
    ## Graph Visualization

    # Visualize the current graph with adjustments to reduce visual clutter
    plt.figure(figsize=(12, 10))
    # Use a spring layout with a smaller 'k' value for a more spaced-out layout
    pos = nx.spring_layout(G, seed=187, k=0.3)
    
    # Determine whether to display node labels based on the node count
    show_labels = len(G.nodes()) <= 20
    
    # Use the node 'feature' attribute for coloring if it exists and is numeric
    if all(isinstance(G.nodes[node].get('feature'), (int, float)) for node in G.nodes()):
        node_color = [G.nodes[node]['feature'] for node in G.nodes()]
    else:
        node_color = "skyblue"
        
    node_color = [G.nodes[n]['y'] for n in G]
    
    # Draw edges with lower alpha and thinner width for clarity
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), alpha=0.5, width=1)
    
    # Draw nodes with adjusted size and edge colors
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.Reds,
                           edgecolors='black', node_size=200)
    
    # Draw labels only if the graph isn't too crowded
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    else:
        print("Node labels skipped due to high node count.")
    
    plt.title(f"Graph Visualization: Period {period_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Save the current figure to the output folder
    fig_filename = os.path.join(output, f"Graph_Figures/graph_{period_name}.png")
    plt.savefig(fig_filename)
    plt.close()

# Save the results as a pickle file 
save_path = os.path.join(input, "Clean/covar_graphs_Regression_change.pkl")
with open(save_path, "wb") as f:
    pickle.dump(covar_graphs, f)



### Sanity Checks

## Double Check that there are no mistakes 

# Double-check that the number of nodes in each covar graph matches the number of unique banks in covar_final for that period
for period, graph in covar_graphs.items():
  # Filter covar_final for the current period and count unique banks by 'gvkey'
  expected_count = covar_final[covar_final["date"] == period]["gvkey"].nunique()
  actual_count = len(graph.nodes())
  
  if actual_count == expected_count:
    print(f"Period {period}: OK - Graph nodes ({actual_count}) match unique banks ({expected_count}).")
  else:
    print(f"Period {period}: MISMATCH - Graph nodes ({actual_count}) do not match unique banks ({expected_count}).")
        


# Print the 'y' attribute for all nodes in the graph
for node_id in G.nodes:
    print(f"Node {node_id}: y = {G.nodes[node_id]['y']}")


    # Assess features for the first graph and ensure correct variables are used
    first_period = list(covar_graphs.keys())[0]  # Get the first period
    first_graph = covar_graphs[first_period]    # Get the corresponding graph

    # Extract all features from the nodes in the first graph
    node_features = {}
    for node_id in first_graph.nodes:
        node_features[node_id] = first_graph.nodes[node_id]['x'].numpy()

    # Convert features to a DataFrame for easier inspection
    features_df = pd.DataFrame.from_dict(node_features, orient='index')

    # Display the first few rows of the features DataFrame
    print("Features for the first graph:")
    print(features_df.head())

    # Check for missing or invalid values in the features
    if features_df.isnull().values.any():
        print("Warning: Missing values detected in the features.")
    else:
        print("No missing values")