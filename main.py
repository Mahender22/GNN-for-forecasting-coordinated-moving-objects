import numpy as np
import data_processing
from GCN_LSTM import GCN_LSTM_Model
import torch
import torch.nn as nn
import pickle
import os

# setting parameters

feature_name = 'pitch_control' 
# ['pitch_control', influence']
game = 1
# [1, 2, 3]
model_name = 'MIv2' 
# ['base', 'MIv1', 'MIv2', 'MIv3', 'MIv4', 'same-0.5opp']
aggr = 'add'
# ['mean', 'add', 'mul'] Note: MIv3 only supports 'mul' and 'mul' is only good for MIv3

# load data
soccer_matrix_filename = r'Processed_data\Coords_Influence_'+str(game)+'.npz'
pitch_control = r'Processed_data\pitch_control_'+str(game)+'.npz'

data = data_processing.load_data(soccer_matrix_filename)
pitch_control_data = data_processing.load_data(pitch_control)

# process data
if feature_name == 'pitch_control':
    X = data_processing.preprocess_data(data, pitch_control_data=pitch_control_data, feature='pitch_control')
elif feature_name == 'influence':
    X = data_processing.preprocess_data(data, feature='influence')

# In this data, the first dimension is time frame (seconds), second dimension is player (22 players in a soccer game), 
# third dimension has player feature (influence or pitch control), x coordinate and y coordinate in respective order

# Create adjacency matrices for each time frame
adj_matrices = data_processing.create_adjacency_matrices(X, distance_threshold=15)

edge_indices, edge_distances = data_processing.create_edge_indices_and_distances(X, distance_threshold=15)

max_edges = max(ei.size(1) for ei in edge_indices)
padded_edge_indices, padded_edge_distances = data_processing.pad_edge_data(edge_indices, edge_distances, max_edges)

# Normalize the feature
feature = X[:, :, 0]
X_influence, min_inf_value, max_inf_value = data_processing.normalize_feature(feature)

# Generate input and output sequences for the GCN-LSTM
n_steps_in, n_steps_out = 3, 3  # Define number of input and output time steps
X_sequences, y_sequences = data_processing.create_sequences(X_influence, X_influence, n_steps_in, n_steps_out)

# Align edge indices and distances with the generated sequences
edge_indices_sequences = [padded_edge_indices[i:(i + n_steps_in)] for i in range(len(X_sequences))]
edge_distances_sequences = [padded_edge_distances[i:(i + n_steps_in)] for i in range(len(X_sequences))]

# Split the data into training and test sets
(X_train, y_train, edge_indices_train, edge_distances_train,
 X_test, y_test, edge_indices_test, edge_distances_test) = data_processing.split_data(
    X_sequences, y_sequences, edge_indices_sequences, edge_distances_sequences, train_split=0.8
)

# Convert to PyTorch tensors
X_train, y_train, X_test, y_test = data_processing.convert_to_tensors(X_train, y_train, X_test, y_test)

# Model parameters
num_node_features = 1
gcn_out_features = 128
lstm_hidden_size = 256
output_features = y_train.shape[-1] 
num_nodes = 22  # Number of players
# n_steps_in = 3
n_steps_out = 3

# Instantiate model
model = GCN_LSTM_Model(num_node_features=num_node_features,
                       gcn_out_features=gcn_out_features,
                       lstm_hidden_size=lstm_hidden_size,
                       output_features=output_features,
                       num_nodes=num_nodes,
                       n_steps_out=n_steps_out,
                       model_name=model_name, 
                       aggr=aggr) 

# Determine the new training and validation sizes
n_train_new = int(0.8 * len(X_train))  # 80% of the current training data for the new training set

# Split the data for training and validation
X_train_new, y_train_new = X_train[:n_train_new], y_train[:n_train_new]
X_val, y_val = X_train[n_train_new:], y_train[n_train_new:]

edge_indices_train_new = edge_indices_train[:n_train_new]
edge_indices_val = edge_indices_train[n_train_new:]

edge_distances_train_new = edge_distances_train[:n_train_new]
edge_distances_val = edge_distances_train[n_train_new:]

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

num_epochs = 40
epochs_no_improve = 0
n_epochs_stop = 5
best_val_loss = 1
stop_at_epoch = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass on the training data
    outputs = model(X_train_new, edge_indices_train_new, edge_distances_train_new)
    loss = criterion(outputs, y_train_new)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        val_outputs = model(X_val, edge_indices_val, edge_distances_val)
        val_loss = criterion(val_outputs, y_val)
    
    scheduler.step(val_loss)
    
    train_losses.append(loss)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the model if you want
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            stop_at_epoch = epoch+1
            break  # Stop training

# Testing phase
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Turn off gradients for testing, saves memory and computations
    # Generate predictions with the test data
    test_outputs = model(X_test, edge_indices_test, edge_distances_test)
    # Compute the loss (or other metrics) on the test set
    test_loss = criterion(test_outputs, y_test)

print(f'Test Loss: {test_loss.item()}')

model_parameters = {
    'model': model_name,
    'aggregation': aggr,
    'game': game,
    'y_true': y_test.detach().numpy(),
    'y_pred': test_outputs.detach().numpy(),
    'train_loss': train_losses,
    'val_loss': val_losses
    # 'residue_all_samples': residue_all_samples,
    # 'mean_residue_per_player': mean_residue_per_player,
    # 'average_residue_all_players': average_residue_all_players,
    # 'median_residue_per_player': median_residue_per_player
}

save_dir = r'saved_'+str(feature_name)+'_results'

# If the directory does not exist, create one
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Specify the file name
filename = save_dir+'\\'+str(model_name)+'_'+str(aggr)+'_'+str(game)+'.pkl'

# Save the variables to a file
with open(filename, 'wb') as file:
    pickle.dump(model_parameters, file)

print(f'Model parameters saved to {filename}')