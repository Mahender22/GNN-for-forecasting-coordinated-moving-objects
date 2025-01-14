import numpy as np
import os
from scipy.spatial.distance import cdist
import torch

def load_data(filename):
    """
    Load soccer matrix data from a .npz file.

    Parameters:
        filename (str): Path to the .npz file.

    Returns:
        np.ndarray: Loaded data from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain the expected 'data' key.
        IOError: If the file cannot be read due to corruption or invalid format.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    try:
        # Attempt to load the file
        data = np.load(filename)
    except Exception as e:
        raise IOError(f"An error occurred while reading the file '{filename}': {e}")

    # Check if the 'data' key exists in the file
    if 'data' not in data:
        raise ValueError(f"The file '{filename}' does not contain the expected 'data' key.")

    return data['data']

def preprocess_data(data, pitch_control_data=None, feature='influence'):
    """
    Preprocess the data based on the selected feature.

    Parameters:
        data (np.ndarray): Input data array.
        pitch_control_data (np.ndarray, optional): Pitch control data array (required if feature is 'pitch_control').
        feature (str): Specifies the feature to process. 
                       - 'influence': Returns the unprocessed data directly.
                       - 'pitch_control': Replaces influence with pitch control and removes NaN rows.
    
    Returns:
        np.ndarray: Processed data.

    Raises:
        ValueError: If pitch_control_data is not provided for 'pitch_control'.
        ValueError: If an unsupported feature is provided.
    """
    X = data

    if feature == 'pitch_control':

        if pitch_control_data is None:
            raise ValueError("pitch_control_data must be provided when feature is not 'influence'.")

        X = X[1:]  # Skip the first time frame
        
        # Replace the influence with pitch control
        X[:, :, 0] = pitch_control_data

        # Identify rows where any pitch control value is NaN
        nan_rows = np.isnan(pitch_control_data).any(axis=1)

        # Filter out rows with NaN values in pitch control
        X = X[~nan_rows]
    
    elif feature != 'influence':
        raise ValueError(f"Unsupported feature: {feature}. Supported features are 'influence' and 'pitch_control'.")

    return X

def create_adjacency_matrices(X, distance_threshold=15, mode='standard'):
    """
    Create adjacency matrices for each time frame based on a distance threshold.

    Parameters:
        X (np.ndarray): Input data array of shape (time_frames, players, features), where features include x and y coordinates.
        distance_threshold (float): Distance threshold for creating adjacency connections.
        mode (str): Specifies the mode of adjacency matrix creation. 
            - 'standard': Create adjacency matrices for all players.
            - 'adversarial': Create separate adjacency matrices for home and away teams.

    Returns:
        list or tuple:
            - If mode is 'standard', returns a list of adjacency matrices for all players.
            - If mode is 'adversarial', returns two lists of adjacency matrices: one for the home team and one for the away team.

    Raises:
        ValueError: If the input array X has an invalid shape.
        ValueError: If distance_threshold is not a positive number.
    """
    # Validate the input array
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 3:
        raise ValueError(f"Input X must be a 3D array, but got {X.ndim}D array.")
    if X.shape[2] < 3:
        raise ValueError(f"Input X must have at least 3 features in the third dimension (influence, x, y), but got {X.shape[2]}.")

    # Validate distance_threshold
    if not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be a positive number.")

    num_time_frames = X.shape[0]
    num_players = X.shape[1]
    adjacency_matrices = []

    for t in range(num_time_frames):
        try:
            coordinates = X[t, :, 1:3]  # Extract x, y coordinates
            distances = cdist(coordinates, coordinates, 'euclidean')
            adjacency_matrix = (distances < distance_threshold) & (distances != 0)
            adjacency_matrices.append(adjacency_matrix.astype(np.int8))
        except Exception as e:
            raise RuntimeError(f"Error processing time frame {t}: {e}")
    if mode == 'standard':
        return adjacency_matrices
    else:
        home_team_matrices = []
        away_team_matrices = []

        for adjacency_matrix in adjacency_matrices:
            # Extract home team (players 0-10) and away team (players 11-21)
            home_team_matrix = adjacency_matrix[:11, :11]
            away_team_matrix = adjacency_matrix[11:, 11:]

            home_team_matrices.append(home_team_matrix.astype(np.int8))
            away_team_matrices.append(away_team_matrix.astype(np.int8))

        return home_team_matrices, away_team_matrices

def split_teams(X):
    """
    Split the input data X into two separate arrays for home and away teams.

    Parameters:
        X (np.ndarray): Input data array of shape (time_frames, players, features)
                        or (time_frames, players), where players are ordered as 
                        home team (0-10) and away team (11-21).

    Returns:
        tuple: Two numpy arrays (home_team_data, away_team_data) of shape:
               - If input is 3D: (time_frames, players_per_team, features)
               - If input is 2D: (time_frames, players_per_team)
    
    Raises:
        ValueError: If the input data does not have 22 players in the second dimension.
    """
    # Validate the input array
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim not in [2, 3]:
        raise ValueError(f"Input X must be a 2D or 3D array, but got {X.ndim}D array.")
    if X.shape[1] != 22:
        raise ValueError(f"Input X must have 22 players in the second dimension, but got {X.shape[1]} players.")

    # Split into home and away teams
    home_team_data = X[:, :11]  # Players 0-10
    away_team_data = X[:, 11:]  # Players 11-21

    return home_team_data, away_team_data

def create_edge_indices_and_distances(X, distance_threshold=15):
    """
    Create edge indices and distances for each time frame based on a distance threshold.
    'edge_indices' is a list of 2-row tensors, one for each time frame
    
    Parameters:
        X (np.ndarray): Input data array of shape (time_frames, players, features), where features include x and y coordinates.
        distance_threshold (float): Distance threshold for creating edges.

    Returns:
        tuple: A list of 2-row tensors (edge indices) and a list of tensors (edge distances) for each time frame.

    Raises:
        ValueError: If the input array X has an invalid shape.
        ValueError: If distance_threshold is not a positive number.
        RuntimeError: If an error occurs during edge creation for a specific time frame.
    """
    # Validate the input array
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 3:
        raise ValueError(f"Input X must be a 3D array, but got {X.ndim}D array.")
    if X.shape[2] < 3:
        raise ValueError(f"Input X must have at least 3 features in the third dimension (influence, x, y), but got {X.shape[2]}.")

    # Validate distance_threshold
    if not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be a positive number.")

    num_time_frames = X.shape[0]
    edge_indices = []
    edge_distances = []

    for t in range(num_time_frames):
        try:
            coordinates = X[t, :, 1:3]  # Extract x, y coordinates
            distances = cdist(coordinates, coordinates, 'euclidean')
            adjacency_matrix = (distances < distance_threshold) & (distances != 0)

            edges = []
            dists = []
            for i in range(adjacency_matrix.shape[0]):
                for j in range(adjacency_matrix.shape[1]):
                    if adjacency_matrix[i, j]:
                        edges.append((i, j))
                        dists.append(distances[i, j])

            edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_distances_tensor = torch.tensor(dists, dtype=torch.float)

            edge_indices.append(edge_index_tensor)
            edge_distances.append(edge_distances_tensor)

        except Exception as e:
            raise RuntimeError(f"Error processing time frame {t}: {e}")

    return edge_indices, edge_distances

def create_interteam_graphs_with_existing_graphs(X, home_team_graphs, away_team_graphs, home_team_X, away_team_X, distance_threshold=15):
    """
    Create inter-team graphs by connecting nodes in home and away team graphs based on a distance threshold.

    Parameters:
        X (np.ndarray): Original input data of shape (time_frames, players, features), 
                        where features include x and y coordinates.
        home_team_graphs (list): Precomputed adjacency matrices for home team graphs.
        away_team_graphs (list): Precomputed adjacency matrices for away team graphs.
        home_team_X (np.ndarray): Data for home team of shape (time_frames, players_per_team, features).
        away_team_X (np.ndarray): Data for away team of shape (time_frames, players_per_team, features).
        distance_threshold (float): Distance threshold for creating edges.

    Returns:
        tuple: Two lists of inter-team edge indices and distances:
            - home_to_away_edges: Edges from home team nodes to nearest away team graphs.
            - away_to_home_edges: Edges from away team nodes to nearest home team graphs.
    """
    # Validate the input array
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 3:
        raise ValueError(f"Input X must be a 3D array, but got {X.ndim}D array.")
    if X.shape[2] < 3:
        raise ValueError(f"Input X must have at least 3 features (influence, x, y), but got {X.shape[2]}.")
    
    num_time_frames = X.shape[0]
    home_to_away_edges = []
    away_to_home_edges = []

    for t in range(num_time_frames):
        # Coordinates for the current time frame
        home_coords = home_team_X[t, :, 1:3]  # x, y coordinates for home team
        away_coords = away_team_X[t, :, 1:3]  # x, y coordinates for away team
        distances_home_to_away = cdist(home_coords, away_coords, 'euclidean')
        distances_away_to_home = cdist(away_coords, home_coords, 'euclidean')

        # Edges and distances for the current time frame
        home_edges = []
        home_dists = []
        away_edges = []
        away_dists = []

        # Iterate over all home nodes
        for home_node in range(home_team_graphs[t].shape[0]):
            # Neighbors of the current home node
            home_neighbors = np.where(home_team_graphs[t][home_node] == 1)[0]
            # Nearest away node
            nearest_away_node = np.argmin(distances_home_to_away[home_node])

            # Check all neighbors of the nearest away node
            away_neighbors = np.where(away_team_graphs[t][nearest_away_node] == 1)[0]
            away_neighbors = np.append(away_neighbors, nearest_away_node)  # Include the node itself

            # Create edges between the current home node's neighbors and the nearest away node's neighbors
            for hn in home_neighbors:
                for an in away_neighbors:
                    if distances_home_to_away[hn, an] < distance_threshold:
                        home_edges.append((hn, an))
                        home_dists.append(distances_home_to_away[hn, an])

        # Iterate over all away nodes
        for away_node in range(away_team_graphs[t].shape[0]):
            # Neighbors of the current away node
            away_neighbors = np.where(away_team_graphs[t][away_node] == 1)[0]
            # Nearest home node
            nearest_home_node = np.argmin(distances_away_to_home[away_node])

            # Check all neighbors of the nearest home node
            home_neighbors = np.where(home_team_graphs[t][nearest_home_node] == 1)[0]
            home_neighbors = np.append(home_neighbors, nearest_home_node)  # Include the node itself

            # Create edges between the current away node's neighbors and the nearest home node's neighbors
            for an in away_neighbors:
                for hn in home_neighbors:
                    if distances_away_to_home[an, hn] < distance_threshold:
                        away_edges.append((an, hn))
                        away_dists.append(distances_away_to_home[an, hn])

        # Store results for the current time frame
        home_to_away_edges.append(torch.tensor(home_edges, dtype=torch.long).t().contiguous())
        away_to_home_edges.append(torch.tensor(away_edges, dtype=torch.long).t().contiguous())

    return home_to_away_edges, away_to_home_edges

def pad_edge_data(edge_indices, edge_distances, max_edges):
    """
    Pad edge indices and edge distances to ensure consistent sizes across all time frames.

    Parameters:
        edge_indices (list): A list of 2-row tensors representing edge indices for each time frame.
        edge_distances (list): A list of tensors representing edge distances for each time frame.
        max_edges (int): The maximum number of edges to pad to.

    Returns:
        tuple: Two lists:
            - padded_edge_indices: List of 2-row tensors with padded edge indices.
            - padded_edge_distances: List of tensors with padded edge distances.

    Raises:
        ValueError: If edge_indices and edge_distances have different lengths.
        ValueError: If max_edges is not a positive integer.
        RuntimeError: If an error occurs during padding.
    """
    # Validate input lengths
    if len(edge_indices) != len(edge_distances):
        raise ValueError("edge_indices and edge_distances must have the same length.")
    
    # Validate max_edges
    if not isinstance(max_edges, int) or max_edges <= 0:
        raise ValueError("max_edges must be a positive integer.")

    padded_edge_indices = []
    padded_edge_distances = []

    for idx, dist in zip(edge_indices, edge_distances):
        try:
            # Padding edge indices
            pad_size_idx = max_edges - idx.size(1)
            if pad_size_idx > 0:
                pad_idx = torch.full((2, pad_size_idx), -1, dtype=idx.dtype, device=idx.device)
                padded_idx = torch.cat([idx, pad_idx], dim=1)
            else:
                padded_idx = idx

            # Padding edge distances
            pad_size_dist = max_edges - dist.size(0)
            if pad_size_dist > 0:
                pad_dist = torch.full((pad_size_dist,), float('inf'), dtype=dist.dtype, device=dist.device)
                padded_dist = torch.cat([dist, pad_dist])
            else:
                padded_dist = dist

            padded_edge_indices.append(padded_idx)
            padded_edge_distances.append(padded_dist)

        except Exception as e:
            raise RuntimeError(f"Error occurred while padding edge data: {e}")

    return padded_edge_indices, padded_edge_distances

def create_sequences(input_data, output_data, n_steps_in, n_steps_out):
    """
    Create sequences for input and output.
    
    Parameters:
        input_data (np.ndarray): Input data array.
        output_data (np.ndarray): Output data array.
        n_steps_in (int): Number of input time steps.
        n_steps_out (int): Number of output time steps.
    
    Returns:
        tuple: Arrays of input and output sequences (X, y).
    """
    X, y = [], []
    for i in range(len(input_data) - n_steps_in - n_steps_out + 1):
        seq_x = input_data[i:(i + n_steps_in)]
        seq_y = output_data[(i + n_steps_in):(i + n_steps_in + n_steps_out)]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def normalize_feature(feature):
    """
    Normalize a feature array using Min-Max normalization.
    
    Parameters:
        feature (np.ndarray): Feature array to be normalized.
    
    Returns:
        np.ndarray: Normalized feature array.
        tuple: Minimum and maximum values used for normalization.
    """
    min_value = np.min(feature)
    max_value = np.max(feature)
    normalized = (feature - min_value) / (max_value - min_value)
    return normalized, min_value, max_value

def split_data(X_sequences, y_sequences, edge_indices_sequences, edge_distances_sequences, train_split=0.8):
    """
    Split data into training and testing sets.
    
    Parameters:
        X_sequences (np.ndarray): Input sequences.
        y_sequences (np.ndarray): Output sequences.
        edge_indices_sequences (list): Edge indices for input sequences.
        edge_distances_sequences (list): Edge distances for input sequences.
        train_split (float): Fraction of data to use for training.
    
    Returns:
        tuple: Training and testing sets for inputs, outputs, edge indices, and edge distances.
    """
    n_train = int(train_split * X_sequences.shape[0])
    X_train, y_train = X_sequences[:n_train], y_sequences[:n_train]
    X_test, y_test = X_sequences[n_train:], y_sequences[n_train:]
    
    edge_indices_train = edge_indices_sequences[:n_train]
    edge_indices_test = edge_indices_sequences[n_train:]
    
    edge_distances_train = edge_distances_sequences[:n_train]
    edge_distances_test = edge_distances_sequences[n_train:]
    
    return (X_train, y_train, edge_indices_train, edge_distances_train,
            X_test, y_test, edge_indices_test, edge_distances_test)

def convert_to_tensors(*arrays):
    """
    Convert arrays to PyTorch tensors.
    
    Parameters:
        *arrays: Input arrays to convert.

    Returns:
        tuple: Converted PyTorch tensors.
    """
    return tuple(torch.tensor(arr, dtype=torch.float) for arr in arrays)

def normalizeAdjacency(W):
    """
    Normalize the given adjacency matrix using symmetric normalization.

    Parameters:
        W (torch.Tensor): A 2D square adjacency matrix of shape (N, N) 
                          where N is the number of nodes.

    Returns:
        torch.Tensor: The normalized adjacency matrix of shape (N, N).

    Raises:
        TypeError: If the input W is not a torch.Tensor.
        ValueError: If the input matrix W is not a square matrix.
        ValueError: If any value on the diagonal of the degree matrix is zero (to avoid division by zero).
    """
    # Check if W is a tensor
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"Expected 'W' to be a torch.Tensor, but got {type(W)} instead.")
    
    # Check if W is a square matrix
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Input matrix 'W' must be a square matrix, but got shape {W.shape}.")
    
    # Compute the degree vector
    d = torch.sum(W, axis=1)

    # Check if the degree vector has any zero values to avoid division by zero
    if torch.any(d == 0):
        raise ValueError("Degree vector contains zero values, leading to division by zero. Check the input adjacency matrix.")
    
    # Invert the square root of the degree
    d = 1 / torch.sqrt(d)

    # Build the square root inverse degree matrix
    D = torch.diag(d)

    # Return the normalized adjacency matrix
    normalized_adjacency = D @ W @ D
    return normalized_adjacency


def snap_to_adjmat(snapshot):
    """
    Convert a snapshot graph into an adjacency matrix.

    Parameters:
        snapshot (torch_geometric.data.Data): A snapshot graph from a temporal dataset, 
                                              containing `edge_index` and `num_nodes`.

    Returns:
        torch.Tensor: The adjacency matrix of shape (num_nodes, num_nodes).

    Raises:
        TypeError: If the input 'snapshot' is not a PyTorch Geometric data object.
        ValueError: If 'snapshot' does not contain 'edge_index' or 'num_nodes' attributes.
    """
    # Validate input is a torch_geometric Data object
    if not hasattr(snapshot, 'edge_index') or not hasattr(snapshot, 'num_nodes'):
        raise ValueError("Input 'snapshot' must be a torch_geometric data object with 'edge_index' and 'num_nodes' attributes.")
    if not isinstance(snapshot.edge_index, torch.Tensor):
        raise TypeError(f"'snapshot.edge_index' must be a torch.Tensor, but got {type(snapshot.edge_index)} instead.")
    if not isinstance(snapshot.num_nodes, int) or snapshot.num_nodes <= 0:
        raise ValueError(f"'snapshot.num_nodes' must be a positive integer, but got {snapshot.num_nodes} instead.")

    # Create an empty adjacency matrix of size (num_nodes, num_nodes)
    num_nodes = snapshot.num_nodes
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Populate the adjacency matrix using edge_index
    for j in range(snapshot.edge_index.size(1)):
        v1 = snapshot.edge_index[:, j][0].item()
        v2 = snapshot.edge_index[:, j][1].item()
        adjacency_matrix[v1][v2] = 1.0

    # Add self-loops to the adjacency matrix
    for i in range(num_nodes):
        adjacency_matrix[i][i] = 1.0

    # Optional: Symmetric normalization of the adjacency matrix (commented out)
    # adjacency_matrix = normalizeAdjacency(adjacency_matrix)

    return adjacency_matrix


def similarity(vec1, vec2):
    """
    Calculate the similarity between two vectors using dot product.

    Parameters:
        vec1 (torch.Tensor): A 1D vector of shape (N,).
        vec2 (torch.Tensor): A 1D vector of shape (N,).

    Returns:
        torch.Tensor: A scalar tensor representing the similarity score.

    Raises:
        TypeError: If the input 'vec1' or 'vec2' is not a torch.Tensor.
        ValueError: If the input vectors 'vec1' and 'vec2' do not have the same shape.
    """
    # Check if both inputs are tensors
    if not isinstance(vec1, torch.Tensor) or not isinstance(vec2, torch.Tensor):
        raise TypeError(f"Both 'vec1' and 'vec2' must be torch.Tensor objects, but got {type(vec1)} and {type(vec2)}.")
    
    # Check if both tensors are 1D and have the same shape
    if vec1.shape != vec2.shape:
        raise ValueError(f"Input vectors 'vec1' and 'vec2' must have the same shape, but got {vec1.shape} and {vec2.shape}.")
    
    # Compute and return the dot product of vec1 and vec2 as similarity
    similarity_score = torch.dot(vec1, vec2)
    return similarity_score