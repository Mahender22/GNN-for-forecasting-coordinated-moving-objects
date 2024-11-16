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

def create_adjacency_matrices(X, distance_threshold=15):
    """
    Create adjacency matrices for each time frame based on a distance threshold.

    Parameters:
        X (np.ndarray): Input data array of shape (time_frames, players, features), where features include x and y coordinates.
        distance_threshold (float): Distance threshold for creating adjacency connections.

    Returns:
        list: A list of adjacency matrices for each time frame.

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

    return adjacency_matrices

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
