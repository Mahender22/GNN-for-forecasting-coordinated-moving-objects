import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def calculate_residue(true_influences, predicted_influences, epsilon=1e-10):
    """
    Calculate the residue between true and predicted influences.
    
    Parameters:
        true_influences (np.ndarray): True influence values of shape [samples, players, time].
        predicted_influences (np.ndarray): Predicted influence values of the same shape.
        epsilon (float): A small value to avoid division by zero.
    
    Returns:
        np.ndarray: Residue values of the same shape.
    
    Raises:
        ValueError: If the shapes of true and predicted influences do not match.
    """
    if true_influences.shape != predicted_influences.shape:
        raise ValueError("Shapes of true and predicted influences must match.")
    
    # Calculate the residue, adding epsilon to avoid division by zero
    residue = np.abs(true_influences - predicted_influences) / (true_influences + epsilon)
    
    return residue

def calculate_residue_metrics(residue_all_samples):
    """
    Calculate various metrics for the residue.
    
    Parameters:
        residue_all_samples (np.ndarray): Residue values of shape [samples, players, time].
    
    Returns:
        dict: Calculated metrics including median and mean residue per player and overall average residue.
    """
    metrics = {
        'median_residue_per_player': np.median(residue_all_samples, axis=(0, 1)),
        'average_residue_all_players': np.mean(residue_all_samples),
        'mean_residue_per_player': np.mean(residue_all_samples, axis=(0, 1))
    }
    return metrics

def load_all_pkl_files(folder_path):
    """
    Load all .pkl files from a folder into a dictionary.
    If metrics are missing, calculate and add them to the results.

    Parameters:
        folder_path (str): Path to the folder containing .pkl files.

    Returns:
        dict: Dictionary where keys are file names (without extensions) and values are loaded data.
    """
    all_models_results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                model_results = pickle.load(file)
            
            # Calculate residue and metrics if missing
            if 'residue_all_samples' not in model_results:
                true_influences = model_results['y_true']
                predicted_influences = model_results['y_pred']
                residue_all_samples = calculate_residue(true_influences, predicted_influences)
                model_results['residue_all_samples'] = residue_all_samples

            if 'median_residue_per_player' not in model_results or \
               'average_residue_all_players' not in model_results or \
               'mean_residue_per_player' not in model_results:
                metrics = calculate_residue_metrics(model_results['residue_all_samples'])
                model_results.update(metrics)

            # Save updated results back to the file
            with open(file_path, 'wb') as file:
                pickle.dump(model_results, file)

            model_name = os.path.splitext(filename)[0]
            all_models_results[model_name] = model_results

    return all_models_results

def plot_median_residue_per_player(all_models_results, title="Median Residue per Player", filter_game=None, filter_models=None):
    """
    Plot median residue per player across different models.

    Parameters:
        all_models_results (dict): Dictionary containing results for all models.
        title (str): Title for the plot.
        filter_game (int): If provided, only plot results for the specified game.
        filter_models (list): If provided, only include specified models.
    """
    plt.figure(figsize=(14, 6))
    players = np.arange(1, 23)
    max_val = 0

    for model_name, results in all_models_results.items():
        if filter_game is not None and results['game'] != filter_game:
            continue
        if filter_models is not None and results['model'] not in filter_models:
            continue

        median_residue_per_player = results['median_residue_per_player']
        plt.plot(players, median_residue_per_player, 'o-', label=model_name)
        max_val = max(max_val, np.max(median_residue_per_player))

    plt.xlabel('Player', fontsize=16)
    plt.ylabel('Median Residue', fontsize=16)
    plt.xticks(players, fontsize=12)
    # plt.yticks(np.arange(0, max_val + 0.1, 0.1), fontsize=12)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=12)
    plt.show()