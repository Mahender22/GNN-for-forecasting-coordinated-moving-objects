import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import preprocessing as p
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy as sp
import scipy.spatial
import sys
import matplotlib.image as mpimg
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve
import matplotlib.patches as patches
import os
from sklearn.neighbors import NearestNeighbors

def data_cleaning(data):
    """
    Clean and process player tracking data
    """
    data = data.copy()
    column_names = data.columns.tolist()
    
    if column_names[3] == 'Player11':  # Home team
        data = data.rename(columns={
            'Player11': '11_x', 'Unnamed: 4': '11_y', 'Player1': '1_x',
            'Unnamed: 6': '1_y', 'Player2': '2_x', 'Unnamed: 8': '2_y', 
            'Player3': '3_x', 'Unnamed: 10': '3_y', 'Player4': '4_x', 
            'Unnamed: 12': '4_y', 'Player5': '5_x', 'Unnamed: 14': '5_y', 
            'Player6': '6_x', 'Unnamed: 16': '6_y', 'Player7': '7_x',
            'Unnamed: 18': '7_y', 'Player8': '8_x', 'Unnamed: 20': '8_y', 
            'Player9': '9_x', 'Unnamed: 22': '9_y', 'Player10': '10_x', 
            'Unnamed: 24': '10_y', 'Player12': '12_x', 'Unnamed: 26': '12_y', 
            'Player13': '13_x', 'Unnamed: 28': '13_y', 'Player14': '14_x', 
            'Unnamed: 30': '14_y', 'Ball': 'Ball_x', 'Unnamed: 32': 'Ball_y'
        }).copy()
        main_players = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        subs = [12, 13, 14]
    else:  # Away team
        data = data.rename(columns={
            'Player25': '25_x', 'Unnamed: 4': '25_y', 'Player15': '15_x',
            'Unnamed: 6': '15_y', 'Player16': '16_x', 'Unnamed: 8': '16_y',
            'Player17': '17_x', 'Unnamed: 10': '17_y', 'Player18': '18_x', 
            'Unnamed: 12': '18_y', 'Player19': '19_x', 'Unnamed: 14': '19_y', 
            'Player20': '20_x', 'Unnamed: 16': '20_y', 'Player21': '21_x', 
            'Unnamed: 18': '21_y', 'Player22': '22_x', 'Unnamed: 20': '22_y',
            'Player23': '23_x', 'Unnamed: 22': '23_y', 'Player24': '24_x', 
            'Unnamed: 24': '24_y', 'Player26': '26_x', 'Unnamed: 26': '26_y', 
            'Player27': '27_x', 'Unnamed: 28': '27_y', 'Player28': '28_x', 
            'Unnamed: 30': '28_y', 'Ball': 'Ball_x', 'Unnamed: 32': 'Ball_y'
        }).copy()
        main_players = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        subs = [26, 27, 28]
    
    sub_dict = {i: subs for i in main_players}
    
    for frame, row in data.iterrows():
        used_subs = set()
        for main in main_players:
            main_x, main_y = f'{main}_x', f'{main}_y'
            if pd.isna(row[main_x]) or pd.isna(row[main_y]):
                found_sub = False
                for sub in sub_dict[main]:
                    sub_x, sub_y = f'{sub}_x', f'{sub}_y'
                    if sub not in used_subs and not pd.isna(row[sub_x]) and not pd.isna(row[sub_y]):
                        data.at[frame, main_x] = row[sub_x]
                        data.at[frame, main_y] = row[sub_y]
                        used_subs.add(sub)
                        found_sub = True
                        break
                if not found_sub:
                    print(f"Frame: {frame}, Player: {main} has NaN values with no valid substitutes.")
    
    return data

def process_game_data(game_number):
    """
    Process data for a specific game with game-specific handling
    """
    # Define paths
    base_path = "Raw_data"
    home_data_path = f"{base_path}/Sample_game_{game_number}/Sample_game_{game_number}_RawTrackingData_Home_Team.csv"
    away_data_path = f"{base_path}/Sample_game_{game_number}/Sample_game_{game_number}_RawTrackingData_Away_Team.csv"
    events_data_path = f"{base_path}/Sample_game_{game_number}/Sample_game_{game_number}_RawEventsData.csv"

    # Game-specific data loading
    if game_number in [1, 2]:
        # Games 1 and 2 have similar format with minor differences
        data_h = pd.read_csv(home_data_path, skiprows=2)
        data_a = pd.read_csv(away_data_path, skiprows=2, low_memory=False)
    else:  # game 3
        # Game 3 has a different format
        data_h = pd.read_csv(home_data_path, skiprows=2)
        data_a = pd.read_csv(away_data_path, skiprows=2)
    
    events = pd.read_csv(events_data_path)

    # Clean data with game-specific handling
    df1 = data_cleaning(data_h)
    df2 = data_cleaning(data_a)

    # Game-specific column handling
    if game_number == 1:
        # Game 1 specific drops
        df1 = df1.drop(['Ball_x', 'Ball_y', '12_x', '12_y', '13_x', '13_y', '14_x', '14_y'], axis=1)
        df2 = df2.drop(['26_x', '26_y', '27_x', '27_y', '28_x', '28_y'], axis=1)
    
    elif game_number == 2:
        # Game 2 specific handling
        df1 = df1.drop(['Ball_x', 'Ball_y', '12_x', '12_y', '13_x', '13_y', '14_x', '14_y'], axis=1)
        
        # Special handling for player 22 substitution in game 2
        if 'Player 26' in df2.columns:
            df2.loc[87985:, '22_x'] = df2.loc[87985:, 'Player 26']
            df2.loc[87985:, '22_y'] = df2.loc[87985:, 'Unnamed: 26']
            df2 = df2.drop(['Player 26', 'Unnamed: 26'], axis=1)
        
        df2 = df2.drop(['26_x', '26_y', '27_x', '27_y', '28_x', '28_y'], axis=1)
    
    else:  # game 3
        # Game 3 specific handling
        df1 = df1.drop(['Ball_x', 'Ball_y', '12_x', '12_y', '13_x', '13_y', '14_x', '14_y'], axis=1)
        df2 = df2.drop(['26_x', '26_y', '27_x', '27_y', '28_x', '28_y'], axis=1)
        # Add any specific game 3 transformations here

    # Validate and merge DataFrames with better error handling
    merge_columns = ['Period', 'Frame', 'Time [s]']
    if not all(df1[col].equals(df2[col]) for col in merge_columns):
        print(f"Game {game_number}: Mismatch in merge columns")
        # Find and report mismatches
        for col in merge_columns:
            if not df1[col].equals(df2[col]):
                print(f"Mismatch in column: {col}")
                mismatched_rows = df1[df1[col] != df2[col]]
                print(f"First few mismatched rows in {col}:")
                print(mismatched_rows[col].head())
        return None

    # Merge the dataframes
    data = df1.join(df2.set_index(merge_columns), on=merge_columns)
    
    # Process time and coordinates
    data['Time [s]'] = data['Time [s]'].astype('int64')
    data = data.drop_duplicates(subset=['Time [s]'], keep='first')
    
    # Denormalize coordinates
    for col in data.columns:
        if '_x' in col:
            data[col] = data[col] * 105
        elif '_y' in col:
            data[col] = data[col] * 68
    
    # Game-specific coordinate transformations
    if game_number == 2:
        # Flip coordinates for second half in game 2
        second_half_mask = data['Period'] == 2
        for col in data.columns:
            if '_x' in col:
                data.loc[second_half_mask, col] = 105 - data.loc[second_half_mask, col]
            elif '_y' in col:
                data.loc[second_half_mask, col] = 68 - data.loc[second_half_mask, col]
    
    return data

def calculate_influence(data):
    """
    Calculate influence values with game-specific handling
    """
    players_id = [1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,24,25]
    influence_data = []
    
    for frame_idx in tqdm(range(len(data)), desc="Calculating influence"):
        frame = data.iloc[frame_idx]
        x, y = [], []
        
        # Handle potential missing columns
        for player_id in players_id:
            x_col, y_col = f'{player_id}_x', f'{player_id}_y'
            if x_col in frame and y_col in frame:
                x.append(frame[x_col])
                y.append(frame[y_col])
            else:
                print(f"Warning: Missing columns {x_col} or {y_col}")
                x.append(0)  # or some other default value
                y.append(0)
        
        points = list(zip(x, y))
        
        # Calculate influence
        random_points = np.random.uniform(low=[0, 0], high=[105, 68], size=(10000, 2))
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(points)
        
        distances, indices = knn.kneighbors(random_points)
        nearest_points_count = np.zeros(len(players_id))
        
        for index in indices.flatten():
            nearest_points_count[index] += 1
        
        frame_influence = [(count/(68*105)*1000) for count in nearest_points_count]
        influence_data.append(frame_influence)
    
    return np.array(influence_data)

def calculate_pitch_control(data, sigma=0.5, lambda_=4.3):
    """
    Calculate pitch control values for each player
    """
    players_id = [1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,24,25]
    pitch_control_data = []
    grid_size = (105, 68)
    x_grid = np.linspace(0, grid_size[0], 50)
    y_grid = np.linspace(0, grid_size[1], 32)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    for frame_idx in tqdm(range(len(data)), desc="Calculating pitch control"):
        frame = data.iloc[frame_idx]
        player_positions = []
        for player_id in players_id:
            x = frame[f'{player_id}_x']
            y = frame[f'{player_id}_y']
            player_positions.append([x, y])
        
        player_positions = np.array(player_positions)
        
        # Calculate distances to all grid points for each player
        distances = cdist(player_positions, grid_points)
        
        # Calculate control values using a Gaussian kernel
        control_values = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize control values
        control_sum = np.sum(control_values, axis=0)
        control_values = control_values / (control_sum + lambda_)
        
        # Calculate total control for each player
        player_control = np.mean(control_values, axis=1)
        pitch_control_data.append(player_control)

    return np.array(pitch_control_data)

def main():
    """
    Main function to process all games and save results
    """
    for game_number in range(1, 4):
        print(f"\nProcessing game {game_number}...")
        
        try:
            # Process game data
            game_data = process_game_data(game_number)
            if game_data is None:
                print(f"Skipping game {game_number} due to data mismatch")
                continue
            
            # Calculate influence values
            print("Calculating influence values...")
            influence_values = calculate_influence(game_data)
            
            # Calculate pitch control values
            print("Calculating pitch control values...")
            pitch_control_values = calculate_pitch_control(game_data)
            
            # Save processed data
            output_path = "Processed_data"
            os.makedirs(output_path, exist_ok=True)
            
            # Save influence values
            print("Saving influence values...")
            np.savez_compressed(
                f"{output_path}/Coords_Influence_{game_number}.npz",
                data=influence_values
            )
            
            # Save pitch control values
            print("Saving pitch control values...")
            np.savez_compressed(
                f"{output_path}/pitch_control_{game_number}.npz",
                data=pitch_control_values
            )
            
            # Save coordinates
            print("Saving processed coordinates...")
            game_data.to_csv(f"{output_path}/processed_coordinates_{game_number}.csv")
            
            print(f"Successfully completed processing game {game_number}")
            
        except Exception as e:
            print(f"Error processing game {game_number}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 