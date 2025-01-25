# GNN for Forecasting Coordinated Moving Objects

A Graph Neural Network (GNN) based approach for forecasting the movement and interactions of coordinated moving objects, with a specific focus on soccer player movements and their spatial relationships.

## Abstract

This project investigates the utility of multiplicative interaction in graph neural networks for forecasting tasks involving non-random interacting moving objects. Unlike traditional traffic flow prediction tasks where nodes are spatially static, this work handles dynamic spatial interactions using multiplicative interaction layers in an attributed graph forecasting model.

The model combines Graph Convolution Networks (GCN) with Long Short-Term Memory (LSTM) neural networks as the base forecasting architecture. The implementation supports both cooperative moving objects and multiple adversarial groups scenarios.

The project includes a novel evolving graph dataset generator based on three 90-minute soccer games, incorporating players' pitch control and influence values. The experimental results demonstrate the feasibility and competitiveness of the proposed method.

## Features

- Multiple GCN model variations:
  - Base model
  - Multiplicative Interaction versions (MIv1, MIv2, MIv3, MIv4)
  - Same-team with opposing team interaction (same-0.5opp)
- Support for different aggregation methods (mean, add, mul)
- Dynamic graph construction based on player positions
- Flexible distance thresholding for edge creation
- Comprehensive data processing pipeline
- Visualization tools for model analysis

## Project Structure

- `main.py`: Main file for training and evaluating the model.
- `main_visualization.py`: Main file for visualization.
- `data_processing.py`: Data processing pipeline for generating the dataset and preprocessing the data.
- `Custom_GCN.py`: Custom GCN implementation, including the multiplicative interaction layers.
- `visualization_methods.py`: Visualization tools for model analysis.
- `GCN_LSTM.py`: Combined GCN and LSTM model.
- `Processed_data`: Directory for the processed game data for three 90 minute soccer games.
- `data_preprocessing.py`: Data preprocessing pipeline for generating the dataset and preprocessing the data.
- `Raw_data`: Directory for the raw tracking and event data for three 90 minute soccer games.

## Data Description

### Input Data Format
The project uses two main types of data files:

1. **Coordinates and Influence Data** (`Coords_Influence_{game}.npz`):
   - Player coordinates (x, y)
   - Influence values for each player
   - Temporal sequence information

2. **Pitch Control Data** (`pitch_control_{game}.npz`):
   - Pitch control values for each player
   - Synchronized with coordinate data

### Data Structure
- Time dimension: Sequences of game frames
- Player dimension: 22 players (both teams)
- Feature dimension: [influence/pitch_control, x_coordinate, y_coordinate]

## Data Preprocessing

### Raw Data Source
The raw tracking and event data used in this project comes from [Metrica Sports' sample data repository](https://github.com/metrica-sports/sample-data). The dataset includes:

- 3 complete soccer games with tracking data for all 22 players
- Data format: CSV files for each game containing:
  - Home team tracking data
  - Away team tracking data
  - Event data
- Field dimensions: 105x68 meters
- Coordinate system: (0,0) is top left, (1,1) is bottom right
- Sampling rate: 25 frames per second

### Data Processing Pipeline

The project includes a comprehensive data preprocessing pipeline (`data_preprocessing.py`) that handles:

1. **Data Loading and Cleaning**
   ```python
   # Game-specific data loading
   data_h = pd.read_csv(home_data_path, skiprows=2)
   data_a = pd.read_csv(away_data_path, skiprows=2)
   ```

2. **Player Tracking Data Processing**
   - Handles player substitutions
   - Normalizes coordinates to field dimensions
   - Manages missing data
   - Game-specific coordinate transformations

3. **Feature Calculation**
   - **Influence Values**: Calculated using k-nearest neighbors approach
     ```python
     # Calculate player influence using KNN
     random_points = np.random.uniform(low=[0, 0], high=[105, 68], size=(10000, 2))
     knn = NearestNeighbors(n_neighbors=1)
     ```
   - **Pitch Control**: Computed using Gaussian kernel
     ```python
     # Calculate control values using Gaussian kernel
     control_values = np.exp(-distances**2 / (2 * sigma**2))
     ```

4. **Game-Specific Handling**
   - Game 1: Standard processing
   - Game 2: Special handling for player substitutions and coordinate flipping
   - Game 3: EPTS FIFA format handling

### Data Structure

#### Input Data
```
Raw_data/
    Sample_game_1/
        Sample_game_1_RawTrackingData_Home_Team.csv
        Sample_game_1_RawTrackingData_Away_Team.csv
        Sample_game_1_RawEventsData.csv
    Sample_game_2/
        ...
    Sample_game_3/
        ...
```

#### Processed Output
```
Processed_data/
    Coords_Influence_1.npz      # Player influence values
    pitch_control_1.npz         # Pitch control values
    processed_coordinates_1.csv  # Cleaned tracking data
    ...
```

#### Data Format
1. **Influence Values Array**: Shape (frames, 22)
   - Each frame contains influence values for all 22 players
   - Values normalized to [0,1] range

2. **Pitch Control Array**: Shape (frames, 22)
   - Each frame contains pitch control values for all 22 players
   - Values represent spatial control of the field

3. **Processed Coordinates**: 
   - Player positions in absolute coordinates (meters)
   - Synchronized with influence and pitch control values

### Running the Preprocessing

1. Install required packages:
```bash
pip install numpy pandas scipy scikit-learn tqdm matplotlib
```

2. Run the preprocessing script with arguments:
```bash
# For influence values:
python data_preprocessing.py influence 1  # Process influence for game 1
python data_preprocessing.py influence 2  # Process influence for game 2
python data_preprocessing.py influence 3  # Process influence for game 3

# For pitch control values:
python data_preprocessing.py pitch_control 1  # Process pitch control for game 1
python data_preprocessing.py pitch_control 2  # Process pitch control for game 2
python data_preprocessing.py pitch_control 3  # Process pitch control for game 3
```

The script accepts two required arguments:
- `feature_type`: Type of feature to calculate ['influence', 'pitch_control']
- `game_number`: Game number to process [1, 2, 3]

3. Monitor progress:
   - Script provides progress bars for each processing step
   - Logs warnings and errors for data quality issues
   - Reports successful completion for each feature/game combination

### Data Validation

The preprocessing pipeline includes several validation steps:
- Column consistency checks
- Missing data detection
- Coordinate range validation
- Frame synchronization verification

### Game-Specific Processing

1. **Game 1**:
   - Standard processing
   - Basic column drops for substitutes and ball data

2. **Game 2**:
   - Special handling for player 22 substitution
   - Coordinate flipping for second half
   - Custom column handling

3. **Game 3**:
   - EPTS FIFA format handling
   - Specific transformations for coordinate system

### Notes
- Coordinate system is converted from normalized [0,1] to actual field dimensions (105x68 meters)
- Missing data is handled through substitution tracking
- Game 2 includes special handling for second-half coordinate flipping
- All processed data is synchronized across different features
- Each feature is processed independently for better memory management

## Model Details

### Base GCN-LSTM Architecture
1. **Graph Construction Layer**
   - Builds dynamic graphs based on player positions
   - Supports both distance-based and k-nearest neighbor approaches
   - Edge weight calculation based on player distances

2. **Graph Convolution Layer**
   - Processes spatial relationships between players
   - Customizable message passing mechanisms
   - Multiple aggregation options (mean, add, multiply)

3. **LSTM Layer**
   - Processes temporal sequences
   - Captures movement patterns over time
   - Maintains temporal dependencies

### Model Variations

1. **Base Model**
   - Standard GCN implementation
   - Basic spatial-temporal processing

2. **MIv1 (Multiplicative Interaction v1)**
   - Basic multiplicative interaction between nodes
   - Enhanced feature combination

3. **MIv2 (Multiplicative Interaction v2)**
   - Advanced feature interaction mechanism
   - Improved spatial relationship modeling

4. **MIv3 (Multiplicative Interaction v3)**
   - Specialized for multiplicative aggregation
   - Optimized for complex interactions

5. **MIv4 (Multiplicative Interaction v4)**
   - Enhanced version with additional interaction layers
   - Better handling of complex movement patterns

6. **same-0.5opp**
   - Specialized handling of team interactions
   - Different weights for same-team vs opposing-team interactions

## Usage Guide

### Training a Model
```python
# Example configuration in main.py
feature_name = 'pitch_control'  # or 'influence'
game = 1                        # Game number (1-3)
model_name = 'base'            # Model variation
aggr = 'mean'                  # Aggregation method
distance_threshold = 15        # or 'k3' for 3-nearest neighbors
```

### Running Experiments
1. **Standard Training**:
```bash
python main.py
```

2. **Visualization**:
```bash
python main_visualization.py
```

### Parameter Tuning
Key parameters that can be adjusted:
- Distance threshold for edge creation
- Number of GCN layers
- LSTM hidden dimensions
- Learning rate
- Batch size
- Sequence length

## Visualization Tools

### Available Visualizations
1. **Player-wise Analysis**
   - Median residue per player
   - Performance comparison across models
   - Team-specific analysis

2. **Model Comparison**
   - Performance across different games
   - Model variation comparison
   - Aggregation method analysis

3. **Temporal Analysis**
   - Prediction accuracy over time
   - Movement pattern visualization
   - Error distribution analysis

## Results and Analysis

### Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Player-specific metrics
- Team-level performance

### Key Findings
- Effectiveness of multiplicative interactions
- Impact of different aggregation methods
- Performance comparison across model variations
- Team-specific insights

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request