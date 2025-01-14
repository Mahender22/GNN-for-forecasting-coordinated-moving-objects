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