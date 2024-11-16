import torch
import torch.nn as nn
from Custom_GCN import CustomGCNLayer  # Assuming this is defined in a separate file

class GCN_LSTM_Model(nn.Module):
    def __init__(self, num_node_features, gcn_out_features, lstm_hidden_size, output_features, num_nodes, n_steps_out):
        """
        Initialize the GCN-LSTM model.

        Parameters:
            num_node_features (int): Number of features per node.
            gcn_out_features (int): Number of output features for the GCN.
            lstm_hidden_size (int): Hidden size for the LSTM.
            output_features (int): Number of output features.
            num_nodes (int): Number of nodes in the graph.
            n_steps_out (int): Number of output time steps.
        """
        super(GCN_LSTM_Model, self).__init__()
        self.gcn = CustomGCNLayer(num_node_features, gcn_out_features)
        self.lstm = nn.LSTM(input_size=gcn_out_features * num_nodes, hidden_size=lstm_hidden_size, 
                            batch_first=True, num_layers=2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(lstm_hidden_size, n_steps_out * num_nodes * num_node_features)
        self.n_steps_out = n_steps_out
        self.num_node_features = num_node_features

    def forward(self, x_sequences, edge_indices_sequences, edge_distances_sequences):
        """
        Forward pass of the GCN-LSTM model.

        Parameters:
            x_sequences (torch.Tensor): Input sequences of shape [batch_size, seq_length, num_nodes, num_features].
            edge_indices_sequences (list): List of edge indices for each time step.
            edge_distances_sequences (list): List of edge distances for each time step.

        Returns:
            torch.Tensor: Model output of shape [batch_size, n_steps_out, num_nodes, num_node_features].
        """
        total_sequences, seq_length, num_nodes = x_sequences.shape
        gcn_outputs = []

        for sequence_idx in range(total_sequences):
            sequence_gcn_outputs = []

            for t in range(seq_length):
                x_t = x_sequences[sequence_idx, t, :].view(-1, 1)  # Reshape to [num_nodes, num_features]
                edge_index_t = edge_indices_sequences[sequence_idx][t]
                edge_distance_t = edge_distances_sequences[sequence_idx][t]
                gcn_output = self.gcn(x_t, edge_index_t, edge_distance_t)
                gcn_output_flat = gcn_output.view(-1)  # Flatten to [gcn_out_features * num_nodes]
                sequence_gcn_outputs.append(gcn_output_flat.unsqueeze(0))

            sequence_gcn_output = torch.cat(sequence_gcn_outputs, dim=0).unsqueeze(0)
            gcn_outputs.append(sequence_gcn_output)

        gcn_outputs_tensor = torch.cat(gcn_outputs, dim=0)
        gcn_outputs_tensor = self.tanh(gcn_outputs_tensor)
        lstm_out, _ = self.lstm(gcn_outputs_tensor)
        lstm_out = self.tanh(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        out = out.view(-1, self.n_steps_out, num_nodes, self.num_node_features)  # Reshape to match target
        return out.squeeze(-1)
