import sqlite3
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Step 1: Load Data from SQLite Database
def load_data_from_folder(folder_path):
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.db'):
            db_path = os.path.join(folder_path, file_name)
            conn = sqlite3.connect(db_path)
            query_check_table = """
            SELECT name FROM sqlite_master WHERE type='table' AND name='ip_stats';
            """
            # Check if the ip_stats table exists
            table_exists = pd.read_sql_query(query_check_table, conn)

            if not table_exists.empty:
                query = """
                SELECT ip, sent_packets, sent_data, bandwidth_sent, recv_packets, recv_data, bandwidth_recv, packet_loss, snr
                FROM ip_stats
                """
                df = pd.read_sql_query(query, conn)
                dfs.append(df)
            else:
                print(f"Skipping {file_name} because it does not contain 'ip_stats' table.")

            conn.close()

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No valid databases with 'ip_stats' table found.")


# Step 2: Preprocess and Scale the Data
def preprocess_data(df):
    X = df[['sent_packets', 'recv_packets', 'bandwidth_sent', 'bandwidth_recv', 'snr']].values
    y = df['packet_loss'].values  # Target: packet loss

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Step 3: Build and Train Neural Network Model
def build_and_train_model(X_scaled, y):
    model = Sequential()
    model.add(Dense(128, input_dim=X_scaled.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2)  # Train for more epochs

    model.save('neuralnetwork_now.keras')
    tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

    return model


# Step 4: Visualize Neural Network Weights
def visualize_neural_network_weights(model, output_file='complex_nn_graph.png'):
    """
    Visualizes the neural network's connections based on their weights.
    Stronger connections are blue, weaker connections are grey.

    Parameters:
    - model: Trained Keras model.
    - output_file: File path to save the visualization.
    """

    # Extract weights from the model layers
    weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):  # We are only interested in Dense layers
            layer_weights = layer.get_weights()[0]  # Get the connection weights
            weights.append(layer_weights)

    # Create a graph using networkx
    G = nx.DiGraph()

    node_pos = {}  # Positions of neurons
    edge_colors = []  # Colors of edges based on weights
    edge_weights = []  # The actual edge weights

    # Variables to keep track of layer sizes and neuron positions
    layer_pos_x = 0
    prev_layer_size = 0
    vertical_spacing = 0.8  # Control vertical spacing between neurons
    horizontal_spacing = 2.0  # Control horizontal spacing between layers

    # Loop through layers and create nodes and edges for each layer's connections
    for i, layer_weights in enumerate(weights):
        layer_size = layer_weights.shape[1]  # Number of neurons in the layer

        # Add neurons to the graph
        for neuron_idx in range(layer_size):
            node_pos[f'layer_{i}_neuron_{neuron_idx}'] = (layer_pos_x, neuron_idx * vertical_spacing)

        # Add connections between neurons and color them based on weights
        if i > 0:
            for src_neuron in range(prev_layer_size):
                for dst_neuron in range(layer_size):
                    weight = layer_weights[src_neuron, dst_neuron]
                    G.add_edge(f'layer_{i - 1}_neuron_{src_neuron}', f'layer_{i}_neuron_{dst_neuron}')

                    # Set the edge color based on the strength of the connection
                    edge_colors.append('blue' if abs(weight) > np.percentile(layer_weights, 90) else 'grey')

                    # Scale the edge width by the weight (to highlight stronger connections)
                    edge_weights.append(abs(weight) * 5)  # Multiply by 5 for better visibility

        prev_layer_size = layer_size
        layer_pos_x += horizontal_spacing

    # Draw the graph using networkx
    plt.figure(figsize=(12, 10))

    # Adjust transparency and node size
    nx.draw(
        G,
        pos=node_pos,
        edge_color=edge_colors,
        node_size=100,  # Smaller nodes for clarity
        node_color="lightgrey",  # Use light grey for the neurons
        alpha=0.8,  # Add transparency to nodes and edges
        with_labels=False,
        edge_cmap=plt.cm.Blues,
        width=edge_weights,  # Scale edge width based on weight strength
    )

    # Save the plot as a PNG file
    plt.savefig(output_file, transparent=True)  # Set background as transparent
    print(f"Neural network graph saved to {output_file}.")


# Step 5: Evaluate and Predict for a New Database
def evaluate_new_db(model, db_path, scaler):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT ip, sent_packets, sent_data, bandwidth_sent, recv_packets, recv_data, bandwidth_recv, packet_loss, snr
    FROM ip_stats
    """
    df_new = pd.read_sql_query(query, conn)
    conn.close()

    # Preprocess new data
    X_new = df_new[['sent_packets', 'recv_packets', 'bandwidth_sent', 'bandwidth_recv', 'snr']].values
    X_new_scaled = scaler.transform(X_new)

    # Predict packet loss
    predictions = model.predict(X_new_scaled)
    df_new['predicted_packet_loss'] = predictions

    # Find the IP with the lowest predicted packet loss (most optimal)
    optimal_ip = df_new.loc[df_new['predicted_packet_loss'].idxmin()]['ip']
    print(f"The most optimal IP address is: {optimal_ip}")

    return df_new[['ip', 'packet_loss', 'predicted_packet_loss']]


# Function to run the entire process
def run_neural_calculations(folder_path, new_db_path):
    df = load_data_from_folder(folder_path)
    X_scaled, y, scaler = preprocess_data(df)
    model = build_and_train_model(X_scaled, y)
    visualize_neural_network_weights(model)  # Visualize the network
    result_df = evaluate_new_db(model, new_db_path, scaler)
    return result_df


# Get the most recent database file
def get_latest_db_file(directory):
    """Get the most recent .db file from the specified directory."""
    db_files = [f for f in os.listdir(directory) if f.endswith('.db')]
    if not db_files:
        return None  # Return None if no .db files are found
    # Get the full path and find the one with the latest modification time
    latest_file = max(db_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)


# Main function (optional)
if __name__ == "__main__":
    folder_path = 'C:/Users/mulle/Desktop/FTDK_NetworkMonitoring/networkmonitoring/OUTPUTS_W1'
    new_db_path = get_latest_db_file("C:/Users/mulle/Desktop/FTDK_NetworkMonitoring/networkmonitoring/recently")
    run_neural_calculations(folder_path, new_db_path)
