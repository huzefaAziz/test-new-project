import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class GraphDiffusionGNN(nn.Module):
    """Graph Neural Network with diffusion process for Mandelbrot approximation"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(nn.Linear(hidden_dim, output_dim))
        
        # Attention mechanism for edge weighting
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, edge_index):
        # x: node features, edge_index: adjacency edges
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            # Graph diffusion with message passing
            x = self.message_passing(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.convs[-1](x)
        return x.squeeze()
    
    def message_passing(self, x, edge_index):
        """Simple message passing with diffusion"""
        # Aggregate messages from neighbors
        row, col = edge_index
        messages = torch.zeros_like(x)
        
        # For each node, aggregate messages from neighbors
        for i in range(len(row)):
            messages[col[i]] += x[row[i]] * 0.5  # Simple diffusion
            messages[row[i]] += x[col[i]] * 0.5
        
        return messages + x * 0.5  # Residual connection

def generate_mandelbrot_points(width=100, height=100, x_range=(-2, 1), y_range=(-1.5, 1.5), max_iter=50):
    """Generate Mandelbrot set points with complex dynamics"""
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Compute Mandelbrot set
    mandelbrot = np.zeros((height, width))
    C = Z.copy()
    
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + C[mask]
        mandelbrot[mask] += 1
    
    mandelbrot = mandelbrot / max_iter
    return X, Y, mandelbrot

def create_graph_from_points(points, k_neighbors=5):
    """Create a graph from 2D points using k-nearest neighbors"""
    n_points = len(points)
    
    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create edges
    edges = []
    for i in range(n_points):
        for j in indices[i]:
            if i != j:
                edges.append((i, j))
    
    # Remove duplicates
    edges = list(set([tuple(sorted(e)) for e in edges]))
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    G.add_edges_from(edges)
    
    return G, np.array(edges)

def add_gaussian_noise_to_graph(G, noise_std=0.05):
    """Add Gaussian noise to graph nodes"""
    noise = np.random.normal(0, noise_std, (G.number_of_nodes(), 2))
    return noise

def graph_diffusion_evolution(G, steps=10, diffusion_rate=0.1):
    """Simulate diffusion process on graph"""
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        # If positions not set, use node indices
        pos = {i: np.array([i % 10, i // 10]) for i in range(G.number_of_nodes())}
    
    pos_array = np.array([pos[i] for i in range(G.number_of_nodes())])
    
    # Diffusion evolution
    evolution_steps = [pos_array.copy()]
    
    for step in range(steps):
        new_pos = pos_array.copy()
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                # Average of neighbor positions
                neighbor_pos = np.mean([pos_array[n] for n in neighbors], axis=0)
                new_pos[node] = (1 - diffusion_rate) * pos_array[node] + diffusion_rate * neighbor_pos
        
        # Add small random perturbation
        new_pos += np.random.normal(0, 0.01, new_pos.shape)
        pos_array = new_pos
        evolution_steps.append(pos_array.copy())
    
    return np.array(evolution_steps)

def train_gnn_on_mandelbrot(G, mandelbrot_values, sample_indices, epochs=100):
    """Train GNN to approximate Mandelbrot values"""
    # Convert graph to PyTorch format
    n_nodes = G.number_of_nodes()
    edge_array = np.array([(u, v) for u, v in G.edges()])
    
    # Prepare node features (positions in complex plane)
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = {i: np.array([i % 10, i // 10]) for i in range(n_nodes)}
    
    node_features = torch.FloatTensor(np.array([pos[i] for i in range(n_nodes)]))
    edge_index = torch.LongTensor(edge_array.T)
    
    # Sample training data (mask for nodes with known mandelbrot values)
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[sample_indices] = True
    
    targets = torch.FloatTensor(mandelbrot_values)
    
    # Initialize and train GNN
    model = GraphDiffusionGNN(input_dim=2, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(node_features, edge_index)
        
        # Only compute loss on sampled nodes
        loss = F.mse_loss(predictions[mask], targets[mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate on all nodes
    model.eval()
    with torch.no_grad():
        all_predictions = model(node_features, edge_index).numpy()
    
    return model, all_predictions, losses

def main():
    # Generate Mandelbrot set
    print("Generating Mandelbrot set...")
    width, height = 100, 100
    X, Y, mandelbrot = generate_mandelbrot_points(width, height)
    
    # Sample points from the Mandelbrot set
    n_samples = 500
    flat_indices = np.random.choice(width * height, n_samples, replace=False)
    sample_points = np.array([(X.flat[i], Y.flat[i]) for i in flat_indices])
    sample_values = mandelbrot.flat[flat_indices]
    
    # Create graph from sampled points
    print("Creating graph from sampled points...")
    G, edges = create_graph_from_points(sample_points, k_neighbors=8)
    
    # Add positions to graph
    for i, (x, y) in enumerate(sample_points):
        G.nodes[i]['pos'] = np.array([x, y])
    
    # Add Gaussian noise
    print("Adding Gaussian noise...")
    noise = add_gaussian_noise_to_graph(G, noise_std=0.03)
    for i in range(G.number_of_nodes()):
        G.nodes[i]['pos'] += noise[i]
        G.nodes[i]['noise'] = noise[i]
    
    # Simulate diffusion
    print("Simulating graph diffusion...")
    evolution = graph_diffusion_evolution(G, steps=15, diffusion_rate=0.15)
    
    # Train GNN on Mandelbrot data
    print("Training GNN for Mandelbrot approximation...")
    model, predictions, losses = train_gnn_on_mandelbrot(
        G, sample_values, list(range(n_samples)), epochs=200
    )
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Original Mandelbrot
    plt.subplot(2, 3, 1)
    plt.imshow(mandelbrot, extent=(-2, 1, -1.5, 1.5), origin='lower', cmap='hot')
    plt.title('Original Mandelbrot Set')
    plt.colorbar()
    
    # Sampled points
    plt.subplot(2, 3, 2)
    plt.scatter(sample_points[:, 0], sample_points[:, 1], 
                c=sample_values, cmap='hot', s=10, alpha=0.8)
    plt.title(f'Sampled Points (n={n_samples})')
    plt.xlim(-2, 1)
    plt.ylim(-1.5, 1.5)
    plt.colorbar()
    
    # Graph structure
    plt.subplot(2, 3, 3)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=20, node_color='blue', edge_color='gray', alpha=0.6)
    plt.title(f'Graph Structure (k={8} nearest neighbors)')
    
    # GNN approximation
    plt.subplot(2, 3, 4)
    # Reconstruct from graph predictions
    full_pred = np.zeros_like(mandelbrot)
    for i, (x, y) in enumerate(sample_points):
        idx_x = np.argmin(np.abs(X[0, :] - x))
        idx_y = np.argmin(np.abs(Y[:, 0] - y))
        full_pred[idx_y, idx_x] = predictions[i]
    
    plt.imshow(full_pred, extent=(-2, 1, -1.5, 1.5), origin='lower', cmap='hot')
    plt.title('GNN Approximation')
    plt.colorbar()
    
    # Training loss
    plt.subplot(2, 3, 5)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Training Loss')
    plt.yscale('log')
    
    # Diffusion evolution
    plt.subplot(2, 3, 6)
    plt.plot(evolution[:, 0, 0], evolution[:, 0, 1], 'r.-', label='Node 1')
    plt.plot(evolution[:, 1, 0], evolution[:, 1, 1], 'b.-', label='Node 2')
    plt.plot(evolution[:, 2, 0], evolution[:, 2, 1], 'g.-', label='Node 3')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Graph Diffusion Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mandelbrot_gnn_diffusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional visualization: Mandelbrot with Gaussian noise overlay
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Add noise to Mandelbrot image
    noisy_mandelbrot = mandelbrot + np.random.normal(0, 0.1, mandelbrot.shape)
    plt.imshow(noisy_mandelbrot, extent=(-2, 1, -1.5, 1.5), origin='lower', cmap='hot')
    plt.title('Mandelbrot with Gaussian Noise')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    # Gaussian filtered version
    filtered = gaussian_filter(noisy_mandelbrot, sigma=1.0)
    plt.imshow(filtered, extent=(-2, 1, -1.5, 1.5), origin='lower', cmap='hot')
    plt.title('Gaussian Filtered Mandelbrot')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('mandelbrot_noise_filter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis complete! Results saved as images.")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Mean prediction error: {np.mean(np.abs(predictions - sample_values)):.4f}")
    print(f"Max prediction error: {np.max(np.abs(predictions - sample_values)):.4f}")
    print(f"Correlation: {np.corrcoef(predictions, sample_values)[0, 1]:.4f}")

if __name__ == "__main__":
    main()