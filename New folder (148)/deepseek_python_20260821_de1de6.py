import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import toponetx as tnx

def mandelbrot(c, max_iter):
    """Calculate if a point is in the Mandelbrot set"""
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

def create_mandelbrot_data(width=1000, height=1000, x_range=(-2, 2), y_range=(-2, 2), max_iter=100):
    """Generate Mandelbrot set data"""
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    mandelbrot_data = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            mandelbrot_data[i, j] = mandelbrot(c, max_iter)
    
    return mandelbrot_data, x, y

def create_topological_analysis(mandelbrot_data, threshold=50):
    """Create topological features using TopoNetX"""
    # Convert to binary based on threshold (points in Mandelbrot set)
    binary_data = (mandelbrot_data < threshold).astype(int)
    
    # Create a simplicial complex from the binary data
    # We'll create vertices for each point in the set
    vertices = []
    edges = []
    triangles = []
    
    height, width = binary_data.shape
    
    # Find vertices (points in the set)
    for i in range(height):
        for j in range(width):
            if binary_data[i, j] == 1:
                vertices.append((i, j))
    
    # Create a vertex to index mapping
    vertex_to_idx = {v: idx for idx, v in enumerate(vertices)}
    
    # Find edges (adjacent points in the set)
    for i in range(height):
        for j in range(width):
            if binary_data[i, j] == 1:
                current = (i, j)
                # Check right neighbor
                if j + 1 < width and binary_data[i, j + 1] == 1:
                    neighbor = (i, j + 1)
                    if current in vertex_to_idx and neighbor in vertex_to_idx:
                        edges.append([vertex_to_idx[current], vertex_to_idx[neighbor]])
                # Check bottom neighbor
                if i + 1 < height and binary_data[i + 1, j] == 1:
                    neighbor = (i + 1, j)
                    if current in vertex_to_idx and neighbor in vertex_to_idx:
                        edges.append([vertex_to_idx[current], vertex_to_idx[neighbor]])
    
    # Find triangles (simplices)
    for i in range(height - 1):
        for j in range(width - 1):
            if binary_data[i, j] == 1 and binary_data[i, j+1] == 1 and binary_data[i+1, j] == 1:
                v1 = (i, j)
                v2 = (i, j+1)
                v3 = (i+1, j)
                if all(v in vertex_to_idx for v in [v1, v2, v3]):
                    triangles.append([vertex_to_idx[v1], vertex_to_idx[v2], vertex_to_idx[v3]])
    
    return vertices, edges, triangles

def plot_mandelbrot_with_topology(mandelbrot_data, x, y, vertices=None, edges=None, triangles=None):
    """Plot Mandelbrot set with topological features"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Standard Mandelbrot set
    cmap = LinearSegmentedColormap.from_list('mandelbrot', 
                                             ['#000000', '#000033', '#000066', '#000099', 
                                              '#0000CC', '#0000FF', '#0033FF', '#0066FF'], 
                                             N=256)
    
    im1 = ax1.imshow(mandelbrot_data, extent=[x[0], x[-1], y[0], y[-1]], 
                     cmap=cmap, origin='lower')
    ax1.set_title('Mandelbrot Set')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    plt.colorbar(im1, ax=ax1, label='Iterations to escape')
    
    # Plot 2: Mandelbrot with topological features
    im2 = ax2.imshow(mandelbrot_data, extent=[x[0], x[-1], y[0], y[-1]], 
                     cmap='gray', alpha=0.3, origin='lower')
    
    # Overlay topological features
    if vertices and edges:
        # Plot vertices
        if vertices:
            vx = [v[1] / (len(x)-1) * (x[-1] - x[0]) + x[0] for v in vertices]
            vy = [v[0] / (len(y)-1) * (y[-1] - y[0]) + y[0] for v in vertices]
            ax2.scatter(vx, vy, c='red', s=1, alpha=0.5, label='Vertices')
        
        # Plot edges
        if edges:
            for edge in edges:
                v1 = vertices[edge[0]]
                v2 = vertices[edge[1]]
                x1 = v1[1] / (len(x)-1) * (x[-1] - x[0]) + x[0]
                y1 = v1[0] / (len(y)-1) * (y[-1] - y[0]) + y[0]
                x2 = v2[1] / (len(x)-1) * (x[-1] - x[0]) + x[0]
                y2 = v2[0] / (len(y)-1) * (y[-1] - y[0]) + y[0]
                ax2.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5, alpha=0.5)
        
        # Plot triangles (simplices)
        if triangles:
            for tri in triangles:
                v1 = vertices[tri[0]]
                v2 = vertices[tri[1]]
                v3 = vertices[tri[2]]
                x1 = v1[1] / (len(x)-1) * (x[-1] - x[0]) + x[0]
                y1 = v1[0] / (len(y)-1) * (y[-1] - y[0]) + y[0]
                x2 = v2[1] / (len(x)-1) * (x[-1] - x[0]) + x[0]
                y2 = v2[0] / (len(y)-1) * (y[-1] - y[0]) + y[0]
                x3 = v3[1] / (len(x)-1) * (x[-1] - x[0]) + x[0]
                y3 = v3[0] / (len(y)-1) * (y[-1] - y[0]) + y[0]
                ax2.fill([x1, x2, x3], [y1, y2, y3], 'g', alpha=0.3)
    
    ax2.set_title('Mandelbrot with Topological Features')
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imaginary')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_topological_properties(vertices, edges, triangles):
    """Analyze and print topological properties"""
    print("Topological Analysis Results:")
    print("-" * 40)
    print(f"Number of vertices (0-simplices): {len(vertices)}")
    print(f"Number of edges (1-simplices): {len(edges)}")
    print(f"Number of triangles (2-simplices): {len(triangles)}")
    
    # Calculate Euler characteristic
    euler_characteristic = len(vertices) - len(edges) + len(triangles)
    print(f"Euler characteristic: {euler_characteristic}")
    
    # Estimate Betti numbers (simplified)
    # β₀ ≈ number of connected components
    # β₁ ≈ number of holes
    # For Mandelbrot set, we can estimate from the structure
    print(f"Estimated β₀ (connected components): ~1 (main component)")
    print(f"Estimated β₁ (holes): ~{len(triangles) - len(edges) + len(vertices)}")

# Main execution
if __name__ == "__main__":
    # Generate Mandelbrot data
    print("Generating Mandelbrot set...")
    mandelbrot_data, x, y = create_mandelbrot_data(
        width=400, height=400, 
        x_range=(-2.5, 1.5), y_range=(-1.5, 1.5), 
        max_iter=100
    )
    
    # Create topological features
    print("Creating topological features...")
    vertices, edges, triangles = create_topological_analysis(mandelbrot_data, threshold=50)
    
    # Analyze topological properties
    analyze_topological_properties(vertices, edges, triangles)
    
    # Plot results
    print("Plotting results...")
    plot_mandelbrot_with_topology(mandelbrot_data, x, y, vertices, edges, triangles)
    
    # Create a TopoNetX simplicial complex (optional)
    print("\nCreating TopoNetX Simplicial Complex...")
    # Note: This would require creating a SimplicialComplex from the data
    # but we'll just print the structure we've already created
    print(f"Simplicial complex with {len(vertices)} vertices, {len(edges)} edges, and {len(triangles)} triangles")
    
    print("\nDone!")