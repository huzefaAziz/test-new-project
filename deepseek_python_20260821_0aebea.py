import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from toponetx.classes import SimplicialComplex

def mandelbrot(c, max_iter):
    """Calculate if a point is in the Mandelbrot set"""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_mandelbrot_data(width, height, x_range, y_range, max_iter):
    """Generate Mandelbrot set data as a 2D array"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Create grid of complex numbers
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    
    # Initialize array for results
    mandelbrot_data = np.zeros((height, width))
    
    # Vectorized computation using numpy
    for i in range(height):
        for j in range(width):
            c = complex(real[j], imag[i])
            mandelbrot_data[i, j] = mandelbrot(c, max_iter)
    
    return mandelbrot_data, real, imag

def create_mandelbrot_with_toponetx():
    """Create Mandelbrot set visualization using TopoNetX"""
    
    # Parameters
    width, height = 800, 600
    x_range = (-2.5, 1.5)
    y_range = (-1.5, 1.5)
    max_iter = 100
    
    # Generate Mandelbrot data
    print("Generating Mandelbrot set...")
    mandelbrot_data, real, imag = generate_mandelbrot_data(
        width, height, x_range, y_range, max_iter
    )
    
    # Create a custom colormap
    colors = ['#000000', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FF00FF']
    cmap = LinearSegmentedColormap.from_list('mandelbrot_cmap', colors, N=256)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Standard Mandelbrot set
    im1 = ax1.imshow(mandelbrot_data, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                     cmap=cmap, origin='lower', interpolation='bilinear')
    ax1.set_title('Mandelbrot Set - Standard View')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Iteration Count')
    
    # Create TopoNetX data structure from the Mandelbrot set
    # We'll identify points in the set (where iterations = max_iter)
    mask = mandelbrot_data >= max_iter
    
    # Extract coordinates of points in the Mandelbrot set
    points_in_set = np.argwhere(mask)
    
    # Convert to actual complex coordinates
    complex_points = []
    for y_idx, x_idx in points_in_set:
        real_val = real[x_idx] if x_idx < len(real) else real[-1]
        imag_val = imag[y_idx] if y_idx < len(imag) else imag[-1]
        complex_points.append((real_val, imag_val))
    
    # Create a SimplicialComplex using TopoNetX
    if complex_points:
        # Take a subset of points for demonstration (for performance)
        sample_size = min(3000, len(complex_points))
        if sample_size > 0:
            sampled_indices = np.random.choice(len(complex_points), sample_size, replace=False)
            sampled_points = [complex_points[i] for i in sampled_indices]
            
            # Create a SimplicialComplex with points as 0-simplices
            simplex_list = []
            for i, point in enumerate(sampled_points):
                simplex_list.append([i])  # Each point is a 0-simplex
            
            # Add some edges (1-simplices) between nearby points
            points_array = np.array(sampled_points)
            for i in range(min(100, len(points_array) - 1)):
                for j in range(i + 1, min(i + 10, len(points_array))):
                    # Connect points that are close to each other
                    dist = np.linalg.norm(points_array[i] - points_array[j])
                    if dist < 0.1:  # Threshold for creating edges
                        simplex_list.append([i, j])
            
            # Create the SimplicialComplex
            try:
                simplex_complex = SimplicialComplex(simplex_list)
                
                # Get simplices for visualization
                # Plot 2: TopoNetX representation
                ax2.scatter(points_array[:, 0], points_array[:, 1], 
                           c=points_array[:, 1], cmap='viridis', s=2, alpha=0.6)
                
                # Draw edges (1-simplices) from the complex
                if hasattr(simplex_complex, 'skeleton') and 1 in simplex_complex.skeleton:
                    edges = simplex_complex.skeleton[1]
                    for edge in edges:
                        if len(edge) == 2:
                            p1 = points_array[edge[0]]
                            p2 = points_array[edge[1]]
                            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3, linewidth=0.5)
                
                ax2.set_title('TopoNetX Simplicial Complex Representation')
                ax2.set_xlabel('Real')
                ax2.set_ylabel('Imaginary')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_xlim(x_range)
                ax2.set_ylim(y_range)
                
                print(f"Number of points in Mandelbrot set: {len(complex_points)}")
                print(f"Sampled points for TopoNetX: {sample_size}")
                print(f"Number of simplices in complex: {len(simplex_list)}")
                
            except Exception as e:
                print(f"Error creating SimplicialComplex: {e}")
                # Fallback: just plot points
                ax2.scatter(points_array[:, 0], points_array[:, 1], 
                           c=points_array[:, 1], cmap='viridis', s=2, alpha=0.6)
                ax2.set_title('TopoNetX - Point Cloud (Fallback)')
                ax2.set_xlabel('Real')
                ax2.set_ylabel('Imaginary')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_xlim(x_range)
                ax2.set_ylim(y_range)
    
    plt.tight_layout()
    plt.show()
    
    return mandelbrot_data, fig

def compute_topological_features(mandelbrot_data, threshold=0.5):
    """Compute topological features using TopoNetX"""
    from toponetx.classes import SimplicialComplex
    
    # Binarize the Mandelbrot set
    binary_data = (mandelbrot_data < mandelbrot_data.max()).astype(int)
    
    # Find connected regions (simplices)
    height, width = binary_data.shape
    simplices = []
    
    # Add points as 0-simplices
    for i in range(height):
        for j in range(width):
            if binary_data[i, j] == 1:
                simplices.append([(i, j)])
    
    # Add edges (1-simplices) between adjacent points
    for i in range(height - 1):
        for j in range(width - 1):
            if binary_data[i, j] == 1 and binary_data[i+1, j] == 1:
                simplices.append([(i, j), (i+1, j)])
            if binary_data[i, j] == 1 and binary_data[i, j+1] == 1:
                simplices.append([(i, j), (i, j+1)])
            if binary_data[i, j] == 1 and binary_data[i+1, j+1] == 1:
                simplices.append([(i, j), (i+1, j+1)])
    
    return simplices

def zoom_mandelbrot(x_center, y_center, zoom_factor, width=800, height=600, max_iter=200):
    """Create a zoomed view of the Mandelbrot set"""
    
    zoom_width = 3.0 / zoom_factor
    zoom_height = 3.0 / zoom_factor
    
    x_range = (x_center - zoom_width/2, x_center + zoom_width/2)
    y_range = (y_center - zoom_height/2, y_center + zoom_height/2)
    
    print(f"Zooming to center ({x_center:.6f}, {y_center:.6f}) with factor {zoom_factor}")
    
    mandelbrot_data, real, imag = generate_mandelbrot_data(
        width, height, x_range, y_range, max_iter
    )
    
    # Create colormap
    colors = ['#000000', '#000033', '#000066', '#000099', '#0000CC', '#0000FF', 
              '#0033FF', '#0066FF', '#0099FF', '#00CCFF', '#00FFFF']
    cmap = LinearSegmentedColormap.from_list('zoom_cmap', colors, N=256)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main zoomed view
    im = ax1.imshow(mandelbrot_data, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                   cmap=cmap, origin='lower', interpolation='bilinear')
    ax1.set_title(f'Mandelbrot Set - Zoom {zoom_factor}x at ({x_center:.6f}, {y_center:.6f})')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.grid(True, alpha=0.3, linestyle='--')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Iteration Count')
    
    # TopoNetX analysis of zoomed region
    mask = mandelbrot_data >= max_iter
    points_in_set = np.argwhere(mask)
    
    if len(points_in_set) > 0:
        # Sample points for visualization
        sample_size = min(2000, len(points_in_set))
        sampled_indices = np.random.choice(len(points_in_set), sample_size, replace=False)
        
        # Convert to coordinates
        sampled_points = []
        for idx in sampled_indices:
            y_idx, x_idx = points_in_set[idx]
            if y_idx < len(imag) and x_idx < len(real):
                sampled_points.append((real[x_idx], imag[y_idx]))
        
        if sampled_points:
            points_array = np.array(sampled_points)
            ax2.scatter(points_array[:, 0], points_array[:, 1], 
                       c=points_array[:, 1], cmap='plasma', s=2, alpha=0.7)
            ax2.set_title('TopoNetX - Zoomed Region Point Cloud')
            ax2.set_xlabel('Real')
            ax2.set_ylabel('Imaginary')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(x_range)
            ax2.set_ylim(y_range)
    
    plt.tight_layout()
    plt.show()
    
    return mandelbrot_data, fig

if __name__ == "__main__":
    # Generate the main Mandelbrot set
    print("Creating Mandelbrot set with TopoNetX...")
    mandelbrot_data, fig = create_mandelbrot_with_toponetx()
    
    # Optional: Create zoomed views
    # Uncomment to see famous regions
    print("\nTry uncommenting the zoom examples to explore famous Mandelbrot regions!")
    # zoom_mandelbrot(-0.75, 0.1, 10, max_iter=300)
    # zoom_mandelbrot(-0.5, 0.0, 50, max_iter=500)
    # zoom_mandelbrot(-1.25, 0.0, 20, max_iter=400)  # Seahorse valley area