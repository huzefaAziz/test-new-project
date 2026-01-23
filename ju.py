
import numpy as np
from sklearn.decomposition import KernelPCA
from typing import Dict, Any, List, Optional, Tuple
import pickle
import os



class KPPPPPA:
    
    def __init__(self):
        self.kpca_models: Dict[Any, KernelPCA] = {}
        self.data: Dict[KernelPCA, np.ndarray] = {}
    
    def add_model(self, key: Any, n_components: int = None, kernel: str = 'rbf', 
                  gamma: float = None, **kwargs) -> None:
        """
        Add a new KernelPCA model to the collection.
        
        Args:
            key: Unique identifier for the model
            n_components: Number of components to keep
            kernel: Kernel type ('rbf', 'poly', 'linear', etc.)
            gamma: Kernel coefficient for 'rbf' and 'poly'
            **kwargs: Additional arguments for KernelPCA
        """
        self.kpca_models[key] = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            **kwargs
        )
    
    def fit(self, key: Any, X: np.ndarray) -> None:
        """
        Fit a KernelPCA model with the given data.
        
        Args:
            key: Model identifier
            X: Training data of shape (n_samples, n_features)
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist. Use add_model() first.")
        
        model = self.kpca_models[key]
        model.fit(X)
        self.data[model] = X
    
    def transform(self, key: Any, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted KernelPCA model.
        
        Args:
            key: Model identifier
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        return model.transform(X)
    
    def fit_transform(self, key: Any, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            key: Model identifier
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Transformed data
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist. Use add_model() first.")
        
        model = self.kpca_models[key]
        transformed = model.fit_transform(X)
        self.data[model] = X
        return transformed
    
    def get_model(self, key: Any) -> KernelPCA:
        """
        Get a KernelPCA model by key.
        
        Args:
            key: Model identifier
            
        Returns:
            The KernelPCA model
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        return self.kpca_models[key]
    
    def remove_model(self, key: Any) -> None:
        """
        Remove a model from the collection.
        
        Args:
            key: Model identifier to remove
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models.pop(key)
        if model in self.data:
            del self.data[model]
    
    def get_all_keys(self) -> list:
        """
        Get all model keys.
        
        Returns:
            List of all model keys
        """
        return list(self.kpca_models.keys())
    
    def inverse_transform(self, key: Any, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            key: Model identifier
            X_transformed: Transformed data
            
        Returns:
            Data in original space
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        return model.inverse_transform(X_transformed)
    
    def get_explained_variance_ratio(self, key: Any) -> np.ndarray:
        """
        Get the explained variance ratio for a model.
        
        Args:
            key: Model identifier
            
        Returns:
            Explained variance ratio array
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        if not hasattr(model, 'eigenvalues_'):
            raise ValueError(f"Model with key '{key}' has not been fitted yet.")
        
        eigenvalues = model.eigenvalues_
        return eigenvalues / eigenvalues.sum()
    
    def get_n_components(self, key: Any) -> int:
        """
        Get the number of components for a model.
        
        Args:
            key: Model identifier
            
        Returns:
            Number of components
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        if hasattr(model, 'eigenvalues_'):
            return len(model.eigenvalues_)
        return model.n_components if model.n_components else 0
    
    def batch_fit(self, X: np.ndarray, keys: Optional[List[Any]] = None) -> None:
        """
        Fit multiple models with the same data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            keys: List of model keys to fit. If None, fits all models.
        """
        if keys is None:
            keys = list(self.kpca_models.keys())
        
        for key in keys:
            if key not in self.kpca_models:
                raise ValueError(f"Model with key '{key}' does not exist.")
            self.fit(key, X)
    
    def batch_transform(self, X: np.ndarray, keys: Optional[List[Any]] = None) -> Dict[Any, np.ndarray]:
        """
        Transform data using multiple models.
        
        Args:
            X: Data to transform
            keys: List of model keys to use. If None, uses all models.
            
        Returns:
            Dictionary mapping keys to transformed data
        """
        if keys is None:
            keys = list(self.kpca_models.keys())
        
        results = {}
        for key in keys:
            if key not in self.kpca_models:
                raise ValueError(f"Model with key '{key}' does not exist.")
            results[key] = self.transform(key, X)
        
        return results
    
    def compare_models(self, X: np.ndarray, keys: Optional[List[Any]] = None) -> Dict[Any, Dict[str, Any]]:
        """
        Compare multiple models by their explained variance.
        
        Args:
            X: Data to evaluate on
            keys: List of model keys to compare. If None, compares all models.
            
        Returns:
            Dictionary with comparison metrics for each model
        """
        if keys is None:
            keys = list(self.kpca_models.keys())
        
        comparisons = {}
        for key in keys:
            if key not in self.kpca_models:
                raise ValueError(f"Model with key '{key}' does not exist.")
            
            model = self.kpca_models[key]
            if not hasattr(model, 'eigenvalues_'):
                comparisons[key] = {'status': 'not_fitted', 'n_components': self.get_n_components(key)}
                continue
            
            X_transformed = self.transform(key, X)
            explained_var = self.get_explained_variance_ratio(key)
            
            comparisons[key] = {
                'n_components': len(explained_var),
                'total_explained_variance': float(explained_var.sum()),
                'mean_explained_variance': float(explained_var.mean()),
                'transformed_shape': X_transformed.shape,
                'kernel': model.kernel,
                'gamma': getattr(model, 'gamma', None)
            }
        
        return comparisons
    
    def save_model(self, key: Any, filepath: str) -> None:
        """
        Save a model to disk.
        
        Args:
            key: Model identifier
            filepath: Path to save the model
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    def load_model(self, key: Any, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            key: Model identifier
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.kpca_models[key] = model
    
    def save_all(self, directory: str) -> None:
        """
        Save all models to a directory.
        
        Args:
            directory: Directory path to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        for key in self.kpca_models.keys():
            filepath = os.path.join(directory, f"model_{key}.pkl")
            self.save_model(key, filepath)
    
    def load_all(self, directory: str) -> None:
        """
        Load all models from a directory.
        
        Args:
            directory: Directory path containing model files
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        for filename in os.listdir(directory):
            if filename.startswith("model_") and filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                key = filename.replace("model_", "").replace(".pkl", "")
                self.load_model(key, filepath)
    
    def get_model_info(self, key: Any) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            key: Model identifier
            
        Returns:
            Dictionary with model information
        """
        if key not in self.kpca_models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        
        model = self.kpca_models[key]
        info = {
            'key': key,
            'kernel': model.kernel,
            'n_components': model.n_components,
            'gamma': getattr(model, 'gamma', None),
            'degree': getattr(model, 'degree', None),
            'coef0': getattr(model, 'coef0', None),
            'alpha': getattr(model, 'alpha', None),
            'fit_inverse_transform': getattr(model, 'fit_inverse_transform', False),
            'is_fitted': hasattr(model, 'eigenvalues_')
        }
        
        if info['is_fitted']:
            info['n_components_actual'] = len(model.eigenvalues_)
            info['eigenvalues'] = model.eigenvalues_.tolist()
            if key in [k for k, m in self.kpca_models.items() if m == model]:
                info['training_data_shape'] = self.data[model].shape if model in self.data else None
        
        return info
    
    def get_all_models_info(self) -> Dict[Any, Dict[str, Any]]:
        """
        Get information about all models.
        
        Returns:
            Dictionary mapping keys to model information
        """
        return {key: self.get_model_info(key) for key in self.kpca_models.keys()}
    
    def clear(self) -> None:
        """
        Clear all models and data.
        """
        self.kpca_models.clear()
        self.data.clear()
    
    def copy_model(self, source_key: Any, target_key: Any) -> None:
        """
        Create a copy of a model with a new key.
        
        Args:
            source_key: Key of the model to copy
            target_key: Key for the new model
        """
        if source_key not in self.kpca_models:
            raise ValueError(f"Source model with key '{source_key}' does not exist.")
        
        if target_key in self.kpca_models:
            raise ValueError(f"Target key '{target_key}' already exists.")
        
        source_model = self.kpca_models[source_key]
        # Create a new model with same parameters
        new_model = KernelPCA(
            n_components=source_model.n_components,
            kernel=source_model.kernel,
            gamma=getattr(source_model, 'gamma', None),
            degree=getattr(source_model, 'degree', None),
            coef0=getattr(source_model, 'coef0', None),
            alpha=getattr(source_model, 'alpha', None),
            fit_inverse_transform=getattr(source_model, 'fit_inverse_transform', False)
        )
        
        # If source model is fitted, fit the new one too
        if hasattr(source_model, 'eigenvalues_') and source_model in self.data:
            new_model.fit(self.data[source_model])
            self.data[new_model] = self.data[source_model]
        
        self.kpca_models[target_key] = new_model
    
    def __len__(self) -> int:
        """
        Return the number of models.
        """
        return len(self.kpca_models)
    
    def __contains__(self, key: Any) -> bool:
        """
        Check if a model key exists.
        """
        return key in self.kpca_models
    
    def __repr__(self) -> str:
        """
        String representation of the class.
        """
        return f"KPPPPPA(n_models={len(self.kpca_models)}, keys={list(self.kpca_models.keys())})"


# Example usage
if __name__ == "__main__":
    # Create an instance
    kpca_manager = KPPPPPA()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Add and configure models
    kpca_manager.add_model('rbf_model', n_components=5, kernel='rbf', gamma=0.1)
    kpca_manager.add_model('poly_model', n_components=5, kernel='poly', degree=3)
    kpca_manager.add_model('linear_model', n_components=5, kernel='linear')
    
    # Fit models
    print("Fitting models...")
    kpca_manager.fit('rbf_model', X)
    kpca_manager.fit('poly_model', X)
    kpca_manager.fit('linear_model', X)
    
    # Transform data
    print("\nTransforming data...")
    X_transformed_rbf = kpca_manager.transform('rbf_model', X)
    print(f"RBF transformed shape: {X_transformed_rbf.shape}")
    
    # Batch transform
    print("\nBatch transforming...")
    all_transformed = kpca_manager.batch_transform(X)
    for key, transformed in all_transformed.items():
        print(f"{key}: {transformed.shape}")
    
    # Get model information
    print("\nModel information:")
    info = kpca_manager.get_model_info('rbf_model')
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Compare models
    print("\nComparing models:")
    comparison = kpca_manager.compare_models(X)
    for model_key, metrics in comparison.items():
        print(f"\n{model_key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Get explained variance
    print("\nExplained variance ratios:")
    for key in kpca_manager.get_all_keys():
        try:
            var_ratio = kpca_manager.get_explained_variance_ratio(key)
            print(f"{key}: {var_ratio[:3]}... (showing first 3)")
        except Exception as e:
            print(f"{key}: {e}")
    
    print(f"\nTotal models: {len(kpca_manager)}")
    print(f"Model keys: {kpca_manager.get_all_keys()}")
    print(f"\n{kpca_manager}")
    