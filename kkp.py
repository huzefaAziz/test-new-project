
import numpy as np
from sklearn.decomposition import KernelPCA
from typing import Dict, Any, List, Optional, Tuple, Union
import pickle
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Model:
    """Container class for GPT2 model, tokenizer, KernelPCA model, and associated data."""
    
    def __init__(self, gpt2_model: Optional[GPT2LMHeadModel] = None,
                 gpt2_tokenizer: Optional[GPT2Tokenizer] = None,
                 kpca_model: Optional[KernelPCA] = None,
                 data: Optional[np.ndarray] = None):
        self.gpt2_model: Optional[GPT2LMHeadModel] = gpt2_model
        self.gpt2_tokenizer: Optional[GPT2Tokenizer] = gpt2_tokenizer
        self.kpca_model: Optional[KernelPCA] = kpca_model
        self.data: Optional[np.ndarray] = data
    
    def __repr__(self) -> str:
        return (f"Model(gpt2_model={self.gpt2_model is not None}, "
                f"tokenizer={self.gpt2_tokenizer is not None}, "
                f"kpca={self.kpca_model is not None}, "
                f"data_shape={self.data.shape if self.data is not None else None})")


class KPPPPPA:
    
    def __init__(self):
        self.models: Dict[Any, Model] = {}
        self.data: Dict[Any, Model] = {}
    
    def add_model(self, key: Any, gpt2_model: Optional[GPT2LMHeadModel] = None,
                  gpt2_tokenizer: Optional[GPT2Tokenizer] = None,
                  kpca_model: Optional[KernelPCA] = None,
                  data: Optional[np.ndarray] = None,
                  gpt2_model_name: Optional[str] = None,
                  kpca_n_components: Optional[int] = None,
                  kpca_kernel: str = 'rbf',
                  kpca_gamma: Optional[float] = None) -> None:
        """
        Add a new Model to the collection.
        
        Args:
            key: Unique identifier for the model
            gpt2_model: Pre-initialized GPT2LMHeadModel (optional)
            gpt2_tokenizer: Pre-initialized GPT2Tokenizer (optional)
            kpca_model: Pre-initialized KernelPCA (optional)
            data: Associated data array (optional)
            gpt2_model_name: HuggingFace model name to load (e.g., 'gpt2')
            kpca_n_components: Number of components for KernelPCA
            kpca_kernel: Kernel type for KernelPCA
            kpca_gamma: Gamma parameter for KernelPCA
        """
        # Initialize GPT2 components
        if gpt2_model_name:
            if gpt2_model is None:
                gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
            if gpt2_tokenizer is None:
                gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        
        # Initialize KernelPCA if not provided
        if kpca_model is None and kpca_n_components is not None:
            kpca_model = KernelPCA(
                n_components=kpca_n_components,
                kernel=kpca_kernel,
                gamma=kpca_gamma
            )
        
        model = Model(
            gpt2_model=gpt2_model,
            gpt2_tokenizer=gpt2_tokenizer,
            kpca_model=kpca_model,
            data=data
        )
        
        self.models[key] = model
        self.data[key] = model
    
    def get_model(self, key: Any) -> Model:
        """Get a Model by key."""
        if key not in self.models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        return self.models[key]
    
    def remove_model(self, key: Any) -> None:
        """Remove a model from the collection."""
        if key not in self.models:
            raise ValueError(f"Model with key '{key}' does not exist.")
        del self.models[key]
        if key in self.data:
            del self.data[key]
    
    def get_all_keys(self) -> List[Any]:
        """Get all model keys."""
        return list(self.models.keys())
    
    # GPT2 Operations
    def generate_text(self, key: Any, prompt: str, max_length: int = 100,
                     num_return_sequences: int = 1, temperature: float = 1.0,
                     top_k: int = 50, top_p: float = 0.95, **kwargs) -> List[str]:
        """Generate text using GPT2 model."""
        model = self.get_model(key)
        if model.gpt2_model is None or model.gpt2_tokenizer is None:
            raise ValueError(f"GPT2 model or tokenizer not available for key '{key}'")
        
        inputs = model.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.gpt2_model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=model.gpt2_tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_texts = [model.gpt2_tokenizer.decode(output, skip_special_tokens=True)
                          for output in outputs]
        return generated_texts
    
    def encode_text(self, key: Any, text: str, return_tensors: str = 'pt') -> torch.Tensor:
        """Encode text using GPT2 tokenizer."""
        model = self.get_model(key)
        if model.gpt2_tokenizer is None:
            raise ValueError(f"Tokenizer not available for key '{key}'")
        return model.gpt2_tokenizer.encode(text, return_tensors=return_tensors)
    
    def decode_text(self, key: Any, tokens: Union[torch.Tensor, List[int], np.ndarray],
                   skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        model = self.get_model(key)
        if model.gpt2_tokenizer is None:
            raise ValueError(f"Tokenizer not available for key '{key}'")
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        return model.gpt2_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    # KernelPCA Operations
    def fit_kpca(self, key: Any, X: Optional[np.ndarray] = None) -> None:
        """Fit KernelPCA model with data."""
        model = self.get_model(key)
        if model.kpca_model is None:
            raise ValueError(f"KernelPCA model not available for key '{key}'")
        
        data = X if X is not None else model.data
        if data is None:
            raise ValueError(f"No data available for fitting. Provide X or set model.data")
        
        model.kpca_model.fit(data)
        model.data = data
    
    def transform_kpca(self, key: Any, X: np.ndarray) -> np.ndarray:
        """Transform data using KernelPCA."""
        model = self.get_model(key)
        if model.kpca_model is None:
            raise ValueError(f"KernelPCA model not available for key '{key}'")
        return model.kpca_model.transform(X)
    
    def fit_transform_kpca(self, key: Any, X: np.ndarray) -> np.ndarray:
        """Fit and transform data using KernelPCA."""
        model = self.get_model(key)
        if model.kpca_model is None:
            raise ValueError(f"KernelPCA model not available for key '{key}'")
        
        transformed = model.kpca_model.fit_transform(X)
        model.data = X
        return transformed
    
    def inverse_transform_kpca(self, key: Any, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform data back to original space."""
        model = self.get_model(key)
        if model.kpca_model is None:
            raise ValueError(f"KernelPCA model not available for key '{key}'")
        return model.kpca_model.inverse_transform(X_transformed)
    
    def get_kpca_explained_variance(self, key: Any) -> np.ndarray:
        """Get explained variance ratio for KernelPCA."""
        model = self.get_model(key)
        if model.kpca_model is None:
            raise ValueError(f"KernelPCA model not available for key '{key}'")
        
        if not hasattr(model.kpca_model, 'eigenvalues_'):
            raise ValueError(f"KernelPCA model for key '{key}' has not been fitted yet.")
        
        eigenvalues = model.kpca_model.eigenvalues_
        return eigenvalues / eigenvalues.sum()
    
    # Combined Operations
    def set_data(self, key: Any, data: np.ndarray) -> None:
        """Set data for a model."""
        model = self.get_model(key)
        model.data = data
    
    def get_data(self, key: Any) -> Optional[np.ndarray]:
        """Get data for a model."""
        model = self.get_model(key)
        return model.data
    
    # Model Information
    def get_model_info(self, key: Any) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model = self.get_model(key)
        
        info = {
            'key': key,
            'has_gpt2_model': model.gpt2_model is not None,
            'has_tokenizer': model.gpt2_tokenizer is not None,
            'has_kpca_model': model.kpca_model is not None,
            'has_data': model.data is not None,
            'data_shape': model.data.shape if model.data is not None else None
        }
        
        if model.gpt2_model is not None:
            info['gpt2_config'] = model.gpt2_model.config.to_dict() if hasattr(model.gpt2_model, 'config') else None
            info['gpt2_num_params'] = sum(p.numel() for p in model.gpt2_model.parameters()) if hasattr(model.gpt2_model, 'parameters') else None
        
        if model.gpt2_tokenizer is not None:
            info['vocab_size'] = model.gpt2_tokenizer.vocab_size
            info['max_length'] = model.gpt2_tokenizer.model_max_length
        
        if model.kpca_model is not None:
            info['kpca_kernel'] = model.kpca_model.kernel
            info['kpca_n_components'] = model.kpca_model.n_components
            info['kpca_gamma'] = getattr(model.kpca_model, 'gamma', None)
            info['kpca_fitted'] = hasattr(model.kpca_model, 'eigenvalues_')
            if info['kpca_fitted']:
                info['kpca_n_components_actual'] = len(model.kpca_model.eigenvalues_)
        
        return info
    
    def get_all_models_info(self) -> Dict[Any, Dict[str, Any]]:
        """Get information about all models."""
        return {key: self.get_model_info(key) for key in self.models.keys()}
    
    # Persistence
    def save_model(self, key: Any, save_directory: str) -> None:
        """Save a model to disk."""
        model = self.get_model(key)
        os.makedirs(save_directory, exist_ok=True)
        
        # Save GPT2 components
        if model.gpt2_model is not None:
            model.gpt2_model.save_pretrained(os.path.join(save_directory, 'gpt2_model'))
        if model.gpt2_tokenizer is not None:
            model.gpt2_tokenizer.save_pretrained(os.path.join(save_directory, 'gpt2_tokenizer'))
        
        # Save KernelPCA
        if model.kpca_model is not None:
            with open(os.path.join(save_directory, 'kpca_model.pkl'), 'wb') as f:
                pickle.dump(model.kpca_model, f)
        
        # Save data
        if model.data is not None:
            np.save(os.path.join(save_directory, 'data.npy'), model.data)
    
    def load_model(self, key: Any, model_directory: str) -> None:
        """Load a model from disk."""
        if not os.path.exists(model_directory):
            raise FileNotFoundError(f"Model directory not found: {model_directory}")
        
        gpt2_model = None
        gpt2_tokenizer = None
        kpca_model = None
        data = None
        
        # Load GPT2
        gpt2_path = os.path.join(model_directory, 'gpt2_model')
        if os.path.exists(gpt2_path):
            gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path)
        
        tokenizer_path = os.path.join(model_directory, 'gpt2_tokenizer')
        if os.path.exists(tokenizer_path):
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        # Load KernelPCA
        kpca_path = os.path.join(model_directory, 'kpca_model.pkl')
        if os.path.exists(kpca_path):
            with open(kpca_path, 'rb') as f:
                kpca_model = pickle.load(f)
        
        # Load data
        data_path = os.path.join(model_directory, 'data.npy')
        if os.path.exists(data_path):
            data = np.load(data_path)
        
        model = Model(
            gpt2_model=gpt2_model,
            gpt2_tokenizer=gpt2_tokenizer,
            kpca_model=kpca_model,
            data=data
        )
        
        self.models[key] = model
        self.data[key] = model
    
    def save_all(self, base_directory: str) -> None:
        """Save all models to a base directory."""
        os.makedirs(base_directory, exist_ok=True)
        for key in self.models.keys():
            model_dir = os.path.join(base_directory, f"model_{key}")
            self.save_model(key, model_dir)
    
    def load_all(self, base_directory: str) -> None:
        """Load all models from a base directory."""
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Directory not found: {base_directory}")
        
        for item in os.listdir(base_directory):
            if item.startswith("model_"):
                model_dir = os.path.join(base_directory, item)
                key = item.replace("model_", "")
                self.load_model(key, model_dir)
    
    # Utility Methods
    def copy_model(self, source_key: Any, target_key: Any) -> None:
        """Create a copy of a model with a new key."""
        if source_key not in self.models:
            raise ValueError(f"Source model with key '{source_key}' does not exist.")
        if target_key in self.models:
            raise ValueError(f"Target key '{target_key}' already exists.")
        
        import copy
        source_model = self.get_model(source_key)
        new_model = Model(
            gpt2_model=copy.deepcopy(source_model.gpt2_model) if source_model.gpt2_model else None,
            gpt2_tokenizer=copy.deepcopy(source_model.gpt2_tokenizer) if source_model.gpt2_tokenizer else None,
            kpca_model=copy.deepcopy(source_model.kpca_model) if source_model.kpca_model else None,
            data=copy.deepcopy(source_model.data) if source_model.data is not None else None
        )
        
        self.models[target_key] = new_model
        self.data[target_key] = new_model
    
    def clear(self) -> None:
        """Clear all models and data."""
        self.models.clear()
        self.data.clear()
    
    def set_gpt2_to_eval(self, key: Any) -> None:
        """Set GPT2 model to evaluation mode."""
        model = self.get_model(key)
        if model.gpt2_model is not None:
            model.gpt2_model.eval()
    
    def set_gpt2_to_train(self, key: Any) -> None:
        """Set GPT2 model to training mode."""
        model = self.get_model(key)
        if model.gpt2_model is not None:
            model.gpt2_model.train()
    
    def __len__(self) -> int:
        """Return the number of models."""
        return len(self.models)
    
    def __contains__(self, key: Any) -> bool:
        """Check if a model key exists."""
        return key in self.models
    
    def __repr__(self) -> str:
        """String representation of the class."""
        return f"KPPPPPA(n_models={len(self.models)}, keys={list(self.models.keys())})"


# Example usage
if __name__ == "__main__":
    # Create an instance
    manager = KPPPPPA()
    
    # Generate sample data for KernelPCA
    np.random.seed(42)
    sample_data = np.random.randn(100, 10)
    
    # Add a model with GPT2 and KernelPCA
    print("Adding model with GPT2 and KernelPCA...")
    manager.add_model(
        key='model1',
        gpt2_model_name='gpt2',  # This will download GPT2 model
        kpca_n_components=5,
        kpca_kernel='rbf',
        kpca_gamma=0.1,
        data=sample_data
    )
    
    # Fit KernelPCA
    print("\nFitting KernelPCA...")
    manager.fit_kpca('model1')
    
    # Transform data with KernelPCA
    print("\nTransforming data with KernelPCA...")
    transformed = manager.transform_kpca('model1', sample_data)
    print(f"Transformed shape: {transformed.shape}")
    
    # Get explained variance
    try:
        var_ratio = manager.get_kpca_explained_variance('model1')
        print(f"\nExplained variance ratio: {var_ratio[:3]}... (first 3)")
    except Exception as e:
        print(f"\nExplained variance error: {e}")
    
    # Generate text with GPT2
    print("\nGenerating text with GPT2...")
    try:
        generated = manager.generate_text('model1', "The future of AI is", max_length=50)
        print(f"Generated: {generated[0]}")
    except Exception as e:
        print(f"Text generation error: {e}")
    
    # Get model information
    print("\nModel information:")
    info = manager.get_model_info('model1')
    for key, value in info.items():
        if key != 'gpt2_config':  # Skip large config dict
            print(f"  {key}: {value}")
    
    # Get all models info
    print("\nAll models info:")
    all_info = manager.get_all_models_info()
    print(f"Number of models: {len(all_info)}")
    
    print(f"\nTotal models: {len(manager)}")
    print(f"Model keys: {manager.get_all_keys()}")
    print(f"\n{manager}")