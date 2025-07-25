import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
import mne

# Add population_transformer to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'population_transformer'))

import registry
from config import ExperimentConfig

# Import PopulationTransformer modules
from population_transformer.models import build_model as pt_build_model
from population_transformer.preprocessors import build_preprocessor as pt_build_preprocessor


class PopulationTransformerDecoder(nn.Module):
    """
    A simple MLP decoder that takes PopulationTransformer embeddings and decodes to word embeddings.
    This is similar to the foundation_model approach but uses PopulationTransformer features.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


@registry.register_model_constructor()
def population_transformer_mlp(model_params: Dict[str, Any]) -> PopulationTransformerDecoder:
    """
    Model constructor for PopulationTransformer MLP decoder.
    
    Args:
        model_params: Dictionary containing:
            - input_dim: Input dimension from PopulationTransformer embeddings
            - output_dim: Output dimension for word embeddings  
            - hidden_dims: List of hidden layer dimensions (optional)
    
    Returns:
        PopulationTransformerDecoder model
    """
    return PopulationTransformerDecoder(
        input_dim=model_params['input_dim'],
        output_dim=model_params['output_dim'],
        hidden_dims=model_params.get('hidden_dims', [256, 128])
    )


@registry.register_data_preprocessor()
def population_transformer_preprocessing_fn(data: np.ndarray, preprocessor_params: Dict[str, Any]) -> np.ndarray:
    """
    Data preprocessing function for PopulationTransformer.
    
    Args:
        data: Neural data of shape [num_words, num_electrodes, timesteps]
        preprocessor_params: Dictionary containing:
            - model_path: Path to PopulationTransformer model weights
            - model_config: PopulationTransformer model configuration
            - batch_size: Batch size for processing (default: 32)
            - device: Device to use ('cuda' or 'cpu')
    
    Returns:
        PopulationTransformer embeddings of shape [num_words, embedding_dim]
    """
    
    print("🧠 PopulationTransformer preprocessing started...")
    device = torch.device(preprocessor_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    batch_size = preprocessor_params.get('batch_size', 32)
    model_path = preprocessor_params['model_path']
    print(f"   📁 Loading model from: {model_path}")
    print(f"   🔧 Using device: {device}")
    print(f"   📦 Batch size: {batch_size}")
    
    # Load PopulationTransformer model
    print("   🔄 Loading model weights...")
    model_state = torch.load(model_path, weights_only=False, map_location='cpu')
    model_config = model_state.get('model_cfg', preprocessor_params['model_config'])
    print("   ✅ Model weights loaded")
    print(f"   📋 Model config: {model_config}")
    
    # Build PopulationTransformer model
    print("   🏗️  Building model architecture...")
    print(f"   🔍 Looking for model name: {model_config.name if hasattr(model_config, 'name') else 'No name attribute'}")
    
    # Debug: Check what models are available
    from population_transformer.models import MODEL_REGISTRY
    print(f"   📋 Available models: {list(MODEL_REGISTRY.keys())}")
    
    pt_model = pt_build_model(model_config)
    pt_model.load_state_dict(model_state['model'])
    pt_model = pt_model.to(device)
    pt_model.eval()
    print("   ✅ Model built and ready")
    
    # Prepare data for PopulationTransformer
    print("   📊 Preparing data for PopulationTransformer...")
    # PopulationTransformer expects data in a specific format
    processed_data = prepare_data_for_population_transformer(data, preprocessor_params)
    print(f"   ✅ Data prepared: {processed_data.shape}")
    
    # Extract embeddings in batches
    print("   🔄 Extracting embeddings in batches...")
    embeddings = []
    total_batches = (len(processed_data) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(processed_data), batch_size):
            batch_data = processed_data[i:i+batch_size]
            
            batch_num = i // batch_size + 1
            print(f"      Processing batch {batch_num}/{total_batches}...")
            
            # Convert to tensor and move to device
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
            
            # Create dummy attention mask (assuming no padding for now)
            attention_mask = torch.ones(batch_tensor.shape[0], batch_tensor.shape[1]).to(device)
            
            # Create dummy position information (subject and coordinate info)
            # This will need to be adapted based on your specific setup
            positions = create_dummy_positions(batch_tensor.shape[0], batch_tensor.shape[1], device)
            
            # Forward pass through PopulationTransformer
            # Using intermediate_rep=True to get embeddings instead of final outputs
            batch_embeddings = pt_model(batch_tensor, attention_mask, positions, intermediate_rep=True)
            
            # Extract CLS token or aggregate embeddings
            if len(batch_embeddings.shape) == 3:  # [batch, seq, dim]
                # Use CLS token (first token) or mean pool
                if preprocessor_params.get('use_cls_token', True):
                    batch_embeddings = batch_embeddings[:, 0, :]  # CLS token
                else:
                    batch_embeddings = batch_embeddings.mean(dim=1)  # Mean pooling
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    # Combine all embeddings
    print("   🔗 Combining all embeddings...")
    embeddings = np.vstack(embeddings)
    print(f"   ✅ PopulationTransformer preprocessing completed!")
    print(f"   📊 Final embeddings shape: {embeddings.shape}")
    
    return embeddings


def prepare_data_for_population_transformer(data: np.ndarray, preprocessor_params: Dict[str, Any]) -> np.ndarray:
    """
    Prepare neural data for PopulationTransformer input format.
    
    Args:
        data: Neural data of shape [num_words, num_electrodes, timesteps]
        preprocessor_params: Preprocessing parameters
    
    Returns:
        Formatted data ready for PopulationTransformer
    """
    
    print(f"   📊 Input data shape: {data.shape}")
    
    # PopulationTransformer expects input_dim=768, but our data has fewer features
    # We need to either:
    # 1. Pad/expand the features to 768 dimensions, or
    # 2. Use a different approach
    
    # For now, let's pad the data to match the expected input dimension
    expected_input_dim = 768  # From PopulationTransformer config
    
    if data.shape[-1] < expected_input_dim:
        # Pad with zeros to reach expected dimension
        padding_needed = expected_input_dim - data.shape[-1]
        print(f"   🔧 Padding data from {data.shape[-1]} to {expected_input_dim} features")
        
        # Reshape to [batch, seq, features] and pad
        padded_data = np.zeros((data.shape[0], data.shape[1], expected_input_dim))
        padded_data[:, :, :data.shape[-1]] = data
        
        print(f"   ✅ Padded data shape: {padded_data.shape}")
        return padded_data
    else:
        # If we have more features, truncate
        print(f"   🔧 Truncating data from {data.shape[-1]} to {expected_input_dim} features")
        return data[:, :, :expected_input_dim]


def create_dummy_positions(batch_size: int, seq_length: int, device: torch.device):
    """
    Create dummy position information for PopulationTransformer.
    This will need to be adapted based on your electrode layout and subject information.
    """
    # Create dummy coordinates and sequence IDs
    # coords should be integer indices for indexing into positional encoding
    # seq_length-1 because first token is CLS token
    coords = torch.randint(0, 100, (batch_size, seq_length-1, 3), device=device, dtype=torch.long)  # Dummy 3D coordinates (X,Y,Z)
    seq_id = torch.randint(0, 10, (batch_size, seq_length-1), device=device, dtype=torch.long)  # Dummy sequence IDs
    
    return (coords, seq_id)


@registry.register_config_setter('population_transformer')
def population_transformer_config_setter(experiment_config: ExperimentConfig, 
                                        raws: list[mne.io.Raw], 
                                        df_word) -> ExperimentConfig:
    """
    Config setter for PopulationTransformer to set runtime parameters.
    
    Args:
        experiment_config: The experiment configuration
        raws: List of MNE Raw objects containing neural data
        df_word: DataFrame with word-level metadata
    
    Returns:
        Updated experiment configuration
    """
    
    # Get channel names for electrode mapping
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    preprocessor_params = experiment_config.data_params.preprocessor_params
    preprocessor_params['ch_names'] = ch_names
    
    # Set input dimensions for the decoder model
    # This will be set based on PopulationTransformer's output dimension
    pt_embedding_dim = preprocessor_params.get('pt_embedding_dim', 512)  # Default PopulationTransformer dimension
    experiment_config.model_params['input_dim'] = pt_embedding_dim
    
    # Set output dimension based on word embeddings
    if hasattr(df_word, 'embedding') and len(df_word['embedding']) > 0:
        embedding_dim = len(df_word['embedding'].iloc[0])
        experiment_config.model_params['output_dim'] = embedding_dim
    
    # Set window width based on PopulationTransformer requirements if specified
    if 'window_width' in preprocessor_params:
        experiment_config.data_params.window_width = preprocessor_params['window_width']
    
    return experiment_config 