import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any
import mne

# Add population_transformer to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'population_transformer'))

import registry
from config import ExperimentConfig

# Import PopulationTransformer modules
from population_transformer.models import build_model as pt_build_model
from population_transformer.preprocessors import build_preprocessor as pt_build_preprocessor

# Copy of Foundation Model's MLP for consistency (avoiding import issues)
class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=F.relu,
        dropout_rate=0.0,
        use_layer_norm=False,
        norm_embedding=False,
    ):
        """
        Initialize a Multi-Layer Perceptron with configurable architecture and LayerNorm.
        This is an exact copy of the Foundation Model's MLP for consistency.

        Args:
            layer_sizes (list): List of integers specifying the size of each layer.
                               First element is input size, last element is output size.
            activation (function): Activation function to use between layers (default: ReLU).
            dropout_rate (float): Dropout probability for regularization (default: 0.).
            use_layer_norm (bool): Whether to use LayerNorm after each hidden layer (default: False).
            norm_embedding (bool): Whether to normalize the output embedding.
        """
        super(MLP, self).__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layer_norm = use_layer_norm
        self.norm_embedding = norm_embedding

        # Create linear layers and layer norms based on specified sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add layer norm for all but the output layer
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(layer_sizes[i + 1]))

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, layer_sizes[0]]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, layer_sizes[-1]]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation, and dropout to all but the final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)

        if self.norm_embedding:
            x = F.normalize(x, dim=1)
        return x


# PopulationTransformerDecoder is now replaced with Foundation Model's MLP copy
# This ensures consistency across all decoders while avoiding import issues


@registry.register_model_constructor()
def population_transformer_mlp(model_params: Dict[str, Any]) -> MLP:
    """
    Model constructor for PopulationTransformer MLP decoder.
    Uses the exact same MLP architecture as Foundation Model for consistency.
    
    Args:
        model_params: Dictionary containing:
            - input_dim: Input dimension from PopulationTransformer embeddings
            - output_dim: Output dimension for word embeddings  
            - hidden_dims: List of hidden layer dimensions (e.g., [256, 100])
            - dropout_rate: Dropout probability (default: 0.2)
            - use_layer_norm: Whether to use LayerNorm (default: True)
    
    Returns:
        MLP model (identical to Foundation Model)
    """
    # Build layer_sizes list: [input_dim, hidden_layers..., output_dim]
    input_dim = model_params['input_dim']
    output_dim = model_params['output_dim']
    
    # Get hidden layers from config, or use default
    hidden_dims = model_params.get('hidden_dims', [256, 100])
    
    # Construct full layer_sizes: [input_dim, hidden..., output_dim]
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    
    return MLP(
        layer_sizes=layer_sizes,
        dropout_rate=model_params.get('dropout_rate', 0.2),
        use_layer_norm=model_params.get('use_layer_norm', True)
    )


class PopulationTransformerEnd2End(nn.Module):
    """
    End-to-end PopulationTransformer model that runs PT in forward() and then
    projects the resulting embedding with the shared MLP head. Supports
    fine-tuning by toggling frozen_weights in model_params.

    Expects inputs shaped [batch, seq_len(electrodes), 768] (already formatted).
    """
    def __init__(self, model_params: Dict[str, Any]):
        super().__init__()

        # Required params
        self.pt_embedding_dim: int = int(model_params.get('pt_embedding_dim', 512))
        output_dim: int = int(model_params['output_dim'])
        hidden_dims: list[int] = model_params.get('hidden_dims', [256, 100])
        dropout_rate: float = float(model_params.get('dropout_rate', 0.2))
        use_layer_norm: bool = bool(model_params.get('use_layer_norm', True))

        # PT runtime config
        self.device = torch.device(model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_cls_token: bool = bool(model_params.get('use_cls_token', True))
        self.frozen_weights: bool = bool(model_params.get('frozen_weights', True))

        # Build PT model
        model_path = model_params['model_path']
        model_state = torch.load(model_path, weights_only=False, map_location='cpu')
        model_config = model_state.get('model_cfg', model_params['model_config'])
        pt_model = pt_build_model(model_config)
        pt_model.load_state_dict(model_state['model'])
        self.pt_model = pt_model.to(self.device)
        self.pt_model.train(not self.frozen_weights)
        if self.frozen_weights:
            for p in self.pt_model.parameters():
                p.requires_grad = False

        # Save raw_data for real positions if available
        self.raw_data = model_params.get('raw_data', None)

        # Build MLP head
        layer_sizes = [self.pt_embedding_dim] + hidden_dims + [output_dim]
        self.mlp = MLP(
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )

    def _make_positions(self, batch_size: int, seq_len_plus_cls: int):
        if self.raw_data is not None:
            coords, seq_id = create_real_positions(self.raw_data, batch_size, seq_len_plus_cls, self.device)
        else:
            coords, seq_id = create_dummy_positions(batch_size, seq_len_plus_cls, self.device)
        return (coords, seq_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, 768]
        x = x.to(self.device)

        # Add CLS token at front
        cls = torch.zeros(x.shape[0], 1, x.shape[2], device=self.device, dtype=x.dtype)
        x_cat = torch.cat([cls, x], dim=1)  # [B, seq_len+1, 768]

        # Attention mask: no padding → all False (bool mask expected by PyTorch)
        attention_mask = torch.zeros(
            x_cat.shape[0], x_cat.shape[1], device=self.device, dtype=torch.bool
        )

        # Positions
        positions = self._make_positions(x_cat.shape[0], x_cat.shape[1])

        # PT forward to get intermediate reps
        pt_out = self.pt_model(x_cat, attention_mask, positions, intermediate_rep=True)
        if pt_out.dim() == 3:
            if self.use_cls_token:
                pt_emb = pt_out[:, 0, :]  # CLS
            else:
                pt_emb = pt_out.mean(dim=1)
        else:
            pt_emb = pt_out

        # Project with MLP head
        return self.mlp(pt_emb)


@registry.register_model_constructor()
def population_transformer_end2end(model_params: Dict[str, Any]) -> PopulationTransformerEnd2End:
    return PopulationTransformerEnd2End(model_params)


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
            - raw_data: MNE Raw object with electrode information (optional)
    
    Returns:
        PopulationTransformer embeddings of shape [num_words, embedding_dim]
    """
    
    print("🧠 PopulationTransformer preprocessing started...")
    device = torch.device(preprocessor_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    batch_size = preprocessor_params.get('batch_size', 32)
    model_path = preprocessor_params['model_path']
    raw_data = preprocessor_params.get('raw_data', None)  # Get MNE Raw data if available
    print(f"   📁 Loading model from: {model_path}")
    print(f"   🔧 Using device: {device}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   🧠 Using real electrode data: {raw_data is not None}")
    
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
            
            # Add CLS token to the beginning of sequence
            # CLS token is a learnable embedding, we'll use zeros for now
            cls_token = torch.zeros(batch_tensor.shape[0], 1, batch_tensor.shape[2]).to(device)
            batch_tensor = torch.cat([cls_token, batch_tensor], dim=1)
            # print(f"      DEBUG: After adding CLS token, batch_tensor.shape = {batch_tensor.shape}")
            
            # Create attention mask: no padding → all False
            attention_mask = torch.zeros(
                batch_tensor.shape[0], batch_tensor.shape[1], device=device, dtype=torch.bool
            )
            
            # Create position information using real electrode data if available
            if raw_data is not None:
                print(f"      Using real electrode coordinates for batch {batch_num}")
                # print(f"      DEBUG: batch_tensor.shape = {batch_tensor.shape}")
                positions = create_real_positions(raw_data, batch_tensor.shape[0], batch_tensor.shape[1], device)
                coords, seq_id = positions
                # print(f"      DEBUG: coords.shape = {coords.shape}, seq_id.shape = {seq_id.shape}")
            else:
                print(f"      Using dummy electrode coordinates for batch {batch_num}")
                positions = create_dummy_positions(batch_tensor.shape[0], batch_tensor.shape[1], device)
                coords, seq_id = positions
                # print(f"      DEBUG: coords.shape = {coords.shape}, seq_id.shape = {seq_id.shape}")
            
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


@registry.register_data_preprocessor('population_transformer_prepare_inputs_fn')
def population_transformer_prepare_inputs_fn(data: np.ndarray, preprocessor_params: Dict[str, Any]) -> np.ndarray:
    """
    Prep-only preprocessor that formats inputs to [num_words, seq_len, 768].
    Does NOT run PopulationTransformer. Use with the end-to-end model.
    """
    return prepare_data_for_population_transformer(data, preprocessor_params)


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
        print(f"   📝 Note: This is a limitation of the current PopulationTransformer model")
        print(f"   💡 Future improvement: Use a model that accepts variable input dimensions")
        
        # Reshape to [batch, seq, features] and pad
        padded_data = np.zeros((data.shape[0], data.shape[1], expected_input_dim))
        padded_data[:, :, :data.shape[-1]] = data
        
        print(f"   ✅ Padded data shape: {padded_data.shape}")
        return padded_data
    else:
        # If we have more features, truncate
        print(f"   🔧 Truncating data from {data.shape[-1]} to {expected_input_dim} features")
        print(f"   📝 Note: This may lose some information")
        return data[:, :, :expected_input_dim]


def create_real_positions(raw_data, batch_size: int, seq_length: int, device: torch.device):
    """
    Create real position information for PopulationTransformer using actual electrode coordinates.
    
    Args:
        raw_data: MNE Raw object containing channel information
        batch_size: Number of samples in batch
        seq_length: Sequence length (number of electrodes + CLS token)
        device: Device to create tensors on
    
    Returns:
        Tuple of (coords, seq_id) where coords are real electrode coordinates
    """
    # Get channel locations from MNE data
    ch_names = raw_data.ch_names
    ch_locs = []
    
    # print(f"      DEBUG: raw_data has {len(ch_names)} channels")
    # print(f"      DEBUG: batch_size={batch_size}, seq_length={seq_length}")
    
    for ch_name in ch_names:
        # Find channel index
        ch_idx = raw_data.ch_names.index(ch_name)
        # Get channel location (first 3 elements are x,y,z coordinates)
        loc = raw_data.info['chs'][ch_idx]['loc'][:3]
        
        # Check if location is valid (not NaN or zero)
        if np.any(np.isnan(loc)) or np.allclose(loc, 0):
            # If no real coordinates, use channel index as proxy
            # This maintains spatial ordering even without real coordinates
            ch_locs.append([ch_idx, ch_idx, ch_idx])
        else:
            # Use real coordinates, convert to integer indices for positional encoding
            # Scale coordinates to reasonable range (0-100) for indexing
            scaled_coords = np.clip((loc + 0.1) * 50, 0, 99).astype(int)
            ch_locs.append(scaled_coords.tolist())
    
    # Convert to tensor format expected by PopulationTransformer
    # print(f"      DEBUG: ch_locs has {len(ch_locs)} entries")
    coords = torch.tensor(ch_locs, dtype=torch.long, device=device)
    # print(f"      DEBUG: coords initial shape = {coords.shape}")
    
    # Repeat for batch size
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    # print(f"      DEBUG: coords final shape = {coords.shape}")
    
    # Create sequence IDs to match the actual number of electrodes (not including CLS token)
    # The CLS token will be handled separately by the positional encoding
    seq_id = torch.zeros(batch_size, len(ch_locs), dtype=torch.long, device=device)
    # print(f"      DEBUG: seq_id shape = {seq_id.shape}")
    
    return (coords, seq_id)


def create_dummy_positions(batch_size: int, seq_length: int, device: torch.device):
    """
    Create dummy position information for PopulationTransformer.
    This is a fallback when real electrode data is not available.
    """
    # Create dummy coordinates and sequence IDs
    # coords should be integer indices for indexing into positional encoding
    # Both coords and seq_id should match the actual sequence length (no CLS token)
    actual_seq_len = seq_length - 1  # Remove CLS token
    coords = torch.randint(0, 100, (batch_size, actual_seq_len, 3), device=device, dtype=torch.long)  # Dummy 3D coordinates (X,Y,Z)
    seq_id = torch.randint(0, 10, (batch_size, actual_seq_len), device=device, dtype=torch.long)  # Dummy sequence IDs
    
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
    
    # Pass the first raw data object to the preprocessor for real electrode coordinates
    # Note: This assumes all subjects have similar electrode layouts
    # For multi-subject data, you might want to handle this differently
    if raws and len(raws) > 0:
        preprocessor_params['raw_data'] = raws[0]  # Use first subject's electrode layout
        print(f"   📍 Using electrode coordinates from subject with {len(raws[0].ch_names)} channels")
        
        # Check if we have real electrode coordinates
        has_real_coords = False
        for ch_idx, ch_name in enumerate(raws[0].ch_names):
            loc = raws[0].info['chs'][ch_idx]['loc'][:3]
            if not (np.any(np.isnan(loc)) or np.allclose(loc, 0)):
                has_real_coords = True
                break
        
        if has_real_coords:
            print(f"   ✅ Found real electrode coordinates in MNE data")
        else:
            print(f"   ⚠️  No real electrode coordinates found, will use channel indices")
    else:
        print(f"   ⚠️  No MNE Raw data available, will use dummy coordinates")
    
    # Copy PT params needed by end-to-end model into model_params
    # Always set embedding dim for MLP head
    pt_embedding_dim = preprocessor_params.get('pt_embedding_dim', 512)
    experiment_config.model_params['pt_embedding_dim'] = pt_embedding_dim
    # Pass through runtime/e2e params if present
    for k in ['model_path', 'model_config', 'device', 'use_cls_token', 'frozen_weights', 'raw_data']:
        if k in preprocessor_params:
            experiment_config.model_params[k] = preprocessor_params[k]
    
    # Set output dimension based on word embeddings
    if hasattr(df_word, 'embedding') and len(df_word['embedding']) > 0:
        embedding_dim = len(df_word['embedding'].iloc[0])
        experiment_config.model_params['output_dim'] = embedding_dim
    
    # Set window width based on PopulationTransformer requirements if specified
    if 'window_width' in preprocessor_params:
        experiment_config.data_params.window_width = preprocessor_params['window_width']
    
    return experiment_config 