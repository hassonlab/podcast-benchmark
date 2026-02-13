"""
Integration code for the Example Foundation Model

This module demonstrates TWO ways to use a foundation model in the benchmark:

1. **Feature Extraction (Frozen)**: Load a pretrained model, freeze it, and use it to
   extract embeddings during preprocessing. Then train a simple decoder on top.

2. **Finetuning**: Include the foundation model as part of your decoder architecture,
   and continue training it (fully or partially) on your downstream task.

Both patterns are registered with the framework's registry system.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from core import registry
from .simple_transformer import load_pretrained_model, SimpleTransformer


# =============================================================================
# PATTERN 1: FEATURE EXTRACTION (FROZEN MODEL)
# =============================================================================


@registry.register_data_preprocessor("population_transformer_feature_extraction")
def extract_foundation_features(data, preprocessor_params):
    """
    Extract frozen features from the foundation model during preprocessing.

    This pattern:
    1. Loads the pretrained foundation model
    2. Freezes all parameters
    3. Runs the data through the model to extract embeddings
    4. Returns embeddings which are then used to train a simple decoder

    Args:
        data: Neural data of shape [num_samples, num_channels, num_timepoints]
        preprocessor_params: Dictionary with:
            - model_dir: Path to pretrained model directory
            - batch_size: Batch size for processing (default: 32)

    Returns:
        embeddings: Numpy array of shape [num_samples, model_dim]
    """
    model_dir = preprocessor_params["model_dir"]
    batch_size = preprocessor_params.get("batch_size", 32)

    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(model_dir, device=device)
    model.eval()
    model.freeze()

    print(f"Loaded foundation model from {model_dir}")
    print(f"Model has {model.get_num_params():,} parameters (all frozen)")

    # Extract embeddings in batches
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
            batch = torch.tensor(
                data[i : i + batch_size], dtype=torch.float32, device=device
            )
            batch_embeddings = model(batch, return_sequence=False)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Extracted embeddings of shape: {embeddings.shape}")

    return embeddings


@registry.register_config_setter("set_sample_rate_for_stft")
def set_sample_rate_for_stft(experiment_config, raws, _df_word):
    """
    Config setter to set sample_rate in data_params for STFT preprocessing.

    This is used to ensure that the STFT preprocessor uses the correct sampling rate.
    """
    # Assume all raws have the same sampling rate
    sample_rate = int(raws[0].info["sfreq"])
    data_params = experiment_config.task_config.data_params
    if not data_params.preprocessor_params:
        # Defaults from popT paper.
        data_params.preprocessor_params = {
            "fs": sample_rate,
            "freq_channel_cutoff": 40,
            "nperseg": 400,
            "noverlap": 350,
            "normalizing": "zscore",
        }
    data_params.preprocessor_params["sample_rate"] = sample_rate
    print(f"Set sample_rate for STFT preprocessing to {sample_rate} Hz")
    return experiment_config


@registry.register_data_preprocessor("stft_preprocessing")
def stft_preprocessing(data, preprocessor_params):
    # Import STFT preprocessor
    from models.popt.preprocessors.stft import STFTPreprocessor

    # Initialize STFT preprocessor
    stft_preprocessor = STFTPreprocessor(**preprocessor_params)
    stft_preprocessor.eval()

    # Apply STFT to all data: [batch, channels, time] → [batch, channels, time_stft, freq_channels]
    # Check if GPU is available for STFT acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert numpy to torch tensor and move to GPU if available
    datas_torch = torch.from_numpy(data).to(device)

    # Move STFT preprocessor to same device (for window function)
    stft_preprocessor = stft_preprocessor.to(device)

    # STFTPreprocessor.forward() automatically uses GPU if input is on GPU
    # Handles batched input: [batch, channels, time] → [batch, channels, time_stft, freq_channels]
    with torch.no_grad():
        stft_output = stft_preprocessor(
            datas_torch
        )  # [batch, channels, time_stft, freq_channels]

    # Convert back to numpy for consistency with rest of pipeline
    datas = stft_output.cpu().numpy()

    print(
        f"Applied STFT preprocessing ({'GPU' if device.type == 'cuda' else 'CPU'}): "
        f"input shape {datas_torch.shape} → output shape {datas.shape}"
    )

    return datas


# Simple MLP decoder to use on top of frozen features
class MLPDecoder(nn.Module):
    """
    Simple MLP decoder for use with frozen foundation features.

    This is used with PATTERN 1 (feature extraction).
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        output_activation: str = "linear",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # Build layers
        prev_dim = input_dim
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_dim, size))
            if use_layer_norm and size != layer_sizes[-1]:
                self.layer_norms.append(nn.LayerNorm(size))
            prev_dim = size

        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        self.output_activation = output_activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Don't apply activation/dropout to final layer
            if i < len(self.layers) - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        # Apply output activation if specified
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "softmax":
            x = F.softmax(x, dim=-1)
        # "linear" means no activation (default)

        # Squeeze the output to match the label shape [batch_size] instead of [batch_size, 1]
        # This matches the behavior of neural_conv_decoder
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x


@registry.register_model_constructor("population_transformer_mlp")
def create_mlp_decoder(model_params):
    """
    Create MLP decoder for frozen foundation features.

    This is used with PATTERN 1 (feature extraction).

    Expected model_params:
        - input_dim: Dimension of foundation model embeddings
        - layer_sizes: List of layer sizes for MLP
        - dropout: Dropout probability (optional)
        - use_layer_norm: Whether to use layer normalization (optional)
    """
    return MLPDecoder(
        input_dim=model_params["input_dim"],
        layer_sizes=model_params["layer_sizes"],
        dropout=model_params.get("dropout", 0.0),
        use_layer_norm=model_params.get("use_layer_norm", False),
    )


# =============================================================================
# PATTERN 2: FINETUNING (TRAINABLE MODEL)
# =============================================================================


class PopulationTransformerDecoder(nn.Module):
    """
    Decoder that includes the PopulationTransformer model as a trainable submodule.

    This pattern:
    1. Loads the PopulationTransformer model with pretrained weights
    2. Optionally freezes some layers (partial finetuning)
    3. Includes the PopulationTransformer model in the decoder architecture
    4. During training, gradients flow through unfrozen parts

    This is used with PATTERN 2 (finetuning).
    """

    def __init__(
        self,
        model_dir: str,
        output_dim: int,
        mlp_layer_sizes: list[int],
        freeze_foundation: bool = False,
        num_frozen_layers: int = 0,
        dropout: float = 0.0,
        input_channels: int = None,
        output_activation: str = "linear",
        use_lip_coords: bool = False,
        max_coord_value: int = 5000,
        use_brainbert: bool = True,
        brainbert_model_dir: str = None,
        stft_config: dict = None,
        sample_rate: int = 2048,
    ):
        """
        Args:
            model_dir: Path to pretrained foundation model directory
            output_dim: Output dimension for final predictions
            mlp_layer_sizes: Layer sizes for decoder head MLP
            freeze_foundation: If True, freeze entire foundation model
            num_frozen_layers: Number of foundation layers to freeze (0 = none)
            dropout: Dropout probability
            input_channels: Number of input channels (if different from pretrained model)
            output_activation: Activation function for output layer ("linear", "sigmoid", "tanh", "softmax")
            use_lip_coords: If True, use LIP coordinates for positional encoding
            max_coord_value: Maximum coordinate value for PE table
            use_brainbert: If True, use BrainBERT to process time sequences (default: True, matching original PopT)
            brainbert_model_dir: Path to BrainBERT pretrained model directory (default: models/brainbert/pretrained_model)
            stft_config: STFT preprocessing configuration dict. If None, uses default values matching original PopT.
            sample_rate: Sampling rate (Hz) for STFT preprocessing. Default: 2048
        """
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BrainBERT model if needed
        self.use_brainbert = use_brainbert
        if use_brainbert:
            if brainbert_model_dir is None:
                # Default BrainBERT path
                brainbert_model_dir = "models/brainbert/pretrained_model"

            try:
                from models.brainbert.simple_transformer import (
                    load_pretrained_model as load_brainbert,
                )

                self.brainbert = load_brainbert(brainbert_model_dir, device=device)
                self.brainbert.eval()
                # BrainBERT is pretrained, so freeze it
                for param in self.brainbert.parameters():
                    param.requires_grad = False
                print(f"Loaded BrainBERT from {brainbert_model_dir}")

                # Initialize STFT preprocessor (matching original PopT)
                if stft_config is None:
                    # Default STFT configuration matching original PopT
                    stft_config = {
                        "fs": sample_rate,
                        "freq_channel_cutoff": 40,
                        "nperseg": 400,
                        "noverlap": 350,
                        "normalizing": "zscore",
                    }

                from .preprocessors.stft import STFTPreprocessor

                self.stft_preprocessor = STFTPreprocessor(**stft_config)
                self.stft_preprocessor.eval()
                # STFT is deterministic, no gradients needed
                for param in self.stft_preprocessor.parameters():
                    param.requires_grad = False
                print(
                    f"Initialized STFT preprocessor with fs={sample_rate}, freq_channels=40"
                )

            except Exception as e:
                print(
                    f"Warning: Failed to load BrainBERT from {brainbert_model_dir}: {e}"
                )
                print("Falling back to direct input processing (use_brainbert=False)")
                self.use_brainbert = False
                self.brainbert = None
                self.stft_preprocessor = None
        else:
            self.brainbert = None
            self.stft_preprocessor = None

        # Load pretrained foundation model
        self.foundation_model = load_pretrained_model(model_dir, device=device)

        # If use_lip_coords is True, we need to replace the positional encoder
        if use_lip_coords:
            from .simple_transformer import MultiSubjBrainPositionalEncoding

            self.foundation_model.use_lip_coords = True
            self.foundation_model.pos_encoder = MultiSubjBrainPositionalEncoding(
                self.foundation_model.model_dim, dropout, max_coord_value
            )
            # Move to device
            self.foundation_model.pos_encoder.to(device)
        else:
            self.foundation_model.use_lip_coords = False

        self.use_lip_coords = use_lip_coords

        # Handle input_projection based on whether BrainBERT is used
        if self.use_brainbert:
            # BrainBERT outputs 768-dimensional embeddings per channel
            expected_input_channels = 768
            if self.foundation_model.input_channels != expected_input_channels:
                print(
                    f"Reinitializing input_projection for BrainBERT output: "
                    f"{self.foundation_model.input_channels} -> {expected_input_channels}"
                )
                model_dim = self.foundation_model.model_dim
                self.foundation_model.input_projection = nn.Linear(
                    expected_input_channels, model_dim
                )
                self.foundation_model.input_channels = expected_input_channels
        else:
            # Direct input processing: use actual channel count
            if (
                input_channels is not None
                and input_channels != self.foundation_model.input_channels
            ):
                print(
                    f"Reinitializing input_projection: {self.foundation_model.input_channels} -> {input_channels}"
                )
                model_dim = self.foundation_model.model_dim
                self.foundation_model.input_projection = nn.Linear(
                    input_channels, model_dim
                )
                self.foundation_model.input_channels = input_channels

        # Handle freezing
        if freeze_foundation:
            self.foundation_model.freeze()
            print("Foundation model completely frozen")
        elif num_frozen_layers > 0:
            self.foundation_model.freeze_layers(num_frozen_layers)
            print(f"Froze first {num_frozen_layers} layers of foundation model")
        else:
            print("Foundation model fully trainable")

        # Decoder head (MLP on top of foundation features)
        foundation_dim = self.foundation_model.model_dim
        self.decoder_head = MLPDecoder(
            input_dim=foundation_dim,
            layer_sizes=mlp_layer_sizes + [output_dim],
            dropout=dropout,
            use_layer_norm=True,
            output_activation=output_activation,
        )

    def forward(self, x, **kwargs):
        """
        Forward pass through STFT → BrainBERT (if enabled) → PopT foundation model.

        Matches original DIVER_CLIP PopT implementation:
        - Supports data_info_list (original approach)
        - Also supports lip_coords directly (backward compatibility)

        Args:
            x: Input tensor [batch_size, num_channels, seq_len] - raw neural signals
            data_info_list: List of dicts with 'LIP_id' key (original DIVER_CLIP approach)
            lip_coords: Optional LIP coordinates [batch_size, num_channels, 3] LongTensor (backward compatibility)
            **kwargs: Additional keyword arguments (e.g., preserve_ensemble for word embedding tasks)

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Extract LIP_id from data_info_list
        data_info_list = kwargs.get("data_info_list", None)
        lip_coords = kwargs.get("lip_coords", None)

        # Extract LIP_id from data_info_list (matching original PopT's datainfo_to_poptpostion_tensor() approach)
        if data_info_list is not None:
            # Reference original PopT.py's datainfo_to_poptpostion_tensor() approach
            coord_tensor = []
            for i in range(len(data_info_list)):
                coord_tensor.append(data_info_list[i]["LIP_id"])
            lip_coords = torch.stack(
                coord_tensor, dim=0
            )  # [batch_size, num_channels, 3]

        # Check if input is already STFT preprocessed (4D tensor: [batch, channels, time_stft, freq_channels])
        is_stft_preprocessed = len(x.shape) == 4

        # Process through BrainBERT if enabled
        if self.use_brainbert:
            if self.brainbert is None:
                raise ValueError("BrainBERT model not loaded but use_brainbert=True")

            if is_stft_preprocessed:
                # Input is already STFT preprocessed: [batch, channels, time_stft, freq_channels]
                batch_size, num_channels, time_stft, freq_channels = x.shape

                # Reshape: [batch, channels, time_stft, freq_channels] → [batch*channels, time_stft, freq_channels]
                x_stft = x.contiguous().view(
                    batch_size * num_channels, time_stft, freq_channels
                )

                # Convert to BrainBERT input shape: [batch*channels, time_stft, freq_channels] → [batch*channels, freq_channels, time_stft]
                # BrainBERT expects [batch, channels, time] where channels=freq_channels (40)
                x_brainbert = x_stft.transpose(
                    1, 2
                )  # [batch*channels, freq_channels, time_stft]

                with torch.no_grad():
                    # BrainBERT inference (frozen, no gradients) - all channels at once
                    # Return sequence to keep time dimension for middle selection (matching original PopT)
                    all_emb_seq = self.brainbert(
                        x_brainbert, return_sequence=True
                    )  # [batch*channels, time_stft, 768]

                    # Time dimension processing: select middle portion and pool (matching original PopT)
                    time_steps_stft = all_emb_seq.shape[1]
                    middle = int(time_steps_stft / 2)
                    # Select middle 10 time steps: [batch*channels, 10, 768]
                    all_emb_selected = all_emb_seq[:, middle - 5 : middle + 5, :]

                    # Mean pooling over time dimension: [batch*channels, 10, 768] → [batch*channels, 768]
                    all_emb = all_emb_selected.mean(dim=1)  # [batch*channels, 768]

                # Reshape back to [batch, channels, 768]
                x_emb = all_emb.view(batch_size, num_channels, 768)
            else:
                # Input is raw signal: [batch, channels, time]
                # STFT preprocessing should have been done in data loading, but fallback to model-level STFT
                if self.stft_preprocessor is None:
                    raise ValueError(
                        "Raw signal input detected but STFT preprocessor not initialized. "
                        "Either enable stft_preprocessing or initialize STFT preprocessor."
                    )

                batch_size, num_channels, time_steps = x.shape

                # Process all channels at once (matching original PopT implementation)
                x_reshaped = x.contiguous().view(
                    batch_size * num_channels, time_steps
                )  # [batch*channels, time]

                with torch.no_grad():
                    # STFT preprocessing: [batch*channels, time] → [batch*channels, time_stft, 40]
                    all_spec = self.stft_preprocessor(
                        x_reshaped
                    )  # [batch*channels, time_stft, 40]

                    # Move STFT output to same device as input (STFT returns CPU tensor)
                    all_spec = all_spec.to(x.device)  # [batch*channels, time_stft, 40]

                    # Convert to BrainBERT input shape: [batch*channels, time_stft, 40] → [batch*channels, 40, time_stft]
                    all_spec_brainbert = all_spec.transpose(
                        1, 2
                    )  # [batch*channels, 40, time_stft]

                    # BrainBERT inference (frozen, no gradients) - all channels at once
                    all_emb_seq = self.brainbert(
                        all_spec_brainbert, return_sequence=True
                    )  # [batch*channels, time_stft, 768]

                    # Time dimension processing: select middle portion and pool (matching original PopT)
                    time_steps_stft = all_emb_seq.shape[1]
                    middle = int(time_steps_stft / 2)
                    all_emb_selected = all_emb_seq[:, middle - 5 : middle + 5, :]
                    all_emb = all_emb_selected.mean(dim=1)  # [batch*channels, 768]

                # Reshape back to [batch, channels, 768]
                x_emb = all_emb.view(batch_size, num_channels, 768)

            # Pass BrainBERT embeddings to PopT foundation model
            if self.use_lip_coords:
                if lip_coords is None:
                    raise ValueError("lip_coords is required when use_lip_coords=True")
                features = self.foundation_model(
                    x_emb, return_sequence=False, lip_coords=lip_coords
                )
            else:
                features = self.foundation_model(x_emb, return_sequence=False)
        else:
            # Direct processing: pass raw signals to PopT (original behavior)
            if self.use_lip_coords:
                if lip_coords is None:
                    raise ValueError("lip_coords is required when use_lip_coords=True")
                features = self.foundation_model(
                    x, return_sequence=False, lip_coords=lip_coords
                )
            else:
                features = self.foundation_model(x, return_sequence=False)

        # Pass through decoder head
        output = self.decoder_head(features)

        return output


@registry.register_model_data_getter("popt_lip_coords")
def get_popt_lip_coords(task_df, raws, model_params):
    """
    Add lip_coords column with LIP coordinates for PopT model.

    The lip_coords tensor contains electrode coordinates for the PopT positional encoding.
    Each sample gets a copy of the same [num_channels, 3] LongTensor since all samples
    share the same electrode configuration.

    Note: This getter is optional for PopT. Only use it when use_lip_coords=True.
    Specify model_data_getter: popt_lip_coords in config to enable.

    Args:
        task_df: DataFrame containing task-specific data
        raws: List of MNE Raw objects
        model_params: Dictionary of model parameters from ModelSpec

    Returns:
        Tuple of (enriched_df, list of added column names)
    """
    from utils.data_utils import extract_subject_id_from_raw, get_lip_coordinates

    # Extract subject ID and channel names from first raw
    subject_id = extract_subject_id_from_raw(raws[0])
    channel_names = raws[0].ch_names

    # Load LIP coordinates
    data_root = model_params.get("data_root", "data")
    lip_df = get_lip_coordinates(subject_id, data_root)
    print(f"PopT: Loaded LIP coordinates for subject {subject_id}")

    # Create channel name to LIP coordinates mapping
    channel_lip_map = {
        row["name"]: [int(row["x"]), int(row["y"]), int(row["z"])]
        for _, row in lip_df.iterrows()
    }

    # Create lip_coords tensor for all samples (same electrode config for all)
    lip_coords = torch.LongTensor(
        [channel_lip_map.get(ch, [0, 0, 0]) for ch in channel_names]
    )  # [num_channels, 3]

    # Repeat for all samples - will be stacked during batching
    num_samples = len(task_df)
    task_df = task_df.copy()  # Avoid modifying original
    task_df["lip_coords"] = [lip_coords.clone() for _ in range(num_samples)]

    return task_df, ["lip_coords"]


@registry.register_model_constructor("population_transformer_finetune")
def create_finetuning_decoder(model_params):
    """
    Create decoder with foundation model for finetuning.

    This is used with PATTERN 2 (finetuning).

    Expected model_params:
        - model_dir: Path to pretrained model directory
        - output_dim: Output dimension
        - mlp_layer_sizes: Layer sizes for decoder head
        - freeze_foundation: Whether to freeze entire foundation (optional)
        - num_frozen_layers: Number of layers to freeze (optional)
        - dropout: Dropout probability (optional)
        - input_channels: Number of input channels (optional, will be set by config setter)
        - output_activation: Output activation function (optional, auto-determined if not provided)
        - use_lip_coords: Whether to use LIP coordinates for positional encoding (optional, default: False)
        - max_coord_value: Maximum coordinate value for PE table (optional, default: 5000)
        - use_brainbert: Whether to use BrainBERT for time sequence processing (optional, default: True)
        - brainbert_model_dir: Path to BrainBERT pretrained model directory (optional)
    """
    output_dim = model_params["output_dim"]

    # Auto-determine output_activation based on output_dim and task type
    # Binary classification (output_dim=1) with BCE loss should use sigmoid
    output_activation = model_params.get("output_activation", None)
    if output_activation is None:
        if output_dim == 1:
            # Binary classification: use sigmoid for BCE loss compatibility
            output_activation = "sigmoid"
        elif output_dim > 1:
            # Multiclass classification: use softmax for cross_entropy loss
            output_activation = "softmax"
        else:
            # Regression or other: no activation
            output_activation = "linear"

    return PopulationTransformerDecoder(
        model_dir=model_params["model_dir"],
        output_dim=output_dim,
        mlp_layer_sizes=model_params.get("mlp_layer_sizes", [128]),
        freeze_foundation=model_params.get("freeze_foundation", False),
        num_frozen_layers=model_params.get("num_frozen_layers", 0),
        dropout=model_params.get("dropout", 0.1),
        input_channels=model_params.get("input_channels", None),
        output_activation=output_activation,
        use_lip_coords=model_params.get("use_lip_coords", False),
        max_coord_value=model_params.get("max_coord_value", 5000),
        # use_brainbert is automatically set by config setter based on use_lip_coords
        # Default here is only used if config setter hasn't set it
        use_brainbert=model_params.get(
            "use_brainbert", False
        ),  # Will be overridden by config setter
        brainbert_model_dir=model_params.get("brainbert_model_dir", None),
        stft_config=model_params.get("stft_config", None),
        sample_rate=model_params.get("sample_rate", 2048),
    )


# =============================================================================
# CONFIG SETTERS
# =============================================================================


@registry.register_config_setter("population_transformer_feature_extraction")
def set_feature_extraction_config(experiment_config, raws, _df_word):
    """
    Config setter for feature extraction pattern.

    Sets the input_dim for the MLP based on the foundation model's dimension.
    """
    from .config import load_config

    model_params = experiment_config.model_spec.params
    model_dir = model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set input_dim to match foundation model output
    model_params["input_dim"] = foundation_config.model_dim

    # Set preprocessor params
    data_params = experiment_config.task_config.data_params
    if not data_params.preprocessor_params:
        data_params.preprocessor_params = {}
    data_params.preprocessor_params["model_dir"] = model_dir

    return experiment_config


@registry.register_config_setter("population_transformer_finetune")
def set_finetuning_config(experiment_config, raws, _df_word):
    """
    Config setter for finetuning pattern.

    Sets the output_dim and loads foundation model config.
    Automatically sets use_brainbert based on use_lip_coords:
    - use_lip_coords=True -> use_brainbert=True (original PopT with LIP)
    - use_lip_coords=False -> use_brainbert=False (direct input processing)
    """
    from .config import load_config

    model_params = experiment_config.model_spec.params
    data_params = experiment_config.task_config.data_params

    # Automatically set use_brainbert based on use_lip_coords
    use_lip_coords = model_params.get("use_lip_coords", False)

    # If use_brainbert is not explicitly set, determine based on use_lip_coords
    if "use_brainbert" not in model_params:
        if use_lip_coords:
            # LIP coordinates require BrainBERT (original PopT implementation)
            model_params["use_brainbert"] = True
            print(
                "use_lip_coords=True: automatically setting use_brainbert=True (original PopT with LIP)"
            )
        else:
            # No LIP: use direct input processing (existing benchmark behavior)
            model_params["use_brainbert"] = False
            print(
                "use_lip_coords=False: automatically setting use_brainbert=False (direct input processing)"
            )
    else:
        # use_brainbert was explicitly set, use that value
        use_brainbert = model_params["use_brainbert"]
        if use_lip_coords and not use_brainbert:
            print(
                "Warning: use_lip_coords=True but use_brainbert=False. "
                "This may not match original PopT implementation."
            )

    use_brainbert = model_params.get("use_brainbert", False)

    if use_brainbert:
        # Enable STFT preprocessing in data loading (matching original neuroprobe implementation)
        data_params.use_stft_preprocessing = True

        # Get sample rate from data for STFT preprocessing
        sample_rate = int(raws[0].info["sfreq"]) if raws else 2048

        # Set STFT configuration in data_params (matching original PopT)
        data_params.stft_config = {
            "freq_channel_cutoff": 40,
            "nperseg": 400,
            "noverlap": 350,
            "normalizing": "zscore",
        }
        print(
            f"STFT preprocessing enabled in data loading: fs={sample_rate}, freq_channels=40, nperseg=400, noverlap=350"
        )

        # BrainBERT outputs 768-dimensional embeddings per channel
        # PopT will receive [batch, channels, 768] instead of [batch, channels, time]
        model_params["input_channels"] = 768
        print("Using BrainBERT: input_channels set to 768 (BrainBERT output dimension)")
    else:
        # Direct input processing: input_channels = time dimension per electrode
        # In PopT, each electrode is a token in the sequence. The feature dimension
        # per token is the number of time samples (window_width * sample_rate).
        # NOTE: set_input_channels sets input_channels to num_electrodes, which is
        # wrong for PopT — that's the sequence length, not the feature dimension.
        pass

    model_dir = model_params["model_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    foundation_config = load_config(config_path)

    # Set window width based on foundation model
    data_params.window_width = foundation_config.window_width

    # For direct input, compute input_channels from window_width and sample rate
    if not use_brainbert:
        sample_rate = int(raws[0].info["sfreq"]) if raws else 512
        time_steps = int(foundation_config.window_width * sample_rate)
        model_params["input_channels"] = time_steps
        print(
            f"Direct input processing: input_channels set to {time_steps} "
            f"(window_width={foundation_config.window_width}s x sample_rate={sample_rate}Hz)"
        )

    # Fix: Copy output_dim to embedding_dim for compatibility with compute_all_metrics
    # This ensures that confusion_matrix can correctly determine num_classes
    if "output_dim" in model_params:
        model_params["embedding_dim"] = model_params["output_dim"]

    return experiment_config
