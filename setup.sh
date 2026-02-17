set -e

# Cross-platform download function
download_file() {
    local url="$1"
    local output_path="$2"
    local show_progress="$3"
    
    # Try wget first (common on Linux)
    if command -v wget >/dev/null 2>&1; then
        if [ "$show_progress" = "true" ]; then
            wget --show-progress "$url" -O "$output_path"
        else
            wget "$url" -O "$output_path"
        fi
    # Fall back to curl (available on macOS by default)
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$output_path"
    else
        echo "Error: Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
}

# Parse command line arguments first
INSTALL_GPU=false
INSTALL_DEV=false
FORCE_VENV=false
ENV_NAME="decoding_env"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --venv)
            FORCE_VENV=true
            shift
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--gpu] [--dev] [--venv] [--env-name NAME]"
            echo "  --gpu       Install GPU dependencies (CUDA packages)"
            echo "  --dev       Install development dependencies (testing)"
            echo "  --venv      Force using python venv instead of conda"
            echo "  --env-name  Specify virtual environment name (default: decoding_env)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Setup Podcast dataset
# Config
DATASET_ID="ds005574"
REPO_URL="https://github.com/OpenNeuroDatasets/${DATASET_ID}.git"
LOCAL_DATA_DIR="data"
OPENNEURO_BASE_URL="https://s3.amazonaws.com/openneuro.org/ds005574"

# Skip data download for development setup
if [ "$INSTALL_DEV" = true ]; then
    echo "Development setup detected. Skipping data downloads."
    echo "To download data later, run without --dev flag."
else
    # Step 1: Clone the dataset if not already present
    if [ ! -d "$DATASET_ID" ]; then
        echo "Cloning $DATASET_ID..."
        git clone "$REPO_URL"
    fi

    # Step 2: Move/rename folder to 'data'
    if [ ! -d "$LOCAL_DATA_DIR" ]; then
        mv "$DATASET_ID" "$LOCAL_DATA_DIR"
    fi

    # Step 3: Download missing files
    echo "Scanning for files to download..."
    cd "$LOCAL_DATA_DIR"

    # Find all files listed in Git (which may be missing)
    git ls-files | while read -r file; do
        if [ ! -f "$file" ]; then
            echo "Downloading missing file: $file"

            # URL-encode the file path
            url_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$file'))")

            # Download the file from OpenNeuro
            file_url="${OPENNEURO_BASE_URL}/${url_path}"
            echo "Downloading: $file_url"

            # Download the file using cross-platform function
            # Remove symlink if it exists
            [ -L "$(basename "$file_url")" ] && rm "$(basename "$file_url")"
            download_file "$file_url" "$(basename "$file_url")" "true"
            
            # Move to proper directory if needed
            if [[ "$file" == */* ]]; then
                # Create parent directories if needed
                mkdir -p "$(dirname "$file")"
                # Move the downloaded file to the correct location
                mv "$(basename "$file_url")" "$file"
            fi
        else
            echo "File already exists, skipping: $file"
        fi
    done

    cd ..

    echo "All missing files downloaded!"

    # Step 3: Download GloVe vectors
    if [ -f data/glove/glove.6B.50d.txt ]; then
        echo "GloVe vectors already prepared. Skipping download and extraction."
    else
        mkdir -p data/glove
        download_file "https://nlp.stanford.edu/data/glove.6B.zip" "data/glove/glove.6B.zip" "false"

        unzip data/glove/glove.6B.zip -d data/glove/

        rm data/glove/glove.6B.zip

        echo "GloVe vectors downloaded and extracted."
    fi
fi

# Step 4: Setup a new virtual environment and install all the necessary packages

# Load anaconda module if available (HPC systems), otherwise use system Python
if command -v module >/dev/null 2>&1; then
    echo "Loading anaconda module (HPC system detected)..."
    module load anaconda3/2024.6
else
    echo "Using system Python (local development system detected)..."
fi

# Function to find and initialize conda
init_conda() {
    # Try to get conda base path dynamically
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    
    # Fallback to common locations
    for conda_path in ~/miniconda3 ~/anaconda3 /usr/anaconda3 /opt/anaconda3; do
        if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
            source "$conda_path/etc/profile.d/conda.sh"
            return 0
        fi
    done
    
    return 1
}

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment: $ENV_NAME"
    
    # Try conda first for better package compatibility (unless --venv is set)
    if [ "$FORCE_VENV" = false ] && init_conda; then
        echo "Using conda for virtual environment creation..."
        # Remove existing venv directory if it exists (cleanup)
        if [ -d "$ENV_NAME" ]; then
            echo "Removing existing $ENV_NAME directory..."
            rm -rf "$ENV_NAME"
        fi
        # Try Python 3.11 first, fallback to 3.10 if it fails
        if conda create -n "$ENV_NAME" python=3.11 -y 2>/dev/null; then
            echo "Created conda environment with Python 3.11"
        else
            echo "Python 3.11 failed, trying Python 3.10..."
            conda create -n "$ENV_NAME" python=3.10 -y
        fi
        conda activate "$ENV_NAME"
    else
        echo "Using python venv..."
        python3 -m venv "$ENV_NAME"
        source "$ENV_NAME/bin/activate"
    fi
else
    echo "Virtual environment $ENV_NAME already exists, using existing one."
    # Try to activate conda env first, fallback to venv
    if [ "$FORCE_VENV" = false ] && init_conda; then
        # Check if conda env exists and activate it
        if conda env list | grep -q "^$ENV_NAME "; then
            echo "Activating conda environment: $ENV_NAME"
            conda activate "$ENV_NAME"
        else
            echo "Conda env not found, using venv: $ENV_NAME"
            source "$ENV_NAME/bin/activate"
        fi
    else
        echo "Conda not found, using venv: $ENV_NAME"
        source "$ENV_NAME/bin/activate"
    fi
fi

# Build dependency installation string
DEPS=""
if [ "$INSTALL_GPU" = true ] && [ "$INSTALL_DEV" = true ]; then
    DEPS="[all]"
    echo "Installing with GPU and development dependencies..."
elif [ "$INSTALL_GPU" = true ]; then
    DEPS="[gpu]"
    echo "Installing with GPU dependencies..."
elif [ "$INSTALL_DEV" = true ]; then
    DEPS="[dev]"
    echo "Installing with development dependencies..."
else
    DEPS="[tensorboard]"
    echo "Installing base dependencies only..."
fi

# Upgrade pip, setuptools, wheel in the virtual environment
# Use python -m pip to avoid permission issues with system Python metadata
echo "Upgrading pip, setuptools, and wheel in virtual environment..."

# Install the package in editable mode (pip will skip already installed conda packages)
pip install -e ".$DEPS"

echo "Setup complete."
echo "To activate the environment later, run: source $ENV_NAME/bin/activate"
