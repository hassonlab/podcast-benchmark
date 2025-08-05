#!/bin/bash

set -e

# Step 1: Setup Podcast dataset
# Config
DATASET_ID="ds005574"
REPO_URL="https://github.com/OpenNeuroDatasets/${DATASET_ID}.git"
LOCAL_DATA_DIR="data"
OPENNEURO_BASE_URL="https://s3.amazonaws.com/openneuro.org/ds005574"

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
        echo "Missing file: $file"

        # URL-encode the file path
        url_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$file'))")

        # Download the file from OpenNeuro
        file_url="${OPENNEURO_BASE_URL}/${url_path}"
        echo "Downloading: $file_url"

	# If file is in this directory we need to delete it first.
	if [[ ! "$file" == */* ]]; then
		rm "$file"
	fi

	# Have to delete existi
        # Download using wget
        wget --show-progress "$file_url"
	
	# Only do this if it's in a directory.
	if [[ "$file" == */* ]]; then
		# Create parent directories if needed
        	mkdir -p "$(dirname "$file")"

		# Move the downloaded file to the correct location
		mv "$(basename "$file_url")" "$file"
	fi
    fi
done

cd ..

echo "All missing files downloaded!"

# Step 3: Download GloVe vectors
if [ -f data/glove/glove.6B.50d.txt ]; then
    echo "GloVe vectors already prepared. Skipping download and extraction."
else
    mkdir -p data/glove
    wget https://nlp.stanford.edu/data/glove.6B.zip -P data/glove/

    unzip data/glove/glove.6B.zip -d data/glove/

    rm data/glove/glove.6B.zip

    echo "GloVe vectors downloaded and extracted."
fi

# Step 4: Setup a new virtual environment and install all the necessary packages
# Parse command line arguments
INSTALL_GPU=false
INSTALL_DEV=false
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
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--gpu] [--dev] [--env-name NAME]"
            echo "  --gpu       Install GPU dependencies (CUDA packages)"
            echo "  --dev       Install development dependencies (testing)"
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

module load anaconda3/2024.6
pip install --user virtualenv

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment: $ENV_NAME"
    virtualenv "$ENV_NAME"
else
    echo "Virtual environment $ENV_NAME already exists, using existing one."
fi

source "$ENV_NAME/bin/activate"

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
    echo "Installing base dependencies only..."
fi

# Install the package in editable mode
pip install -e ".$DEPS"

echo "Setup complete."
echo "To activate the environment later, run: source $ENV_NAME/bin/activate"
