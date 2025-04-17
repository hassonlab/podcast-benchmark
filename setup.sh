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
module load anaconda3/2024.6
pip install --user virtualenv

mkdir decoding_env
virtualenv decoding_env
source decoding_env/bin/activate

pip install -r requirements.txt

echo "Setup complete."
