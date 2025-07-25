# PopulationTransformer Integration with podcast-benchmark

This branch adds PopulationTransformer support to the podcast-benchmark repository, enabling brain-to-word decoding using pre-trained PopulationTransformer models.

## 🎯 What's New

### **PopulationTransformer Module**
- **Location:** `population_transformer_module/`
- **Purpose:** Integration layer between PopulationTransformer and podcast-benchmark
- **Components:**
  - `population_transformer_utils.py`: Core integration logic
  - `__init__.py`: Module initialization

### **Configuration Files**
- **Location:** `configs/population_transformer/`
- **Files:**
  - `population_transformer_cpu.yml`: CPU-optimized configuration
  - `population_transformer_base.yml`: Base configuration
  - `population_transformer_frozen.yml`: Frozen PopulationTransformer weights
  - `population_transformer_finetune.yml`: End-to-end fine-tuning

### **PopulationTransformer Submodule**
- **Location:** `population_transformer/`
- **Source:** [PopulationTransformer Repository](https://github.com/czlwang/PopulationTransformer)
- **Purpose:** Pre-trained models and architecture definitions

## 🚀 Quick Start

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv decoding_env
source decoding_env/bin/activate  # Linux/Mac
# or
decoding_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install omegaconf  # Required for PopulationTransformer
```

### **2. Download PopulationTransformer Weights**
```bash
# Create directory
mkdir -p population_transformer/pretrained_weights/popt_brainbert_stft/

# Download from HuggingFace or PopulationTransformer repository
# Place: pretrained_popt_brainbert_stft.pth in the above directory
```

### **3. Download Dataset**
```bash
# Run setup script to download podcast dataset
./setup.sh
```

### **4. Run PopulationTransformer Experiment**
```bash
# CPU-optimized experiment
python main.py --config configs/population_transformer/population_transformer_cpu.yml

# Or use Makefile
make population-transformer-cpu
```

## 📊 What the Integration Does

### **Pipeline Overview**
```
Neural Data (iEEG/ECoG) 
    ↓
PopulationTransformer (frozen feature extractor)
    ↓
Neural Embeddings (512-dim)
    ↓
MLP Decoder (trained)
    ↓
Predicted Word Embeddings (50-dim)
    ↓
Compare with GPT-2 Embeddings (ground truth)
```

### **Key Features**
- **Transfer Learning:** Uses pre-trained PopulationTransformer as feature extractor
- **Cross-Validation:** Tests performance across different time lags
- **CPU Optimization:** Configured for CPU-only environments
- **Modular Design:** Easy to extend and modify

## 🔧 Configuration Options

### **CPU Configuration (`population_transformer_cpu.yml`)**
```yaml
training_params:
  batch_size: 8
  epochs: 10
  n_folds: 2
  early_stopping_patience: 5

data_params:
  preprocessor_params:
    device: cpu
    batch_size: 4
    frozen_weights: true
```

### **Model Parameters**
- **Input Dimension:** 512 (PopulationTransformer output)
- **Output Dimension:** 50 (GPT-2 word embeddings)
- **Hidden Layers:** [256, 128] (MLP decoder)

## 📈 Performance Metrics

### **Evaluation Metrics**
- **Cosine Similarity:** Measures correlation between predicted and actual word embeddings
- **Word-Level Analysis:** Performance per word with minimum repetition threshold
- **Cross-Validation:** Robust performance estimation

### **Expected Results**
- **Range:** -1 to +1 (perfect negative to perfect positive correlation)
- **Good Performance:** > 0.3 cosine similarity
- **Lag Analysis:** Performance varies by time window around word onset

## 🛠️ Technical Details

### **PopulationTransformer Architecture**
- **Model:** `pt_model_custom` (6 layers, 8 heads, 512 hidden dim)
- **Position Encoding:** Multi-subject brain positional encoding
- **Input:** 768-dimensional features (padded from neural data)
- **Output:** 512-dimensional neural embeddings

### **Data Processing**
- **Neural Data:** iEEG/ECoG recordings from podcast listening
- **Word Data:** GPT-2 embeddings for transcript words
- **Alignment:** Temporal alignment of neural activity with word onsets

### **Training Process**
1. **Feature Extraction:** PopulationTransformer processes neural data
2. **Cross-Validation:** 2-fold CV over different time lags
3. **Decoder Training:** MLP learns neural → word embedding mapping
4. **Evaluation:** Cosine similarity between predicted and actual embeddings

## 🔍 Files Added/Modified

### **New Files**
- `population_transformer_module/`
- `configs/population_transformer/`
- `POPULATION_TRANSFORMER_INTEGRATION.md` (this file)

### **Modified Files**
- `main.py`: Added PopulationTransformer module import
- `Makefile`: Added PopulationTransformer targets
- `.gitignore`: Updated to exclude large files and environments
- `.gitmodules`: Added PopulationTransformer submodule

### **Excluded from Repository**
- `data/`: Neural data and embeddings (large files)
- `models/`: Trained model checkpoints
- `results/`: Experiment results
- `decoding_env/`: Virtual environment
- `population_transformer/pretrained_weights/`: Model weights
- `*.ipynb`: Jupyter notebooks

## 🐛 Troubleshooting

### **Common Issues**
1. **Missing PopulationTransformer weights:** Download from HuggingFace
2. **Missing dependencies:** Install `omegaconf` and other requirements
3. **Memory issues:** Reduce batch size in config
4. **Import errors:** Ensure PopulationTransformer submodule is initialized

### **Debug Commands**
```bash
# Test basic integration
python test_cpu_ready.py

# Test imports
python test_population_transformer_imports.py

# Check config loading
python -c "import yaml; yaml.safe_load(open('configs/population_transformer/population_transformer_cpu.yml'))"
```

## 🤝 Contributing

This integration follows the podcast-benchmark modular design:
1. **Model Constructor:** `@registry.register_model_constructor()`
2. **Data Preprocessor:** `@registry.register_data_preprocessor()`
3. **Config Setter:** `@registry.register_config_setter()`
4. **Configuration:** YAML config files
5. **Makefile:** Easy-to-use targets
