from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

import mne
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

from scipy.spatial.distance import cosine

import data_utils
from config import TrainingParams, DataParams

def train_decoding_model(X: np.array,
                         Y: np.array,
                         selected_words: list[str],
                         model_constructor_fn,
                         lag: int,
                         model_params: dict,
                         training_params: TrainingParams,
                         model_dir,
                         plot_results=False):
    """
    Train decoding model on data using 5-fold cross-validation.
    Uses 3 folds for training, 1 fold for validation (early stopping), and 1 fold for testing.
    Uses cosine similarity as a metric and for early stopping.
    
    Args:
        X: Input data of shape [num_examples, num_electrodes, num_timepoints]
        Y: Target embeddings of shape [num_examples, embedding_dim]
        selected_words: String representation of words of shape [num_examples]
        model_constructor_fn: Function constructor for model. Should have function arguments model_constructor_fn(model_params) -> Model.
        lag: Current lag being trained over
        model_params: Dictionary of model parameters
        training_params: Training parameters
        model_dir: Directory to write models to. 
        plot_results: If true then will plot data relevant to each fold, upon finishing each fold. (default: False)
        
    Returns:
        models: List of trained PitomModels (one per fold)
        histories: List of dictionaries containing training histories
        cv_results: Dictionary with cross-validation metrics
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Convert numpy arrays to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float32)
    
    # Initialize cross-validation results
    models = []
    histories = []
    cv_results = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_cosine': [],
        'val_cosine': [],
        'test_cosine': []
    }
    roc_results = []

    kf = KFold(n_splits=5, shuffle=False)
    fold_indices = list(kf.split(range(X.shape[0])))
    best_epoch = 0

    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        model_path = os.path.join(model_dir, f'best_pitom_model_fold{fold+1}.pt')
        
        # For each fold, we need to:
        # 1. Use 3 folds for training
        # 2. Use 1 fold for validation (early stopping)
        # 3. Use 1 fold for testing
        
        train_idx, val_idx = train_test_split(
            np.array(train_val_idx),
            test_size=0.25,  # Equivalent to 1 fold out of 4
            shuffle=False
        )
        
        # Create data loaders for train, validation, and test sets
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_params.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_params.batch_size, 
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_params.batch_size, 
            shuffle=False
        )
        
        # Initialize model for this fold
        model = model_constructor_fn(model_params).to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_params.learning_rate,
            weight_decay=training_params.weight_decay  # L2 regularization
        )
        
        # Initialize variables for early stopping (now based on cosine similarity)
        best_val_cosine = -float('inf')  # We want to maximize cosine similarity
        patience_counter = 0
        
        # Initialize history dictionary to track metrics for this fold
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_cosine': [],
            'val_cosine': []
        }
        
        # Training loop for this fold
        progress_bar = tqdm(range(training_params.epochs), desc=f"Lag {lag}, Fold {fold + 1}")
        for epoch in progress_bar:
            # Training phase
            model.train()
            train_loss = 0.0
            train_cosine = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                
                # Calculate cosine similarity between predictions and targets
                batch_cosines = calculate_cosine_similarity(outputs.detach().cpu(), targets.detach().cpu())
                train_cosine += sum(batch_cosines)

            train_loss = train_loss / len(train_loader.dataset)
            train_cosine = train_cosine / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_cosine = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Calculate cosine similarity
                    batch_cosines = calculate_cosine_similarity(outputs.cpu(), targets.cpu())
                    val_cosine += sum(batch_cosines)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_cosine = val_cosine / len(val_loader.dataset)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_cosine'].append(train_cosine)
            history['val_cosine'].append(val_cosine)
            
            # Early stopping based on cosine similarity (higher is better)
            if val_cosine > best_val_cosine:
                best_val_cosine = val_cosine
                patience_counter = 0
                best_epoch = epoch
                # Save best model for this fold
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= training_params.early_stopping_patience:
                    break

            # Update progress bar
            progress_bar.set_postfix({'train_loss': train_loss , 'train_cosine': train_cosine, 'val_loss': val_loss, 'val_cosine': val_cosine, 'best_epoch': best_epoch + 1, 'epoch': epoch})
        
        # Load best model for this fold
        model.load_state_dict(torch.load(model_path))
        
        # Test the model on the test set
        model.eval()
        test_loss = 0.0
        test_cosine = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                test_loss += loss.item() * inputs.size(0)
                
                # Calculate cosine similarity
                batch_cosines = calculate_cosine_similarity(outputs.cpu(), targets.cpu())
                test_cosine = sum(batch_cosines)

            roc_result = calculate_word_embeddings_roc_auc_logits(model, X_test, Y_test, selected_words[test_idx], device)
            roc_results.append(roc_result)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_cosine = test_cosine / len(test_loader.dataset)
        
        print(f"\nFold {fold+1} Test Results: Loss = {test_loss:.4f}, Cosine Similarity = {test_cosine:.4f}")
        
        # Store fold results
        cv_results['train_loss'].append(history['train_loss'][-1])  # Last epoch
        cv_results['val_loss'].append(history['val_loss'][history['val_cosine'].index(max(history['val_cosine']))])  # Loss at best cosine
        cv_results['test_loss'].append(test_loss)
        cv_results['train_cosine'].append(history['train_cosine'][-1])  # Last epoch
        cv_results['val_cosine'].append(best_val_cosine)  # Best validation cosine
        cv_results['test_cosine'].append(test_cosine)
        
        # Store model and history for this fold
        models.append(model)
        histories.append(history)
        
        # Plot training history for this fold
        # TODO: Add support for this to write plots to file.
        if plot_results:
            plot_training_history(history, fold=fold+1)
    
    # Calculate and print cross-validation results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Mean Train Loss: {np.mean(cv_results['train_loss']):.4f} ± {np.std(cv_results['train_loss']):.4f}")
    print(f"Mean Val Loss: {np.mean(cv_results['val_loss']):.4f} ± {np.std(cv_results['val_loss']):.4f}")
    print(f"Mean Test Loss: {np.mean(cv_results['test_loss']):.4f} ± {np.std(cv_results['test_loss']):.4f}")
    print(f"Mean Train Cosine: {np.mean(cv_results['train_cosine']):.4f} ± {np.std(cv_results['train_cosine']):.4f}")
    print(f"Mean Val Cosine: {np.mean(cv_results['val_cosine']):.4f} ± {np.std(cv_results['val_cosine']):.4f}")
    print(f"Mean Test Cosine: {np.mean(cv_results['test_cosine']):.4f} ± {np.std(cv_results['test_cosine']):.4f}")
    
    # Plot overall cross-validation results
    # TODO: Add support for this to write plots to file.
    if plot_results:
        plot_cv_results(cv_results)

    final_word_auc = {}
    for roc_result in roc_results:
        final_word_auc.update(roc_result['word_aucs'])
    weighted_roc_mean = summarize_roc_results(final_word_auc, selected_words)
    
    return models, histories, cv_results, roc_results, weighted_roc_mean


def calculate_cosine_similarity(predictions, targets):
    """
    Calculate cosine similarity between predictions and targets.
    
    Args:
        predictions: Tensor of shape [batch_size, embedding_dim]
        targets: Tensor of shape [batch_size, embedding_dim]
        
    Returns:
        cosine_similarities: List of cosine similarities for each example
    """
    cosine_similarities = []
    
    # Convert to numpy if tensors
    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()
    
    # Calculate cosine similarity for each example
    for i in range(predictions.shape[0]):
        # Normalize vectors
        pred_norm = np.linalg.norm(predictions[i])
        target_norm = np.linalg.norm(targets[i])
        
        # Handle zero vectors
        if pred_norm == 0 or target_norm == 0:
            cosine_similarities.append(0.0)
        else:
            # Calculate cosine similarity
            similarity = np.dot(predictions[i], targets[i]) / (pred_norm * target_norm)
            cosine_similarities.append(similarity)
    
    return cosine_similarities


def plot_training_history(history, fold=None):
    """
    Plot the training and validation loss and cosine similarity.
    
    Args:
        history: Dictionary containing training history
        fold: Fold number (optional)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    title = 'Training and Validation Loss'
    if fold is not None:
        title = f'Fold {fold}: {title}'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot cosine similarity
    ax2.plot(history['train_cosine'], label='Training Cosine Similarity')
    ax2.plot(history['val_cosine'], label='Validation Cosine Similarity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Similarity')
    title = 'Training and Validation Cosine Similarity'
    if fold is not None:
        title = f'Fold {fold}: {title}'
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_cv_results(cv_results):
    """
    Plot cross-validation results.
    
    Args:
        cv_results: Dictionary containing cross-validation results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Prepare data
    folds = range(1, len(cv_results['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(folds, cv_results['train_loss'], 'o-', label='Training Loss')
    ax1.plot(folds, cv_results['val_loss'], 'o-', label='Validation Loss')
    ax1.plot(folds, cv_results['test_loss'], 'o-', label='Test Loss')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Cross-Validation Loss')
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True)
    
    # Plot cosine similarity
    ax2.plot(folds, cv_results['train_cosine'], 'o-', label='Training Cosine Similarity')
    ax2.plot(folds, cv_results['val_cosine'], 'o-', label='Validation Cosine Similarity')
    ax2.plot(folds, cv_results['test_cosine'], 'o-', label='Test Cosine Similarity')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cross-Validation Cosine Similarity')
    ax2.set_xticks(folds)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def calculate_word_embeddings_roc_auc_logits(model, X, Y, selected_words, device, min_repetitions=5):
    """
    Calculate ROC-AUC for word embedding predictions using logits approach.
    
    This follows the described method more closely, converting distances to logits
    via softmax transformation.
    """
    # Step 1: Get word frequency counts and filter for words with minimum repetitions
    word_counts = Counter(selected_words)
    frequent_words = [word for word, count in word_counts.items() if count >= min_repetitions]
    print(f"Found {len(frequent_words)} words with at least {min_repetitions} repetitions ({len(frequent_words)/len(set(selected_words))*100:.1f}% of unique words)")
    
    X = X.to(device)
    
    # Step 2: Get predicted embeddings for all neural data
    model_predictions = []
    for i in range(len(X)):
        with torch.no_grad():
            input_data = X[i:i+1]
            pred = model(input_data).cpu().numpy()
        
        model_predictions.append(pred.squeeze())
    
    predicted_embeddings = np.array(model_predictions)
    
    # Step 3: Group all embeddings for each unique word
    # Y should be on cpu for comparisons.
    Y = Y.cpu()
    unique_words = list(set(selected_words))
    word_to_embeddings = {word: np.array(Y[np.array(selected_words) == word]) for word in unique_words}
    
    # Step 4: Calculate average embeddings for each unique word
    avg_word_embeddings = {word: np.mean(embs, axis=0) for word, embs in word_to_embeddings.items()}
    
    # Step 5: Calculate cosine distances and convert to logits
    word_aucs = {}
    word_to_idx = {}
    
    for idx, pred_embedding in enumerate(predicted_embeddings):
        # Calculate distances to all unique words
        distances = []
        for word in unique_words:
            avg_embedding = avg_word_embeddings[word]
            distance = cosine(pred_embedding, avg_embedding)
            # Convert distance to similarity
            similarity = 1 - distance
            distances.append(similarity)
        
        # Convert similarities to logits using softmax
        logits = torch.tensor(distances)
        logits = F.softmax(logits, dim=0).numpy()
        
        # For each instance, collect logits for the correct label and all other labels
        true_word = selected_words[idx]
        true_word_idx = unique_words.index(true_word)
        
        # Update the logits for each word
        if true_word not in word_to_idx:
            word_to_idx[true_word] = {'logits': [], 'is_true': []}
        
        for word_idx, word in enumerate(unique_words):
            if word not in word_to_idx:
                word_to_idx[word] = {'logits': [], 'is_true': []}
            
            word_to_idx[word]['logits'].append(logits[word_idx])
            word_to_idx[word]['is_true'].append(1 if word == true_word else 0)
    
    # Step 6: Calculate ROC-AUC for each frequent word
    for word in frequent_words:
        try:
            roc_auc = roc_auc_score(
                np.array(word_to_idx[word]['is_true']),
                np.array(word_to_idx[word]['logits'])
            )
            word_aucs[word] = roc_auc
        except ValueError:
            print(f"Skipping ROC-AUC calculation for '{word}' - insufficient class variety")
    
    # Step 7: Calculate weighted ROC-AUC based on word frequency
    total_count = sum(word_counts[word] for word in frequent_words if word in word_aucs)
    weighted_auc = sum(word_aucs[word] * word_counts[word] for word in word_aucs) / total_count
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(word_aucs.keys(), word_aucs.values())
    plt.axhline(y=weighted_auc, color='r', linestyle='-', label=f'Weighted Average: {weighted_auc:.3f}')
    plt.xticks(rotation=90)
    plt.ylabel('ROC-AUC')
    plt.xlabel('Words')
    plt.title('ROC-AUC for Word Predictions (Logits Approach)')
    plt.legend()
    plt.tight_layout()
    
    return {
        'word_aucs': word_aucs,
        'weighted_auc': weighted_auc,
        'frequent_words': frequent_words
    }


def summarize_roc_results(word_aucs, selected_words, min_repetitions=5):
    word_counts = Counter(selected_words)
    frequent_words = [word for word, count in word_counts.items() if count >= min_repetitions]

    total_count = sum(word_counts[word] for word in frequent_words if word in word_aucs)
    weighted_auc = sum(word_aucs[word] * word_counts[word] for word in word_aucs) / total_count

    return weighted_auc


def run_training_over_lags(lags,
                           raws: list[mne.io.Raw],
                           df_word: pd.DataFrame,
                           word_embeddings: np.array,
                           preprocessing_fn,
                           model_constructor_fn,
                           model_params: dict,
                           training_params: TrainingParams,
                           data_params: DataParams,
                           trial_name,
                           output_dir="results/",
                           model_dir="models/"):
    """
    Args:
        lags: array of lags to run training over
        raws: list of mne.Raw objects for each subject,
        df_word: dataframe containing columns word, start, and end corresponding to words in the transcript,
        word_embeddings: np.array of embeddings of each word in df_word,
        preprocessing_fn: function to preprocess data for each lag.
            Should have contract 
            preprocessing_fn(data: np.array of shape [num_words, num_electrodes, timesteps],
                             params: dictionary in data_params['preprocessor_params']) -> array of shape [num_words, ...]
        model_constructor_fn: Function constructor for model. Should have function arguments model_constructor_fn(model_params) -> Model.
        model_params: Dictionary of model parameters
        training_params: Training parameters
        data_params: Parameters for get_data function call
        trial_name: Name of trial to be used for file writing
        output_dir: Name of folder to write results to.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    filename = os.path.join(output_dir, f"lag_performance.csv")
    if os.path.exists(filename):
        roc_df = pd.read_csv(filename)
        weighted_roc_means = roc_df.rocs.tolist()
        already_read_lags = roc_df.lags.tolist()
    else:
        weighted_roc_means = []
        already_read_lags = []

    for lag in lags:
        if lag in already_read_lags:
            print(f'Lag {lag} already done, skipping...')
            continue
        # Maybe make it so only words in all lags are included.
        print('=' * 60)
        print('running lag:', lag)
        print('=' * 60)
        X, Y, selected_words = data_utils.get_data(lag,
                                                   raws, 
                                                   df_word, 
                                                   word_embeddings,
                                                   data_params.window_width,
                                                   preprocessing_fn,
                                                   data_params.preprocessor_params)
    
        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y)
        
        print(f"X_tensor shape: {X_tensor.shape}, Y_tensor shape: {Y_tensor.shape}")
    
        models, histories, cv_results, roc_results, weighted_roc_mean = train_decoding_model(
            X_tensor, Y_tensor, selected_words, model_constructor_fn, lag,
            model_params=model_params,
            training_params=training_params,
            model_dir=os.path.join(model_dir, f"lag_{lag}")
        )
        weighted_roc_means.append(weighted_roc_mean)
    
        # Write file
        pd.DataFrame({'lags': lags[:len(weighted_roc_means)],
                      'rocs': np.array(weighted_roc_means)}).to_csv(filename, index=False)

    return weighted_roc_means