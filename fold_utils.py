# Code to gather folds. For now only two options (sequential vs word based) so no need for a registry.

import numpy as np

from sklearn.model_selection import KFold, train_test_split


# TODO: Test these!
def get_sequential_folds(X, num_folds=5):
    """Get folds separated by time with no preference for word selection."""
    kf = KFold(n_splits=num_folds, shuffle=False)
    fold_indices = list(kf.split(range(X.shape[0])))
    final_folds = []
    for train_val_idx, test_idx in fold_indices:
        train_idx, val_idx = train_test_split(
            np.array(train_val_idx),
            test_size=0.25,
            shuffle=False,
        )
        final_folds.append((train_idx, val_idx, test_idx))

    return final_folds


def get_zero_shot_folds(selected_words, num_folds=5):
    # Build folds by ensuring that all occurences of a particular word fall within one fold.
    def _word_folds_to_example_folds(word_fold_idx, word_to_idx, unique_words):
        example_fold_idx = []
        for i in word_fold_idx:
            example_fold_idx.extend(word_to_idx[unique_words[i]])
        return example_fold_idx

    unique_words = list(set(selected_words))
    word_to_idx = {}
    for i, word in enumerate(selected_words):
        if word not in word_to_idx:
            word_to_idx[word] = []
        word_to_idx[word].append(i)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_indices = list(kf.split(range(len(unique_words))))

    final_folds = []
    for fold, (train_val_word_idx, test_word_idx) in enumerate(fold_indices):
        train_val_word_indices = np.array(train_val_word_idx)
        train_word_idx, val_word_idx = train_test_split(
            train_val_word_indices,
            test_size=0.25,  # Equivalent to 1 fold out of 4
            random_state=42 + fold,  # Different seed for each fold
        )

        train_idx = _word_folds_to_example_folds(
            train_word_idx, word_to_idx, unique_words
        )
        val_idx = _word_folds_to_example_folds(val_word_idx, word_to_idx, unique_words)
        test_idx = _word_folds_to_example_folds(
            test_word_idx, word_to_idx, unique_words
        )
        final_folds.append((train_idx, val_idx, test_idx))

    return final_folds
