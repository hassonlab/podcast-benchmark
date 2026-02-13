import numpy as np 
import copy 

class EarlyStopPruner:
    def __init__(self, params, verbose=True):
        self.verbose = verbose 
        self.best_score = None
        self.early_stop_counter = 0
        self.early_stop_criterion=getattr(params, 'early_stop_criteria', 'val_loss')
        self.patience = params.early_stop_patience
        self.best_epoch = -1
        self.best_model_states = None
        
        if any(s in self.early_stop_criterion.lower() for s in ['loss', 'error', 'rmse', 'mae']):
            self.is_minimize = True
        elif any(s in self.early_stop_criterion.lower() for s in ['accuracy', 'kappa', 'f1', 'auroc', 'cosine_similarity']):
            self.is_minimize = False
        else:
            raise ValueError(f"Unsupported early_stop_criteria: {self.early_stop_criterion}. Supported: ['loss', 'error', 'rmse', 'mae'] for minimize, ['acc', 'kappa', 'f1', 'auroc'] for maximize")
    
        self._early_best = np.inf if self.is_minimize else -np.inf
    
    def __call__(self, epoch, model_state_dict, val_loss, val_metrics):
        if self.early_stop_criterion not in val_metrics:
            raise ValueError(f"Unsupported early_stop_criteria: {self.early_stop_criterion}. Supported: {list(val_metrics.keys())}")
        
        current_score = val_metrics[self.early_stop_criterion]
        print(f"Early Best:{self._early_best}")
        print(f"Current Score:{current_score}")
        improved = (current_score <= self._early_best) if self.is_minimize else (current_score >= self._early_best)
        if improved:
            print(f"Validation {self.early_stop_criterion} {'decreasing' if self.is_minimize else 'increasing'}....saving weights !!")
            self.best_epoch = epoch + 1
            self.best_metrics = val_metrics
            self.best_score = current_score
            self.early_stop_counter = 0
            self._early_best = current_score
            self.best_model_states = copy.deepcopy(model_state_dict)
        else:
            self.early_stop_counter += 1
            if self.verbose:
                print(f"No improvement in {self.early_stop_criterion}. "
                      f"Counter: {self.early_stop_counter}/{self.patience}")
            if self.early_stop_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                return True
            
        return False
