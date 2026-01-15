import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error, precision_score, recall_score, mean_absolute_error
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from utils.podcast_metrics import cosine_similarity, cosine_distance, compute_nll_contextual, similarity_entropy

class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
    
    def _process_predictions(self, task_type, pred_logits, y_for_metrics):
            scores = []
            if "classification" in task_type:
                pred_y = torch.max(pred_logits, dim=-1)[1]
                if task_type == "classification_binary":
                    score_y = torch.sigmoid(pred_logits[:, 1]).to(dtype=torch.float32)
                    scores += score_y.cpu().numpy().tolist()
                else: 
                    prob = torch.softmax(pred_logits, dim=1).to(dtype=torch.float32)
                    scores += prob.detach().cpu().numpy().tolist()
            elif task_type == "regression":
                pred_y = pred_logits
            else:
                pred_y = None
            truths = np.atleast_1d(y_for_metrics.detach().cpu().numpy()).tolist()  
            preds = np.atleast_1d(pred_y.detach().cpu().numpy()).tolist()
            return truths, preds, scores
        
    def _compute_metrics(self, task_type, truths, preds, scores, losses, mode):
        metrics = {}
        if "classification" in task_type:
            metrics['acc'] = balanced_accuracy_score(truths, preds)
            metrics['f1'] = f1_score(truths, preds, average='weighted', zero_division=0)
            metrics['kappa'] = cohen_kappa_score(truths, preds)
            metrics['precision'] = precision_score(truths, preds, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(truths, preds, average='weighted', zero_division=0)
            metrics['cm'] = confusion_matrix(truths, preds)
        if task_type == "classification_binary" and scores is not None and len(scores) > 0:
            if len(np.unique(truths)) > 1:
                metrics['auroc'] = roc_auc_score(truths, scores)
                precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
                metrics['pr_auc'] = auc(recall, precision)
            else:
                metrics['auroc'] = np.nan
        elif task_type == "regression":
            metrics['corrcoef'] = np.corrcoef(truths, preds)[0]
            metrics['r2'] = r2_score(truths, preds)
            metrics['rmse'] = mean_squared_error(truths, preds) ** 0.5
            metrics['mae'] = mean_absolute_error(truths, preds)
            if self.params.dataset_name == 'Podcast':
                metrics['cosine_similarity'] = cosine_similarity(torch.Tensor(preds), torch.Tensor(truths)).item()
                metrics['cosine_distance'] = cosine_distance(torch.Tensor(preds), torch.Tensor(truths)).item()
                metrics['nll_contextual'] = compute_nll_contextual(preds, truths).item()
                metrics['similarity_entropy'] = similarity_entropy(torch.Tensor(preds), torch.Tensor(truths)).item()
        
        if losses:
            metrics['loss'] = np.mean(losses)
            
        prefix = f"{mode}_"
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return metrics
    
    def evaluate(self, model, task_type, criterion=None, mode="val", use_amp=False, amp_dtype=None):
        model.eval()
        
        truths = []
        preds = []
        scores = []
        losses = []
        
        with torch.no_grad(), torch.autocast(device_type = 'cuda', dtype=amp_dtype, enabled=use_amp):
            for x, y, data_info_list in tqdm(self.data_loader, mininterval=1):
                x = x.cuda()
                y = y.cuda()

                pred_logits = model(x, data_info_list=data_info_list)
                
                if criterion is not None:
                    y_original = y
                    if isinstance(criterion, BCEWithLogitsLoss): 
                        y = F.one_hot(y, num_classes=2).float()
                        
                    
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = criterion(pred_logits.transpose(1, 2), y)
                    else:
                        loss = criterion(pred_logits, y)
                    losses.append(loss.item())
                    y_for_metrics = y_original
                else:
                    y_for_metrics = y

                b_truths, b_preds, b_scores =  self._process_predictions(task_type, pred_logits.to(torch.float32), y_for_metrics.to(torch.float32))

                truths.extend(b_truths)
                preds.extend(b_preds)
                if b_scores:
                    scores.extend(b_scores)
                
        truths = np.asarray(truths)
        preds = np.asarray(preds)
        scores = np.asarray(scores) if len(scores) > 0 else None        

        metrics = self._compute_metrics(task_type, truths, preds, scores, losses, mode)
        
        return metrics