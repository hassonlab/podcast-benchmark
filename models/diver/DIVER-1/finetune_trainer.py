import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm
from timeit import default_timer as timer
import numpy as np
import os
from utils.earlystop_pruner import EarlyStopPruner
from finetune_evaluator import Evaluator

                      
class Trainer(object):
    def __init__(self, params, data_loader, model, downstream_task_info=None, clip_value=1):

        self.params = params
        self.data_loader = data_loader
        self.model = model.cuda()

        self.downstream_task_info = downstream_task_info
        self.clip_value = clip_value

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])
        
        self.early_stopper = EarlyStopPruner(params)
        
        self.best_model_states = None

        if params.frozen:
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=self.params.lr*1e-2
            )
        
        if params.downstream_dataset in ["Neuroprobe", "MentalArithmetich", "Mumtaz2016", "SHU-MI", "CHB-MIT", "TUAB"]:
            self.task_type = "classification_binary"
            self.criterion  = BCEWithLogitsLoss().cuda() 
        elif params.downstream_dataset in ["FACED", "PhysioNet-MI", "BCIC2020-3", "ISRUC", "SEED-V", "TUEV"]:
            self.task_type = "classification_multi"
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()

        elif params.downstream_dataset in ["Podcast", "SEED-VIG"]:
            self.task_type = "regression"
            if params.podcast_loss_fn == 'nll_contextual':
                from utils.podcast_metrics import compute_nll_contextual
                self.criterion = compute_nll_contextual
            else:
                self.criterion = MSELoss().cuda()

    def train(self, input_norm_factor=100):
        for epoch in range(self.params.epochs):
            self.model.train()

            if self.params.frozen: self.model.backbone.eval() 
            
            start_time = timer()
            losses = []
            amp_dtype = torch.bfloat16 
            
            for x, y, data_info_list in tqdm(self.data_loader['train'], mininterval=10): 
                self.optimizer.zero_grad()
                
                if self.params.precompute_features:
                    x = x.cuda()
                else:
                    x = x.cuda() / input_norm_factor
                y = y.cuda()

                with torch.autocast(device_type = 'cuda', dtype=amp_dtype, enabled=self.params.use_amp): 
                    pred = self.model(x, data_info_list)
                    if isinstance(self.criterion, BCEWithLogitsLoss):
                        y = F.one_hot(y, num_classes=2).float()

                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            self.model.eval()

            with torch.no_grad(), torch.autocast(device_type = 'cuda', dtype=amp_dtype, enabled=self.params.use_amp):
                val_metrics = self.val_eval.evaluate(self.model, task_type=self.task_type, criterion=self.criterion, mode="val", use_amp=self.params.use_amp, amp_dtype=amp_dtype)
                val_loss = val_metrics.get('val_loss', float('nan'))

                self._print_metrics(epoch, losses, val_loss, start_time, val_metrics)
            
                base_model = self._unwrap(self.model) 
                should_stop = self.early_stopper(
                    epoch=epoch, 
                    model_state_dict=base_model.state_dict(), 
                    val_loss=val_loss,
                    val_metrics=val_metrics, 
                )
            
                if should_stop:
                    print("Early stopping triggered.")
                    break

        base_model = self._unwrap(self.model)
        base_model.load_state_dict(self.early_stopper.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            test_metrics = self.test_eval.evaluate(self.model, task_type=self.task_type, criterion=self.criterion, mode="test", use_amp=self.params.use_amp, amp_dtype=amp_dtype)
            print("***************************Test results************************")
            self._print_metrics(epoch, losses, val_loss, start_time, test_metrics)
            model_path = os.path.join(self.params.model_dir, "best_state.pth")
            torch.save(base_model.state_dict(), model_path)
        final_results = {f"{k}": v for k, v in self.early_stopper.best_metrics.items()}
        final_results.update({f"{k}": v for k, v in test_metrics.items()})
        final_results['best_epoch'] = self.early_stopper.best_epoch
        return final_results 
    
    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _print_metrics(self, epoch, losses, val_loss, start_time, metrics):
        metrics_log_parts = []
        for k, v in metrics.items():
            if k in ['val_loss', 'cm', 'test_loss', 'test_cm', 'val_cm']:
                continue
            
            if isinstance(v, (list, np.ndarray)):
                if "corrcoef" in k and isinstance(v, np.ndarray) and v.size > 1:
                    val = v[1] if v.size == 2 else np.mean(v) 
                else:
                    val = np.mean(v)
                metrics_log_parts.append(f"{k} (avg): {val:.5f}")
            else:
                metrics_log_parts.append(f"{k}: {v:.5f}")

        metrics_log_str = ", ".join(metrics_log_parts)

        
        print(
            f"Epoch {epoch + 1}: Train Loss: {np.mean(losses):.5f}, Val loss: {val_loss:.5f}, "
            f"{metrics_log_str}, LR: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}, "
            f"Time: {(timer() - start_time) / 60:.2f} mins"
        )

        print_cm = next((metrics[k] for k in ["cm", "test_cm", "val_cm"] if k in metrics), None)
        if print_cm is not None:
            print("Confusion Matrix:\n", print_cm)