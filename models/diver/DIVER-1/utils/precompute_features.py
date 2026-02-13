import torch
from torch import nn
from typing import Dict, Optional
import os
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from datasets.dataloader_utils import collate_fn_for_data_info_finetuning

Tensor = torch.Tensor

def _hash_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    return batch

def _detach_to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _detach_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_detach_to_cpu(v) for v in x)
    return x

def _get_labels(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for k in ("y", "label", "labels", "target", "targets"):
            if k in batch:
                return batch[k]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[1]
    raise KeyError("Could not find labels in batch (expected keys y/label/target or tuple index 1).")

class PrecomputedFeatureManager:
    def __init__(self, cache_root: str):
        self.cache_root = cache_root

    def make_cache_id(
        self,
        backbone_config_name: Optional[str],
        ft_config_name: Optional[str],
        feature_extraction_type: str,
        foundation_dir: Optional[str],
        patch_sampling_rate: Optional[int],
        splits: Dict[str, Optional[int]],
    ) -> str:
        payload = f"{backbone_config_name}|{ft_config_name}|{feature_extraction_type}|{patch_sampling_rate}|{foundation_dir}|{splits}"
        return _hash_str(payload)

    def _cache_dir(self, cache_id: str) -> str:
        return os.path.join(self.cache_root, "features", cache_id)

    def _split_file(self, cache_id: str, split: str) -> str:
        return os.path.join(self._cache_dir(cache_id), f"{split}.pt")

    def has_cache(self, cache_id: str, required_splits: Iterable[str]) -> bool:
        cache_dir = self._cache_dir(cache_id)
        if not os.path.isdir(cache_dir):
            return False
        return all(os.path.isfile(self._split_file(cache_id, s)) for s in required_splits)
    
    @torch.no_grad()
    def compute_and_save_with_cache_id(
        self,
        model: torch.nn.Module,
        data_loaders: Dict[str, DataLoader],
        device: torch.device,
        cache_id: str,
        feature_key: str = "encoder_output",
        fallback_token_key: Tuple[str, str] = ("token_manager_output", "org_x_position_features"),
        input_norm_factor=100,
        use_amp=False, 
        amp_dtype=None
    ) -> None:
        model_was_training = model.training
        model.eval()
        os.makedirs(self._cache_dir(cache_id), exist_ok=True)

        print("PRECOMPUTING IS RUNNING")

        for split, loader in data_loaders.items():
            X_list: List[Tensor] = []
            y_list: List[Tensor] = []
            info_list: List[Any] = []

            for batch in loader:
                batch_dev = _to_device(batch, device)
                x, y, data_info_list = batch_dev

                x = x / input_norm_factor

                with torch.no_grad(), torch.autocast(device_type = 'cuda', dtype=amp_dtype, enabled=use_amp):
                    backbone_output = model.backbone(
                        x,
                        data_info_list=data_info_list,
                        use_mask=False,
                        return_encoder_output=True,
                    )
                    model_out = model.feature_extraction_func(backbone_output, data_info_list=data_info_list)

                if isinstance(model_out, dict) and feature_key in model_out:
                    feat = model_out[feature_key]
                elif (
                    isinstance(model_out, dict)
                    and fallback_token_key[0] in model_out
                    and isinstance(model_out[fallback_token_key[0]], dict)
                    and fallback_token_key[1] in model_out[fallback_token_key[0]]
                ):
                    feat = model_out[fallback_token_key[0]][fallback_token_key[1]]
                elif isinstance(model_out, Tensor):
                    feat = model_out
                else:
                    raise KeyError(
                        f"Could not locate precompute feature in model output. "
                        f"Tried '{feature_key}' and '{fallback_token_key[0]}:{fallback_token_key[1]}'"
                    )

                y = _get_labels(batch_dev)
                info = data_info_list

                X_list.append(_detach_to_cpu(feat))
                y_list.append(_detach_to_cpu(y))
                if info is not None:
                    if isinstance(info, (list, tuple)):
                        info_list.extend(info)

            X = torch.cat(X_list, dim=0) if len(X_list) > 1 else X_list[0]
            y = torch.cat(y_list, dim=0) if len(y_list) > 1 else y_list[0]

            obj = {"X": X, "y": y}
            if len(info_list) > 0:
                obj["info"] = info_list 
            torch.save(obj, self._split_file(cache_id, split))

        if model_was_training:
            model.train()

    def load_arrays(self, cache_id: str, split: str) -> Tuple[Tensor, Tensor, Optional[List[Any]]]:
        obj = torch.load(self._split_file(cache_id, split), map_location="cpu", weights_only=False)
        return obj["X"], obj["y"], obj.get("info", None)

    def get_or_create(
        self,
        model: torch.nn.Module,
        data_loaders: Dict[str, DataLoader],
        device: torch.device,
        cache_id: str,
        feature_key: str = "encoder_output",
        fallback_token_key: Tuple[str, str] = ("token_manager_output", "org_x_position_features"),
        use_amp=False,
        amp_dtype=None
    ):
        splits = list(data_loaders.keys()) 
        
        if not self.has_cache(cache_id, splits):
            self.compute_and_save_with_cache_id(
                model=model,
                data_loaders=data_loaders,
                device=device,
                cache_id=cache_id,
                feature_key=feature_key,
                fallback_token_key=fallback_token_key,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
        return {s: self.load_arrays(cache_id, s) for s in splits} 

class _PrecomputedDictDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor, info: Optional[List[Any]] = None):
        assert X.shape[0] == y.shape[0], "X and y must have same N"
        self.X, self.y, self.info = X, y, info

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.info is not None:
            return x, y, self.info[idx]
        else:
            return x, y

def build_precomputed_feature_dataloaders(
    arrays_by_split: Dict[str, Tuple[Tensor, Tensor, Optional[List[Any]]]],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    dls: Dict[str, DataLoader] = {}
    for split, (X, y, info) in arrays_by_split.items():
        ds = _PrecomputedDictDataset(X, y, info)
        shuffle = split == "train"
        dls[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_for_data_info_finetuning,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return dls
