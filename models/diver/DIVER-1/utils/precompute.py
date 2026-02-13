import torch
from utils.precompute_features import PrecomputedFeatureManager, build_precomputed_feature_dataloaders

def bring_feature_dataloaders(params, data_loader, model, device):
    cache_root = params.model_dir
    manager = PrecomputedFeatureManager(cache_root=cache_root)

    split_sizes = {}
    for split_name, loader in data_loader.items():
        try:
            split_sizes[split_name] = len(loader.dataset)
        except Exception:
            split_sizes[split_name] = None

    cache_id = manager.make_cache_id(
        backbone_config_name=params.backbone_config,
        ft_config_name=params.ft_config,
        feature_extraction_type=params.feature_extraction_type,
        foundation_dir=params.foundation_dir,
        patch_sampling_rate=params.patch_sampling_rate,
        splits=split_sizes,
    )

    arrays_by_split = manager.get_or_create(
        model=model,
        data_loaders=data_loader,
        device=device,
        cache_id=cache_id,
        feature_key="encoder_output",
        fallback_token_key=("token_manager_output", "org_x_position_features"),
        use_amp=params.use_amp,
        amp_dtype=torch.bfloat16
    ) 

    feature_loaders = build_precomputed_feature_dataloaders(
        arrays_by_split,
        batch_size=params.batch_size,
        num_workers=getattr(params, "num_workers", 4),
        pin_memory=True,
    )
    return feature_loaders