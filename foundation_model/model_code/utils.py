from foundation_model.model_code.config import VideoMAEExperimentConfig
from foundation_model.model_code.models_mae import MaskedAutoencoderViT

def create_model(config: VideoMAEExperimentConfig):
    model_config = config.video_mae_task_config.vit_config
    num_frames = int(
        config.ecog_data_config.sample_length * config.ecog_data_config.new_fs
    )
    model = MaskedAutoencoderViT(
        img_size=[8, 8],
        patch_size=model_config.patch_size,
        in_chans=len(config.ecog_data_config.bands),
        embed_dim=model_config.dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        decoder_embed_dim=model_config.decoder_embed_dim,
        decoder_depth=model_config.decoder_depth,
        decoder_num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.mlp_ratio,
        num_frames=num_frames,
        t_patch_size=model_config.frame_patch_size,
        no_qkv_bias=model_config.no_qkv_bias,
        sep_pos_embed=model_config.sep_pos_embed,
        trunc_init=model_config.trunc_init,
        cls_embed=model_config.use_cls_token,
        # TODO: Make this configurable.
        pred_t_dim=num_frames,
        img_mask=None,
        pct_masks_to_decode=config.video_mae_task_config.pct_masks_to_decode,
    )
    return model