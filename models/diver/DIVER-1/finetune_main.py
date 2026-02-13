import argparse
import random
import os
import numpy as np
import torch

from datasets.datasets_loaders import get_loader
from finetune_trainer import Trainer
from models.model_builders import CustomIdentity
from utils.precompute import bring_feature_dataloaders


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_parser_args(parser):
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD, or multi_lr_AdamW, multi_lr_AdamW_2)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetich, TUEV, TUAB, BCIC-IV-2a, Neuroprobe, MOABB, Podcast]')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--patch_sampling_rate', type=int, default=500, help='patch_sampling_rate (default: 500Hz)')
    parser.add_argument('--datasets_dir', type=str,default='/data/datasets/BigDownstream/Faced/processed', help='datasets_dir')
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir')
    parser.add_argument('--foundation_dir', type=str, default=None, help='foundation_dir')
    parser.add_argument('--width', type=int, default=None, help='Override d_model width')
    parser.add_argument('--depth', type=int, default=None, help='Override e_layers depth')
    parser.add_argument('--patch_size', type=int, default=500, help='patch_size, default 500 (1 second at 500Hz)')
    parser.add_argument('--ft_config', type=str, default='flatten_linear', help='flatten_linear or flatten_mlp)')
    parser.add_argument('--early_stop_criteria', type=str, default='val_f1', help='early stopping criteria (default: val_f1)')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='early stopping patience (default 50)')
    parser.add_argument('--mup_weights', type=str2bool, default=False, help='Whether the pretraining (and hence its weight) was done with mup')
    parser.add_argument('--ft_mup', type=str2bool, default=False, help='use MUP for fine-tuning model')
    parser.add_argument('--use_amp', type=str2bool, default=True, help='whether to use automatic mixed precision (AMP). By default True ')
    parser.add_argument('--deepspeed_pth_format', type=str2bool, default = True, help='whether the pth is saved using the deepspeed method or not')    
    parser.add_argument('--frozen', type=str2bool, default=False, help='frozen')
    parser.add_argument('--precompute_features', type=str2bool, default=False, help='If True and frozen, cache adapter-stage features and train ft head on arrays')
    parser.add_argument('--precompute_only', type=str2bool, default=False, help='If True, running is ended when precomputing is finished')
    parser.add_argument('--precompute_batch_size', type=int, default=64, help='Batch size in precompute phase')
    parser.add_argument('--optimize_criteria', type=str, default='val_f1', help='optimization criteria (default: val_f1), could use val_auroc, val_acc, val_kappa,  or sth')
    parser.add_argument('--num_mlp_layers', type=int, default=None, help='If using flatten mlp, set num of mlp layers')
    
    parser.add_argument('--neuroprobe_task', type=str, default='', help='[frame_brightness, global_flow, local_flow, global_flow_angle,local_flow_angle ,face_num  ,volume  ,pitch ,delta_volume ,delta_pitch ,speech,onset,gpt2_surprisal,word_length,word_gap,word_index,word_head_pos,word_part_speech,speaker')
    parser.add_argument('--neuroprobe_evaltype', type=str, default='sssm', help='[sssm, ssdm, dsdm]') 
    parser.add_argument('--neuroprobe_subject', type=int, default=1, help='neuroprobe subject no') 
    parser.add_argument('--neuroprobe_trial', type=int, default=1, help='neuroprobe trial no') 
    parser.add_argument('--neuroprobe_lite', type=str2bool, default=True, help='neuroprobe lite or full (full when false)') 
    parser.add_argument('--neuroprobe_fold', type=int, default=0, help='neuroprobe fold idx (5fold, start from 0 to 4)') 
    parser.add_argument('--neuroprobe_preprocess', type=str, default='downsample_500', help='neuroprobe preprocess type') 
    parser.add_argument('--neuroprobe_model_type', type=str, default='DIVER', help='neuroprobe model type (DIVER, BrainBERT, PopT)')
    parser.add_argument('--neuroprobe_allow_corrupted', type=str2bool, default=False, help='whether to allow corrupted data in neuroprobe dataset') 
    parser.add_argument('--neuroprobe_clip', type=int, default=200, help='neuroprobe data clip range (pretraining was done in 200uv)')
    parser.add_argument('--neuroprobe_patchsize', type=int, default=500, help="neuroprobe patchsize")
    parser.add_argument('--moabb_task', type=str, default='', help='set benchmark in moabb')
    parser.add_argument('--paradigm', type=str, default='motor_imagery', choices=['motor_imagery', 'leftright_imagery'] , help='set prediction paradigm for moabb')
    parser.add_argument('--num_channels', type=int, default=22, help='set num of channels for moabb sub dataset')
    parser.add_argument('--resample_rate', type=int, default=500, help='set resample rate for moabb dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='set train test split ratio')
    
    parser.add_argument('--podcast_embedding_layer', type=int, default=24, help='embedding layer where GPT embedding')
    parser.add_argument('--podcast_embedding_type', type=str, default='gpt-2xl', choices=['gpt-2xl', 'arbitrary'], help='which model to get embedding')
    parser.add_argument('--podcast_embedding_pca_dim', type=int, default=50, help='embedding dimension reduction')
    parser.add_argument('--podcast_subject_ids', type=list, default=[9], help='podcast subject id to finetune')
    parser.add_argument('--podcast_add_highpass', type=str2bool, default=False, help='add highpass filter to iEEG data')
    parser.add_argument('--podcast_lag', type=int, default=0, help='time lag for language and iEEG')
    parser.add_argument('--podcast_window_width', type=float, default=0.5, help='window width from language')
    parser.add_argument('--podcast_word_column', type=str, default='lemmatized_word', choices=['lemmatized_word', 'norm_word'], help='word type in word dictionary')
    parser.add_argument('--podcast_foldidx', type=int, default=0, help='fold idx to use')
    parser.add_argument('--podcast_fold_num', type=int, default=5, help='fold num to use')
    parser.add_argument('--podcast_loss_fn', type=str, default='mse', help='set loss function for podcast dataset')
    print("WARNING : PODCAST BASIC SETTING SHOULD BE SAVED")
    
    return parser

def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser = get_parser_args(parser)
    params = parser.parse_args()

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")
    
    dataset_name = params.downstream_dataset 
    data_loader = get_loader(dataset_name, params)

    from datasets.task_info import task_dict
    downstream_task_info = task_dict(params)

    base_shape_save_dir = params.model_dir 
    if not os.path.exists(base_shape_save_dir):
        os.makedirs(base_shape_save_dir, exist_ok=True)

    from models.finetune_model import flatten_linear_finetune, flatten_mlp_finetune
    if params.ft_config == "flatten_linear":
        model = flatten_linear_finetune(params, downstream_task_info)
    elif params.ft_config == "flatten_mlp":
        model = flatten_mlp_finetune(params, downstream_task_info)

    model.to(device)

    if not params.frozen or (params.frozen and not params.precompute_features):
        trainer = Trainer(params, data_loader, model, downstream_task_info=downstream_task_info, clip_value=params.clip_value)
        best_validation_metric = trainer.train()
        return best_validation_metric[params.optimize_criteria]

    feature_loader = bring_feature_dataloaders(params, data_loader, model, device)
    model.backbone = CustomIdentity().to(device)
    model.feature_extraction_func = CustomIdentity().to(device) 
    trainer = Trainer(params, feature_loader, model, downstream_task_info=downstream_task_info, clip_value=params.clip_value)
    best_validation_metric = trainer.train()
    return best_validation_metric[params.optimize_criteria]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()