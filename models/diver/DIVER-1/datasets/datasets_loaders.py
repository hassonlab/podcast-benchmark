import lmdb
import pickle
from scipy import signal
import torch
import random
import json
import itertools
import pandas as pd
import re 
import os
import sys
import numpy as np
from typing import Any, Callable, Tuple, Union
from datasets.dataloader_utils import collate_fn_for_data_info_finetuning
from utils.util import to_tensor

# insert your neuroprobe module path
# module_path = "/your/path/to/neuroprobe"
# if module_path not in sys.path:
#     sys.path.append(module_path)
# import neuroprobe


from moabb.datasets import BNCI2014_001
from moabb.datasets import PhysionetMI
from moabb.datasets import Stieger2021
from moabb.datasets import Cho2017
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery, MotorImagery
from moabb.datasets.base import CacheConfig
from moabb.datasets.bnci import load_data
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.datasets.base import BaseDataset as _MOABBDataset
from moabb.paradigms.base import BaseParadigm as _MOABBParadigm

from torcheeg.datasets.constants import STANDARD_1005_CHANNEL_LOCATION_DICT
from torcheeg import transforms
from torcheeg.transforms.base_transform import EEGTransform
import torcheeg.datasets.moabb as moabb_dataset
from torcheeg.datasets import DEAPDataset as deap_dataset
from torcheeg.model_selection import train_test_split_cross_subject, train_test_split_cross_trial
from torch.utils.data import DataLoader, Dataset, Subset

def get_loader(dataset_name, params):

    dataset_name = re.sub(r'-mi$', '', dataset_name, flags=re.IGNORECASE)
    dataset_name = re.sub(r'[^a-zA-Z0-9]', '', dataset_name).lower()

    data_loader = None
    if dataset_name == 'moabb':
        data_loader = get_moabb_loader(dataset_name, params)
    elif dataset_name == 'deap':
        data_loader = get_deap_loader(dataset_name, params)
    elif dataset_name in {'neuroprobe', 'faced', 'physionet', 'podcast', 'isruc', 'chbmit', 'mumtaz2016', 'mentalarithmetich', 'tuev', 'tuab'}:
        loader = get_dataset_loader(dataset_name, params)
        data_loader = loader.get_data_loader()
    return data_loader
    
class NewMOABBDataset(moabb_dataset.MOABBDataset):
    def __init__(self, dataset_name, params, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.params = params

        print("NewMOABB is working")

    @staticmethod
    def process_record(file: Any = None,
                       chunk_size: int = -1,
                       overlap: int = 0,
                       offline_transform: Union[None, Callable] = None,
                       dataset: _MOABBDataset = None,
                       paradigm: _MOABBParadigm = None,
                       before_trial: Union[None, Callable] = None,
                       after_trial: Union[None, Callable] = None,
                       **kwargs):

        subject_id, session_id = file

        write_pointer = 0
        session_signal = dataset.get_data(
            subjects=[subject_id])[subject_id][session_id]

        for run_id, run_signal in session_signal.items():
            if before_trial is not None:
                run_signal = before_trial(run_signal)

            proc = paradigm.get_data(dataset=dataset)

            if proc is None:
                continue

            trial_queue = []
            roi_signals, labels, metadatas = proc

            idx = metadatas.index[metadatas['run'] == run_id].to_numpy()
            if idx.size == 0:
                continue

            roi_signals_run = roi_signals[idx]
            labels_run = labels[idx]
            metadatas_run = metadatas.loc[idx]  

            for roi_id, (roi_signal, label, (_, meta_row)) in enumerate(
                    zip(roi_signals_run, labels_run, metadatas_run.iterrows())):

                start_at = 0
                if chunk_size <= 0:
                    chunk_size = roi_signal.shape[1] - start_at
                end_at = start_at + chunk_size

                step = chunk_size - overlap

                while end_at <= roi_signal.shape[1]:
                    clip_sample = roi_signal[:, start_at:end_at]

                    t_eeg = clip_sample
                    if not offline_transform is None:
                        t = offline_transform(eeg=t_eeg)
                        t_eeg = t['eeg']

                    clip_id = f'{subject_id}_{session_id}_{write_pointer}'
                    write_pointer += 1

                    dataset_name = str(type(dataset)).split('.')[-1].replace("'>", "")

                    meta_dict = meta_row.to_dict()
                    meta_dict['modality'] = 'EEG'
                    meta_dict['Dataset'] = f'MOABB_{dataset_name}'

                    record_info = {
                        'subject_id': meta_dict['subject'],
                        'session_id': meta_dict['session'],
                        'trial_id': run_id,
                        'roi_id': roi_id,
                        'clip_id': clip_id,
                        'label': label,
                        'start_at': start_at,
                        'end_at': end_at,
                        'metadata': meta_dict
                    }
                    start_at = start_at + step
                    end_at = start_at + chunk_size

                    yield {
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        }

    def set_records(self,
                    dataset: _MOABBDataset,
                    io_path: str = '.torcheeg/io/moabb_bnci2014001',
                    download_path: str = '.torcheeg/io/moabb_bnci2014001/raw',
                    **kwargs):
        subject_id_list = dataset.subject_list
        if download_path is None:
            download_path = os.path.join(io_path, 'raw')
        if not os.path.exists(download_path):
            dataset.download(subject_list=subject_id_list,
                             path=download_path,
                             verbose=False)

        subject_id = subject_id_list[0]
        session_id_list = list(
            dataset.get_data(subjects=[subject_id])[subject_id].keys())

        subject_id_session_id_list = list(
            itertools.product(subject_id_list, session_id_list))
        return subject_id_session_id_list[:1]

    def _add_meta_info_dataset_specific(self, meta_dict, dataset_name):
        if dataset_name == 'bnci2014001':
            channel_list = ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'PZ', 'P4', 'POZ']
            
        elif dataset_name == 'physionetMI':
            channel_list = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2',
                     'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2',
                     'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
                     'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5',
                     'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                     'O1', 'OZ', 'O2', 'IZ']      
        elif dataset_name == 'cho2017':
            channel_list = [
            "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2",
            "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]
        elif dataset_name == 'lee2019':
            channel_list = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8' 'FC5', 'FC1', 'FC2', 'FC6',
                            'T7', 'C3', 'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                            'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'FC3', 'FC4',
                            'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'FTT9H',
                            'TPP7H', 'TP7', 'TPP9H', 'FT10', 'FTT10H', 'TPP8H', 'TP8', 'TPP10H',
                            'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

        meta_dict['channel_names'] = channel_list
        meta_dict['xyz_id'] =  np.array([
                                    np.array(STANDARD_1005_CHANNEL_LOCATION_DICT[ch]) * 1000
                                    for ch in channel_list if ch in STANDARD_1005_CHANNEL_LOCATION_DICT
                                    ])
        meta_dict['resample_rate'] = self.params.resample_rate
        return meta_dict
                    
    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info
        meta_data = info['metadata']

        meta_data2dict = meta_data.replace("'", '"')

        meta_dict = json.loads(meta_data2dict)

        meta_dict = self._add_meta_info_dataset_specific(meta_dict, self.dataset_name)

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label, meta_dict

class CustomResample(EEGTransform):
    def __init__(self, origin_sampling_rate: int, target_sampling_rate: int,
                 apply_to_baseline=False,
                 axis=-1,
                 scale: bool = False):

        super(CustomResample, self).__init__(
            apply_to_baseline=apply_to_baseline)
        self.original_rate = origin_sampling_rate
        self.new_rate = target_sampling_rate
        self.axis = axis
        self.scale = scale

    def apply(self,
              eeg: np.ndarray,
              **kwargs
              ) -> np.ndarray:

        eeg = eeg.astype(np.float32)

        if self.original_rate == self.new_rate:
            return eeg

        ratio = float(self.new_rate) / self.original_rate

        if int(np.ceil(eeg.shape[self.axis])) % self.original_rate == 0:
            eeg = eeg
        
        else:
            trash = int(int(np.ceil(eeg.shape[self.axis])) % self.original_rate)
            eeg = eeg[:,:-trash]
        
        self.original_rate = int(self.original_rate)
        self.new_rate = int(self.new_rate)
        gcd = np.gcd(self.original_rate, self.new_rate)
        EEG_res = signal.resample_poly(
            eeg, self.new_rate // gcd, self.original_rate // gcd, axis=self.axis
        )
        
        if self.scale:
            EEG_res /= np.sqrt(ratio)

        return np.asarray(EEG_res, dtype=eeg.dtype)

    @property
    def __repr__(self) -> any:
        return f'''{
                'original_sampling_rate': self.original_rate,
                'target_sampling_rate': self.new_rate,
                'apply_to_baseline':self.apply_to_baseline
                'axis': self.axis,
                'scale': self.scale,
                'res_type': self.res_type
            }'''

def train_test_split_default_dataset(dataset, split_path):

    info = dataset.info

    train_row_ids = info.index[info['session_id'].str.contains('train')].tolist()
    test_row_ids = info.index[info['session_id'].str.contains('test')].tolist()

    train_info = info.iloc[train_row_ids]
    test_info = info.iloc[test_row_ids]

    train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
    test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    from copy import copy

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset

class MyBNCI2014_001(BNCI2014_001):
    def __init__(self, path=None):
        super().__init__()
        self.default_path = path

    def data_path(self, subject, path=None, **kwargs):
        if path is None and self.default_path is not None:
            path = self.default_path
        kwargs.setdefault("update_path", False)
        return super().data_path(subject=subject, path=path, **kwargs)

    def _get_single_subject_data(self, subject):
        return load_data(
            subject=subject,
            dataset=self.code,
            path=self.default_path,    
            update_path=False,         
            verbose=False
        )

    def get_data(self, subjects=None, cache_config=None, process_pipeline=None):
        cache_config = CacheConfig.make(cache_config)
        if cache_config.path is None and self.default_path is not None:
            cache_config.path = self.default_path
        return super().get_data(subjects=subjects,
                                cache_config=cache_config,
                                process_pipeline=process_pipeline)


class MyPhysionetMI(PhysionetMI):
    def __init__(self, path=None, imagined=True, executed=False):
        super().__init__(imagined=imagined, executed=executed)
        self.default_path = path 

    def data_path(self, subject, path=None, force_update=False, update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        runs = [1, 2] + self.hand_runs + self.feet_runs
        use_path = path or self.default_path
        if use_path is None:
            raise ValueError("MyPhysionetMI: default_path needed.")
        os.makedirs(use_path, exist_ok=True)
        return self._load_data(subject, runs=runs, path=use_path,
                               force_update=force_update, verbose=verbose)

    def _load_data(self, subject, runs, path=None, force_update=False, verbose=None):
        if path is None:
            if self.default_path is None:
                raise ValueError("MyPhysionetMI: default_path needed.")
            path = self.default_path
        return super()._load_data(subject, runs, path=path,
                                  force_update=force_update, verbose=verbose)

    def _get_single_subject_data(self, subject):
        data = {}
        idx = 0

        for run in self.hand_runs:
            raw = self._load_one_run(subject, run)  
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(raw, desired_event_id=self.events)
            idx += 1

        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(raw, desired_event_id=self.events)
            idx += 1

        return {"0": data}

class MyCho2017(Cho2017):
    def __init__(self, path=None):
        super().__init__()
        self.default_path = path

    def data_path(self, subject, path=None, force_update=False, update_path=False, verbose=None):
        if path is None and self.default_path is not None:
            path = self.default_path
        os.makedirs(path, exist_ok=True)
        return super().data_path(subject=subject,
                                 path=path,
                                 force_update=force_update,
                                 update_path=update_path,
                                 verbose=verbose)

class MyLee2019_MI(Lee2019_MI):
    def __init__(self, path=None, **kwargs):
        super().__init__(**kwargs)
        self.default_path = path

    def data_path(self, subject, path=None, force_update=False, update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        use_path = path or self.default_path
        if use_path is None:
            raise ValueError("MyLee2019_MI: default_path needed.")
        os.makedirs(use_path, exist_ok=True)
        return super().data_path(
            subject=subject,
            path=use_path,
            force_update=force_update,
            update_path=update_path,
            verbose=verbose,
        )
    
def get_moabb_loader(dataset_name, params):
    if params.moabb_task == 'bnci2014001':
        moabb_dataset = MyBNCI2014_001(path=f'.torcheeg/io/moabb_{params.moabb_task}/raw')
        original_samplingrate = 250
    elif params.moabb_task == 'physionetMI':
        moabb_dataset = MyPhysionetMI(path=f'.torcheeg/io/moabb_{params.moabb_task}/raw')
        original_samplingrate = 160
    elif params.moabb_task == 'stieger2021':
        raise NotImplementedError('Stieger dataset in not yet implemented')
        original_samplingrate = 1000
    elif params.moabb_task == 'cho2017':
        moabb_dataset = MyCho2017(path=f'.torcheeg/io/moabb_{params.moabb_task}/raw')
        original_samplingrate = 512
    elif params.moabb_task == 'lee2019':
        moabb_dataset = MyLee2019_MI(path=f'.torcheeg/io/moabb_{params.moabb_task}/raw')
        original_samplingrate = 1000
    else:
        raise ValueError('Given dataset name is not included in MOABB')

    if params.paradigm == 'leftright_imagery':
        paradigm = LeftRightImagery()
        label_mapping = {'left_hand': 0,
                        'right_hand': 1,}
    elif params.paradigm == 'motor_imagery':
        paradigm = MotorImagery()
        if params.moabb_task == 'bnci2014001':
            label_mapping = {'left_hand': 0,
                            'right_hand': 1,
                            'feet': 2,
                            'tongue': 3,}
        elif params.moabb_task == 'physionetMI':
            paradigm = MotorImagery(resample=128)
            label_mapping = {'left_hand': 0,
                             'right_hand': 1,
                             'hands': 2,
                             'feet': 3,
                             }
        elif params.moabb_task == 'cho2017':
            label_mapping = {'left_hand': 0,
                             'right_hand': 1}
        elif params.moabb_task == 'stieger2021':
            label_mapping = {'left_hand': 0,
                             'right_hand': 1,
                             'both_hand': 2,
                             'rest': 3}
        elif params.moabb_task == 'lee2019':
            label_mapping = {'left_hand': 0,
                             'right_hand': 1}
        else:
            raise ValueError('Given paradigm and dataset are not matched')
    
    else:
        raise ValueError('Given paradigm is not included in MOABB')


    dataset = NewMOABBDataset(dataset_name=params.moabb_task,
                              params=params,
                              dataset=moabb_dataset,
                            paradigm=paradigm,
                            io_path=f'.torcheeg/io/moabb_{params.moabb_task}',
                            download_path=f'.torcheeg/io/moabb_{params.moabb_task}/raw', 
                            offline_transform=transforms.Compose([
                                CustomResample(origin_sampling_rate=original_samplingrate, target_sampling_rate=params.resample_rate),
                                transforms.Reshape((params.num_channels, -1, params.resample_rate)),
                                transforms.ToTensor()
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select('label'),
                                transforms.Mapping(label_mapping)
                            ]),
                            num_worker=params.num_workers,)
    
    if params.moabb_task == 'bnci2014001':
        train_val_dataset, test_dataset = train_test_split_default_dataset(dataset=dataset, split_path=f'./.torcheeg/model_selection_{params.moabb_task}')
        train_dataset, val_dataset = train_test_split_cross_trial(dataset=train_val_dataset, test_size=params.test_ratio, split_path=f'./.torcheeg/model_selection_{params.moabb_task}_trainval')

    else:
        train_val_dataset, test_dataset = train_test_split_cross_subject(dataset=dataset, test_size=params.test_ratio, split_path=f'./.torcheeg/model_selection_{params.moabb_task}')
        train_dataset, val_dataset = train_test_split_cross_subject(dataset=train_val_dataset, test_size=params.test_ratio, split_path=f'./.torcheeg/model_selection_{params.moabb_task}_trainval')

    if params.precompute_features:
        batch_size_to_use = params.precompute_batch_size
    else:
        batch_size_to_use = params.batch_size
    train_loader = DataLoader(train_dataset, 
                              batch_size = batch_size_to_use,             
                              collate_fn=collate_fn_for_data_info_finetuning,
                              shuffle=True,
                              num_workers=getattr(params, 'num_workers', 0))
    val_loader = DataLoader(val_dataset,
                             batch_size = batch_size_to_use,
                             collate_fn=collate_fn_for_data_info_finetuning,
                             shuffle=False,
                             num_workers=getattr(params, 'num_workers', 0))
    test_loader = DataLoader(test_dataset,
                             batch_size = batch_size_to_use,
                             collate_fn=collate_fn_for_data_info_finetuning,
                             shuffle=False,
                             num_workers=getattr(params, 'num_workers', 0))

    data_loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return data_loader

def get_deap_loader(dataset_name, params):

    original_sampling_rate = 128
    target_sampling_rate = params.resample_rate

    dataset = deap_dataset(root_path='./data_preprocessed_python', 
                            offline_transform=transforms.Compose([
                                CustomResample(origin_sampling_rate=original_sampling_rate, target_sampling_rate=target_sampling_rate),
                                transforms.Reshape((params.num_channels, -1, params.resample_rate)),
                                transforms.ToTensor()
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select('valence'),
                            ]),
                            num_worker=params.num_workers)

    train_val_dataset, test_dataset = train_test_split_cross_subject(dataset=dataset, test_size=params.test_ratio)
    train_dataset, val_dataset = train_test_split_cross_subject(dataset=train_val_dataset, test_size=params.test_ratio)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size = params.batch_size,             
                              collate_fn=collate_fn_for_data_info_finetuning,
                              shuffle=True,
                              num_workers=getattr(params, 'num_workers', 0))
    val_loader = DataLoader(val_dataset,
                             batch_size = params.batch_size,
                             collate_fn=collate_fn_for_data_info_finetuning,
                             shuffle=False,
                             num_workers=getattr(params, 'num_workers', 0))
    test_loader = DataLoader(test_dataset,
                             batch_size = params.batch_size,
                             collate_fn=collate_fn_for_data_info_finetuning,
                             shuffle=False,
                             num_workers=getattr(params, 'num_workers', 0))

    data_loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return data_loader


class BaseCustomDataset(Dataset):
    def __init__(self, data_dir, mode=None, files=None, transform=None, collate_fn = collate_fn_for_data_info_finetuning, patch_size= 500, db=None, subject_id=None, trial_id=None, eval_name=None, preprocess = None, model_type = None, electrode_coord_dict = None):
        super(BaseCustomDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.files = files
        self.keys = None
        self.transform = transform
        self.collate = collate_fn 
        self.patch_size = patch_size
        self.db = db
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.preprocess = preprocess
        self.model_type = model_type
        self.electrode_coord_dict = electrode_coord_dict
        self.initialize_dataset()
    
    def initialize_dataset(self):
        """Initialize dataset-specific structures (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement initialize_dataset")
        
    def __len__(self):
        if self.keys is not None:
            return len(self.keys)
        elif self.files is not None:
            return len(self.files)
        elif self.eval_name is not None:
            return len(self.db)
        return 0
    
    def __getitem__(self, idx):
        """Get item implementation (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement __getitem__")    


class BaseLoadDataset:
    """
    Base data loader class that creates and manages DataLoaders.
    """
    def __init__(self, dataset_class, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.dataset_class = dataset_class
    
    def prepare_datasets(self):
        """Prepare train, val, test datasets (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement prepare_datasets")
    
    def get_data_loader(self):
        """Create data loaders from datasets"""
        train_set, val_set, test_set = self.prepare_datasets()
        
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        
        if self.params.precompute_features:
            batch_size_to_use = self.params.precompute_batch_size
        else:
            batch_size_to_use = self.params.batch_size

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=batch_size_to_use,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'val': DataLoader(
                val_set,
                batch_size=batch_size_to_use,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'test': DataLoader(
                test_set,
                batch_size=batch_size_to_use,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
        }
        return data_loader

class LMDBCustomDataset(BaseCustomDataset):
    """Dataset implementation for LMDB-based datasets (FACED, PHYSIO, SEEDV, etc.)"""
    def initialize_dataset(self, return_info = True):
        self.db = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.return_info = return_info
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[self.mode]
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        data_info = pair.get('data_info', {})
        
        data = to_tensor(data)
        if self.transform is not None:
            data = self.transform(data)
        if self.return_info:
            return data, label, data_info
        else:
            return data, label

class LMDBLoadDataset(BaseLoadDataset):
    """Load dataset implementation for LMDB-based datasets"""
    
    def prepare_datasets(self):
        transform=None        
        train_set = self.dataset_class(self.datasets_dir, mode='train', transform=transform)
        val_set = self.dataset_class(self.datasets_dir, mode='val', transform=transform)
        test_set = self.dataset_class(self.datasets_dir, mode='test')
        return train_set, val_set, test_set

class NeuroprobeCustomDataset(BaseCustomDataset):
    """Dataset implementation for neuroprobe Benchtree dataset """
    def initialize_dataset(self, return_info = True):
        self.return_info = return_info
        self.ieeg_scaling_factor = 2.0
        self.lip_from_btb={}
        for s in range(1,11): 
            csv_path = f"/sub_{s}.csv" #insert the preprocessed csv file
            df = pd.read_csv(csv_path)
            order = ['L', 'I', 'P'] 
            channel_dict = {}
            for _, row in df.iterrows():
                key = row['Electrode']
                vals = [row[c] for c in order]
                key = key.replace('*', '')
                channel_dict[key] = vals  
            self.lip_from_btb[s] = channel_dict  


    def _get_data_info_dict(self, item):
        data_info_dict={}
        data_info_dict['xyz_id'] = item['electrode_coordinates']
        data_info_dict['LIP_id'] = torch.LongTensor(np.array([self.lip_from_btb[self.subject_id][ch] for ch in item['electrode_labels_subset']]))
        data_info_dict['modality'] = 'iEEG'
        data_info_dict['coord_subtype'] = ['depth' for i in range(len(item['electrode_labels_subset']))]
        data_info_dict['subject_id'] = "BT_" + str(self.subject_id)
        data_info_dict['Dataset'] = 'Brainbanktree'
    
        return data_info_dict

    def _data_patcher(self, X, patchsize):
        if len(X.shape) != 2:
            raise ValueError("Input data should be 2D array (C, N) for patching, but got 3D") 
        elif len(X.shape) == 2:    
            C, N = X.shape
            if N // patchsize < 1:
                raise ValueError(f"Input length {N} is shorter than the patch size {patchsize}.")
            num_patches = N // patchsize
            X = X[ :, :num_patches * patchsize]  
            X = X.reshape( C, num_patches, patchsize)
            return X
    
    def __getitem__(self, idx):
        item = self.db[idx]
        orig_data = item['data']
        label = item['label']
        elec_label =  item['electrode_labels_subset']
        if self.model_type == "DIVER":
            from eval_utils import preprocess_data
            preprocessed_data = preprocess_data(orig_data, electrode_labels=elec_label, preprocess=self.preprocess, preprocess_parameters=None)
            data = self._data_patcher(preprocessed_data,self.patch_size).to(torch.float) / self.ieeg_scaling_factor
            if data.shape[2] != self.patch_size:
                raise ValueError(f"{data.shape[2]} length,but {self.patch_size} size patch is need")
            data_info = self._get_data_info_dict(item)
        elif self.model_type == "DIVER_but_no_preprocess":
            from eval_utils import preprocess_data
            preprocessed_data = preprocess_data(orig_data, electrode_labels=elec_label, preprocess='remove_line_noise-downsample_500', preprocess_parameters=None)
            data = self._data_patcher(preprocessed_data,self.patch_size).to(torch.float) / self.ieeg_scaling_factor 
            data_info = self._get_data_info_dict(item)
        elif self.model_type in ["BrainBERT_raw", "BrainBERT_mean"]:
            from eval_utils import preprocess_data
            data = preprocess_data(orig_data, electrode_labels=elec_label, preprocess='remove_line_noise-laplacian', preprocess_parameters=None)
            data_info = self._get_data_info_dict(item)
        elif self.model_type == "PopT":
            from eval_utils import preprocess_data
            data = preprocess_data(orig_data, electrode_labels=elec_label, preprocess='remove_line_noise-laplacian', preprocess_parameters=None)
            data_info = self._get_data_info_dict(item)

        if self.return_info:
            return data, label, data_info
        else:
            return data, label
        
class NeuroprobeLoadDataset(BaseLoadDataset):
    """Load dataset implementation for LMDB-based datasets"""
    def __init__(self, dataset_class, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.dataset_class = dataset_class
        self.seed = params.seed
        self.subject_id = params.neuroprobe_subject
        self.trial_id = params.neuroprobe_trial
        self.eval_name = params.neuroprobe_task
        self.kfold_idx = params.neuroprobe_fold
        self.allow_corrupted = params.neuroprobe_allow_corrupted
        self.model_type = params.neuroprobe_model_type
        self.preprocess = params.neuroprobe_preprocess
        self.np_evaltype = params.neuroprobe_evaltype
        self.np_clip = params.neuroprobe_clip
        self.lite = params.neuroprobe_lite
        self.patch_size = params.neuroprobe_patchsize
        self.lite_dict = {1:[1,2],2:[0,4],3:[0,1],4:[0,1],7:[0,1],10:[0,1]}
        self.full_dict =  {1:[0,1,2],2:[0,1,2,3,4,6],3:[0,1,2],4:[0,1,2],5:[0],6:[0,1,2],7:[0,1],8:[0],9:[0],10:[0,1]}

        print("neuroprobe subject, trial, task, fold, allow_corrupted, model_type, preprocess", self.subject_id, self.trial_id, self.eval_name, self.kfold_idx, self.allow_corrupted, self.model_type, self.preprocess)
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
        if self.model_type == "DIVER":
            if self.preprocess == "laplacian-downsample_500":
                assert self.np_clip == 1 
            self.subject = BrainTreebankSubject(self.subject_id, allow_corrupted=self.allow_corrupted, allow_missing_coordinates=False, cache=False, dtype=torch.float32, DIVER_preprocess=True, clip =self.np_clip) 
        else:
            self.subject = BrainTreebankSubject(self.subject_id, allow_corrupted=self.allow_corrupted, allow_missing_coordinates=False, cache=False, dtype=torch.float32, DIVER_preprocess=False) 
        self.nano = False
        self.electrode_coord_dict=self.subject.get_electrode_coordinates_dict()
        self.output_indices = False
        
        self.start_neural_data_before_word_onset = 0
        self.end_neural_data_after_word_onset = 2048 

        

    def prepare_datasets(self):
        if self.np_evaltype == 'sssm' :
            transform=None
            if self.subject_id not in self.lite_dict.keys():
                raise ValueError(f"sub{self.subject_id} is not on lite")
            if self.trial_id not in self.lite_dict[self.subject_id] :
                raise ValueError("no matching trial id in lite")
            from neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
            self.dataset = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, self.trial_id, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                        start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                        lite=self.lite, nano=self.nano, output_dict=True)
            self.len_dataset = len(self.dataset)

            from sklearn.model_selection import KFold
            if self.lite :
                kf = KFold(n_splits=2, shuffle=False)
            else:
                raise ValueError("only lite is ready")
            train_val_idx, test_idx =list(kf.split(self.dataset))[self.kfold_idx]
            random.seed(self.seed)
            random.shuffle(train_val_idx)        
            n1 = int(len(train_val_idx) * 0.9)
            train_idx = train_val_idx[:n1]
            val_idx = train_val_idx[n1:]

            self.db_train = Subset(self.dataset,train_idx)
            self.db_val = Subset(self.dataset,val_idx)
            self.db_test = Subset(self.dataset,test_idx)

            train_set = self.dataset_class(data_dir = None, db=self.db_train, subject_id=self.subject_id, trial_id=self.trial_id, eval_name=self.eval_name, model_type = self.model_type, preprocess = self.preprocess, electrode_coord_dict = self.electrode_coord_dict)
            val_set = self.dataset_class(data_dir = None, db=self.db_val, subject_id=self.subject_id, trial_id=self.trial_id, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess, electrode_coord_dict = self.electrode_coord_dict)
            test_set = self.dataset_class(data_dir = None,db=self.db_test, subject_id=self.subject_id, trial_id=self.trial_id, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess, electrode_coord_dict = self.electrode_coord_dict)

            return train_set, val_set, test_set
        
        elif self.np_evaltype == 'ssdm':
            from torch.utils.data import ConcatDataset
            from neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
            transform=None
            if self.subject_id not in self.lite_dict.keys():
                raise ValueError(f"sub{self.subject_id} is not on lite")
            if self.kfold_idx > 1 :
                raise ValueError("k_fold must below 2")
            else:
                if self.lite :
                    trial_list = self.lite_dict[self.subject_id]
                    for fold in range(2):
                        trial = trial_list[fold]
                        if fold != self.kfold_idx :
                            db_train_val = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                        start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                        lite=self.lite, nano=self.nano, output_dict=True)
                        else:
                            self.db_test = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                            start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                            lite=self.lite, nano=self.nano, output_dict=True)
                else:
                    trial_list = self.full_dict[self.subject_id]
                    if len(trial_list) < 2:
                        raise ValueError(f"sub{self.subject_id} has less than two trial.")
                    if len(trial_list) > 2:
                        db_train_list = []
                        for trial in trial_list:
                            if trial != self.trial_id :
                                db_train_list.append(BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                            start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                            lite=self.lite, nano=self.nano, output_dict=True))
                            else:
                                self.db_test = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                            start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                            lite=self.lite, nano=self.nano, output_dict=True) 
                                
                        db_train_val = ConcatDataset(db_train_list)      
                    elif len(trial_list) == 2:
                        for trial in trial_list:
                            if trial != self.trial_id :
                                db_train_val = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                            start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                            lite=self.lite, nano=self.nano, output_dict=True)
                            else:
                                self.db_test = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                            start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                            lite=self.lite, nano=self.nano, output_dict=True)                                                        

                
                self.len_dataset = len(self.db_test) + len(db_train_val)
                random.seed(self.seed)
                train_val_idx = [i for i in range(len(db_train_val))]
                random.shuffle(train_val_idx)        
                n1 = int(len(train_val_idx) * 0.9)
                train_idx = train_val_idx[:n1]
                val_idx = train_val_idx[n1:]
                self.db_train = Subset(db_train_val,train_idx)
                self.db_val = Subset(db_train_val,val_idx)

                train_set = self.dataset_class(data_dir = None, db=self.db_train, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name, model_type = self.model_type, preprocess = self.preprocess, electrode_coord_dict = self.electrode_coord_dict)
                val_set = self.dataset_class(data_dir = None, db=self.db_val, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess,electrode_coord_dict = self.electrode_coord_dict)
                test_set = self.dataset_class(data_dir = None,db=self.db_test, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess,electrode_coord_dict = self.electrode_coord_dict)
                  
        elif self.np_evaltype == 'dsdm':
            from torch.utils.data import ConcatDataset
            from neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset
            from neuroprobe.braintreebank_subject import BrainTreebankSubject
            if self.model_type == "DIVER":
                if self.preprocess == "laplacian-downsample_500":
                    assert self.np_clip == 1 
                self.train_subject = BrainTreebankSubject(2, allow_corrupted=self.allow_corrupted, allow_missing_coordinates=False, cache=False, dtype=torch.float32, DIVER_preprocess=True, clip =self.np_clip)
            else:
                self.train_subject = BrainTreebankSubject(2, allow_corrupted=self.allow_corrupted, allow_missing_coordinates=False, cache=False, dtype=torch.float32, DIVER_preprocess=False)
            
            transform=None
            if self.subject_id not in self.lite_dict.keys():
                raise ValueError(f"sub{self.subject_id} is not on lite")
            if self.subject_id == 2:
                raise ValueError(f"sub{self.subject_id} is use only for training in dsdm")
            if self.kfold_idx > 1 :
                raise ValueError("k_fold must below 2")
            else:
                if self.lite :
                    trial_list = self.lite_dict[self.subject_id]
                    trial = trial_list[self.kfold_idx]
                    self.db_test = BrainTreebankSubjectTrialBenchmarkDataset(self.subject, trial, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                        start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                        lite=self.lite, nano=self.nano, output_dict=True)
                    db_train_val = BrainTreebankSubjectTrialBenchmarkDataset(self.train_subject, 4, dtype=torch.float32, eval_name=self.eval_name, output_indices=self.output_indices, 
                                                        start_neural_data_before_word_onset=self.start_neural_data_before_word_onset, end_neural_data_after_word_onset=self.end_neural_data_after_word_onset,
                                                        lite=self.lite, nano=self.nano, output_dict=True)
                else:
                    raise ValueError("only lites is raedy for dsdm")
                self.len_dataset = len(self.db_test) + len(db_train_val)
                random.seed(self.seed)
                train_val_idx = [i for i in range(len(db_train_val))]
                random.shuffle(train_val_idx)        
                n1 = int(len(train_val_idx) * 0.9)
                train_idx = train_val_idx[:n1]
                val_idx = train_val_idx[n1:]
                self.db_train = Subset(db_train_val,train_idx)
                self.db_val = Subset(db_train_val,val_idx)

                train_set = self.dataset_class(data_dir = None, db=self.db_train, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name, model_type = self.model_type, preprocess = self.preprocess, electrode_coord_dict = self.electrode_coord_dict)
                val_set = self.dataset_class(data_dir = None, db=self.db_val, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess,electrode_coord_dict = self.electrode_coord_dict)
                test_set = self.dataset_class(data_dir = None,db=self.db_test, subject_id=self.subject_id, trial_id=None, eval_name=self.eval_name,  model_type = self.model_type, preprocess = self.preprocess,electrode_coord_dict = self.electrode_coord_dict)
                return train_set, val_set, test_set   

class PodcastAdhocDataset(Dataset):
    def __init__(self, dataset_class, params):
        self.params = params
        self.dataset_class = dataset_class
        self.datasets_dir = params.datasets_dir
        self.embedding_layer = params.podcast_embedding_layer
        self.embedding_type = params.podcast_embedding_type
        self.embedding_pca_dim = params.podcast_embedding_pca_dim
        self.subject_ids = params.podcast_subject_ids
        self.per_subject_electrodes = None 
        self.channel_reg_ex = None 
        self.lag = params.podcast_lag 
        self.window_width = params.podcast_window_width 
        self.word_column = params.podcast_word_column
        self.XY_already_exist = getattr(params, "XY_already_exist", False)

        ids = params.podcast_subject_ids
        if isinstance(ids, (list, tuple)):
            subs_tag = "-".join(f"{int(s):02d}" if str(s).isdigit() else str(s) for s in ids)
        else:
            subs_tag = f"{int(ids):02d}" if str(ids).isdigit() else str(ids)

        self.subs_tag = subs_tag
        import hashlib, json, os
        key = {
            "layer": self.embedding_layer,
            "etype": self.embedding_type,
            "pca": self.embedding_pca_dim,
            "subs": tuple(self.subject_ids),
            "lag": self.lag,
            "ww": self.window_width,
            "wcol": self.word_column,
        }
        digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
        self.cache_path = os.path.join(self.datasets_dir, f"cache_sub{subs_tag}_XY_{digest}.pt")

        self.X = None
        self.Y = None
        self.words = None
        self.data_info = None

        self._prepare_data_once()

    def _prepare_data_once(self):
        if self.X is not None: 
            return

        if self.XY_already_exist and os.path.exists(self.cache_path):
            data = torch.load(self.cache_path, map_location="cpu")
            self.X, self.Y, self.words = data["X"], data["Y"], data["words"]
            return

        podcastdataprepare = PodcastdataprepareAdhoc(self.params)
        podcast_raws = podcastdataprepare.load_raws(subject_electrode_mapping=None)
        df_word = podcastdataprepare.word_embedding_decoding_task()
        X, Y, words = podcastdataprepare.get_data(self.lag,
                                                    podcast_raws,
                                                    df_word,
                                                    self.window_width,
                                                    word_column=self.word_column)

        channel_info_tsv_path = os.path.join(self.datasets_dir, f'sub-{self.subs_tag:02}/ieeg/sub-{self.subs_tag:02}_space-MNI152NLin2009aSym_electrodes.tsv')
        bad_channel_info_tsv_path = os.path.join(self.datasets_dir, f'sub-{self.subs_tag:02}/ieeg/sub-{self.subs_tag:02}_task-podcast_channels.tsv')

        

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(np.asarray(list(Y)), dtype=torch.float32)
        self.words = np.asarray(list(words)) if words is not None else None
        self.data_info = self.load_data_info_from_tsv(channel_info_tsv_path, bad_channel_info_tsv_path)

        try:
            torch.save({"X": self.X, "Y": self.Y, "words": self.words, "data_info": self.data_info}, self.cache_path)
        except Exception:
            pass

    def load_data_info_from_tsv(self, coords_tsv_path: str, meta_tsv_path: str | None = None):
        df = pd.read_csv(coords_tsv_path, sep="\t")

        required = {"name", "x", "y", "z"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"TSV missing columns: {missing}")

        df = df.copy()
        df["name"] = df["name"].astype(str).str.strip()
        for c in ["x", "y", "z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if meta_tsv_path is not None and os.path.exists(meta_tsv_path):
            meta = pd.read_csv(meta_tsv_path, sep="\t")
            if {"name", "status"}.issubset(meta.columns):
                meta = meta.copy()
                meta["name"] = meta["name"].astype(str).str.strip()
                meta["status"] = meta["status"].astype(str).str.strip().str.lower()
                bad_names = set(meta.loc[meta["status"] == "bad", "name"])
                if bad_names:
                    df = df[~df["name"].isin(bad_names)].reset_index(drop=True)

        if "group" not in df.columns:
            df["group"] = "unknown"
        df["group"] = df["group"].astype(str)

        df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

        mapping = {"G": "Grid", "S": "Strip", "D": "Depth"}
        coord_subtype = (
            df["group"].astype(str).str.strip().str.upper()
            .map(mapping)
            .fillna("Unknown")
            .to_numpy()
        )

        channel_names = df["name"].tolist()
        xyz_id = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        self.data_info = {
            "Dataset": "Podcast",
            "modality": "iEEG",
            "release": None,
            "subject_id": getattr(self, "subs_tag", None),
            "task": "language decoding",
            "resampling_rate": 500,
            "original_sampling_rate": 512,
            "channel_names": channel_names,
            "xyz_id": xyz_id,
            "coord_subtype": coord_subtype,
            "coordinate_space": "MNI152",
        }
        return self.data_info

    def __len__(self):
        self._prepare_data_once()
        return self.X.shape[0]

    def __getitem__(self, idx):
        self._prepare_data_once()
        if self.words is None:
            return self.X[idx], self.Y[idx]
        return self.X[idx], self.Y[idx], self.words[idx], self.data_info

def collate_for_podcast(batch):
        """
        Handles both:
        - (X, Y)
        - (X, Y, words, data_info)
        """
        first = batch[0]
        if len(first) == 2:
            xs, ys = zip(*batch)
            return torch.stack(xs, 0), torch.stack(ys, 0)
        elif len(first) == 4:
            xs, ys, words, data_info = zip(*batch)
            patches = torch.stack(xs, 0)
            targets = torch.stack(ys, 0)
            return patches, targets, data_info 
        
        else:
            raise ValueError("Unsupported batch tuple structure.")

    
class PodcastAdhocLoadDataset(BaseLoadDataset):

    def prepare_datasets(self):
        base_ds = self.dataset_class(self.dataset_class, self.params)

        fold_type = "sequential_folds" 

        N = len(base_ds)
        if N == 0:
            raise ValueError("Empty dataset.")

        if fold_type == "sequential_folds":
            fold_indices = get_sequential_folds(N, num_folds=self.params.podcast_fold_num)
        elif fold_type == "zero_shot_folds":
            raise NotImplementedError('zero shot folds not fully implemented yet')
            fold_indices = get_zero_shot_folds(
                selected_words, num_folds=n_folds
            )
        else:
            raise ValueError(f"Unknown fold_type: {fold_type}")

        use_fold_indices = fold_indices[self.params.podcast_foldidx]

        train_id, val_id, test_id = use_fold_indices

        train_set = Subset(base_ds, list(map(int, train_id)))
        val_set   = Subset(base_ds, list(map(int, val_id)))
        test_set  = Subset(base_ds, list(map(int, test_id)))

        return train_set, val_set, test_set

    def get_data_loader(self):
        """Create data loaders from datasets"""
        train_set, val_set, test_set = self.prepare_datasets()
        
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))

        if self.params.precompute_features:
            batch_size_to_use = self.params.precompute_batch_size
        else:
            batch_size_to_use = self.params.batch_size
        
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=batch_size_to_use,
                collate_fn=collate_for_podcast,
                shuffle=True,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'val': DataLoader(
                val_set,
                batch_size=batch_size_to_use,
                collate_fn=collate_for_podcast,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'test': DataLoader(
                test_set,
                batch_size=batch_size_to_use,
                collate_fn=collate_for_podcast,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
        }
        return data_loader

class TUEVCustomDataset(BaseCustomDataset):    
    def initialize_dataset(self):
        pass
    
    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
        data = data_dict['signal']
        label = int(data_dict['label'][0] - 1)
        data = signal.resample(data, 1000, axis=-1)
        data = data.reshape(16, 5, 200)
        return data * 10000, label


class TUEVLoadDataset(BaseLoadDataset):
    """Load dataset implementation for TUEV dataset"""
    
    def prepare_datasets(self):
        train_files = os.listdir(os.path.join(self.datasets_dir, "processed_train"))
        train_sub = list(set([f.split("_")[0] for f in train_files]))
        print("train sub", len(train_sub))
        test_files = os.listdir(os.path.join(self.datasets_dir, "processed_eval"))

        val_sub = np.random.choice(train_sub, size=int(
            len(train_sub) * 0.1), replace=False)
        train_sub = list(set(train_sub) - set(val_sub))
        val_files = [f for f in train_files if f.split("_")[0] in val_sub]
        train_files = [f for f in train_files if f.split("_")[0] in train_sub]

        train_set = self.dataset_class(
            os.path.join(self.datasets_dir, "processed_train"), 
            files=train_files
        )
        val_set = self.dataset_class(
            os.path.join(self.datasets_dir, "processed_train"), 
            files=val_files
        )
        test_set = self.dataset_class(
            os.path.join(self.datasets_dir, "processed_eval"), 
            files=test_files
        )
        
        return train_set, val_set, test_set

class ISRUCCustomDataset(BaseCustomDataset):
    def __init__(self, seqs_labels_path_pair, transform=None):
        self.seqs_labels_path_pair = seqs_labels_path_pair
        self.transform = transform
        
    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        seq = np.load(seq_path)
        label = np.load(label_path)
        return seq, label
            
class ISRUCLoadDataset(BaseLoadDataset):
    def __init__(self, dataset_class, params):
        self.params = params
        self.dataset_class = dataset_class
        self.seqs_dir = os.path.join(params.datasets_dir, 'seq')
        self.labels_dir = os.path.join(params.datasets_dir, 'label')
        self.seqs_labels_path_pair = self.load_path()
        

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)
        train_set = ISRUCCustomDataset(train_pairs)
        val_set = ISRUCCustomDataset(val_pairs)
        test_set = ISRUCCustomDataset(test_pairs)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=1,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=1,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
    def load_path(self):
        seqs_labels_path_pair = []
        subject_dirs_seq = []
        subject_dirs_labels = []
        for subject_num in range(1, 101):
            subject_dirs_seq.append(os.path.join(self.seqs_dir, f'ISRUC-group1-{subject_num}'))
            subject_dirs_labels.append(os.path.join(self.labels_dir, f'ISRUC-group1-{subject_num}'))

        for subject_seq, subject_label in zip(subject_dirs_seq, subject_dirs_labels):
            subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            for seq_fname, label_fname in zip(seq_fnames, label_fnames):
                subject_pairs.append((os.path.join(subject_seq, seq_fname), os.path.join(subject_label, label_fname)))
            seqs_labels_path_pair.append(subject_pairs)
        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair):
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for i in range(100):
            if i < 80:
                train_pairs.extend(seqs_labels_path_pair[i])
            elif i < 90:
                val_pairs.extend(seqs_labels_path_pair[i])
            else:
                test_pairs.extend(seqs_labels_path_pair[i])
        return train_pairs, val_pairs, test_pairs

class SEEDVCustomDataset(LMDBCustomDataset):
    """SEEDV dataset implementation"""
    pass

class FACEDCustomDataset(LMDBCustomDataset):
    """FACED dataset implementation"""
    pass

class PhysioCustomDataset(LMDBCustomDataset):
    """PhysioNet dataset implementation"""
    pass

class SHUCustomDataset(LMDBCustomDataset):
    """SHU dataset implementation"""
    assert NotImplementedError("SHU dataset is not implemented yet")
    pass

class CHBCustomDataset(LMDBCustomDataset):
    """CHB dataset implementation"""
    assert NotImplementedError("CHB dataset is not implemented yet")
    pass


class MumtazCustomDataset(LMDBCustomDataset):
    """Mumtaz dataset implementation"""
    assert NotImplementedError("Mumtaz dataset is not implemented yet")
    pass


class SEEDVIGCustomDataset(LMDBCustomDataset):
    """SEEDVIG dataset implementation"""
    assert NotImplementedError("SEEDVIG dataset is not implemented yet")
    pass


class StressCustomDataset(LMDBCustomDataset):
    """Stress dataset implementation"""
    assert NotImplementedError("Stress dataset is not implemented yet")
    pass


class TUABCustomDataset(TUEVCustomDataset):
    """TUAB dataset implementation"""
    pass


class BCICIVCustomDataset(LMDBCustomDataset):
    """BCIC-IV dataset implementation"""
    assert NotImplementedError("BCIC-IV dataset is not implemented yet")
    pass

def get_dataset_loader(dataset_name, params):
    dataset_name = dataset_name.lower()
    
    datasets = {
        'seedv': (LMDBLoadDataset, SEEDVCustomDataset),
        'faced': (LMDBLoadDataset, FACEDCustomDataset),
        'physionet': (LMDBLoadDataset, PhysioCustomDataset),
        'isruc': (ISRUCLoadDataset, ISRUCCustomDataset),
        'chbmit': (LMDBLoadDataset, CHBCustomDataset),
        'mumtaz2016': (LMDBLoadDataset, MumtazCustomDataset),
        'mentalarithmetich': (LMDBLoadDataset, StressCustomDataset), 
        'tuev': (TUEVLoadDataset, TUEVCustomDataset),
        'neuroprobe': (NeuroprobeLoadDataset,NeuroprobeCustomDataset),
        'podcast': (PodcastAdhocLoadDataset, PodcastAdhocDataset),
        'tuab': (TUEVLoadDataset, TUABCustomDataset),
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(datasets.keys())}")
    
    loader_class, dataset_class = datasets[dataset_name]
    
    return loader_class(dataset_class, params)