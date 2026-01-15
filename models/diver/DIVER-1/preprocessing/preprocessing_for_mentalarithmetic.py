import os
import lmdb
import pickle
import mne
import numpy as np


def load_xyz_from_elc(elc_path: str,
                      want_channels: list[str]) -> np.ndarray:
    want_up = [ch.upper() for ch in want_channels]

    with open(elc_path, 'r') as f:
        lines = f.readlines()[4:]

    positions_start = labels_start = None
    for i, ln in enumerate(lines):
        ll = ln.strip().lower()
        if ll == "positions":
            positions_start = i + 1
        elif ll == "labels":
            labels_start = i + 1
            break
    if positions_start is None or labels_start is None:
        raise RuntimeError("ELC file does not have Positions/Labels")

    positions = []
    for ln in lines[positions_start:labels_start-1]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        xyz = [float(p) for p in ln.split()[:3]]
        positions.append(np.array(xyz, dtype=float))

    labels = [ln.strip().upper() for ln in lines[labels_start:]
              if ln.strip() and not ln.startswith('#')]

    if len(labels) != len(positions):
        raise RuntimeError("Labels != Positions")

    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))
    return np.vstack(xyz_list)  


root_dir = '/your/MentalArithmetic/path'
files = [file for file in os.listdir(root_dir) if file.endswith('.edf')]
files = sorted(files)
print(f'files: {files}')
print(f'======= # files: {len(files)} =======')

files_dict = {
    'train':files[:56],
    'val':files[56:64],
    'test':files[64:],
}
print(files_dict)
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}


selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
                     'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
                     'EEG Fz', 'EEG Cz', 'EEG Pz'] 
RESAMPLE_RATE = 500 


saveDir = '/your/lmdb/path'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
db = lmdb.open(saveDir, map_size=5737418240) 


for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True)
        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        print(f'channels after selection: {raw.info["ch_names"]}')
        original_sampling_rate = int(raw.info['sfreq'])
        raw.resample(RESAMPLE_RATE)

        wanted_channels = [ch.replace('EEG ', '') for ch in raw.info["ch_names"]]
        print(f'wanted_channels: {wanted_channels}')

        elc_file = "/standard_1005.elc"
        xyz_array = load_xyz_from_elc(elc_file, wanted_channels)

        eeg = raw.get_data(units='uV')
        chs, points = eeg.shape
        print(f'eeg_array before reshaping: {eeg.shape}')
        a = points % (5 * RESAMPLE_RATE)
        if a != 0:
            eeg = eeg[:, :-a]
        eeg = eeg.reshape(chs, -1, 5, RESAMPLE_RATE).transpose(1, 0, 2, 3)
        print(f'eeg_array after reshaping: {eeg.shape}')
        label = int(file[-5])

        for i, sample in enumerate(eeg):
            sample_key = f'{file[:-4]}-{i}'
            data_dict = {
                'sample':sample, 
                'label':label-1,
                "data_info": { 
                    "Dataset": "MentalArith",
                    "modality": "EEG",
                    "release": None,            
                    "subject_id": file,          
                    "task": 'MentalArith',
                    "resampling_rate": RESAMPLE_RATE, 
                    "original_sampling_rate": original_sampling_rate,
                    "segment_index": i,                
                    "start_time": i*4,
                    "channel_names": wanted_channels,
                    "xyz_id": xyz_array
                }
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()