from scipy import signal
import os
import lmdb
import pickle
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
        raise RuntimeError("ELC file does not have positions/Labels")

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

labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])

root_dir='your/FACED/path'

files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train':files[:80],
    'val':files[80:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open('/your/LMDB/path/FACED.lmdb', map_size=18612500172)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        f = open(os.path.join(root_dir, file), 'rb')
        array = pickle.load(f)
        print(array.shape)
        eeg = signal.resample(array, 6000, axis=2)
        print(eeg.shape)
        eeg_ = eeg.reshape(28, 32, 30, 200)

        elc_file = "/standard_1005.elc"

        want_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Fc1', 'Fc2', 'Fc5', 'Fc6',
    'Cz', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'Cp1', 'Cp2', 'Cp5', 'Cp6', 'Pz', 'P3', 'P4',
    'P7', 'P8', 'Po3', 'Po4', 'Oz', 'O1', 'O2']
        
        xyz_array = load_xyz_from_elc(elc_file, want_channels)
        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10*j:10*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                subject_id = os.path.splitext(file)[0]
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label,
                    "data_info": { 
                    "Dataset": "FACED",
                    "modality": "EEG",
                    "release": None,            
                    "subject_id": subject_id,          
                    "task": 'FACED',
                    "resampling_rate": 200, 
                    "original_sampling_rate": 250,
                    "segment_index": i,                
                    "start_time": i * 30,
                    "channel_names": want_channels,
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