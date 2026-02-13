import os
import lmdb
import pickle
import numpy as np
import mne

import pdb

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
tasks = ['04', '06', '08', '10', '12', '14']

root_dir = 'your/Physionet-MI/path'

files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train': files[:70],
    'val': files[70:89],
    'test': files[89:109],
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}



selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']

db = lmdb.open('your/LMBD/path', map_size=12614542346)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        file_dir = os.path.join(root_dir, file)

        if not os.path.isdir(file_dir):
            print(f"not a dir: {file_dir}")
            continue
        for task in tasks:
            edf_path = os.path.join(file_dir, f'{file}R{task}.edf')

            if not os.path.exists(edf_path):
                print(f"no EDF files {edf_path}")
                continue
            raw = mne.io.read_raw_edf(os.path.join(root_dir, file, f'{file}R{task}.edf'), preload=True)
            raw.pick_channels(selected_channels, ordered=True)
            if len(raw.info['bads']) > 0:
                print('interpolate_bads')
                raw.interpolate_bads()
            original_sampling_rate = int(raw.info['sfreq'])
            raw.set_eeg_reference(ref_channels='average')
            raw.filter(l_freq=4, h_freq=40)
            raw.notch_filter((60))
            raw.resample(500)
            
            elc_file = "/standard_1005.elc" 

            want_channels = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                     'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2',
                     'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
                     'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5',
                     'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
                     'O1', 'Oz', 'O2', 'Iz']
            
            xyz_array = load_xyz_from_elc(elc_file, want_channels)

            events_from_annot, event_dict = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw,
                                events_from_annot,
                                event_dict,
                                tmin=0,
                                tmax=4. - 1.0 / raw.info['sfreq'],
                                baseline=None,
                                preload=True)
            data = epochs.get_data(units='uV')
            pdb.set_trace()
            events = epochs.events[:, 2]
            print(data.shape, events)
            data = data[:, :, -2000:]
            print(data.shape)
            bz, ch_nums, _ = data.shape
            data = data.reshape(bz, ch_nums, 4, 500)
            print(data.shape)
            for i, (sample, event) in enumerate(zip(data, events)):
                if event != 1:
                    sample_key = f'{file}R{task}-{i}'
                    data_dict = {
                        'sample': sample, 'label': event - 2 if task in ['04', '08', '12'] else event,
                        "data_info": { 
                            "Dataset": "PhysioNet-MI",
                            "modality": "EEG",
                            "release": None,            
                            "subject_id": file,          
                            "task": 'PhysioNet-MI',
                            "resampling_rate": 500, 
                            "original_sampling_rate": original_sampling_rate,
                            "segment_index": i,                
                            "start_time": i*4,
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