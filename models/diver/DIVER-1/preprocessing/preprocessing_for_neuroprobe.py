import os
import h5py
from mne.filter import filter_data, notch_filter

root_dir = "your/BTB/path"
orig_sr = 2048

subject_id = 1
trial_id = 0

neural_data_file = os.path.join(root_dir, f'sub_{subject_id}_trial{trial_id:03}_preprocessed.h5')
with h5py.File(neural_data_file, 'r') as f:
    print(f['data']['electrode_0'])
    print(f.keys())
    print(len(f['data']))
    for key, data in f['data'].items():
        print(key, data.shape)

neural_data_file = os.path.join(root_dir, f'sub_{subject_id}_trial{trial_id:03}.h5')
with h5py.File(neural_data_file, 'r') as f:
    print(f['data']['electrode_0'])
    print(f.keys())
    print(len(f['data']))
    for key, data in f['data'].items():
        print(key, data.shape)

def preprocessing_and_save(subject_id, trial_id):
    print("===============================================")
    print(f"Processing subject {subject_id}, trial {trial_id}")
    neural_data_file = os.path.join(root_dir, f'sub_{subject_id}_trial{trial_id:03}.h5')
    preprocessed_data_file = os.path.join(root_dir, f'sub_{subject_id}_trial{trial_id:03}_preprocessed.h5')
    with h5py.File(neural_data_file, 'r') as f:
        preprocessed_data = {}
        neural_data = f['data']
        electrode_key_list =  list(neural_data.keys())
        for elec_key in electrode_key_list:
            print(elec_key)
            elec_data = neural_data[elec_key][:]
            freqs_to_notch = [60, 120, 180]
            filtered_data = filter_data(elec_data, sfreq=orig_sr, l_freq=0.5, h_freq=None, verbose=False)
            elec_data_notched = notch_filter(filtered_data, Fs=orig_sr, freqs=freqs_to_notch, verbose=False)
            preprocessed_data[elec_key] = elec_data_notched
    with h5py.File(preprocessed_data_file, 'w') as f:
        data_group = f.create_group('data')
        for elec_key, elec_data in preprocessed_data.items():
            data_group.create_dataset(elec_key, data=elec_data, compression="gzip")
    print(f"Preprocessed data saved to {preprocessed_data_file}")    
        
lite_list = [[1,1],[1,2],[2,0],[2,4],[3,0],[3,1],[4,0],[4,1],[7,0],[7,1],[10,0],[10,1]]
for lite in lite_list:
        subject_id = lite[0]
        trial_id = lite[1]
        preprocessing_and_save(subject_id, trial_id)


