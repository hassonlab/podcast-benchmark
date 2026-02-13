torcheeg_task_dict = {
    "FACED" : {
        "task_type": "classification",
        "target_dynamics": "discrete",  
        "consistent_channels": True,  
        "num_channels": 32,
        "num_seconds": 10,  
        "num_targets": 9,  
    },
    "PhysioNet-MI": {
        "task_type": "classificatidon",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 64,
        "num_seconds": 4,
        "num_targets": 4,
    }
}

add_task_dict = {
    "SEED_V": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 62,
        "num_seconds": 1,
        "num_targets": 5,
    },
    "SHU_MI": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 32,
        "num_seconds": 4,
        "num_targets": 2,
    },
    "ISRUC": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 6,
        "num_seconds": 30,
        "num_targets": 5,
    },
    "CHB_MIT": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 16,
        "num_seconds": 10,
        "num_targets": 2,
    },
    "BCIC2020_3": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 64,
        "num_seconds": 3,
        "num_targets": 5,
    },
    "Mumtaz2016": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 19,
        "num_seconds": 5,
        "num_targets": 2,
    },
    "SEED_VIG": {
        "task_type": "regression",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 17,
        "num_seconds": 8,
        "num_targets": 1,
    },
    "MentalArithmetich": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 19,
        "num_seconds": 5,
        "num_targets": 2,
    },
    "TUEV": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 16,
        "num_seconds": 5,
        "num_targets": 6,
    },
    "TUAB": {
        "task_type": "classification",
        "target_dynamics": "discrete",
        "consistent_channels": True,
        "num_channels": 16,
        "num_seconds": 10,
        "num_targets": 2,
    }
    
    
}

neuroprobe_channel_dict_allow_corrupted_full_True = {
    "sub_1": 153,
    "sub_2": 162,
    "sub_3": 122,
    "sub_4": 186,
    "sub_5": 156,
    "sub_6": 164,
    "sub_7": 244,
    "sub_8": 160,
    "sub_9": 103,
    "sub_10": 214
}

neuroprobe_channel_dict_allow_corrupted_full_False= {
    "sub_1": 129,
    "sub_2": 135,
    "sub_3": 112,
    "sub_4": 183,
    "sub_5": 140,
    "sub_6": 161,
    "sub_7": 238,
    "sub_8": 151,
    "sub_9": 96,
    "sub_10": 205
}

neuroprobe_channel_dict_allow_corrupted_False = {
    "sub_1": 120, 
    "sub_2": 119,
    "sub_3": 109, 
    "sub_4": 120, 
    "sub_7": 119,
    "sub_10": 120
}

neuroprobe_channel_dict_allow_corrupted_True= {
    "sub_1": 120,
    "sub_2": 120, 
    "sub_3": 120,
    "sub_4": 120,
    "sub_7": 120,
    "sub_10": 120
}


common_neuroprobe_setting = {
    "task_type": "classification",
    "target_dynamics": "discrete",
    "consistent_channels": False,
    "num_channels": {'lite_false' : neuroprobe_channel_dict_allow_corrupted_False, 'lite_true' : neuroprobe_channel_dict_allow_corrupted_True,
                     'full_false' : neuroprobe_channel_dict_allow_corrupted_full_False, 'full_true' : neuroprobe_channel_dict_allow_corrupted_full_True},
    "num_seconds": 1,
    "num_targets": 2,
}

neuroprobe_task_dict = {
    "NEUROPROBE_volume": common_neuroprobe_setting,
    "NEUROPROBE_delta_volume": common_neuroprobe_setting,
    "NEUROPROBE_onset": common_neuroprobe_setting,
    "NEUROPROBE_speech": common_neuroprobe_setting,
    "Neuroprobe": common_neuroprobe_setting
}

podcast_channel_dict = {
    "sub_1": 99,
    "sub_2": 90,
    "sub_3": 235,
    "sub_4": 143,
    "sub_5": 159,
    "sub_6": 166,
    "sub_7": 116,
    "sub_8": 72,
    "sub_9": 188,
}

common_podcast_setting = {
    "task_type": "regression",
    "target_dynamics": "discrete",
    "target_sampling_rate" : 250, 
    "consistent_channels": True, 
    "num_channels": 188, 
    "num_seconds" : 1,
    "num_targets": 50
}

podcast_task_dict = {
    "Podcast": common_podcast_setting
}

for task_dict in [torcheeg_task_dict, add_task_dict]:
    for task in task_dict:
        task_dict[task]["modality"] = "EEG"
        
for task in [neuroprobe_task_dict, podcast_task_dict]:
    for task in task_dict:
        task_dict[task]["modality"] = "iEEG"

EEG_TASK_DICT = {**torcheeg_task_dict, **add_task_dict}
iEEG_TASK_DICT = {**neuroprobe_task_dict, **podcast_task_dict}


FINAL_TASK_DICT = {
    **EEG_TASK_DICT,
    **iEEG_TASK_DICT,    
}
