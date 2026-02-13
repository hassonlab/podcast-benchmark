from datasets.task_attributes import FINAL_TASK_DICT

def task_dict(params):
    downstream_task_info = FINAL_TASK_DICT[params.downstream_dataset]
    downstream_task_info['patch_sampling_rate'] = params.patch_sampling_rate
    if params.downstream_dataset == "Neuroprobe": 
        if params.neuroprobe_evaltype in ['ssdm','sssm']: 
            downstream_task_info['consistent_channels'] = True
            if not isinstance(downstream_task_info['num_channels'], int):
                if params.neuroprobe_lite:  
                    if params.neuroprobe_allow_corrupted == False:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['lite_false'][f'sub_{params.neuroprobe_subject}']
                    if params.neuroprobe_allow_corrupted == True:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['lite_true'][f'sub_{params.neuroprobe_subject}']
                else:
                    if params.neuroprobe_allow_corrupted == False:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['full_false'][f'sub_{params.neuroprobe_subject}']
                    if params.neuroprobe_allow_corrupted == True:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['full_true'][f'sub_{params.neuroprobe_subject}']
      
        elif params.neuroprobe_evaltype == 'dsdm':
            downstream_task_info['consistent_channels'] = False
            if not isinstance(downstream_task_info['num_channels'], int):
                if params.neuroprobe_lite:
                    if params.neuroprobe_allow_corrupted == False:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['lite_false']
                    if params.neuroprobe_allow_corrupted == True:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['lite_true']
                else:
                    if params.neuroprobe_allow_corrupted == False:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['full_false']
                    if params.neuroprobe_allow_corrupted == True:
                        downstream_task_info['num_channels'] = downstream_task_info['num_channels']['full_true']
        else:
            raise ValueError("neuroprobe_evaltype must in ssdm, sssm, dsdm ")
    return downstream_task_info