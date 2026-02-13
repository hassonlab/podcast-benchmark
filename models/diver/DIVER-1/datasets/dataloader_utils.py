import torch

def collate_fn_for_data_info_finetuning(batch):
    patches, labels, data_infos = zip(*batch)
    patches = torch.stack(patches, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return patches, labels, data_infos