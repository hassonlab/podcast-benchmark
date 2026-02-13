# DIVER-1
EPHYS FOUNDATION MODEL 

## SETUP
- Before finetuning, set the following requirements. 
```bash
conda create --name my-env-name --file requirements.txt 
conda activate my-env-name
```
## PREPROCESSING
- We have implemented our own preprocessing methods for the downsteam task datasets, therefore to utilize our model, use our preprocessing files. 
- Standard elc files for preprocessing are from https://github.com/fieldtrip/fieldtrip/blob/master/template/electrode/standard_1005.elc

## FINETUNE 
- Example scripts are currently implemented on the following downstream tasks; Neuroprobe, FACED, Physionet-MI, MentalArithmetic.
- For Neuroprobe downstream tasks, cloning the Neuroprobe's github is needed; (https://github.com/insight-neuro/neuroprobe/tree/main). 
After cloning the directory, paste this code in to the ./datasets/datasets_loaders.py file.
```python
#datasets/datasets_loaders.py 
import sys
module_path = "/your/path/to/neuroprobe"
if module_path not in sys.path:
    sys.path.append(module_path)
import neuroprobe
```

## WEIGHTS 
- pretrained weights for iEEG downstream tasks: ``` ./weights/ieeg_pretrained_weights.pt```
- pretrained weights for EEG downstream tasks: ``` ./weights/i_eeg_pretrained_weights.pt```
