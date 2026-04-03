# 1. env setup 
## Solution: create a local venv manually

```bash
cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark

# Data + GloVe should already be downloaded from a prior setup.sh run.
# If not, run: ./setup.sh  (it will fail after downloading data, which is fine)

# Create venv on scratch (avoids home quota issues)
python3 -m venv decoding_env
source decoding_env/bin/activate
pip install -e "."          # base deps only
# pip install -e ".[gpu]"   # if you need bundled CUDA libs (usually not needed on Perlmutter)
```

1-1. try with podcast main code.
source decoding_env/bin/activate
python main.py --config configs/baselines/neural_conv_decoder/glove.yml

Below is for reference.
```
source /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark/decoding_env/bin/activate
cd /pscratch/sd/a/ahhyun/EcoGFound/PODCAST/podcast-benchmark
python main.py --config configs/baselines/neural_conv_decoder/glove.yml
```
Or for a CPU-only test without needing a compute node, you can force CPU:
```
CUDA_VISIBLE_DEVICES="" python main.py --config configs/baselines/neural_conv_decoder/glove.yml
```


# 2. generate NERSC style script
- `/commands/run-local.sh` : login node / inside interactive node
- `/commands/submit-task-nersc.sh`
- `/commands/debug/submit-task-nersc.sh` ; debug version submit task

Test by below
```
cd models/brainbert/pretrained_model && python generate_checkpoint.py
./commands/run-local.sh --model brainbert --task content_noncontent --variant supersubject --epochs 1

./commands/run-local.sh --model popt --task content_noncontent --variant supersubject --epochs 1
./commands/run-local.sh --model diver --task content_noncontent --variant supersubject --epochs 1
```

cf) get interactive node.
salloc -A m5187_g -C gpu -q shared_interactive -N 1 --gpus=1 -t 00:30:00