# Run all commands in one shell
.ONESHELL:-

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H:%M:%S")

# To run locally
# CMD = python
# To batch on slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

neural-conv:
	mkdir -p logs
	$(CMD) main.py \
		--config configs/neural_conv_decoder/neural_conv_decoder.yml

foundation-model:
	mkdir -p logs
	$(CMD) main.py \
		--config configs/foundation_model/foundation_model.yml
