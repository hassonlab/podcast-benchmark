# Run all commands in one shell
.ONESHELL:-

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H:%M:%S")

PREFIX = decoder-training
JOB_NAME = "$(PREFIX)-$(USR)-$(DT)"

# To run locally
# CMD = python
# To batch on slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

FROZEN := frozen_attention frozen_linear
FINETUNE := finetune_attention finetune

FOUNDATION_MODEL_RUNS := finetune frozen_linear

MODEL_DIR_PATH := /scratch/gpfs/zparis/ECoG-foundation-pretraining/checkpoints/
# MODEL_CHECKPOINT_NAMES := \
#   model=patch_dim_1_large-grad_accum=2-encoder_mask_ratio=0.5_29115592 \
#   model=patch_dim_1_medium-grad_accum=2-encoder_mask_ratio=0.5_29115507 \
#   model=patch_dim_1_small-grad_accum=2-encoder_mask_ratio=0.5_29115501 \
#   model=patch_dim_2_large-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
#   model=patch_dim_2_medium-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
#   model=patch_dim_2_small-grad_accum=2-encoder_mask_ratio=0.5_29115813

MODEL_CHECKPOINT_NAMES := \
  model=patch_dim_2_small-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
  model=patch_dim_1_small-grad_accum=2-encoder_mask_ratio=0.5_29115501
  

run-foundation-model-many:
	@for model in $(MODEL_CHECKPOINT_NAMES); do \
	  for config in $(FOUNDATION_MODEL_RUNS); do \
	    echo "Running model=$$model with config=$$config"; \
	    $(MAKE) foundation-model-override PREFIX=$$config CONFIG_SUFFIX=_$$config MODEL_DIR=$(MODEL_DIR_PATH)$$model/best_checkpoint; \
	  done; \
	done

foundation-model-frozen:
	for item in $(FROZEN); do \
		$(MAKE) foundation-model PREFIX=$$item CONFIG_SUFFIX=_$$item; \
	done

foundation-model-finetune:
	for item in $(FINETUNE); do \
		$(MAKE) foundation-model PREFIX=$$item CONFIG_SUFFIX=_$$item; \
	done

neural-conv:
	mkdir -p logs
	$(CMD) main.py \
		--config "configs/neural_conv_decoder/neural_conv_decoder$(CONFIG_SUFFIX).yml"

foundation-model-override:
	mkdir -p logs
		$(CMD) main.py \
			--config "configs/foundation_model/foundation_model$(CONFIG_SUFFIX).yml" \
			--model_params.model_dir="$(MODEL_DIR)"

foundation-model:
	mkdir -p logs
	$(CMD) main.py \
		--config "configs/foundation_model/foundation_model$(CONFIG_SUFFIX).yml"


transform-checkpoints:
	@for model in $(MODEL_CHECKPOINT_NAMES); do \
		python temp_checkpoint_override.py --input=$(MODEL_DIR_PATH)$$model/best_checkpoint --output=$(MODEL_DIR_PATH)$$model/best_checkpoint/checkpoint.pth; \
	done

# Development and testing targets
setup:
	./setup.sh

setup-gpu:
	./setup.sh --gpu

setup-dev:
	./setup.sh --dev

setup-all:
	./setup.sh --gpu --dev

test-env:
	./setup.sh --dev --env-name test_env

test:
	@if [ -d "test_env" ]; then \
		source test_env/bin/activate && python -m pytest tests/ -v; \
	elif [ -d "decoding_env" ]; then \
		source decoding_env/bin/activate && python -m pytest tests/ -v; \
	else \
		echo "No virtual environment found. Run 'make test-env' or 'make setup-dev' first."; \
		exit 1; \
	fi

clean-env:
	rm -rf decoding_env test_env

.PHONY: setup setup-gpu setup-dev setup-all test-env test clean-env
