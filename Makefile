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


# To activate the environment later, run: source decoding_env/bin/activate
#--config "configs/neural_conv_decoder/neural_conv_decoder$(CONFIG_SUFFIX).yml"

FROZEN := frozen_attention frozen_linear
FINETUNE := finetune_attention finetune

FOUNDATION_MODEL_RUNS := nll_finetune

MODEL_DIR_PATH := /scratch/gpfs/zparis/ECoG-foundation-pretraining/checkpoints/
# MODEL_CHECKPOINT_NAMES := \
#   model=patch_dim_1_large-grad_accum=2-encoder_mask_ratio=0.5_29115592 \
#   model=patch_dim_1_medium-grad_accum=2-encoder_mask_ratio=0.5_29115507 \
#   model=patch_dim_1_small-grad_accum=2-encoder_mask_ratio=0.5_29115501 \
#   model=patch_dim_2_large-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
#   model=patch_dim_2_medium-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
#   model=patch_dim_2_small-grad_accum=2-encoder_mask_ratio=0.5_29115813

# MODEL_CHECKPOINT_NAMES := \
#   model=patch_dim_2_small-grad_accum=2-encoder_mask_ratio=0.5_29115813 \
#   model=patch_dim_1_small-grad_accum=2-encoder_mask_ratio=0.5_29115501 \
#   model=patch_dim_2_medium-grad_accum=2-encoder_mask_ratio=0.5_29115813

# MODEL_CHECKPOINT_NAMES := \
#   model=patch_dim_1_medium-grad_accum=2-encoder_mask_ratio=0.5_29115507 \
#   model=patch_dim_2_large-grad_accum=2-encoder_mask_ratio=0.5_29115813

MODEL_CHECKPOINT_NAMES := \
	model=test_model_small_short-encdr_mask=0.25-new_fs=64-bs=64-grad-acc=4-alpha=0.8-datasize=1.0-decoder=1.0-lr=5.00e-04-noreg_29201634

GRAD_ACCUM_STEPS := 1 2

RUN_NAME := model=test_model_large-encdr_mask=0.25-new_fs=64-bs=64-grad-acc=4-alpha=0.8-datasize=1.0-decoder=1.0-noreg_29189824
MODEL_DIR := $(MODEL_DIR_PATH)/$(RUN_NAME)/best_checkpoint

LAG_STEP_SIZE = 200
LAG_MINS = -2000 -1400 -800 -200 400
LAG_MAXES = -1400 -800 -200 400 1100

DECODE_FROM_LAYERS = 3 7 11 15 19 23


run-foundation-model-layers:
	@for layer in $(DECODE_FROM_LAYERS); do \
		echo "Running model from encoder layer=$$layer"; \
		$(MAKE) foundation-model-override PREFIX=$$layer_layer DECODE_LAYER=$$layer; \
	done


run-foundation-model-lags:
	@$(foreach i,$(shell seq 1 $(words $(LAG_MINS))), \
		$(eval MIN_LAG = $(word $(i),$(LAG_MINS))) \
		$(eval MAX_LAG = $(word $(i),$(LAG_MAXES))) \
		$(eval CURRENT_PREFIX = run_$(MIN_LAG)_$(MAX_LAG)) \
		$(MAKE) foundation-model-lags PREFIX=$(CURRENT_PREFIX) MIN_LAG=$(MIN_LAG) MAX_LAG=$(MAX_LAG); \
	)

run-foundation-model-many:
	@for model in $(MODEL_CHECKPOINT_NAMES); do \
	  for config in $(FOUNDATION_MODEL_RUNS); do \
	    echo "Running model=$$model with config=$$config"; \
	    $(MAKE) foundation-model-override PREFIX=$$config CONFIG_SUFFIX=_$$config MODEL_DIR=$(MODEL_DIR_PATH)$$model/best_checkpoint; \
	  done; \
	done

run-foundation-model-grad:
	@for model in $(MODEL_CHECKPOINT_NAMES); do \
	  for steps in $(GRAD_ACCUM_STEPS); do \
	    echo "Running model=$$model with steps=$$steps"; \
	    $(MAKE) foundation-model-override PREFIX=nll_$$model CONFIG_SUFFIX=_nll_finetune MODEL_DIR=$(MODEL_DIR_PATH)$$model/best_checkpoint GRAD_ACCUM=$$steps; \
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
	$(CMD) main.py --config "configs/neural_conv_decoder/neural_conv_decoder_content_noncontent.yml"
# 	$(CMD) main.py --config "configs/neural_conv_decoder/neural_conv_decoder_pos.yml"
		

foundation-model-lags:
	mkdir -p logs
	$(CMD) main.py \
		--config "configs/foundation_model/foundation_model$(CONFIG_SUFFIX).yml" \
		--model_params.model_dir="$(MODEL_DIR)" \
		--training_params.min_lag=$(MIN_LAG) \
		--training_params.max_lag=$(MAX_LAG) \
		--training_params.lag_step_size=$(LAG_STEP_SIZE)

foundation-model-override:
	mkdir -p logs
		$(CMD) main.py \
			--config "configs/foundation_model/foundation_model$(CONFIG_SUFFIX).yml" \
			--model_params.model_dir="$(MODEL_DIR)" \
			--model_params.decode_from_layer=$(DECODE_LAYER)
# --training_params.grad_accumulation_steps=$(GRAD_ACCUM)

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
	@if [ -f ~/miniconda3/etc/profile.d/conda.sh ] || [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then \
		if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then \
			source ~/miniconda3/etc/profile.d/conda.sh; \
		elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then \
			source ~/anaconda3/etc/profile.d/conda.sh; \
		fi; \
		if conda env list | grep -q "^test_env "; then \
			conda activate test_env && python -m pytest tests/ -v; \
		elif conda env list | grep -q "^decoding_env "; then \
			conda activate decoding_env && python -m pytest tests/ -v; \
		elif [ -d "test_env" ]; then \
			source test_env/bin/activate && python -m pytest tests/ -v; \
		elif [ -d "decoding_env" ]; then \
			source decoding_env/bin/activate && python -m pytest tests/ -v; \
		else \
			echo "No virtual environment found. Run 'make test-env' or 'make setup-dev' first."; \
			exit 1; \
		fi; \
	elif [ -d "test_env" ]; then \
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
