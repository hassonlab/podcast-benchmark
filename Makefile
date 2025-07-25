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

population-transformer:
	mkdir -p logs
	$(CMD) main.py \
		--config "configs/population_transformer/population_transformer$(CONFIG_SUFFIX).yml"

population-transformer-frozen:
	$(MAKE) population-transformer CONFIG_SUFFIX=_frozen

population-transformer-finetune:
	$(MAKE) population-transformer CONFIG_SUFFIX=_finetune

population-transformer-base:
	$(MAKE) population-transformer CONFIG_SUFFIX=_base

population-transformer-cpu:
	$(MAKE) population-transformer CONFIG_SUFFIX=_cpu

transform-checkpoints:
	@for model in $(MODEL_CHECKPOINT_NAMES); do \
		python temp_checkpoint_override.py --input=$(MODEL_DIR_PATH)$$model/best_checkpoint --output=$(MODEL_DIR_PATH)$$model/best_checkpoint/checkpoint.pth; \
	done
