# Run all commands in one shell
.ONESHELL:-

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H:%M:%S")

PREFIX = decoder-training
JOB_NAME ?= "$(PREFIX)-$(USR)-$(DT)"

# To run locally
# CMD = python
# To batch on slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

# Specify model and a single config to use.
# Usage:
#   make train MODEL_NAME=neural_conv_decoder CONFIG=neural_conv_decoder_base.yml
train-config:
	mkdir -p logs
	$(CMD) main.py \
		--config "configs/$(MODEL_NAME)/$(CONFIG)"

# Train all configs from training_matrix.yaml, optionally filtered by MODELS and/or TASKS
# Usage:
#   make train-all                                                    # Run all configs
#   make train-all MODELS=example_foundation_model                    # Run only example_foundation_model configs
#   make train-all TASKS=sentence_onset_task                          # Run only sentence_onset_task configs
#   make train-all MODELS=example_foundation_model,neural_conv_decoder TASKS=word_embedding_decoding_task
# To run locally instead of with slurm, comment out CMD in this file to use python directly
train-all:
	@mkdir -p logs
	@echo "Generating training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg 2>&1 | grep -v "^Generated" | grep -v "^Warning" | grep -v "Models:" | grep -v "Tasks:"); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		job_name="$(PREFIX)-$$config_tag-$(USR)-$(DT)"; \
		echo "Submitting: $$model / $$task / $$config"; \
		JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config MODEL_NAME="$$model" CONFIG="$$config"; \
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

.PHONY: setup setup-gpu setup-dev setup-all test-env test clean-env train-config train-all
