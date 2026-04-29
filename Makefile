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

SUBJECTS ?= 1 3 4 5 6 7 8 9
REGIONS ?= EAC MTG ITG TP IFG TPJ PRC PC RIGHT
SUBJECT_GROUPS ?=
REGION_GROUPS ?=
SUBJECT_BATCH_SIZE ?=
REGION_BATCH_SIZE ?=
TASKS ?= word_embedding_decoding_task,sentence_onset_task,gpt_surprise_task,gpt_surprise_multiclass_task,content_noncontent_task,pos_task,llm_decoding_task,whisper_embedding_decoding_task,iu_boundary_task,volume_level_decoding_task
CONFIG_OVERRIDES ?=

# Specify a single config to use.
# Usage:
#   make train-config CONFIG=configs/baselines/word_embedding_decoding_task/neural_conv_decoder_gpt2_supersubject.yml
#   make train-config CONFIG=configs/baselines/sentence_onset_task/neural_conv_decoder_per_subject.yml CONFIG_OVERRIDES='--training_params.epochs=5'
train-config:
	@if [ -z "$(CONFIG)" ]; then \
		echo "CONFIG is required"; \
		exit 1; \
	fi
	mkdir -p logs
	$(CMD) main.py \
		--config "$(CONFIG)" \
		$(CONFIG_OVERRIDES) \
		$(OVERRIDES)

# Train all configs from training_matrix.yaml, optionally filtered by MODELS and/or TASKS
# Usage:
#   make train-all                                                    # Run all configs
#   make train-all MODELS=baselines/neural_conv_decoder               # Run only neural conv baseline configs
#   make train-all TASKS=sentence_onset_task                          # Run only sentence_onset_task configs
#   make train-all CONFIG_OVERRIDES='--training_params.epochs=5'       # Apply overrides to every config
#   make train-all MODELS=baselines/neural_conv_decoder,baselines/time_pooling_model TASKS=word_embedding_decoding_task
# To run locally instead of with slurm, comment out CMD in this file to use python directly
train-all:
	@mkdir -p logs
	@echo "Generating training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		job_name="$(PREFIX)-$$config_tag-$(USR)-$(DT)"; \
		echo "Submitting: $$model / $$task / $$config"; \
		JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config"; \
	done

# Train each supersubject config one job per config.
# Usage:
#   make train-all-supersubjects                         # Uses TASKS default
#   make train-all-supersubjects TASKS=sentence_onset_task
#   make train-all-supersubjects MODELS=baselines/neural_conv_decoder
train-all-supersubjects:
	@mkdir -p logs
	@echo "Generating supersubject training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg | grep '_supersubject\.yml'); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		job_name="$(PREFIX)-$$config_tag-$(USR)-$(DT)"; \
		echo "Submitting: $$model / $$task / $$config"; \
		JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config"; \
	done

# Train each per-subject config one subject per job.
# Usage:
#   make train-all-per-subjects                         # Uses SUBJECTS and TASKS defaults
#   make train-all-per-subjects SUBJECTS="1 3 4"
#   make train-all-per-subjects MODELS=baselines/neural_conv_decoder TASKS=sentence_onset_task
train-all-per-subjects:
	@mkdir -p logs
	@echo "Generating per-subject training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg | grep '_per_subject\.yml'); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		is_multi=$$(python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print("1" if isinstance(cfg, dict) and "tasks" in cfg else "0")' "$$config"); \
		subjects="$(SUBJECTS)"; \
		for subject in $$subjects; do \
			if [ "$$is_multi" = "1" ]; then \
				overrides="--shared_params.task_config.data_params.subject_ids=[$$subject]"; \
			else \
				overrides="--task_config.data_params.subject_ids=[$$subject]"; \
			fi; \
			job_name="$(PREFIX)-$$config_tag-sub$$subject-$(USR)-$(DT)"; \
			echo "Submitting: $$model / $$task / $$config / subject $$subject"; \
			JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config" OVERRIDES="$$overrides"; \
		done; \
	done

# Train each per-subject config with subjects batched into one job per group.
# Groups may be a Python/JSON-like nested list or semicolon-separated groups.
# If SUBJECT_GROUPS is empty and SUBJECT_BATCH_SIZE is set, SUBJECTS is chunked.
# Usage:
#   make train-all-subject-groups SUBJECT_GROUPS='[[1, 3, 4], [5, 6], [7, 8, 9]]'
#   make train-all-subject-groups SUBJECT_GROUPS='1 3 4;5 6;7 8 9'
#   make train-all-subject-groups SUBJECT_BATCH_SIZE=3
#   make train-all-subject-groups MODELS=baselines/neural_conv_decoder TASKS=sentence_onset_task
train-all-subject-groups:
	@mkdir -p logs
	@echo "Generating grouped per-subject training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	subject_specs=$$(python scripts/format_make_groups.py --groups '$(SUBJECT_GROUPS)' --items '$(SUBJECTS)' --batch-size '$(SUBJECT_BATCH_SIZE)' --kind int --tag-prefix sub); \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg | grep '_per_subject\.yml'); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		is_multi=$$(python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print("1" if isinstance(cfg, dict) and "tasks" in cfg else "0")' "$$config"); \
		for subject_spec in $$subject_specs; do \
			subject_group=$${subject_spec%%|*}; \
			subject_tag=$${subject_spec#*|}; \
			if [ "$$is_multi" = "1" ]; then \
				overrides="--shared_params.task_config.data_params.subject_ids=$$subject_group"; \
			else \
				overrides="--task_config.data_params.subject_ids=$$subject_group"; \
			fi; \
			job_name="$(PREFIX)-$$config_tag-$$subject_tag-$(USR)-$(DT)"; \
			echo "Submitting: $$model / $$task / $$config / subjects $$subject_group"; \
			JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config" OVERRIDES="$$overrides"; \
		done; \
	done

train-all-per-subject-groups: train-all-subject-groups

# Train each per-region config one atlas region per job.
# Usage:
#   make train-all-per-regions                          # Uses REGIONS and TASKS defaults
#   make train-all-per-regions REGIONS="EAC MTG RIGHT"
#   make train-all-per-regions MODELS=baselines/neural_conv_decoder TASKS=sentence_onset_task
train-all-per-regions:
	@mkdir -p logs
	@echo "Generating per-region training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	regions="$(REGIONS)"; \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg | grep '_per_region\.yml'); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		is_multi=$$(python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print("1" if isinstance(cfg, dict) and "tasks" in cfg else "0")' "$$config"); \
		for region in $$regions; do \
			if [ "$$is_multi" = "1" ]; then \
				overrides="--shared_params.regions=['$$region']"; \
			else \
				overrides="--regions=['$$region']"; \
			fi; \
			job_name="$(PREFIX)-$$config_tag-$$region-$(USR)-$(DT)"; \
			echo "Submitting: $$model / $$task / $$config / region $$region"; \
			JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config" OVERRIDES="$$overrides"; \
		done; \
	done

# Train each per-region config with atlas regions batched into one job per group.
# Groups may be a Python/JSON-like nested list or semicolon-separated groups.
# If REGION_GROUPS is empty and REGION_BATCH_SIZE is set, REGIONS is chunked.
# Usage:
#   make train-all-region-groups REGION_GROUPS='[[EAC, ITG, MTG], [IFG, TP, TPJ], [PRC, PC, RIGHT]]'
#   make train-all-region-groups REGION_GROUPS='EAC ITG MTG;IFG TP TPJ;PRC PC RIGHT'
#   make train-all-region-groups REGION_BATCH_SIZE=3
#   make train-all-region-groups MODELS=baselines/neural_conv_decoder TASKS=sentence_onset_task
train-all-region-groups:
	@mkdir -p logs
	@echo "Generating grouped per-region training targets..."
	@models_arg=""; \
	if [ -n "$(MODELS)" ]; then models_arg="--models $(MODELS)"; fi; \
	tasks_arg=""; \
	if [ -n "$(TASKS)" ]; then tasks_arg="--tasks $(TASKS)"; fi; \
	region_specs=$$(python scripts/format_make_groups.py --groups '$(REGION_GROUPS)' --items '$(REGIONS)' --batch-size '$(REGION_BATCH_SIZE)' --kind str); \
	for target in $$(python scripts/generate_training_targets.py $$models_arg $$tasks_arg | grep '_per_region\.yml'); do \
		model=$$(echo $$target | cut -d'|' -f1); \
		task=$$(echo $$target | cut -d'|' -f2); \
		config=$$(echo $$target | cut -d'|' -f3); \
		config_tag=$$(basename $$config .yml); \
		is_multi=$$(python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print("1" if isinstance(cfg, dict) and "tasks" in cfg else "0")' "$$config"); \
		for region_spec in $$region_specs; do \
			region_group=$${region_spec%%|*}; \
			region_tag=$${region_spec#*|}; \
			if [ "$$is_multi" = "1" ]; then \
				overrides="--shared_params.regions=$$region_group"; \
			else \
				overrides="--regions=$$region_group"; \
			fi; \
			job_name="$(PREFIX)-$$config_tag-region$$region_tag-$(USR)-$(DT)"; \
			echo "Submitting: $$model / $$task / $$config / regions $$region_group"; \
			JOB_NAME="$$job_name" $(MAKE) --no-print-directory train-config CONFIG="$$config" OVERRIDES="$$overrides"; \
		done; \
	done

train-all-per-region-groups: train-all-region-groups

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

.PHONY: setup setup-gpu setup-dev setup-all test-env test clean-env train-config train-all train-all-supersubjects train-all-per-subjects train-all-subject-groups train-all-per-subject-groups train-all-per-regions train-all-region-groups train-all-per-region-groups
