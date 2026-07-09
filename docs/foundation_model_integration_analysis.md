# Foundation Model Integration Analysis

## Scope

This document records the current analysis and follow-up TODO for integrating the model-specific implementation that exists in `references_podcast/podcast-benchmark` into the latest root codebase at `podcast-benchmark`.

The root codebase is the source of truth. The reference code is used only as an implementation source for:

- `brainbert`
- `popt`
- `diver`
- selected `diverclip` execution settings that should be folded into `diver`

`diverclip` itself is not planned as a separate model family in the root codebase.

## High-Level Findings

### 1. The root framework is structurally newer than the reference implementation

The latest root benchmark has already moved to a newer execution model:

- task registration uses `task_registry`
- model construction uses `ModelSpec`
- model-specific DataFrame enrichment uses `model_data_getter`
- task config is nested under `task_config`
- multi-stage runs are supported via `MultiTaskConfig`

Relevant files:

- [`main.py`](/storage/connectome/jmhan/podcast-benchmark/main.py)
- [`core/config.py`](/storage/connectome/jmhan/podcast-benchmark/core/config.py)
- [`core/registry.py`](/storage/connectome/jmhan/podcast-benchmark/core/registry.py)

This means the integration target is not "copy reference code as-is", but "port reference functionality into the root execution model".

### 2. BrainBERT and PopT in root currently contain placeholder/example-style integration

The root implementations of `brainbert` and `popt` are not faithful ports of the reference integration. They are closer to an example foundation-model onboarding pattern than the reference wrappers used in the old benchmark.

Relevant files:

- [`models/brainbert/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/brainbert/integration.py)
- [`models/popt/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/popt/integration.py)

By contrast, the reference versions are tied to the actual intended upstream wrappers and checkpoint handling:

- [`references_podcast/podcast-benchmark/models/brainbert/integration.py`](/storage/connectome/jmhan/references_podcast/podcast-benchmark/models/brainbert/integration.py)
- [`references_podcast/podcast-benchmark/models/popt/integration.py`](/storage/connectome/jmhan/references_podcast/podcast-benchmark/models/popt/integration.py)

### 3. DIVER in root is partially ported, but not fully aligned with the reference setup

The root `diver` implementation already adopts the new framework better than BrainBERT and PopT:

- it registers a `model_data_getter`
- it works with `task_config`
- it injects `xyz_id` into the task DataFrame for DIVER input

Relevant file:

- [`models/diver/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/diver/integration.py)

However, the reference tree still contains practical setup that is not fully represented in root:

- pretrained checkpoint variants
- execution/config conventions stabilized under `diverclip`
- strategy-based model path selection

Relevant reference files:

- [`references_podcast/podcast-benchmark/models/diver/integration.py`](/storage/connectome/jmhan/references_podcast/podcast-benchmark/models/diver/integration.py)
- [`references_podcast/podcast-benchmark/models/diverclip/integration.py`](/storage/connectome/jmhan/references_podcast/podcast-benchmark/models/diverclip/integration.py)
- [`references_podcast/podcast-benchmark/models/diverclip/model_registry.py`](/storage/connectome/jmhan/references_podcast/podcast-benchmark/models/diverclip/model_registry.py)

### 4. Diverclip should be treated as a configuration and initialization source, not a new root model

The current desired direction is:

- do not re-introduce `diverclip` as its own root model family
- reuse `diverclip`'s practical execution settings inside root `diver`

This means the useful part of `diverclip` is:

- checkpoint selection logic
- stable initialization strategies
- execution-time options known to work under the benchmark loop

This is not the same as restoring:

- `diverclip`
- `diverclip_flatten`
- `diverclip_attn`

as separate constructors.

### 5. LLM decoding is already integrated into the root benchmark

The root benchmark already has a native LLM decoding task:

- [`tasks/llm_decoding.py`](/storage/connectome/jmhan/podcast-benchmark/tasks/llm_decoding.py)

The reference benchmark used a separate `llm_decoding_pipeline`, but that separation is no longer desirable. The root task system should remain the only task pipeline. Only model-specific LLM config setters and constructors should be ported if needed.

### 6. The current root framework already provides most of the hooks needed for an integration.py-centered port

The latest root benchmark already has the two key extension points needed for a low-diff port:

- model-specific extra inputs can be injected through `model_data_getter`
- neural preprocessing can be injected through `preprocessing_fn_name`

Relevant files:

- [`main.py`](/storage/connectome/jmhan/podcast-benchmark/main.py)
- [`utils/decoding_utils.py`](/storage/connectome/jmhan/podcast-benchmark/utils/decoding_utils.py)
- [`utils/data_utils.py`](/storage/connectome/jmhan/podcast-benchmark/utils/data_utils.py)
- [`core/registry.py`](/storage/connectome/jmhan/podcast-benchmark/core/registry.py)

This is important because it means the intended port does not need to start by rewriting shared utilities.

Specifically:

- LIP or MNI coordinates can be added to the task DataFrame via model-specific getters
- those added columns are automatically appended to `input_fields`
- the batched tensors are then passed into model `forward(**kwargs)` automatically
- STFT can be handled through a model-registered preprocessor instead of rewriting `utils/data_utils.py`

### 7. Shared utility edits should be avoided unless a concrete runtime blocker appears

Current preference for the port:

- modify `models/brainbert/integration.py`
- modify `models/popt/integration.py`
- modify `models/diver/integration.py`
- allow a small targeted extension to [`core/config.py`](/storage/connectome/jmhan/podcast-benchmark/core/config.py) if YAML support requires it

Avoid by default:

- `utils/data_utils.py`
- `utils/decoding_utils.py`
- `metrics/classification_metrics.py`
- `models/shared_preprocessors.py`

These files should only be changed if a specific required reference behavior cannot be expressed using the existing integration hooks.

### 8. Not every gap in the reference implementation needs to be repaired

The goal is not to improve the reference codebase beyond what was actually used.

That means the port does not need to proactively fix reference-side incompleteness such as:

- strict `checkpoint_file` YAML enforcement everywhere
- global removal of `electrode_file_path`
- making DIVER code defaults perfectly match all past configs
- cleaning up older legacy configs that still use `bce`

The target is fidelity to the practical reference behavior, not cleanup beyond reference.

## Implementation Constraints

### Preferred edit surface

The preferred implementation surface is:

- [`models/brainbert/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/brainbert/integration.py)
- [`models/popt/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/popt/integration.py)
- [`models/diver/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/diver/integration.py)

This should cover:

- checkpoint and wrapper loading
- model-specific config setters
- model-specific preprocessors
- model-specific data getters
- model-specific output-shape and activation behavior

### YAML-controlled loss and output activation

The root benchmark already resolves losses from YAML through `training_params.losses`.

Relevant files:

- [`utils/decoding_utils.py`](/storage/connectome/jmhan/podcast-benchmark/utils/decoding_utils.py)
- [`metrics/classification_metrics.py`](/storage/connectome/jmhan/podcast-benchmark/metrics/classification_metrics.py)

Therefore the integrations should follow this rule:

- if the configured loss is `bce_with_logits`, model outputs must remain raw logits
- if the configured loss is `cross_entropy`, model outputs must remain raw logits
- if a legacy probability-based loss such as `bce` is intentionally used, sigmoid output may still be allowed

This activation decision should be implemented inside each model integration, not by modifying the shared training loop.

### Minimal `core/config.py` extension that may be worth making

If YAML-native control is desired, the minimal additional `DataParams` fields worth adding are:

- `use_stft_preprocessing: bool = False`
- `stft_config: Optional[dict] = None`
- `use_lip_coords: bool = False`

These are enough to express the reference-style model behavior cleanly in YAML while keeping the rest of the root config schema intact.

`loading_options` is not required for the planned port because the root already has first-class fields for:

- `target_sr`
- `signal_unit`
- `do_drop_bads`

So `loading_options` should only be added if reproducing the reference YAML shape becomes more important than minimizing schema churn.

## Model-Specific Analysis

### BrainBERT

Current root state:

- implemented as a generic example foundation model
- not yet aligned with the reference wrapper-based loading path
- likely not preserving reference checkpoint/config semantics

Needs from reference:

- wrapper path setup
- checkpoint resolution and load semantics
- reference config mapping
- root-compatible config setter behavior
- input shape adapter between root benchmark tensors and BrainBERT expectations

### PopT

Current root state:

- also example-style at the top level
- contains more practical logic than BrainBERT, but still needs verification against the reference behavior
- includes BrainBERT-assisted mode and LIP-related hooks, but these need to be validated against the intended reference pipeline

Needs from reference:

- correct upstream/checkpoint semantics
- explicit handling of reference-style electrode sequence behavior
- robust LIP coordinate injection via `model_data_getter`
- root-compatible config setters and model specs

### DIVER

Current root state:

- closer to production than BrainBERT and PopT
- already using `model_data_getter`
- already adapted to the new `task_config` shape

Still missing or incomplete:

- reference-level convenience and variant support
- strategy-based checkpoint selection
- practical settings proven under the old `diverclip` path

### DIVERCLIP-derived settings to absorb into DIVER

What is worth porting:

- strategy-based pretrained path resolution
- known-good execution defaults
- compatible checkpoint naming support
- any adapter or auxiliary weight loading that is genuinely useful in root `diver`

What should not be ported as separate root model families:

- `diverclip`
- `diverclip_flatten`
- `diverclip_attn`

## Concrete TODO

The near-term goal is no longer just "make one task run". The goal is:

- keep `brainbert`, `popt`, and `diver` integrated into the root codebase
- define a `supersubject` evaluation variant
- make all root tasks runnable under that variant through model-specific YAML configs

For the current planning pass, YAML deduplication through a separate `common.yml` file is deferred. The loader does not currently support file-level include or inheritance. Therefore the first complete pass should prefer clarity and completeness over DRYness.

### Phase 0: Freeze the configuration policy

1. Treat `supersubject` as the canonical pooled setting that uses all electrodes listed in [`processed_data/all_subject_sig.csv`](/storage/connectome/jmhan/podcast-benchmark/processed_data/all_subject_sig.csv).
2. In `supersubject` configs, do not hardcode a single subject id.
3. Let `electrode_file_path: processed_data/all_subject_sig.csv` determine the included subject and electrode set.
4. Do not introduce a separate YAML inheritance layer yet.
5. Accept repetition across YAML files in the first complete implementation pass.

### Phase 1: Finalize model-side integration so YAMLs can target all tasks

1. Keep the edit surface centered on:
   - [`models/brainbert/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/brainbert/integration.py)
   - [`models/popt/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/popt/integration.py)
   - [`models/diver/integration.py`](/storage/connectome/jmhan/podcast-benchmark/models/diver/integration.py)
2. Keep shared utility edits minimal and only for concrete blockers.
3. Ensure all three models can consume YAML-controlled task settings for:
   - regression losses
   - binary classification losses
   - multiclass losses
   - LLM decoding task inputs
4. Keep loss-to-activation behavior correct:
   - `mse` and embedding regression -> linear output
   - `bce_with_logits` -> linear/raw logit output
   - `cross_entropy` -> linear/raw logit output
5. Ensure the following model-specific data paths are stable:
   - BrainBERT STFT preprocessing
   - PopT STFT preprocessing
   - PopT BrainBERT-upstream execution
   - PopT LIP coordinate injection
   - DIVER MNI coordinate injection
   - DIVER strategy-based pretrained path selection

### Phase 2: Inventory the full target task set

The final YAML plan should cover all tasks currently present in the root benchmark, not just the ones that existed in the reference implementation.

Root task inventory:

- `word_embedding_decoding_task`
- `whisper_embedding_decoding_task`
- `content_noncontent_task`
- `sentence_onset_task`
- `gpt_surprise_task`
- `gpt_surprise_multiclass_task`
- `pos_task`
- `volume_level_decoding_task`
- `iu_boundary_task`
- `llm_decoding_task`
- `llm_embedding_pretraining_task`

Reference-backed task group:

- `word_embedding_decoding_task`
- `whisper_embedding_decoding_task`
- `content_noncontent_task`
- `sentence_onset_task`
- `gpt_surprise_task`
- `gpt_surprise_multiclass_task`
- `pos_task`
- `volume_level_decoding_task`

Root-only task group that still needs new YAML design:

- `iu_boundary_task`
- `llm_decoding_task`
- `llm_embedding_pretraining_task`

### Phase 3: Bring over YAMLs for reference-backed tasks

1. Review the reference configs used in practice for `brainbert`, `popt`, and `diver`.
2. For each task that was already implemented in reference, create root-native YAML configs under the new model/task/variant-file layout.
3. Translate reference-era fields into the current root schema rather than copying old YAMLs verbatim.
4. Preserve task semantics from reference where possible:
   - same task name
   - same output dimension intent
   - same loss family
   - same coordinate / STFT behavior
5. The first YAML migration target set should be:
   - `word_embedding_decoding_task`
   - `whisper_embedding_decoding_task`
   - `content_noncontent_task`
   - `sentence_onset_task`
   - `gpt_surprise_task`
   - `gpt_surprise_multiclass_task`
   - `pos_task`
   - `volume_level_decoding_task`

### Phase 4: Design YAMLs for tasks not implemented in reference

1. Create new root-native YAMLs for:
   - `iu_boundary_task`
   - `llm_decoding_task`
   - `llm_embedding_pretraining_task`
2. Do not wait for perfect deduplication before writing these configs.
3. For each new task, define:
   - canonical `trial_name`
   - correct `task_specific_config`
   - correct `training_params.losses`
   - correct `training_params.metrics`
   - correct `model_spec.params.output_dim`원하면 다음 답변에서 “어떤 검증 로그를 찍어야 이 수정이 맞게 들어갔는지”까지 같이 제안하겠습니다.
   
   
   
4. Ensure these root-only task YAMLs are consistent with the already-ported model integrations rather than inventing a separate execution path.

### Phase 5: Build the supersubject config matrix

For each of the following models:

- `brainbert`
- `popt`
- `diver`

Create a `supersubject` config set that covers the full task list, with one canonical YAML per model-task pair.

Planned minimum matrix:

- `word_embedding`
- `whisper_embedding`
- `content_noncontent`
- `sentence_onset`
- `gpt_surprise`
- `gpt_surprise_multiclass`
- `pos`
- `volume_level`
- `iu_boundary`
- `llm_decoding`
- `llm_embedding_pretraining`

Notes:

- PopT may later gain secondary variants such as LIP vs no-LIP, but the first complete pass should establish one canonical config per task.
- If a task is not meaningful for a given model after validation, that should be recorded explicitly rather than silently omitted.

### Phase 6: Validate all YAMLs without introducing new code paths

For every new YAML:

1. confirm config load succeeds
2. confirm config setters run
3. confirm task getter resolves
4. confirm model constructor resolves
5. confirm required model data getters resolve
6. confirm expected loss and metric names resolve

At this stage the goal is not yet benchmark-quality results. The goal is complete runnable coverage.

### Phase 7: Run task-by-task smoke tests

1. For each model, run a small smoke-test subset of tasks first:
   - `word_embedding`
   - `content_noncontent`
   - `pos`
   - `llm_decoding`
2. After those pass, extend to the rest of the task set.
3. Record any task that requires additional integration changes and feed it back into Phase 1.

### Phase 8: Prepare final measurement runs

Once all YAMLs load and smoke tests pass:

1. choose canonical supersubject YAMLs for actual measurement
2. define fold / lag / channel overrides through CLI rather than hardcoding them into YAML
3. run the requested BrainBERT / PopT / DIVER measurements
4. archive the exact command lines used for reproducibility

## Target Configuration Strategy

For now, do not attempt to split shared config blocks into a separate `common.yml`.

Reason:

- the current config loader does not support YAML include or inheritance across files
- YAML anchors only help within a single file
- introducing a new config composition system would require touching code beyond the currently preferred scope

The more important structural decision is the directory axis order.

### Preferred axis order

The preferred final structure is:

- `model > task > evaluation-variant-file`

For example:

- `configs/foundation_models/brainbert/word_embedding/supersubject.yml`
- `configs/foundation_models/popt/content_noncontent/supersubject.yml`
- `configs/foundation_models/diver/llm_decoding/supersubject.yml`

### Why this is preferred

This is better than `model > subject > task` for the current benchmark because:

- task differences are larger than subject-scope differences
- `supersubject` is not a literal subject, but one evaluation variant among many possible electrode-selection profiles
- future variants such as `sub01.yml`, `sub02.yml`, or `supersubject_with_lip.yml` fit naturally beside the same task
- it keeps all configs relevant to one model-task pair together
- it avoids a large `supersubject/` directory full of unrelated task files

This is also slightly better than a fully nested `model > task > subject/` directory because:

- the evaluation variant currently maps to one config file, not a set of many files
- using `supersubject.yml` as the filename keeps the tree shallower
- future single-subject configs can be added as sibling files without creating unnecessary directories

Therefore the immediate strategy is:

- use explicit per-task YAMLs
- place them under model-first, task-second directories
- encode evaluation variant, subject scope, or electrode profile in the filename
- keep some duplication if needed
- optimize for readability and reliability first

Recommended filename convention:

- `supersubject.yml` for the pooled all-electrode setting
- `sub01.yml`, `sub02.yml`, ... for whole-subject variants
- `sub01_sig.yml`, `sub01_all.yml`, `sub01_custom_<tag>.yml` for electrode-profile variants

If deduplication becomes necessary later, it should be addressed as a separate config-system task rather than folded into the current integration pass.

## Planned Final Directory Structure

After the YAML migration is complete, the intended `foundation_models` layout is model-first and task-second, with evaluation variant encoded in the filename:

```text
configs/
  foundation_models/
    brainbert/
      word_embedding/
        supersubject.yml
      whisper_embedding/
        supersubject.yml
      content_noncontent/
        supersubject.yml
      sentence_onset/
        supersubject.yml
      gpt_surprise/
        supersubject.yml
      gpt_surprise_multiclass/
        supersubject.yml
      pos/
        supersubject.yml
      volume_level/
        supersubject.yml
      iu_boundary/
        supersubject.yml
      llm_decoding/
        supersubject.yml
      llm_embedding_pretraining/
        supersubject.yml
    popt/
      word_embedding/
        supersubject.yml
      whisper_embedding/
        supersubject.yml
      content_noncontent/
        supersubject.yml
      sentence_onset/
        supersubject.yml
      gpt_surprise/
        supersubject.yml
      gpt_surprise_multiclass/
        supersubject.yml
      pos/
        supersubject.yml
      volume_level/
        supersubject.yml
      iu_boundary/
        supersubject.yml
      llm_decoding/
        supersubject.yml
      llm_embedding_pretraining/
        supersubject.yml
    diver/
      word_embedding/
        supersubject.yml
      whisper_embedding/
        supersubject.yml
      content_noncontent/
        supersubject.yml
      sentence_onset/
        supersubject.yml
      gpt_surprise/
        supersubject.yml
      gpt_surprise_multiclass/
        supersubject.yml
      pos/
        supersubject.yml
      volume_level/
        supersubject.yml
      iu_boundary/
        supersubject.yml
      llm_decoding/
        supersubject.yml
      llm_embedding_pretraining/
        supersubject.yml
```

Possible future extensions under the same scheme:

- `configs/foundation_models/brainbert/word_embedding/sub01.yml`
- `configs/foundation_models/brainbert/word_embedding/sub02.yml`
- `configs/foundation_models/brainbert/word_embedding/sub01_sig.yml`
- `configs/foundation_models/brainbert/word_embedding/sub01_custom_lgab.yml`
- `configs/foundation_models/popt/word_embedding/supersubject_with_lip.yml`

Optional cleanup after migration:

- move current flat files in [`configs/foundation_models`](/storage/connectome/jmhan/podcast-benchmark/configs/foundation_models) into a temporary `legacy/` directory
- update docs and example commands to point to the new model-first, task-second layout

## Execution Note

The work should be considered complete only when:

- the three model integrations are stable
- the supersubject YAML matrix exists for all root tasks under the model/task/variant-file layout
- each YAML can at least complete a smoke-test run
- the requested final measurements can be launched from those YAMLs without additional ad hoc config edits
