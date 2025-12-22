# Baseline Results Summary

This page summarizes the baseline results for all tasks in the podcast benchmark.

## Overview

Total number of baseline results: 10

## Content Noncontent Task

### content_noncontent_task_sig_elecs_mlp_early_stop_roc

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_content_noncontent.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_content_noncontent.yml)

![content_noncontent_task_sig_elecs_mlp_early_stop_roc Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/content_noncontent_task_sig_elecs_mlp_early_stop_roc_2025-12-19-00-34-17_lag_performance.png)

**Best Performance:**

- **Lag:** 200ms
- **test_roc_auc_mean:** 0.5900

---

## Ensemble Model

### ensemble_model_10_arbitrary

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_arbitrary.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_arbitrary.yml)

![ensemble_model_10_arbitrary Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/ensemble_model_10_arbitrary_2025-12-19-00-17-32_lag_performance.png)

**Best Performance:**

- **Lag:** 400ms
- **test_word_avg_auc_roc_mean:** 0.5549
- **test_word_top_5_mean:** 0.0415

---

### ensemble_model_10_glove

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_glove.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_glove.yml)

![ensemble_model_10_glove Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/ensemble_model_10_glove_2025-12-19-00-17-41_lag_performance.png)

**Best Performance:**

- **Lag:** 400ms
- **test_word_avg_auc_roc_mean:** 0.6046
- **test_word_top_5_mean:** 0.0357

---

### ensemble_model_10_gpt2

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt2.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt2.yml)

![ensemble_model_10_gpt2 Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/ensemble_model_10_gpt2_2025-12-19-00-17-43_lag_performance.png)

**Best Performance:**

- **Lag:** 400ms
- **test_word_avg_auc_roc_mean:** 0.6057
- **test_word_top_5_mean:** 0.0254

---

## Gpt Surprise

### gpt_surprise

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt_surprise_multiclass.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt_surprise_multiclass.yml)

![gpt_surprise Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/gpt_surprise_2025-12-19-00-18-44_lag_performance.png)

**Best Performance:**

- **Lag:** 400ms
- **test_corr_mean:** 0.0591

---

## Gpt Surprise Multiclass

### gpt_surprise

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt_surprise_multiclass.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_gpt_surprise_multiclass.yml)

![gpt_surprise Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/gpt_surprise_2025-12-19-00-18-43_lag_performance.png)

**Best Performance:**

- **Lag:** 200ms
- **test_roc_auc_multiclass_mean:** 0.5333

---

## Pos Task

### pos_task_sig_elecs_without_other_classes

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_pos.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_pos.yml)

![pos_task_sig_elecs_without_other_classes Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/pos_task_sig_elecs_without_other_classes_2025-12-19-00-34-17_lag_performance.png)

**Best Performance:**

- **Lag:** 600ms
- **test_roc_auc_multiclass_mean:** 0.5305

---

## Sentence Onset

### sentence_onset_lr

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_sentence_onset.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/neural_conv_decoder/neural_conv_decoder_sentence_onset.yml)

![sentence_onset_lr Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/sentence_onset_lr_2025-12-19-00-18-44_lag_performance.png)

**Best Performance:**

- **Lag:** 0ms
- **test_roc_auc_mean:** 0.8800

---

## Volume Level

### volume_level_simple

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/time_pooling_model/simple_model.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/time_pooling_model/simple_model.yml)

![volume_level_simple Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/volume_level_simple_2025-12-19-00-34-56_lag_performance.png)

**Best Performance:**

- **Lag:** 200ms
- **test_corr_mean:** 0.4479

---

### volume_level_torch_ridge

**Config:** [`/Users/zachparis/Documents/projects/podcast-benchmark/configs/time_pooling_model/simple_model.yml`](/Users/zachparis/Documents/projects/podcast-benchmark/configs/time_pooling_model/simple_model.yml)

![volume_level_torch_ridge Performance](/Users/zachparis/Documents/projects/podcast-benchmark/docs/baseline_plots/volume_level_torch_ridge_2025-12-19-00-42-42_lag_performance.png)

**Best Performance:**

- **Lag:** 200ms
- **test_corr_mean:** 0.4476

---

