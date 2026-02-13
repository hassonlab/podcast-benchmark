subject_list=(1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10 1 1 2 2 3 3 4 4 7 7 10 10)
total_trial_list=(1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1 1 2 0 4 0 1 0 1 0 1 0 1)
total_task_list=(0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 8 8 8 8 8 8 8 8 8 8 8 8 9 9 9 9 9 9 9 9 9 9 9 9 10 10 10 10 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 11 11 12 12 12 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 13 13 14 14 14 14 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 17 17 17 17 17 17 17 17 17 17 17 17 18 18 18 18 18 18 18 18 18 18 18 18)

subject_no=${subject_list[$id]}
trial_no=${total_trial_list[$id]}
task_no=${total_task_list[$id]}
neuroprobe_task_list=("frame_brightness" "global_flow" "local_flow" "global_flow_angle" "local_flow_angle" "face_num" "volume" "pitch" "delta_volume" "delta_pitch" "speech" "onset" "gpt2_surprisal" "word_length" "word_gap" "word_index" "word_head_pos" "word_part_speech" "speaker")
neuroprobe_task=${neuroprobe_task_list[$task_no]} 
fold_list=(0 1)

run_finetuning() {
    local neuroprobe_task=$1
    local neuroprobe_trial=$2
    local neuroprobe_fold=$3
    local gpu_id=$4
    python finetune_main.py \
        --seed 41 \
        --cuda  0 \
        --epochs 40 \
        --batch_size 32 \
        --lr 2.0e-03 \
        --weight_decay 1e-02 \
        --downstream_dataset Neuroprobe \
        --datasets_dir /your/path/to/dataset \
        --model_dir /your/path/to/save/modelstate \
        --foundation_dir ./weights/ieeg_pretrained_weights.pt \
        --width 256 \
        --depth 12 \
        --patch_size 50 \
        --num_workers 8 \
        --label_smoothing 0.1 \
        --ft_config flatten_linear \
        --early_stop_criteria val_auroc \
        --early_stop_patience 10 \
        --mup_weights True \
        --ft_mup False \
        --use_amp True \
        --deepspeed_pth_format True \
        --frozen False \
        --neuroprobe_task "$neuroprobe_task" \
        --neuroprobe_subject "$subject_no" \
        --neuroprobe_trial "$neuroprobe_trial" \
        --neuroprobe_fold "$neuroprobe_fold" \
        --neuroprobe_evaltype sssm \
        --neuroprobe_model_type DIVER \
        --neuroprobe_allow_corrupted False \
        --neuroprobe_clip 1 \
        --neuroprobe_patchsize 50 \
        --neuroprobe_preprocess "laplacian-downsample_500" 

for fold_id in "${fold_list[@]}"; do
   run_finetuning "${neuroprobe_task}" "${trial_no}" "${fold_id}" 0
done