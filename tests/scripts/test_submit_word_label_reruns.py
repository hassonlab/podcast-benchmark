from scripts.submit_word_label_reruns import build_jobs, command_for


def test_word_label_rerun_matrix_is_complete_and_collision_free():
    jobs = build_jobs()

    assert len(jobs) == 400
    assert len({job.trial_name for job in jobs}) == len(jobs)
    assert all(f"--trial_name={job.trial_name}" in command_for(job) for job in jobs)


def test_lag_chunks_cover_each_requested_grid_once():
    jobs = build_jobs()

    cnn_subject = [
        job for job in jobs
        if "neural_conv_decoder" in job.trial_name
        and "-pos-per-subject-all-" in job.trial_name
    ]
    covered = []
    for job in cnn_subject:
        values = {item.split("=", 1)[0]: int(item.split("=", 1)[1]) for item in job.overrides
                  if item.startswith("--training_params.")}
        covered.extend(range(
            values["--training_params.min_lag"],
            values["--training_params.max_lag"],
            values["--training_params.lag_step_size"],
        ))
    assert sorted(covered) == list(range(-1000, 1001, 25))
    assert len(covered) == len(set(covered))

    foundation = [
        job for job in jobs
        if "brainbert-pos-subject-1-" in job.trial_name
    ]
    covered = []
    for job in foundation:
        values = {item.split("=", 1)[0]: int(item.split("=", 1)[1]) for item in job.overrides}
        covered.extend(range(
            values["--training_params.min_lag"],
            values["--training_params.max_lag"],
            values["--training_params.lag_step_size"],
        ))
    assert sorted(covered) == list(range(-1000, 1001, 100))
    assert len(covered) == len(set(covered))
