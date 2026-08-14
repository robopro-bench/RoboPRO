#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3:-0}

./script/.update_path.sh > /dev/null 2>&1

if [[ "${gpu_id}" == *,* ]]; then
    # Multi-GPU: dynamic per-seed dispatch across the listed GPUs (e.g. 0,1).
    # episode_num is read from the task config, so the amount collected matches
    # the single-GPU path. All episodes land in one run dir (episode0..N-1).
    PYTHONWARNINGS=ignore::UserWarning \
    python -u script/collect_parallel.py "${task_name}" "${task_config}" --gpus "${gpu_id}"
else
    # Single-GPU: stock sequential collection.
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    PYTHONWARNINGS=ignore::UserWarning \
    python -u script/collect_data.py $task_name $task_config
    rm -rf data/${task_name}/${task_config}/.cache
fi
