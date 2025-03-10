#!/bin/bash

# Usage: ./run_tmux_training.sh "0,1"
#        ./run_tmux_training.sh "0,1,2"
rm -r grads/
GPU_IDS=$1

# Split the GPU IDs into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"

# Name the tmux session
SESSION_NAME="training_session"

# Create a new tmux session (detached) that shows nvidia-smi in the first window
tmux new-session -d -s "$SESSION_NAME" "watch -n0.1 nvidia-smi"

# Loop through each GPU ID and start the training process in a new tmux window
for GPU_ID in "${GPU_ARRAY[@]}"; do
    WINDOW_NAME="gpu_$GPU_ID"
    tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME"

    # We'll run train_mmap.py, passing in:
    #   1) the positional GPU index
    #   2) the comma-separated GPU IDs as --gpus
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" \
        "python train_mmap.py --gpu_index $GPU_ID --visible_devices ${GPU_IDS//,/ } ${@:2}" C-m}
done

# Attach to the tmux session
tmux attach-session -t "$SESSION_NAME"
