#!/bin/bash

# Define the GPU IDs
GPU_IDS=$1

# Split the GPU IDs into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"

# Create a new tmux session
SESSION_NAME="training_session"
tmux new-session -d -s $SESSION_NAME

# Loop through each GPU ID and start the training process in a new tmux window
for GPU_ID in "${GPU_ARRAY[@]}"; do
    WINDOW_NAME="gpu_$GPU_ID"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python train.py $GPU_ID -g $GPU_IDS" C-m
done

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME