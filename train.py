"""
Multi-GPU training script for Unsloth models.
Distributes training across specified GPUs with weight synchronization.
"""
import os
import argparse
from typing import List, Tuple
from speedy_utils import setup_logger

def parse_arguments() -> Tuple[int, List[int]]:
    """
    Parse command line arguments for GPU configuration.
    
    Returns:
        Tuple containing current GPU ID and list of all GPU IDs
    """
    parser = argparse.ArgumentParser(description="Distributed training for Unsloth models")
    parser.add_argument('gpu_index', type=int, help='Index of current GPU in the GPU list')
    parser.add_argument('--gpus', '-g', type=str, default='0,2', 
                        help='Comma separated list of all GPUs to use')
    args = parser.parse_args()

    all_gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    current_gpu_id = all_gpu_ids[int(args.gpu_index)]
    
    return current_gpu_id, all_gpu_ids

def main():
    """Main execution function for the training script."""
    current_gpu_id, all_gpu_ids = parse_arguments()
    
    # Configure environment for this GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(current_gpu_id)
    setup_logger('D' if current_gpu_id == all_gpu_ids[0] else 'I')
    
    # Import here to ensure environment variables are set before importing
    from unsloth_trainer_multi_gpus.training_utils import setup_model_and_training
    
    # Setup and start training
    trainer = setup_model_and_training(
        current_gpu_id, all_gpu_ids
    )
    trainer.train()

if __name__ == "__main__":
    main()
