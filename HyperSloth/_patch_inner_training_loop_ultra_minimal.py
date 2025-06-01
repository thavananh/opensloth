import os
from typing import Optional, Dict, Any
from fastcore.all import patch
from transformers.trainer import Trainer, TrainerState
from HyperSloth.logging_config import get_hypersloth_logger
from HyperSloth._patch_log import _patch_log


def patch_inner_training_loop(trainer):
    """
    Ultra-minimal patch that only adds essential HyperSloth customizations.
    This approach patches specific methods instead of duplicating the entire training loop.
    """
    # Get environment variables
    hp_local_rank = int(os.getenv('HYPERSLOTH_LOCAL_RANK', '0'))
    hp_num_gpus = int(os.getenv('HYPERSLOTH_NUM_GPUS', '1'))
    
    # Apply log patch
    trainer_class = type(trainer)
    _patch_log(trainer_class)
    
    # Get enhanced logger
    enhanced_logger = get_hypersloth_logger(gpu_id=str(hp_local_rank))
    
    # Patch 1: TrainerState creation with HyperSloth fields
    original_trainer_state_init = TrainerState.__init__
    
    def enhanced_trainer_state_init(self, **kwargs):
        original_trainer_state_init(self, **kwargs)
        # Add HyperSloth custom fields
        self.is_world_process_zero = (hp_local_rank == 0)
    
    TrainerState.__init__ = enhanced_trainer_state_init
    
    # Patch 2: TrainerState loading with field preservation
    original_load_from_json = TrainerState.load_from_json
    
    @staticmethod
    def enhanced_load_from_json(json_path):
        state = original_load_from_json(json_path)
        # Ensure HyperSloth fields exist after loading
        if not hasattr(state, 'is_world_process_zero'):
            state.is_world_process_zero = (hp_local_rank == 0)
        return state
    
    TrainerState.load_from_json = enhanced_load_from_json
    
    # Patch 3: GPU-specific batch slicing (if multi-GPU)
    if hp_num_gpus > 1 and hasattr(trainer_class, 'get_batch_samples'):
        original_get_batch_samples = trainer_class.get_batch_samples
        
        def select_gpu_slice(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Select this GPU's slice of the batch data."""
            if inputs is None:
                return inputs
                
            result = {}
            for key, value in inputs.items():
                if key in ['input_ids', 'attention_mask', 'labels'] and hasattr(value, '__getitem__'):
                    result[key] = value[hp_local_rank::hp_num_gpus]
                else:
                    result[key] = value
            return result
        
        @patch  
        def get_batch_samples(self: Trainer, epoch_iterator, num_batches, device=None):
            """Enhanced batch sampling with GPU slicing for HyperSloth."""
            batch_samples, num_items_in_batch = original_get_batch_samples(
                self, epoch_iterator, num_batches, device
            )
            
            # Apply GPU-specific slicing
            processed_samples = [
                select_gpu_slice(inputs) for inputs in batch_samples
            ]
            return processed_samples, num_items_in_batch
    
    enhanced_logger.info(
        f'HyperSloth ultra-minimal patches applied successfully '
        f'(rank {hp_local_rank}/{hp_num_gpus})'
    )
