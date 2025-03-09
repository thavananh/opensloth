import os
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=str, default='0')
parser.add_argument('--all_gpus', type=str, default='0,1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


from unsloth_trainer_multi_gpus.training_utils import *

model, tokenizer, dataset, trainer = prepare_input(args.gpu, args.all_gpus)

trainer.train()
