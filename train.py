import os
import argparse
import time
from speedy_utils import setup_logger
parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int)
parser.add_argument('--gpus','-g', type=str, default='0,2', help='Comma separated list of all gpus')
args = parser.parse_args()

gpus = [int(i) for i in args.gpus.split(',')]
current_gpu = gpus[int(args.gpu)]

os.environ['CUDA_VISIBLE_DEVICES'] = str(current_gpu)
setup_logger('D' if current_gpu == gpus[0] else 'I')

from unsloth_trainer_multi_gpus.training_utils import *

model, tokenizer, dataset, trainer = prepare_input(current_gpu, gpus)

trainer.train()
