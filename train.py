#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Training script for pipeline leakage detection model.
Supports both JSON config and command-line arguments.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warnings
warnings.filterwarnings('ignore')

from config import TrainConfig
from utils.logger import setlogger
from utils.train_utils_leak import train_utils
import logging


def main():
    args = TrainConfig.get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    
    save_dir = os.path.join(args.checkpoint_dir, 
                            datetime.strftime(datetime.now(), '%m%d-%H%M%S') + '_' + args.model_name + '_' + args.task)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    setlogger(os.path.join(save_dir, 'train.log'))
    
    config_dict = {k: v for k, v in vars(args).items()}
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    for k, v in config_dict.items():
        logging.info("{}: {}".format(k, v))
    
    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
    
    return save_dir


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    save_dir = main()
    print(f"Training complete. Results saved to: {save_dir}")