#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.tune_utils import tune_utils

import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


import sys
sys.setrecursionlimit(100000)


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    # parser.add_argument('--model_name', type=str, default='LORA_Net_12345', help='the name of the model')

    parser.add_argument('--model_Fine_name', type=str, default='LORA_Net_12345', help='加载微调模型')
    parser.add_argument('--data_name', type=str, default='leak_signals', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=r'E:\projects\UDTL-LoRA\data\leak_signals', help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[0],[1], [2], [3]], help='transfer learning tasks')
    parser.add_argument('--task', type=str, default='F0-123-[1]', help='transfer learning tasks')
    parser.add_argument('--set_input', type=str, default='T F', help='nomalization type')
 

    parser.add_argument('--note', type=str, default='no.1,3,5    数据增强改动 原始:(-100,400),(0.5,1.5)     [1,1,1,1] bs=128,2 ')
    parser.add_argument('--num_layers', type=str, default='2', help='num_layers in model')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')

    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='.\checkpoint_adabn', help='the directory to save the model')


   # fine paramenters
    parser.add_argument('--source_num_classes', type=int, default=12, help='源域泄露孔径位置类别')
    parser.add_argument('--target_num_classes', type=int, default=12, help='目标域泄露孔径位置数目')
    parser.add_argument('--target_label', type=str, default=['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13' ] ,help='')
    parser.add_argument('--target_classes', type=str, default=[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13] ], help='')


    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')


    
    
    parser.add_argument('--Fine_number', type=int, default=1, help='use one-shot、three-shot、five-shot、ten-shot for every classes in taget domain')
    parser.add_argument('--model_Fine', type=str, default=r'E:\projects\UDTL-LoRA\pth\0-123\12345\41-0.9812-best_model.pth' , help='')
    parser.add_argument('--param_zero', type=bool, default=False , help='参数清零')
    parser.add_argument('--Fine_epoch', type=int, default='50' , help='')
    parser.add_argument('--Fine_lr', type=float, default='0.01' , help='')


    parser.add_argument('--Fine_classes', type=bool, default=True, help='是否微调类别数')
    parser.add_argument('--Fine_num_classes', type=int, default=12, help='微调任务中泄露孔径类别数')




    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    print("1"==1)

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S') + '_' + args.model_Fine_name + '_' + args.task
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    

    test = tune_utils(args, save_dir)
    test.setup()
    test.test()


    


