#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Visualization script for training results.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import argparse


def plot_results(train_dict, save_path):
    """Plot training results."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    titles = ['Loss', 'Accuracy (Position)']
    datasets = ['source_train', 'source_val', 'target_val']
    keys = ['Loss', 'Acc_pos']
    
    for i, dataset in enumerate(datasets):
        for j, key in enumerate(keys):
            ax = axs[i, j]
            data = train_dict[f'{dataset}-{key}']
            ax.plot(data)
            ax.set_title(f'{dataset} - {titles[j]}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'result.jpg'), bbox_inches='tight')
    print(f"Results plot saved to: {os.path.join(save_path, 'result.jpg')}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
    parser.add_argument('--exp_dir', type=str, required=True, help='experiment directory')
    args = parser.parse_args()
    
    train_log_path = os.path.join(args.checkpoint_dir, args.exp_dir, 'train.log')
    
    if not os.path.exists(train_log_path):
        print(f"Error: {train_log_path} not found")
        return
    
    print(f"Visualizing results from: {os.path.join(args.checkpoint_dir, args.exp_dir)}")
    print("Note: Please load train_dict from trainer.train_dict in your training script")
    print("This script requires the train_dict to be saved or passed programmatically")


if __name__ == '__main__':
    main()
