#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
import torch.nn as nn
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import OrderedDict
from scipy.signal import resample
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch.nn.functional as F

import datasets
import models


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


class train_utils:
    """Training utilities for pipeline leakage detection model."""

    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.train_dict = OrderedDict()
        self.train_dict['source_train-Loss'] = []
        self.train_dict['source_train-Acc_pos'] = []
        self.train_dict['source_val-Loss'] = []
        self.train_dict['source_val-Acc_pos'] = []
        self.train_dict['target_val-Loss'] = []
        self.train_dict['target_val-Acc_pos'] = []
        self.batch_loss = 0.0
        self.batch_acc_pos = 0
        self.batch_count = 0

    def setup(self):
        """Initialize datasets, model, loss and optimizer."""
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        Dataset = getattr(datasets, args.data_name)

        if isinstance(args.transfer_task[0], str):
            args.transfer_task = eval("".join(args.transfer_task))

        self.datasets = {}
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_val'] = Dataset(
            args.data_dir, args.transfer_task, args.normlizetype, args.source_num_classes
        ).data_split(transfer_learning=False)

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=(True if x.split('_')[1] == 'train' else False),
                num_workers=args.num_workers,
                pin_memory=(True if self.device == 'cuda' else False)
            )
            for x in ['source_train', 'source_val', 'target_val']
        }

        self.model = getattr(models, args.model_name)(args.pretrained)
        self.model_test = getattr(models, args.model_name)(args.pretrained)

        if args.adabn:
            self.model_eval = getattr(models, args.model_name)(args.pretrained)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.adabn:
                self.model_eval = torch.nn.DataParallel(self.model_eval)

        if args.opt == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.opt == 'sgd':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        else:
            raise Exception("optimizer not implement")

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 150, 1e-4)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0
        self.model.to(self.device)
        if args.adabn:
            self.model_eval.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, input_data):
        """Apply data augmentation to input."""
        data_set = []
        for data in input_data:
            shift = np.random.randint(-30, 30)
            scale = np.random.uniform(0.9, 1.1)
            data = np.roll(data, shift) * scale
            
            data_set.append(data)
        return torch.tensor(data_set, dtype=torch.float32)

    def train(self):
        """Main training loop."""
        args = self.args
        step = 0
        best_acc_pos = 0.0
        best_loss_pos = float('inf')
        step_start = time.time()
        
        # Early stopping parameters
        patience = 30  # 如果30个epoch验证集loss没改善则停止
        patience_counter = 0
        early_stop = False

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            for phase in ['source_train', 'source_val', 'target_val']:
                epoch_start = time.time()
                epoch_loss = 0.0
                epoch_acc_pos = 0
                self.batch_count = 0
                self.batch_loss = 0.0
                self.batch_acc_pos = 0

                if phase != 'target_val':
                    if phase == 'source_train':
                        self.model.train()
                    else:
                        self.model.eval()
                else:
                    if args.adabn:
                        torch.save(
                            self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict(),
                            os.path.join(self.save_dir, 'model_temp.pth')
                        )
                        self.model_eval.load_state_dict(
                            torch.load(os.path.join(self.save_dir, 'model_temp.pth'))
                        )
                        self.model_eval.train()
                        self.model_eval.apply(apply_dropout)
                        with torch.set_grad_enabled(False):
                            for _ in range(args.adabn_epochs):
                                if args.eval_all:
                                    inputs_all = None
                                    for batch_idx, (inputs, _, _) in enumerate(self.dataloaders['target_val']):
                                        if inputs_all is None:
                                            inputs_all = inputs
                                        else:
                                            inputs_all = torch.cat((inputs_all, inputs), dim=0)
                                    if inputs_all is not None:
                                        inputs_all = inputs_all.to(self.device)
                                        _ = self.model_eval(inputs_all)
                                else:
                                    for batch_idx, (inputs, _, _) in enumerate(self.dataloaders['target_val']):
                                        inputs = inputs.to(self.device)
                                        _ = self.model_eval(inputs)
                        self.model_eval.eval()
                    else:
                        self.model.eval()

                for batch_idx, (inputs, label_pos, _) in enumerate(self.dataloaders[phase]):
                    label_pos = label_pos.to(self.device)
                    with torch.set_grad_enabled(phase == 'source_train'):
                        if args.adabn:
                            if phase != 'target_val':
                                pos, _ = self.model(self.set_input(inputs).to(self.device))
                            else:
                                pos, _ = self.model_eval(inputs.to(self.device))
                        else:
                            if phase != 'target_val':
                                pos, _ = self.model(self.set_input(inputs).to(self.device))
                            else:
                                pos, _ = self.model(inputs.to(self.device))

                        loss = self.criterion(pos, label_pos)
                        pred_pos = pos.argmax(dim=1)
                        correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)

                        epoch_loss += loss_temp
                        epoch_acc_pos += correct_pos

                        if phase == 'source_train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            self.batch_loss += loss_temp
                            self.batch_acc_pos += correct_pos
                            self.batch_count += inputs.size(0)
                            step += 1

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc_pos = epoch_acc_pos / len(self.dataloaders[phase].dataset)

                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc_pos: {:.4f}, Cost {:.1f} sec'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc_pos, time.time() - epoch_start
                    )
                )

                if phase == 'source_train':
                    self.train_dict['source_train-Loss'].append(epoch_loss)
                    self.train_dict['source_train-Acc_pos'].append(epoch_acc_pos)
                elif phase == 'source_val':
                    self.train_dict['source_val-Loss'].append(epoch_loss)
                    self.train_dict['source_val-Acc_pos'].append(epoch_acc_pos)
                elif phase == 'target_val':
                    self.train_dict['target_val-Loss'].append(epoch_loss)
                    self.train_dict['target_val-Acc_pos'].append(epoch_acc_pos)

                    # Early stopping based on validation loss
                    if epoch_loss < best_loss_pos:
                        best_loss_pos = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logging.info(f"Early stopping at epoch {epoch}, best val loss: {best_loss_pos:.4f}")
                            early_stop = True

                    if epoch_acc_pos > best_acc_pos:
                        best_acc_pos = epoch_acc_pos
                        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                        if hasattr(self, 'prev_best_model_path') and os.path.exists(self.prev_best_model_path):
                            os.remove(self.prev_best_model_path)

                        logging.info("save best model epoch {}, acc_pos {:.4f}".format(epoch, epoch_acc_pos))
                        model_path = os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc_pos))
                        torch.save(model_state_dic, model_path)
                        self.prev_best_model_path = model_path

                        if args.eval_test_all:
                            self._evaluate_and_log(epoch, model_state_dic)

            if early_stop:
                break
                
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _evaluate_and_log(self, epoch, model_state_dic):
        """Evaluate on target validation set and log metrics."""
        args = self.args
        test_start = time.time()

        torch.save(model_state_dic, os.path.join(self.save_dir, 'model_test.pth'))
        self.model_test.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_test.pth')))
        self.model_test.eval()

        all_label_pos = []
        all_pred_pos = []
        count = 0

        for batch_idx, (inputs, label_pos, _) in enumerate(self.dataloaders['target_val']):
            inputs = inputs.to(self.device)
            label_pos = label_pos.to(self.device)
            self.model_test.to(self.device)

            pos_test, _ = self.model_test(inputs)
            pred_pos = pos_test.argmax(dim=1)

            all_label_pos.extend(label_pos.cpu().numpy())
            all_pred_pos.extend(pred_pos.cpu().numpy())
            count += inputs.size(0)

        all_label_pos = np.array(all_label_pos)
        all_pred_pos = np.array(all_pred_pos)

        acc_pos = sum(p == l for p, l in zip(all_pred_pos, all_label_pos)) / count

        logging.info('epoch: {} , acc_pos : {:.4f}, Cost {:.1f} sec'.format(
            epoch, acc_pos, time.time() - test_start
        ))

        precision_macro = precision_score(all_label_pos, all_pred_pos, average='macro')
        precision_micro = precision_score(all_label_pos, all_pred_pos, average='micro')
        precision_weighted = precision_score(all_label_pos, all_pred_pos, average='weighted')

        recall_macro = recall_score(all_label_pos, all_pred_pos, average='macro')
        recall_micro = recall_score(all_label_pos, all_pred_pos, average='micro')
        recall_weighted = recall_score(all_label_pos, all_pred_pos, average='weighted')

        f1_macro = f1_score(all_label_pos, all_pred_pos, average='macro')
        f1_micro = f1_score(all_label_pos, all_pred_pos, average='micro')
        f1_weighted = f1_score(all_label_pos, all_pred_pos, average='weighted')

        logging.info('epoch: {} , Precision (Macro): {:.4f}, Recall (Macro): {:.4f}, F1 (Macro): {:.4f}'.format(
            epoch, precision_macro, recall_macro, f1_macro))
        logging.info('epoch: {} ,Precision (Micro): {:.4f}, Recall (Micro): {:.4f}, F1 (Micro): {:.4f}'.format(
            epoch, precision_micro, recall_micro, f1_micro))
        logging.info('epoch: {} ,Precision (Weighted): {:.4f}, Recall (Weighted): {:.4f}, F1 (Weighted): {:.4f}'.format(
            epoch, precision_weighted, recall_weighted, f1_weighted))

        logging.info("\nClassification Report:\n%s", classification_report(all_label_pos, all_pred_pos))