#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import random

import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


import torch.nn.functional as F

import datasets as datasets
import models.Net as models
import models.Diff_UNet as Diff_UNet



import os  
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from collections import OrderedDict


from sklearn.preprocessing import MultiLabelBinarizer

# def set_seed(seed=99):
#     random.seed(seed)  # 设置 Python 随机种子
#     np.random.seed(seed)  # 设置 NumPy 随机种子
#     torch.manual_seed(seed)  # 设置 PyTorch CPU 的随机种子
#     torch.cuda.manual_seed(seed)  # 如果使用 GPU，设置 GPU 的随机种子
#     torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，设置所有 GPU 的随机种子

#     # 保证结果的可复现性
#     torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
#     torch.backends.cudnn.benchmark = False     # 如果为 True，系统会自动寻找最适合当前配置的高效算法，但会引入随机性

# # 使用特定的随机种子
# set_seed(99)



def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        # self.loss_values = []

        # self.mlb = MultiLabelBinarizer(classes=np.arange(args.class_num))
        self.train_dict = OrderedDict()
        self.train_dict['source_train-Loss'] = []
        self.train_dict['source_train-Acc_pos'] = []
  
        self.train_dict['source_val-Loss'] = []
        self.train_dict['source_val-Acc_pos'] = []
        
        self.train_dict['target_val-Loss'] = []
        self.train_dict['target_val-Acc_pos'] = []   

        self.Fine_dict = OrderedDict() 
        self.Fine_dict['Fine_Acc'] = []  
       
    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets,args.data_name)
        self.datasets = {}


        if isinstance(args.transfer_task[0], str):
           #print( args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))


        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype, args.source_num_classes).data_split(transfer_learning=False)
        # 加载所有目标域的数据集
        # self.data_test = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_test()     

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size, drop_last=True,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           #worker_init_fn=lambda worker_id: np.random.seed(99)
                                                           )
                            for x in ['source_train', 'source_val', 'target_val']}
        
        # Define the model
        self.model = getattr(models,args.model_name)(args.pretrained)
#      
        self.model_test = getattr(models, args.model_name)(args.pretrained)
        
        if args.adabn:
            self.model_eval = getattr(models, args.model_name)(args.pretrained)
        
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.adabn:
                self.model_eval = torch.nn.DataParallel(self.model_eval)

        # Define the optimizer
        if args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters(self)), lr=args.lr,
                                        weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")


        # Define the learning rate decay
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

        # Invert the model and define the loss
        self.model.to(self.device)
        if args.adabn:
            self.model_eval.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.SoftmaxCrossEntropyLoss()
    

    def set_input(self, input):
        """input: a dictionary that contains the data itself and its metadata information.
        example:
            input_signals.size() = B, C, 1, S

        """
        data_set = []
        for data in input:
            shift = np.random.randint(-100, 400)
            # 随机生成一个偏移量shift
            scale = np.random.uniform(0.5, 1.5)
            # 随机生成一个缩放因子scale
            data_ = np.roll( data , shift ) * scale
            data_set.append(data_)
            self.input_signals = torch.tensor(data_set)     
        return self.input_signals
    
    # def wd(self,input):

    #   self.wd = getattr(model_wd,'WaveletGatedNet')(signal_length=1792, wavelet_name='db1',level=8)  
      
    #   return self.wd(input) 
from torch.utils.data import DataLoader, TensorDataset
class DiffUNetTrainer:
    def __init__(self, model, target_data, device='cuda', lr=1e-3, batch_size=32, epochs=100):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # 如果是 DDPM 风格，后面可以换成噪声预测的损失

        # 创建 DataLoader
        self.dataset = TensorDataset(target_data, target_data)  # 使用目标域数据进行训练
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                noise = torch.randn_like(x)
                noised_input = x + noise * 0.1  # 模拟扩散初始噪声

                # 使用 noised_input 作为输入进行训练
                output = self.model(noised_input, x)  # 条件是原始信号
                loss = self.criterion(output, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {total_loss/len(self.dataloader):.6f}")

    def save_model(self, path='diff_unet_pretrained.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='diff_unet_pretrained.pth'):
        self.model.load_state_dict(torch.load(path))
   

    # ===== MMD Loss Function =====
    def mmd_loss(self, source, target):
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        return torch.norm(source_mean - target_mean)

    # ===== Diffusion反向过程（简化版） =====
    def reverse_process(self, model, condition, timesteps=50):
        noise = torch.randn_like(condition)
        x = noise
        for _ in range(timesteps):
            x = model(x, condition)
        return x

    # ===== 生成函数：从目标域抽样条件，扩散生成样本 =====
    def generate_and_filter_samples(self, model, target_data, target_labels, mmd_threshold=0.3, num_generated_per_class=10, timesteps=50):
        model.eval()
        generated_samples = []
        generated_labels = []

        for label in target_labels.unique():
            class_data = target_data[target_labels == label]
            if class_data.size(0) < 1:
                continue

            for _ in range(num_generated_per_class):
                condition = class_data[torch.randint(0, len(class_data), (1,))]
                generated = self.reverse_process(model, condition, timesteps)
                
                # ===== MMD 打分 =====
                mmd = self.mmd_loss(generated, class_data)
                if mmd.item() < mmd_threshold:
                    generated_samples.append(generated)
                    generated_labels.append(label)

        if len(generated_samples) == 0:
            return None, None

        return torch.cat(generated_samples, dim=0), torch.tensor(generated_labels).to(target_data.device)


    # ===== 数据拼接函数 =====
    def combine_generated_and_original(generated, original):
        return torch.cat([generated, original], dim=0)
    
    # 假设你已有数据如下：
    B, C, T = 32, 3, 1792  # 批量，通道，时间步
    source_data = torch.randn(B, C, T)
    target_data = torch.randn(B, C, T)
    target_labels = torch.randint(0, 11, (B,))
    # 初始化 UNet 模型
    model = Diff_UNet(in_channels=C, condition_channels=C)
   
    # 训练模型（使用目标域数据进行训练）
    trainer = Diff_UNet(model, target_data, device='cuda', lr=1e-3, batch_size=32, epochs=100)
    trainer.train()  # 开始训练

    # 保存模型
    trainer.save_model('diff_unet_pretrained.pth') # 权重文件
    # 训练结束后，加载模型（用于生成）
    trainer.load_model('diff_unet_pretrained.pth')

    # 生成样本（比如每类生成10个）
    generated_samples = generate_and_filter_samples(model, target_data, target_labels, num_generated_per_class=10)

    # 合并数据，用于后续训练分类器
    augmented_data = combine_generated_and_original(generated_samples, source_data)


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        batch_count = 0
        batch_acc_pos = 0
        best_acc_pos = 0.0
        step_start = time.time()

        # 1. 记忆池（保存高准确率数据）
        memory_buffer = {"x": [], "y": []}  
        


        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()

                epoch_loss = 0.0
                batch_loss =0
                epoch_acc_pos =0
                epoch_acc_cls =0
                #######################################
                # # 检查数据加载器
                # print(type(self.dataloaders))
                # print(self.dataloaders)

                # # 检查数据加载器中的数据数量
                # for phase, dataloader in self.dataloaders.items():
                #     print(f"Phase: {phase}, Number of samples: {len(dataloader.dataset)}")

                # # 打印一些样本数据
                # for phase, dataloader in self.dataloaders.items():
                #     print(f"Phase: {phase}")
                #     for i, (inputs, label_pos, label_cls) in enumerate(dataloader):
                #         print(f"Sample {i+1}: inputs={inputs}, label_pos={label_pos},label_cls={label_cls}")
                #         if i >= 3:  # 打印前三个样本
                #             break
                # ###############################################
                # Set model to train mode or test mode
                if phase != 'target_val':
                    if phase=='source_train':
                       self.model.train()
                    if phase=='source_val':
                       self.model.eval()
                else:
                    if args.adabn:
                        torch.save(self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict(),
                                   os.path.join(self.save_dir, 'model_temp.pth'))
                        #保存本轮的参数文件
                        self.model_eval.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_temp.pth')))
                        self.model_eval.train()
                        self.model_eval.apply(apply_dropout)
                        with torch.set_grad_enabled(False):

                            for i in range(args.adabn_epochs):
                                if args.eval_all:
                                    for batch_idx, (inputs, _,_) in enumerate(self.dataloaders['target_val']):
                                        if batch_idx == 0:
                                            inputs_all = inputs
                                        else:
                                            inputs_all = torch.cat((inputs_all, inputs), dim=0)
                                    inputs_all = inputs_all.to(self.device)
                                    _ = self.model_eval(inputs_all)
                                else:
                                    for i in range(args.adabn_epochs):
                                        for batch_idx, (inputs, _,_) in enumerate(self.dataloaders['target_val']):
                                            inputs = inputs.to(self.device)
                                            _ = self.model_eval(inputs)
                        self.model_eval.eval()
                    else:
                        self.model.eval()

                for batch_idx, (inputs, label_pos,_) in enumerate(self.dataloaders[phase]):
                    # print(inputs.shape)
                    # inputs = self.set_input(inputs)    
                    label_pos = label_pos.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        if args.adabn:  
                            if phase != 'target_val':
                                pos,__annotations__ = self.model(self.set_input(inputs).to(self.device))
                            else:
                                pos,_ = self.model_eval(inputs.to(self.device))
                        else: 
                            if  phase != 'target_val':
                                pos,_ = self.model(self.set_input(inputs).to(self.device)) # phase = 'source_train'
                            else:
                                pos,_ = self.model(inputs.to(self.device)) # phase = 'target_val'

                       
                        label_pos = label_pos.to(self.device)                  
                        loss_pos = self.criterion(pos , label_pos) 
                        loss = loss_pos 
                        pred_pos = pos.argmax(dim=1)
                        correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        # loss_temp_cls = loss_cls.item() * inputs.size(0)
      
                        epoch_loss += loss_temp
                        epoch_acc_pos += correct_pos
                        #在这个epoch中正确的次数
                        # precision = precision_score(label_pos, pred_pos, average='weighted')
                        # recall = recall_score(label_pos, pred_pos, average='weighted')
                        # f1 = f1_score(label_pos, pred_pos, average='weighted')



                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss_pos.backward()
                
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc_pos += correct_pos
                            batch_count += inputs.size(0)
                           
                            # Print the training information
                            # if step % args.print_step == 0:
                            #     batch_loss = batch_loss / batch_count
                            #     batch_acc_pos = batch_acc_pos / batch_count
                            #     temp_time = time.time()
                            #     train_time = temp_time - step_start
                            #     step_start = temp_time
                            #     batch_time = train_time / args.print_step if step != 0 else train_time
                            #     sample_per_sec = 1.0*batch_count/train_time  
                            #     logging.info('Epoch: {} [{}/{}], Train Loss : {:.4f} Train Acc pos: {:.4f} ,''{:.1f} examples/sec {:.2f} sec/batch'.format(
                            #         epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),batch_loss, batch_acc_pos,  sample_per_sec, batch_time)
                            #         )
                            #     batch_acc_pos = 0
                            #     batch_loss =  0.0
                            #     batch_count = 0
                            step += 1


                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc_pos = epoch_acc_pos / len(self.dataloaders[phase].dataset)
                epoch_acc_cls = epoch_acc_cls / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc_pos: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc_pos, time.time() - epoch_start
                ))

                if phase == 'source_train':
                    self.train_dict['source_train-Loss'].append(epoch_loss)
                    self.train_dict['source_train-Acc_pos'].append(epoch_acc_pos)
               
                elif phase == 'source_val':
                    self.train_dict['source_val-Loss'].append(epoch_loss)
                    self.train_dict['source_val-Acc_pos'].append(epoch_acc_pos)
       
                elif phase == 'target_val':
                    self.train_dict['target_val-Loss'].append(epoch_loss)
                    self.train_dict['target_val-Acc_pos'].append(epoch_acc_pos)

                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    # if epoch_acc_pos > best_acc_pos   or epoch > args.max_epoch-2:
                    if epoch_acc_pos > best_acc_pos:
                        best_acc_pos = epoch_acc_pos

                        logging.info("save best model epoch {}, acc_pos {:.4f}".format(epoch, epoch_acc_pos))
                        torch.save(model_state_dic,os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc_pos)))

                         # 对目标域所有的数据进行测试
                        if args.eval_test_all:
                            test_start = time.time()                            
                            torch.save(model_state_dic,os.path.join(self.save_dir, 'model_test.pth'))
                            self.model_test.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_test.pth')))
                            self.model_test.eval()
                            correct_pos=0
                            inputs_all = label_pos =[]
                            acc_pos = 0
                            count = 0 
                            
                            # 存储所有真实标签和预测标签用来计算精确度、召回率、F1
                            all_label_pos = []
                            all_pred_pos = []   

                            # 将数据传递给模型进行测试,假设你的模型是 model,对数据进行预测
                            for batch_idx, (inputs, label_pos,label_cls) in enumerate(self.dataloaders['target_val']):
                                inputs_all = inputs
                                label_pos = label_pos
                                inputs_all = inputs_all.to(self.device)
                                label_pos = label_pos.to(self.device)    
                                self.model_test.to(self.device)
                                pos_test,_ = self.model_test(inputs_all)
                                pred_pos = pos_test.argmax(dim=1)
                                
                                correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()
                                acc_pos += correct_pos
                                count += inputs.size(0)
                                
                                # 收集当前 batch 的预测和真实标签
                                all_label_pos.extend(label_pos.cpu().numpy())
                                all_pred_pos.extend(pred_pos.cpu().numpy())

                            acc_pos = acc_pos / count 


                            logging.info('epoch: {} , acc_pos : {:.4f}, Cost {:.1f} sec'.format(
                                epoch, acc_pos , time.time() - test_start)
                                ) 
                            # 转换为 numpy 数组
                            all_label_pos = np.array(all_label_pos)
                            all_pred_pos = np.array(all_pred_pos)

                            # 计算精确率（Precision）
                            precision_macro = precision_score(all_label_pos, all_pred_pos, average='macro')  # 宏平均
                            precision_micro = precision_score(all_label_pos, all_pred_pos, average='micro')  # 微平均
                            precision_weighted = precision_score(all_label_pos, all_pred_pos, average='weighted')  # 加权平均

                            # 计算召回率（Recall）
                            recall_macro = recall_score(all_label_pos, all_pred_pos, average='macro')
                            recall_micro = recall_score(all_label_pos, all_pred_pos, average='micro')
                            recall_weighted = recall_score(all_label_pos, all_pred_pos, average='weighted')

                            # 计算 F1 值
                            f1_macro = f1_score(all_label_pos, all_pred_pos, average='macro')
                            f1_micro = f1_score(all_label_pos, all_pred_pos, average='micro')
                            f1_weighted = f1_score(all_label_pos, all_pred_pos, average='weighted')

                            # 打印日志
                            logging.info('epoch: {} , Precision (Macro): {:.4f}, Recall (Macro): {:.4f}, F1 (Macro): {:.4f}'.format(
                                epoch, precision_macro, recall_macro, f1_macro))
                            logging.info('epoch: {} ,Precision (Micro): {:.4f}, Recall (Micro): {:.4f}, F1 (Micro): {:.4f}'.format(
                                epoch, precision_micro, recall_micro, f1_micro))
                            logging.info('epoch: {} ,Precision (Weighted): {:.4f}, Recall (Weighted): {:.4f}, F1 (Weighted): {:.4f}'.format(
                                epoch, precision_weighted, recall_weighted, f1_weighted)) 
                            
                            classification_report_str = classification_report(all_label_pos, all_pred_pos)

                            # 记录分类报告
                            logging.info("\nClassification Report:\n%s", classification_report_str)

                
                        
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        


       







