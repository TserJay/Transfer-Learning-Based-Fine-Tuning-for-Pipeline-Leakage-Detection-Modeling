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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F

import datasets as datasets
import models.LORA_Net_12345 as models
#import models.LORA_Net_12345 as model_Ftest
#import models.wd as model_wd


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
#        self.model_Ftest = getattr(model_Ftest, args.model_Fine_name)(args.pretrained)


        
        self.model_test = getattr(models, args.model_name)(args.pretrained)

        
        
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, Dataset.num_classes)
    
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
            
            # transformed_signal1 = np.roll(data[:, 0], shift) * scale
            # transformed_signal2 = np.roll(data[:, 1], shift) * scale
            # transformed_signal3 = np.roll(data[:, 2], shift) * scale
            # # 分别对data数据集的第一、第二、第三列进行处理
                   
            # data_ = np.concatenate([transformed_signal1.reshape(-1, 1), transformed_signal2.reshape(-1, 1),
            #                         transformed_signal3.reshape(-1, 1)], axis=1)
            # data_ = np.concatenate([transformed_signal1.reshape(-1, 1), transformed_signal2.reshape(-1, 1),
            #                         transformed_signal3.reshape(-1, 1)], axis=1)
            # data_ = np.concatenate([transformed_signal1.reshape(-1, 1),
            #                         transformed_signal2.reshape(-1, 1)], axis=1)
            # 变为多行一列的形式，沿x轴作为一组

            data_set.append(data_)
            self.input_signals = torch.tensor(data_set)
            
        # self.input_signals = torch.from_numpy(np.transpose(np.array(data_set), (0, 2, 1))).float()#.to(self.device)
        # np.transpose改变数组的维度位置;(0,1,2)->(0,2,1)

        # self.pos = (torch.tensor(labels).float() / self.args.dot_num).reshape(-1, 1)


        # self.cla = torch.from_numpy(self.mlb.fit_transform(labels)).float()#.to(self.args.device) #类别任务
        # 使用 mlb.fit_transform 方法将目标数据中的第二列进行 one-hot 编码,并转换格式
        ############################################################################################################################

        return self.input_signals
    
    # def wd(self,input):

    #   self.wd = getattr(model_wd,'WaveletGatedNet')(signal_length=1792, wavelet_name='db1',level=8)  
      
    #   return self.wd(input) 
    


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
                                pos,_ = self.model(inputs.to(self.device)) ###
                            else:
                                pos,_ = self.model(inputs.to(self.device))

                        # inputs = inputs.to(self.device)
                        label_pos = label_pos.to(self.device)
                        # label_cls = label_cls.to(self.device)
                      
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
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc_pos = batch_acc_pos / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time  
                                logging.info('Epoch: {} [{}/{}], Train Loss : {:.4f} Train Acc pos: {:.4f} ,''{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),batch_loss, batch_acc_pos,  sample_per_sec, batch_time)
                                    )
                                batch_acc_pos = 0
                                batch_loss =  0.0
                                batch_count = 0
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
                        if  args.Fine == False:
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

                   
                    #################  Fine start ######################        
                    if epoch == args.max_epoch-1 and args.Fine:

                    
                       
                        class MLP(nn.Module):
                            """ Very simple multi-layer perceptron (also called FFN)   多层感知器FFN"""

                            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                                super().__init__()
                                self.num_layers = num_layers
                                h = [hidden_dim] * (num_layers - 1)
                                self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

                            def forward(self, x):
                                for i, layer in enumerate(self.layers):
                                    x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                                return x

                                                                
                        # 定义微调的模型结构                        
                        class MLPModel(nn.Module):
                            def __init__(self, base_model):
                                super(MLPModel, self).__init__()
                                self.base_model = base_model
                                # if args.Fine_classes:
                                #     self.tar_MLP = MLP(input_dim=32, hidden_dim=64, output_dim=16, num_layers=1)
                                #     self.fc = nn.Linear(16,args.num_classes)
				
                            def forward(self, x):
                                # with torch.no_grad():

                                pos,out = self.base_model(x)
                                # if args.Fine_classes:
                                #     x = self.tar_MLP(out)
                                #     pos = self.fc(out)
                                return pos

                        Final_acc=  0
                        acc_list = []

                        Fine_precision_macro_list = [] 
                        Fine_precision_micro_list = [] 
                        Fine_precision_weighted_list = []  
                        
                        Fine_recall_macro_list = [] 
                        Fine_recall_micro_list = [] 
                        Fine_recall_weighted_list = [] 
                    
                        Fine_f1_macro_list = [] 
                        Fine_f1_micro_list = [] 
                        Fine_f1_weighted_list = []  
                        for i in range(0,3):
                            test_start = time.time() 
                            self.Fine_datasets = []        
                            target_dataset = self.datasets['target_val']
                            # 创建集合，用于存储不同组合的(pos, cls)标签
                            unique_combinations = set()

                            # 遍历数据集，找到每个不同的(pos, cls)组合
                            for sample in target_dataset:
                                features, pos, cls = sample
                                unique_combinations.add((pos, cls))
                        

                            Fine_number = args.Fine_number

                            for pos, cls in unique_combinations:
                                
                                # 收集所有匹配当前 (pos, cls) 组合的样本
                                matching_samples = [sample for sample in target_dataset if sample[1] == pos and sample[2] == cls]
                                
                                # 随机选择指定数量的样本
                                selected_samples = random.sample(matching_samples, Fine_number)
                                
                                # 将选择的样本添加到 Fine_datasets 中
                                self.Fine_datasets.extend(selected_samples)

                                
                                                    
                            # # 设置每个组合中要选择的样本数量
                            # Fine_number = [args.Fine_number]
                            # for i in Fine_number:
                            #     num_samples_per_combination = i  # 修改这个值以选择每个组合的样本数量
                            #     # 从每个不同的组合中选择多个样本
                            #     for pos, cls in unique_combinations:
                            #         count = 0
                            #         for sample in target_dataset:
                            #             if sample[1] == pos and sample[2] == cls:
                            #                 self.Fine_datasets.append(sample)
                            #                 count += 1
                            #                 if count >= num_samples_per_combination:
                            #                     break  # 选择指定数量的样本后跳出内循环

                                
                            # 打印数据和标签
                            # for i, (random_data, random_pos,random_cls) in enumerate(self.Fine_datasets):
                            #     print(f"Random Sample {i+1}:")
                            #     print("Data:", random_data)
                            #     print("pos:", random_pos)
                            #     print("cls:", random_cls)              


                            few_shot_loader = torch.utils.data.DataLoader(self.Fine_datasets, batch_size=32, shuffle=True, drop_last=True)

                            # 加载预训练好的模型并创建MLP模型
                            # base_model = self.model

                            base_model = self.model_Ftest

                            # 读取模型的权重
                            base_model.load_state_dict(torch.load(args.model_Fine))
                            # base_model.eval()
                            # 参数重置
                            if args.param_zero:
                                for param in base_model.parameters():
                                    param.data.zero_()
                                    

                            # mlp_model = MLPModel(base_model).to(self.device)
                            base_model = base_model.to(self.device)

                            # 冻结部分层的参数
                            layers_to_freeze = [base_model.backbone.conv1, base_model.backbone.bn1,base_model.backbone.relu,
                                                # mlp_model.base_model.backbone.layer1,mlp_model.base_model.backbone.BiLSTM1,
                                                # mlp_model.base_model.backbone.layer2,mlp_model.base_model.backbone.BiLSTM2,
                                                # mlp_model.base_model.backbone.layer3,mlp_model.base_model.backbone.BiLSTM3, 
                                                # mlp_model.base_model.backbone.layer4,mlp_model.base_model.BiLSTM1,
                                                # mlp_model.base_model.ap,

                                                # mlp_model.base_model.backbone.LoraLayer_1,mlp_model.base_model.backbone.LoraLayer_2,
                                                # mlp_model.base_model.backbone.LoraLayer_3,
                                            
                                                # mlp_model.base_model.backbone.LoraLayer_4,

                                                # mlp_model.base_model.backbone.conv2,mlp_model.base_model.backbone.bn2,mlp_model.base_model.backbone.relu,

                                                # mlp_model.base_model.enc_embedding_en,mlp_model.base_model.transformer_encoder,
                                                # mlp_model.base_model.backbone1,
                                                # mlp_model.base_model.projetion_cls,

                                                # mlp_model.base_model.projetion_pos_1,
                                                # mlp_model.base_model.fc1


                                                ]
                            for layer in layers_to_freeze:
                                for param in layer.parameters():
                                    param.requires_grad = False
                            # for param in mlp_model.base_model.parameters():
                            #     param.requires_grad = False
                            for name, param in base_model.named_parameters():
                                if param.requires_grad:
                                    logging.info(f"{name}: Train")
                                else:
                                    logging.info(f"{name}: Freeze")

                            # 只训练模型中的MLP层
                            # optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)
                            Fine_acc_pos=0

                            Fine_precision_macro = 0
                            Fine_precision_micro = 0 
                            Fine_precision_weighted = 0 

                            Fine_recall_macro = 0
                            Fine_recall_micro = 0
                            Fine_recall_weighted = 0
               
                            Fine_f1_macro = 0 
                            Fine_f1_micro = 0
                            Fine_f1_weighted = 0

                            if args.Fine_1:
                                optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), lr=args.Fine_lr,momentum=0.9)
                                criterion = nn.CrossEntropyLoss()
                        
                                # 在目标域数据上微调模型
                                num_epochs = args.Fine_epoch
                                for epoch in range(num_epochs):
                                    base_model.train()
                                    running_loss = 0.0
                                    for inputs, pos, _ in few_shot_loader:
                                        optimizer.zero_grad()
                                        inputs = inputs.cuda()
                                        pos = pos.cuda()
                                        outputs,_ = base_model(inputs)
                                        # print(outputs)
                                        # outputs = args.target_label[args.target_classes[0][mlp_model(inputs).item()]][0]
                                        loss = criterion(outputs, pos)
                                        loss.backward()
                                        optimizer.step()
                                        running_loss += loss.item()

                                    logging.info('Fine-Epoch {}/{}, Fine-Loss: {}'.format(epoch+1,num_epochs,loss))
                                    
                                    # 最后10轮对Fine后的模型进行测试
                                    if epoch>num_epochs-11:    
                                        base_model.eval()
                                        correct_pos=0
                                        inputs_all = []
                                        label_pos =[]
                                        all_label_pos = []
                                        all_pred_pos = []
                                        acc_pos = 0
                                        
                                        count = 0         
                                        for batch_idx, (inputs, label_pos,label_cls) in enumerate(self.dataloaders['target_val']):
                                            inputs_all = inputs
                                            label_pos = label_pos
                                            inputs_all = inputs_all.to(self.device)
                                            label_pos = label_pos.to(self.device) 

                                            pos_test,_ = base_model(inputs_all)
                                            pos_test = pos_test.to(self.device)
                                            pred_pos = pos_test.argmax(dim=1)

                                            correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()

                                            

                                            # 累加标签和预测值
                                            all_label_pos.extend(label_pos.cpu().numpy())  # 添加当前批次标签
                                            all_pred_pos.extend(pred_pos.cpu().numpy())    # 添加当前批次预测
                            
                                            acc_pos += correct_pos
                                            # precision_value += precision
                                            # recall_value += recall
                                            # f1_value += f1
                                            count += inputs.size(0)
                                        acc_pos = acc_pos / count

                                        # 计算指标
                                        all_label_pos_cpu = np.array(all_label_pos)  # 转换为NumPy数组
                                        all_pred_pos_cpu = np.array(all_pred_pos)
                                  

                                        # 计算精确率（Precision） - 多分类
                                        precision_macro = precision_score(all_label_pos_cpu, all_pred_pos_cpu, average='macro')  # 宏平均
                                        precision_micro = precision_score(all_label_pos_cpu, all_pred_pos_cpu, average='micro')  # 微平均
                                        precision_weighted = precision_score(all_label_pos_cpu, all_pred_pos_cpu, average='weighted')  # 加权平均

                                        # 计算召回率（Recall） - 多分类
                                        recall_macro = recall_score(all_label_pos_cpu, all_pred_pos_cpu, average='macro')
                                        recall_micro = recall_score(all_label_pos_cpu, all_pred_pos_cpu, average='micro')
                                        recall_weighted = recall_score(all_label_pos_cpu, all_pred_pos_cpu, average='weighted')

                                        # 计算F1值 - 多分类
                                        f1_macro = f1_score(all_label_pos_cpu, all_pred_pos_cpu, average='macro')
                                        f1_micro = f1_score(all_label_pos_cpu, all_pred_pos_cpu, average='micro')
                                        f1_weighted = f1_score(all_label_pos_cpu, all_pred_pos_cpu, average='weighted')

                                     

                                        self.Fine_dict['Fine_Acc'].append(acc_pos)
                                        Fine_acc_pos += acc_pos
                                        Fine_precision_macro += precision_macro
                                        Fine_precision_micro += precision_micro
                                        Fine_precision_weighted += precision_micro

                                       
                                        Fine_recall_macro += recall_macro
                                        Fine_recall_micro += recall_micro
                                        Fine_recall_weighted += recall_weighted

                                        # 计算F1值 - 多分类
                                        Fine_f1_macro += f1_macro
                                        Fine_f1_micro += f1_micro
                                        Fine_f1_weighted += f1_weighted


                                        logging.info('Fine-epoch {}/{}, Fine_number: {}, Fine-acc_pos: {:.4f},precision_macro: {:.4f}, precision_micro: {:.4f},precision_weighted: {:.4f}, recall_macro: {:.4f}, recall_micro: {:.4f}, recall_weighted: {:.4f},f1_macro: {:.4f}, f1_micro: {:.4f}, f1_weighted: {:.4f},Cost: {:.1f} sec' 
                                                     .format(epoch+1, num_epochs, Fine_number, acc_pos, precision_macro, precision_micro,precision_weighted,recall_macro, recall_micro, recall_weighted,f1_macro, f1_micro, f1_weighted, time.time() - test_start))
                                        
                                        classification_report_str = classification_report(all_label_pos_cpu, all_pred_pos_cpu)

                                        # 记录分类报告
                                        logging.info("\nClassification Report:\n%s", classification_report_str)


                                        # 生成、可视化混淆矩阵
                                        conf_matrix = confusion_matrix(all_label_pos_cpu, all_pred_pos_cpu)
                                        plt.figure(figsize=(8, 6))
                                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                                                    
                                                    xticklabels=['0','1','2','3','4','5','6','7','8','9','10'],  # 根据实际类别标签修改
                                                    yticklabels=['0','1','2','3','4','5','6','7','8','9','10'])  # 根据实际类别标签修改
                                        plt.ylabel('True label')
                                        plt.xlabel('Predicted label')
                                        plt.title('Confusion Matrix')
                                        #plt.show()
                                        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix_{}_{}.png'.format(i,epoch)))
                                        
                                
                                        
                                        # torch.save(mlp_model.state_dict(),os.path.join(self.save_dir,'{}-fine_tuned_{}_{}.pth'.format(epoch+1, acc_pos, Fine_number)))
                                    
                                    if epoch== num_epochs-1:
                                        Fine_acc_pos = Fine_acc_pos/10

                                        Fine_precision_macro = Fine_precision_macro/10
                                        Fine_precision_micro = Fine_precision_micro/10
                                        Fine_precision_weighted = Fine_precision_weighted/10
                                       
                                        Fine_recall_macro = Fine_recall_macro/10
                                        Fine_recall_micro = Fine_recall_micro/10
                                        Fine_recall_weighted = Fine_recall_weighted/10
                                   
                                        Fine_f1_macro = Fine_f1_macro/10
                                        Fine_f1_micro = Fine_f1_micro/10
                                        Fine_f1_weighted = Fine_f1_weighted/10
                                        

                                        logging.info('Fina-i:{},Fina-Fine-acc_pos : {:.4f},Fine_precision_macro: {:.4f}, Fine_precision_micro: {:.4f},Fine_precision_weighted: {:.4f}, Fine_recall_macro: {:.4f}, Fine_recall_micro: {:.4f}, Fine_recall_weighted: {:.4f}, Fine_f1_macro: {:.4f}, Fine_f1_micro: {:.4f}, Fine_f1_weighted: {:.4f}'
                                                     .format(i,Fine_acc_pos,Fine_precision_macro,Fine_precision_micro,Fine_precision_weighted,Fine_recall_macro,Fine_recall_micro,Fine_recall_weighted,Fine_recall_weighted,Fine_f1_macro,Fine_f1_micro,Fine_f1_weighted))
                                        
                                        torch.save(base_model.state_dict(),os.path.join(self.save_dir,'fine_tuned_{}_{}_{}.pth'.format(i,Fine_acc_pos,Fine_number)))
                                        
                                        Final_acc += Fine_acc_pos
                                        acc_list.append(Fine_acc_pos)

                                        

                                        
                                        Fine_precision_macro_list.append(Fine_precision_macro) 
                                        Fine_precision_micro_list.append(Fine_precision_micro)  
                                        Fine_precision_weighted_list.append(Fine_precision_weighted)  
                                       
                                        Fine_recall_macro_list.append(Fine_recall_macro)  
                                        Fine_recall_micro_list.append(Fine_recall_micro)  
                                        Fine_recall_weighted_list.append(Fine_recall_weighted) 
                                   
                                        Fine_f1_macro_list.append(Fine_f1_macro) 
                                        Fine_f1_micro_list.append(Fine_f1_micro) 
                                        Fine_f1_weighted_list.append(Fine_f1_weighted) 
                            # # 对Fine后的模型进行测试
                            # mlp_model.eval()
                            # correct_pos=0
                            # inputs_all = label_pos =[]
                            # acc_pos = 0
                            # count = 0         
                            # for batch_idx, (inputs, label_pos,label_cls) in enumerate(self.dataloaders['target_val']):
                            #     inputs_all = inputs
                            #     label_pos = label_pos
                            #     inputs_all = inputs_all.to(self.device)
                            #     label_pos = label_pos.to(self.device)    
                            #     pos_test= mlp_model(inputs_all)
                            #     pred_pos = pos_test.argmax(dim=1)
                            #     correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()
                            #     acc_pos += correct_pos
                            #     count += inputs.size(0)
                            # acc_pos = acc_pos / count
                            # self.Fine_dict['Fine_Acc'].append(acc_pos)
                            # logging.info(' Fine_number : {}, Fine-acc_pos : {:.4f}, Cost {:.1f} sec'.format(i, acc_pos , time.time() - test_start))  
                            # # 保存微调后的模型
                            # torch.save(mlp_model.state_dict(),os.path.join(self.save_dir,'fine_tuned_{}_{}.pth'.format(Fine_acc_pos,Fine_number)))
                        
                        
                        
                        #logging.info(acc_list)
                        # 计算样本标准差
                        sample_std = np.std(acc_list, ddof=1)  # ddof=1 表示样本标准差
                        # 计算总体标准差
                        population_std = np.std(acc_list)  # 默认计算总体标准差
                        mean_value = np.mean(acc_list)
                        #logging.info('mean_value:{:.4f}, sample_std:{:.4f}, sample_std:{:.4f}'.format(mean_value, sample_std, population_std))
                        #logging.info(acc_list)

                    
                                
                        mean_Fine_precision_macro = np.mean(Fine_precision_macro_list)
                        mean_Fine_precision_micro = np.mean(Fine_precision_micro_list)
                        mean_Fine_precision_weighted = np.mean(Fine_precision_weighted_list)


                        mean_Fine_recall_macro = np.mean(Fine_recall_macro_list)
                        mean_Fine_recall_micro = np.mean(Fine_recall_micro_list)
                        mean_Fine_recall_weighted = np.mean(Fine_recall_weighted_list)

                        mean_Fine_f1_macro = np.mean(Fine_f1_macro_list)
                        mean_Fine_f1_micro = np.mean(Fine_f1_micro_list)
                        mean_Fine_f1_weighted = np.mean(Fine_f1_weighted_list)

                        # 计算样本标准差 (ddof=1)
                        std_sample_Fine_precision_macro = np.std(Fine_precision_macro_list, ddof=1)* 100
                        std_sample_Fine_precision_micro = np.std(Fine_precision_micro_list, ddof=1)* 100
                        std_sample_Fine_precision_weighted = np.std(Fine_precision_weighted_list, ddof=1)* 100

                        std_sample_Fine_recall_macro = np.std(Fine_recall_macro_list, ddof=1)* 100
                        std_sample_Fine_recall_micro = np.std(Fine_recall_micro_list, ddof=1)* 100
                        std_sample_Fine_recall_weighted = np.std(Fine_recall_weighted_list, ddof=1)* 100

                        std_sample_Fine_f1_macro = np.std(Fine_f1_macro_list, ddof=1)* 100
                        std_sample_Fine_f1_micro = np.std(Fine_f1_micro_list, ddof=1)* 100
                        std_sample_Fine_f1_weighted = np.std(Fine_f1_weighted_list, ddof=1)* 100

                        # 计算总体标准差 (ddof=0)
                        std_population_Fine_precision_macro = np.std(Fine_precision_macro_list, ddof=0)* 100
                        std_population_Fine_precision_micro = np.std(Fine_precision_micro_list, ddof=0)* 100
                        std_population_Fine_precision_weighted = np.std(Fine_precision_weighted_list, ddof=0)* 100

                        std_population_Fine_recall_macro = np.std(Fine_recall_macro_list, ddof=0)* 100
                        std_population_Fine_recall_micro = np.std(Fine_recall_micro_list, ddof=0)* 100
                        std_population_Fine_recall_weighted = np.std(Fine_recall_weighted_list, ddof=0)* 100

                        std_population_Fine_f1_macro = np.std(Fine_f1_macro_list, ddof=0)* 100
                        std_population_Fine_f1_micro = np.std(Fine_f1_micro_list, ddof=0)* 100
                        std_population_Fine_f1_weighted = np.std(Fine_f1_weighted_list, ddof=0)* 100


                        sample_std_percentage = sample_std * 100
                        population_std_percentage = population_std * 100
                        logging.info('---------------------------------------result---------------------------------------')
                        logging.info('Final-acc : {:.4f}, sample_std:{:.2f}%, population_std:{:.2f}%'.format(mean_value,sample_std_percentage,population_std_percentage))

                        logging.info('----------------------- macro -----------------------')
                        logging.info('Fine_precision_macro: {:.4f}, {:.2f}%, {:.2f}%, Fine_recall_macro: {:.4f}, {:.2f}%, {:.2f}%, Fine_f1_macro: {:.4f}, {:.2f}%, {:.2f}%'
                                                     .format(mean_Fine_precision_macro,std_sample_Fine_precision_macro, std_population_Fine_precision_macro, 
                                                             mean_Fine_recall_macro, std_sample_Fine_recall_macro, std_population_Fine_recall_macro,
                                                             mean_Fine_f1_macro, std_sample_Fine_f1_macro, std_population_Fine_f1_macro))
                        
                        logging.info('----------------------- micro -----------------------')
                        logging.info('Fine_precision_micro: {:.4f}, {:.2f}%, {:.2f}%, Fine_recall_micro: {:.4f}, {:.2f}%, {:.2f}%, Fine_f1_micro: {:.4f}, {:.2f}%, {:.2f}%'
                                                     .format(mean_Fine_precision_micro, std_sample_Fine_precision_micro, std_population_Fine_precision_micro,
                                                             mean_Fine_recall_micro, std_sample_Fine_recall_micro, std_population_Fine_recall_micro,
                                                             mean_Fine_f1_micro, std_sample_Fine_f1_micro, std_population_Fine_f1_micro))
                        
                        logging.info('----------------------- weighted -----------------------')
                        logging.info('Fine_precision_weighted: {:.4f}, {:.2f}%, {:.2f}%, Fine_recall_weighted: {:.4f}, {:.2f}%, {:.2f}%, Fine_f1_weighted: {:.4f}, {:.2f}%, {:.2f}%,'
                                                     .format(mean_Fine_precision_weighted, std_sample_Fine_precision_weighted, std_population_Fine_precision_weighted,
                                                             mean_Fine_recall_weighted, std_sample_Fine_recall_weighted, std_population_Fine_recall_weighted,
                                                             mean_Fine_f1_weighted, std_sample_Fine_f1_weighted, std_population_Fine_f1_weighted))

                        # logging.info('---------------------------------------result---------------------------------------')

                        # logging.info('Final-acc : {:.4f}, sample_std:{:.2f}%, population_std:{:.2f}%'.format(mean_value,sample_std_percentage,population_std_percentage))
                        
                        # logging.info('Fine_precision_macro: {:.4f}, Fine_precision_micro: {:.4f},Fine_precision_weighted: {:.4f},Fine_recall_macro: {:.4f}, Fine_recall_micro: {:.4f}, Fine_recall_weighted: {:.4f}, Fine_f1_macro: {:.4f}, Fine_f1_micro: {:.4f}, Fine_f1_weighted: {:.4f}'
                        #                              .format(mean_Fine_precision_macro,mean_Fine_precision_micro,mean_Fine_precision_weighted,mean_Fine_recall_macro,mean_Fine_recall_micro,mean_Fine_recall_weighted,mean_Fine_f1_macro,mean_Fine_f1_micro,mean_Fine_f1_weighted))
                        # logging.info('Fine_precision_macro: {:.4f}, Fine_precision_micro: {:.4f},Fine_precision_weighted: {:.4f}, Fine_recall_macro: {:.4f}, Fine_recall_micro: {:.4f}, Fine_recall_weighted: {:.4f}, Fine_f1_macro: {:.4f}, Fine_f1_micro: {:.4f}, Fine_f1_weighted: {:.4f}'
                        #                              .format(mean_Fine_precision_macro,mean_Fine_precision_micro,mean_Fine_precision_weighted,mean_Fine_recall_macro,mean_Fine_recall_micro,mean_Fine_recall_weighted,mean_Fine_f1_macro,mean_Fine_f1_micro,mean_Fine_f1_weighted))
                        
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        


       







