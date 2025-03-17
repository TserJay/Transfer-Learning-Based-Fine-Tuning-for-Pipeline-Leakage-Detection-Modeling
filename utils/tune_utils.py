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

from collections import OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer


import models.LORA_Net_12345 as model_test
from itertools import cycle  # 让 target batch 轮流使用

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



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
# #set_seed(99)



def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()
class tune_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

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

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,drop_last=True,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                        #    num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           worker_init_fn=lambda worker_id: np.random.seed(99))
                            for x in ['source_train', 'source_val', 'target_val']}
        
        
        # Define the model
        # self.model = getattr(models,args.model_name)
        self.model_test = getattr(model_test, args.model_Fine_name)(args.pretrained)
    
    

    def test(self):
        """
        Training process
        :return:
        """
        args = self.args

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
            
        # define mmd_loss
        def compute_mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5):
                    
            batch_size = source_features.size(0)
            
            # 合并源域和目标域特征
            total_features = torch.cat([source_features, target_features], dim=0)  # [2*batch_size, feature_dim]
            
            # 计算两两欧式距离：利用广播机制，计算 L2 距离矩阵
            total0 = total_features.unsqueeze(0).expand(total_features.size(0), total_features.size(0), total_features.size(1))
            total1 = total_features.unsqueeze(1).expand(total_features.size(0), total_features.size(0), total_features.size(1))
            L2_distance = ((total0 - total1) ** 2).sum(2)
            
            # 带宽设置：这里取中值，可以防止带宽太小或者太大
            bandwidth = torch.median(L2_distance.detach())
            if bandwidth.item() == 0:
                bandwidth = torch.tensor(1.0).to(source_features.device)
            # 生成一组带宽参数
            bandwidth_list = [bandwidth * (kernel_mul**(i - kernel_num//2)) for i in range(kernel_num)]
            
            # 对每个带宽计算 RBF 核，并求均值
            kernels = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
            kernel_matrix = sum(kernels)  # [2*batch_size, 2*batch_size]
            
            # 划分核矩阵
            XX = kernel_matrix[:batch_size, :batch_size]
            YY = kernel_matrix[batch_size:, batch_size:]
            XY = kernel_matrix[:batch_size, batch_size:]
            YX = kernel_matrix[batch_size:, :batch_size]
            
            # 计算 MMD：源域内部、目标域内部和跨域部分
            mmd_loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
            return mmd_loss
            

                                                
        # 定义微调的模型结构                        
        # class MLPModel(nn.Module):
        #     def __init__(self, base_model, num_classes):
        #         super(MLPModel, self).__init__()
        #         self.base_model = base_model
        #         if args.Fine_classes:
        #             self.tar_MLP = MLP(input_dim=32, hidden_dim=64, output_dim=16, num_layers=1)
        #             self.fc = nn.Linear(16,num_classes)

        #     def forward(self, x):
        #         # with torch.no_grad():

        #         pos,out = self.base_model(x)
        #         if args.Fine_classes:
        #             x = self.tar_MLP(out)
        #             pos = self.fc(x)
        #         return pos
            
        class MMDModel(nn.Module):
            def __init__(self, base_model, num_classes):

                super(MMDModel, self).__init__()
                self.feature_extractor = base_model
                self.tar_MLP = MLP(input_dim=32, hidden_dim=64, output_dim=16, num_layers=1)
                self.fc = nn.Linear(16,num_classes)
               
            def forward(self, x):
               
                pos,out = self.feature_extractor(x)
                x = self.tar_MLP(out)
                pos = self.fc(x)
                return pos,out
            
                # # 分别提取源域和目标域特征
                # source_features = self.feature_extractor(x_source)
                # target_features = self.feature_extractor(x_target)
                
                # # 对源域特征进行分类预测
                # logits = self.classifier(source_features)
                
                # # 计算 MMD 损失
                # mmd_loss = compute_mmd_loss(source_features, target_features)
                
                # return logits, mmd_loss

        # def train_epoch(model, source_loader, target_loader, optimizer, alpha=1.0, lambda_mmd=0.1, device='cuda'):
      
        #     logits, mmd_loss = model(x_source, x_target)
            
        #     # 交叉熵损失：使用源域真实标签
        #     ce_loss = ce_loss_fn(logits, y_source)
            
        #     # 联合损失
        #     loss = alpha * ce_loss + lambda_mmd * mmd_loss
            
        #     loss.backward()
        #     optimizer.step()
            
        #     total_loss_value += loss.item()
        #     total_batches += 1

        # return total_loss_value / total_batches



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

        for i in range(0,10):
            test_start = time.time() 
            # 初始化微调数据集和测试数据集
            self.Fine_datasets = []  # 用于微调的样本
            self.Test_datasets = []  # 用于测试的样本

            # 创建集合，用于存储不同组合的(pos, cls)标签
            unique_combinations = set()

            # num_epochs = 10
            # alpha = 1.0      # 交叉熵损失权重
            lambda_mmd = 0.1 # MMD 损失权重

            # 获取数据加载器
            source_dataset = self.dataloaders['source_train']

            # 遍历目标数据集，找到每个不同的(pos, cls)组合
            target_dataset = self.datasets['target_val']
            for sample in target_dataset:
                features, pos, cls = sample
                unique_combinations.add((pos, cls))

            # 设置每个(pos, cls)组合的微调样本数量
            Fine_number = args.Fine_number

            # 临时变量，用于记录已选中的样本
            selected_samples_set = set()  # 使用集合以便快速检查已选样本

            # 遍历所有 (pos, cls) 组合并选择样本
            for pos, cls in unique_combinations:
                # 收集所有匹配当前 (pos, cls) 组合的样本
                matching_samples = [sample for sample in target_dataset if sample[1] == pos and sample[2] == cls]               
                # 随机选择指定数量的样本
                selected_samples = random.sample(matching_samples, min(Fine_number, len(matching_samples)))                
                # 将选择的样本添加到 Fine_datasets 中
                self.Fine_datasets.extend(selected_samples) 
                # 记录选中的样本
                selected_samples_set.update([tuple(sample[0].flatten()) for sample in selected_samples])  # .flatten() 确保数据是1D

            # 将未被选中的样本添加到测试集
            self.Test_datasets = [sample for sample in target_dataset if tuple(sample[0].flatten()) not in selected_samples_set]

            from torch.utils.data import RandomSampler
            sampler_few = RandomSampler(self.Fine_datasets, replacement=True, num_samples=32)
            sampler_test = RandomSampler(self.Fine_datasets, replacement=True, num_samples=32)
            few_shot_loader = torch.utils.data.DataLoader(self.Fine_datasets, batch_size=32, sampler=sampler_few)
            test_loader = torch.utils.data.DataLoader(self.Test_datasets, batch_size=32, drop_last=True)

            # few_shot_loader = torch.utils.data.DataLoader(self.Fine_datasets, batch_size=32, sampler=sampler, drop_last=True)

            # 加载预训练好的模型并创建MLP模型
            # base_model = self.model

            base_model = self.model_test
            # base_model = MLPModel

            # 读取模型的权重
            base_model.load_state_dict(torch.load(args.model_Fine))
            # base_model.eval()

            model_tune = MMDModel(base_model, num_classes=args.Fine_num_classes).to(self.device)
            # 冻结部分层的参数
            layers_to_freeze = [#model_tune.base_model.backbone.conv1, model_tune.base_model.backbone.bn1,model_tune.base_model.backbone.relu,
                                
                                # model_tune.base_model.backbone.layer1,model_tune.base_model.backbone.BiLSTM1,
                                # model_tune.base_model.backbone.layer2,model_tune.base_model.backbone.BiLSTM2,
                                # model_tune.base_model.backbone.layer3,model_tune.base_model.backbone.BiLSTM3, 
                                # model_tune.base_model.backbone.layer4,model_tune.base_model.BiLSTM1,
                                # model_tune.base_model.ap,

                                # model_tune.base_model.backbone.LoraLayer_1,model_tune.base_model.backbone.LoraLayer_2,
                                # model_tune.base_model.backbone.LoraLayer_3,
                            
                                # model_tune.base_model.backbone.LoraLayer_4,

                                # model_tune.base_model.backbone.conv2,model_tune.base_model.backbone.bn2,model_tune.base_model.backbone.relu,

                                
                                
                                # model_tune.base_model.projetion_cls,

                                # model_tune.base_model.projetion_pos_1,
                                # model_tune.base_model.fc1


                                ]
            

            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            for name, param in model_tune.named_parameters():
                if param.requires_grad:
                    logging.info(f"{name}: Train")
                else:
                    logging.info(f"{name}: Freeze")

     
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

           
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_tune.parameters()), lr=args.Fine_lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
    
            # 在目标域数据上微调模型
            num_epochs = args.Fine_epoch
            
            for epoch in range(num_epochs):
                model_tune.train()
                
                running_loss = 0.0
                #loss = torch.tensor(0.0).cuda()  # 初始化 loss
                #print(f"Number of batches in loader: {len(few_shot_loader)}")
                
                # 让 source_loader 的 batch 轮流使用
                source_loader_iter = cycle(source_dataset)  
                
                for inputs, pos, _ in few_shot_loader:  # few_shot_loader 是目标域
                    optimizer.zero_grad()

                    # 目标域数据
                    inputs, pos = inputs.cuda(), pos.cuda()
                    outputs, out_target = model_tune(inputs)  # 提取目标域特征

                    # 取一个源域 batch
                    source_inputs, _, _ = next(source_loader_iter)
                    source_inputs = source_inputs.cuda()
                    _, out_source = model_tune(source_inputs)  # 提取源域特征

                    # 计算 MMD 损失
                    mmd_loss = compute_mmd_loss(out_source, out_target)

                    # 计算交叉熵损失
                  
                    
                    # print(f"pos shape: {pos.shape}")

                    # # 如果 pos 是 one-hot，转换成类别索引
                    # if len(pos.shape) > 1 and pos.shape[1] > 1:
                    #     pos = pos.argmax(dim=1)

                    # 计算交叉熵损失
                    ce_loss = criterion(outputs, pos)
                    # ce_loss = criterion(outputs, pos.argmax(dim=1))

                    # 总损失
                    loss = ce_loss + lambda_mmd * mmd_loss
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

    
                logging.info('Fine-Epoch {}/{}, Fine-Loss: {}'.format(epoch+1,num_epochs,running_loss))
                
                # 最后10轮对Fine后的模型进行测试
                if epoch>num_epochs-11:    
                    model_tune.eval()
                    correct_pos=0
                    inputs_all = []
                    label_pos =[]
                    all_label_pos = []
                    all_pred_pos = []
                    acc_pos = 0
                    
                    count = 0         
                    for batch_idx, (inputs, label_pos,label_cls) in enumerate(test_loader):
                        inputs_all = inputs
                        label_pos = label_pos
                        inputs_all = inputs_all.to(self.device)
                        label_pos = label_pos.to(self.device) 

                        pos_test,_ = model_tune(inputs_all)
                        pred_pos = pos_test.argmax(dim=1)

                        correct_pos = torch.eq(pred_pos, label_pos).float().sum().item()

                        # 累加标签和预测值
                        all_label_pos.extend(label_pos.cpu().numpy())  # 添加当前批次标签
                        all_pred_pos.extend(pred_pos.cpu().numpy())    # 添加当前批次预测
        
                        acc_pos += correct_pos
                        
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
                    
            
                    
                    #torch.save(model_tune.state_dict(),os.path.join(self.save_dir,'{}-fine_tuned_{}_{}.pth'.format(epoch+1, acc_pos, Fine_number)))
                
                if epoch == num_epochs-1:
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
        
                    # 保存微调后的模型
                    # torch.save(model_tune.state_dict(),os.path.join(self.save_dir,'fine_tuned_{}_{}.pth'.format(acc_pos,Fine_number)))
                
    
    
    
        # 计算样本标准差
        sample_std = np.std(acc_list, ddof=1)  # ddof=1 表示样本标准差
        # 计算总体标准差
        population_std = np.std(acc_list)  # 默认计算总体标准差
        mean_value = np.mean(acc_list)
        

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











