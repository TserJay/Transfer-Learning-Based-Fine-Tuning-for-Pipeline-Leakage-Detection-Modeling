import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
import logging
 

signal_size = 1024



#Three working conditions

signal_size = 1024
# work_condition=['0','1','2','3']
# dataname= {0:work_condition[0],
#            1:work_condition[1],
#            2:work_condition[2],
#            3:work_condition[3]
#           }  #四个泄露孔径类别的路径




dataname = ['0' , '1' , '2' , '3']
label_condition=['2','3','4','5','6','7','8','9','10','11','12','13']
# label = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 ]
label_pos = [0,1,2,3,4,5,6,7,8,9,10,11]
label_cls = [0,1,2,3]



label_source_condition=[ ]
label_source_pos = []




def get_source_files(root, N,source_num_classes):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab_pos = []
    lab_cls = []
    # print(class_id)
    # for id in class_id:
    #     file_list_path = os.path.join()
    
    k=0
    random_numbers = random.sample(range(0, 12), source_num_classes)
    logging.info("source_pos:{}".format(random_numbers))
    for i in tqdm(random_numbers):
    #for i in tqdm(range(0,11)):
        path1 = os.path.join('/tmp',root,dataname[N[k]][0],label_condition[i])
        data1, lab_pos1, lab_cls1 = data_load(path1 , label_pos[i], label_cls[ N[k] ])
        data += data1
        lab_pos +=lab_pos1
        lab_cls +=lab_cls1

    return [data, lab_pos,lab_cls]
#generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab_pos = []
    lab_cls = []
    # print(class_id)
    # for id in class_id:
    #     file_list_path = os.path.join()
    
    k=0
    for i in tqdm(range(0,12)):
        path1 = os.path.join('/tmp',root,dataname[N[k]][0],label_condition[i]) 
        data1, lab_pos1, lab_cls1 = data_load(path1 , label_pos[i] , label_cls[ N[k] ])
        data += data1
        lab_pos +=lab_pos1
        lab_cls +=lab_cls1

    return [data, lab_pos, lab_cls]


def data_load(filename,label_pos,label_cls):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''

    data , lab_pos , lab_cls, data_path= [] , [] , [] , []
    file_list =  os.listdir(filename)

    for file in file_list:  
        finale_path = os.path.join(filename, file)
        data_path.append(finale_path)

    # data = np.concatenate((np.array(data_path).reshape(-1, 1), np.array(label).reshape(-1, 1)), axis=1)
        # 将all_data_path转换为一列，-1代表未指定行数
        # 输出实例：all_files[路径，类别]
        # axis表示级联坐标，axis = 1为沿着x轴为一组
    for path in data_path:
        df = pd.read_csv(path, header=0)
        data_temp = df.values[:, [1, 3, 5]][:1792, ].astype(float)

        data.append(data_temp)
        lab_pos.append(label_pos)
        lab_cls.append(label_cls)

    
    # start,end=0,signal_size
    # while end<=fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start +=signal_size
    #     end +=signal_size

    return data, lab_pos, lab_cls

#--------------------------------------------------------------------------------------------------------------------
class leak_signals(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1" ,source_num_classes=12):

        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.source_num_classes = source_num_classes
        # print(self.source_N)   #源域数据

        
        self.target_N = transfer_task[1:4]
        # print(self.target_N)   #目标域数据
        

        self.normlizetype = normlizetype
        
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True,x=False,i=0):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label_pos": list_data[1], "lebel_cls":list_data[2]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val     
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            
            return source_train, source_val, target_train, target_val
        else:

            
            #get source train and val
            list_data = get_source_files(self.data_dir, self.source_N, self.source_num_classes)
            data_pd_1 = pd.DataFrame({"data": list_data[0], "label_pos": list_data[1], "label_cls": list_data[2]})
            train_pd, val_pd = train_test_split(data_pd_1, test_size=0.2, random_state=40,stratify=data_pd_1["label_pos"])
            #, stratify=[data_pd["label_pos"], data_pd["label_cls"]]
            # print(train_pd)
            # print(val_pd)
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])


            # get target train and val
            all_data_pd = []
            data_target = []

            for target in self.target_N:
                list_data = get_files(self.data_dir, target)
                # 假设get_files返回的是一个包含三个元素的列表，分别是数据、位置标签和分类标签  
                # 直接创建一个DataFrame，而不需要在循环中重复创建  
                data_pd_temp = pd.DataFrame({  
                    "data": list_data[0],  
                    "label_pos": list_data[1],  
                    "label_cls": list_data[2]  
                    })
                # data_target[target] = data_pd_temp 
                
                all_data_pd.append(data_pd_temp)  
                # print(data_pd_temp)
  
                # 在循环结束后，合并所有的data_pd_temp  
                data_pd_final = pd.concat(all_data_pd, ignore_index=True)

            # print(data_pd_final)

            target_val = dataset(list_data=data_pd_final, transform=self.data_transforms['val'])
            return source_train, source_val, target_val
        



    def data_test(self,i=0):
            # get target train and val
            all_data_pd = []
            

            for target in self.target_N:
                
                list_data = get_files(self.data_dir, target)
                # 假设get_files返回的是一个包含三个元素的列表，分别是数据、位置标签和分类标签  
                # 直接创建一个DataFrame，而不需要在循环中重复创建  
                data_pd_temp = pd.DataFrame({  
                    "data": list_data[0],  
                    "label_pos": list_data[1],  
                    "label_cls": list_data[2]  
                    })
                # data_target[target] = data_pd_temp 
                
                all_data_pd.append(data_pd_temp)  
                # print(data_pd_temp)
  
                # 在循环结束后，合并所有的data_pd_temp  
                data_pd_final = pd.concat(all_data_pd, ignore_index=True)

            print(data_pd_final)

           
            return data_pd_final



