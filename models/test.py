import torch
import torch.nn as nn

class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = {}
        self.lora_alpha = {}
        self.lora_dropout = nn.ModuleDict()
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        # 这里可以定义权重初始化的方法
        pass

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.lora_layer = LoraLayer(input_size, output_size)

    def forward(self, x, adapter_name):
        # 在这里可以调用 LoraLayer 的前向传播逻辑
        # 需要根据你具体的需求实现前向传播
        pass

    def update_lora(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.lora_layer.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)


# 创建模型实例
model = MyModel(input_size=128, output_size=10)

# 更新 LoRA 适配器
model.update_lora(adapter_name='adapter1', r=4, lora_alpha=0.1, lora_dropout=0.2, init_lora_weights=True)

# 进行前向传播
# 假设 x 是输入数据
# output = model(x, adapter_name='adapter1')




import torch
import torch.nn as nn
import torch.nn.init as init
import math

class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = {}
        self.lora_alpha = {}
        self.lora_dropout = nn.ModuleDict()
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]

        init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        if lora_A.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(lora_A.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(lora_A.bias, -bound, bound)

        init.kaiming_uniform_(lora_B.weight, a=math.sqrt(5))
        if lora_B.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(lora_B.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(lora_B.bias, -bound, bound)

    def forward(self, x, adapter_name):
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        lora_dropout = self.lora_dropout[adapter_name]

        x = lora_A(x)
        x = lora_dropout(x)
        x = lora_B(x)

        x = x * self.scaling[adapter_name]

        return x








class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyModel, self).__init__()
        self.lora_layer = LoraLayer(in_features, out_features)
        # 添加适配器示例
        self.lora_layer.update_layer('adapter1', r=4, lora_alpha=1.0, lora_dropout=0.1, init_lora_weights=True)

    def forward(self, x, adapter_name):
        # 确保传递适配器名称
        return self.lora_layer(x, adapter_name)

# 示例用法
model = MyModel(in_features=128, out_features=64)
input_tensor = torch.randn(32, 128)  # 示例输入
output = model(input_tensor, 'adapter1')  # 使用适配器名称调用
