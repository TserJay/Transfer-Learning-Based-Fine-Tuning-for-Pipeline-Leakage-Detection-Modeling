import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 加载源域模型和tokenizer（例如使用BERT）
source_model = AutoModel.from_pretrained("bert-base-uncased")
source_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 加载目标域模型和tokenizer
target_model = AutoModel.from_pretrained("bert-base-uncased")
target_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 1. 提取源域特征
def extract_source_features(texts):
    inputs = source_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = source_model(**inputs).last_hidden_state  # 获取源域特征
    return features

# 示例源域数据
source_texts = ["source example 1", "source example 2"]
source_features = extract_source_features(source_texts)

# 2. 设置LoRA配置并应用到目标域模型
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none")
target_model = get_peft_model(target_model, lora_config)

# 加载目标域数据集（以Hugging Face数据集为例）
dataset = load_dataset("imdb", split="train[:1%]")  # 加载IMDB数据集的一小部分作为示例

# 数据预处理
def preprocess_function(examples):
    return target_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)

# 4. 使用Trainer进行微调
trainer = Trainer(
    model=target_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 开始训练
trainer.train()

# 5. 微调后从目标域提取特征
def extract_target_features(texts):
    inputs = target_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = target_model(**inputs).last_hidden_state  # 获取目标域特征
    return features

target_texts = ["target example 1", "target example 2"]
target_features = extract_target_features(target_texts)

# 6. 特征融合（可以使用拼接、加权平均等方法）
def fuse_features(source_features, target_features):
    # 简单拼接特征向量
    fused = torch.cat((source_features, target_features), dim=-1)
    return fused

fused_features = fuse_features(source_features, target_features)

# 7. 将融合后的特征用于下游任务（如分类）
print("Fused Features Shape:", fused_features.shape)
