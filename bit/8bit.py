
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 启用4-bit量化
    bnb_4bit_quant_type="nf4",           # 使用NormalFloat4量化  
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用bfloat16精度
    bnb_4bit_use_double_quant=True,      # 启用双重量化  
    bnb_4bit_double_quant_type="nf4",    # 双重量化也使用NF4格式
)

# 2. 模型加载与量化
model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,    
    device_map="auto"                   
)

# 准备模型进行k-bit训练（重要步骤！）
model = prepare_model_for_kbit_training(model)

# 3. LoRA配置（QLoRA核心）
lora_config = LoraConfig(
    r=16,                              # 低秩矩阵的秩  
    lora_alpha=32,                     # 缩放因子
    lora_dropout=0.05,                 # Dropout概率
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                    # 需要适配的模块
        "q_proj", "v_proj", "k_proj", "o_proj"  # 注意力机制相关模块
    ]
)

# 应用LoRA到量化模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 输出可训练参数比例


# 4. 数据准备（示例使用小样本数据）
dataset = load_dataset("imdb")  # 示例数据集
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
    batched=True
)

# 5. 分页优化器配置
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # 有效batch=16
    optim="paged_adamw_8bit",          # 启用分页优化器 
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,                         # 混合精度训练
    save_steps=500,
    logging_steps=100,
)


# 6. 训练执行

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./qlora-finetuned-model")
tokenizer.save_pretrained("./qlora-finetuned-model")