import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,Trainer,DataCollatorForLanguageModeling
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_double_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                             quantization_config=bnb_config,
                                             device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model,LoraConfig(r=16,lora_alpha=32,
                                        lora_dropout=0.05,bias="none",
                                        task_type="CAUSAL_LM",
                                        target_modules=["q_proj","v_proj","k_proj","o_proj"]))
tokenized_dataset = load_dataset("imdb").map(lambda x: AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                                             (x["text"],padding="max_length",truncation=True),batched=True)
trainer = Trainer(model=model,args=TrainingArguments(output_dir="./qlora-output",
                                                     per_device_train_batch_size=4,
                                                     gradient_accumulation_steps=4,optim="paged_adamw_8bit",
                                                     learning_rate=2e-4,num_train_epochs=3,fp16=True,save_steps=500,logging_steps=100),
                                                     train_dataset=tokenized_dataset["train"],
                  data_collator=DataCollatorForLanguageModeling(AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf"),mlm=False))
trainer.train()
model.save_pretrained("./qlora-finetuned-model")
AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf").save_pretrained("./qlora-finetuned-model")