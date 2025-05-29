# Installazioni necessarie (da eseguire solo una volta)
# pip install peft transformers trl huggingface_hub accelerate --upgrade torch

import os
import torch
import gc
import pandas as pd
import re
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configurazione gestione memoria PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16 if torch.cuda.is_available() else torch.float32)

# Login HuggingFace
login(token="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")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Paths
data_path_datasets = './datasets/multipico'
data_path_models = './models/rhetorical-figures'
data_path_prediction = './models/multipico_classification'
os.makedirs(data_path_models, exist_ok=True)

models_name = [
    'mistralai/Ministral-8B-Instruct-2410',
]

# LoRA parameters
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# TrainingArguments parameters
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Ridotto
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    eval_strategy="steps",  # Calcolo della validation loss ad ogni step
    save_strategy="no",
    learning_rate=2e-4,
    weight_decay=0.001,
    eval_steps=500,  # Numero di passaggi prima di valutare il modello
    logging_steps= 500,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

print("Inizializzazione completata")

def prepare_dataset(split):
    print(f"Preparing {data_path_datasets}/{split}.csv dataset per la baseline...")
    df = pd.read_csv(f'{data_path_datasets}/{split}.csv')

    instruction = (
        "Analyze the ironic sentence (INPUT) and identify which rhetorical figure (OUTPUT) it uses. "
        "Explain your choice briefly."
    )
    dataset = []
    for _, row in df.iterrows():
        input = row['reply'] + row['post']
        dataset.append({
            'instruction': instruction,
            'reply_id': row['reply_id'],
            'post_id': row['post_id'],
            'input': input,
        })
    return Dataset.from_pandas(pd.DataFrame(dataset))

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}   # Forza su cuda:0
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def formatting_prompts_func(example):
    return [f'''<s> [INST] {instr} [/INST] [INPUT] {inp} [/INPUT] [OUTPUT] {outp} [/OUTPUT] </s>'''
            for instr, inp, outp in zip(example['instruction'], example['input'], example['output'])]

def train(model, tokenizer, train_dataset, eval_dataset):
    collator = DataCollatorForCompletionOnlyLM(" [OUTPUT]", tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train()
    return trainer

def load_fine_tuned_model(model_name, fine_tuned_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}   # Anche qui forzato su cuda:0
    )
    model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def clear_output(output):
    match = re.search(r'\[OUTPUT\](.*?)\[/OUTPUT\]', output, re.DOTALL)
    return match.group(1).strip() if match else "Output not found"

def model_generation(pipe, test_dataset, output_path):
    records = []
    with torch.no_grad():
        for i, record in enumerate(test_dataset, 1):
            print(f'record #{i}')
            result = pipe(f"<s> [INST] {record['instruction']} [/INST] [INPUT] {record['input']} [/INPUT] [OUTPUT]")
            prediction = clear_output(result[0]['generated_text'])
            print(f'Result: {result} | Output: {result[0]["generated_text"]} | Prediction: {prediction}')
            records.append((record['reply_id'], record['post_id'], record['input'], prediction, result[0]['generated_text']))

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        df = pd.DataFrame(records, columns=['reply_id','post_id', 'input', 'prediction', 'raw_generation'])
        df.to_csv(output_path, index=False)

# Caricamento dataset
dataset = prepare_dataset('multipico_ita_filtered')

print("Dataset caricato correttamente")

print("Inizio generazione")
for model_name in models_name:
    print(f"Generazione {model_name}...")
    clean_model_name = model_name.split('/')[1]
    output_path = f'{data_path_prediction}/fine-tuned-{clean_model_name}'

    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_name, f'{output_path}')

    pipe = pipeline(
        'text-generation',
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        max_new_tokens=200, 
        temperature=0.1,
        torch_dtype=torch.bfloat16
    )

    for i in range(1):
        model_generation(pipe, dataset, f'{output_path}-decoding-{i+1}.csv')

    print(f"Generazione completata per {model_name}")

    del fine_tuned_model, fine_tuned_tokenizer, pipe
    gc.collect()
    torch.cuda.empty_cache()
