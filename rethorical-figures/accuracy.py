import pandas as pd
import os
from glob import glob
from collections import defaultdict

def calculate_accuracy(file_path):
    df = pd.read_csv(file_path)
    df['correct'] = df.apply(
        lambda row: str(row['prediction']).lower() in str(row['actual']).lower(),
        axis=1
    )
    return df['correct'].mean()

def collect_model_accuracies(folder_path):
    decoding_files = glob(os.path.join(folder_path, "*-decoding-[1-3].csv"))
    model_groups = defaultdict(list)

    for file_path in decoding_files:
        filename = os.path.basename(file_path)
        model_prefix = filename.rsplit("-decoding-", 1)[0]
        model_groups[model_prefix].append(file_path)

    for model_prefix, file_list in model_groups.items():
        print(f"\n🔍 Modello: {model_prefix}")
        scores = []
        for file in sorted(file_list):
            acc = calculate_accuracy(file)
            scores.append(acc)
            print(f"  {os.path.basename(file)} → Accuratezza (inclusiva, case-insensitive): {acc:.2%}")
        if scores:
            avg_accuracy = sum(scores) / len(scores)
            print(f"📊 Accuratezza media: {avg_accuracy:.2%}")

# Directory corrente
folder_path = "."
collect_model_accuracies(folder_path)