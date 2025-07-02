import pandas as pd
import os
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import re

def extract_figure(text):
    match = re.search(r"that's why is an example of\s+(.*)", str(text), re.IGNORECASE)
    if match:
        figure = match.group(1).strip().lower().rstrip(".")
        return figure
    return ""

def round_sig(x, sig=3):
    if isinstance(x, (float, int)):
        return float(f"{x:.{sig}g}")
    return x

def format_report(report_dict):
    rounded = {}
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rounded[label] = {k: round_sig(v) for k, v in metrics.items()}
        else:
            rounded[label] = round_sig(metrics)
    return rounded

def print_formatted_report(report_dict):
    print("\n📈 Report di classificazione (precision, recall, f1-score, support):")
    labels = [label for label in report_dict if label not in ('accuracy', 'macro avg', 'weighted avg')]
    header = f"{'Label':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}"
    print(header)
    print("-" * len(header))
    for label in labels + ['macro avg', 'weighted avg']:
        row = report_dict[label]
        print(f"{label:<20} {row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1-score']:>8.3f} {row['support']:>8.0f}")
    print(f"{'Accuracy':<20} {'':>8} {'':>8} {'':>8} {report_dict['accuracy']:>8.3f}")

def calculate_metrics(file_path, print_confusion=False):
    df = pd.read_csv(file_path)

    df['extracted_prediction'] = df['prediction'].apply(extract_figure)
    df['rhetorical_figure'] = df['rhetorical_figure'].astype(str).str.lower().str.strip()

    if print_confusion:
        print(f"\n🧩 Matrice di confusione per {os.path.basename(file_path)}:")
        confusion = pd.crosstab(df['rhetorical_figure'], df['extracted_prediction'],
                                rownames=['Actual'], colnames=['Predicted'])
        print(confusion)

    report = classification_report(df['rhetorical_figure'], df['extracted_prediction'], output_dict=True, zero_division=0)
    return report

def average_reports(reports):
    avg_report = {}
    keys = reports[0].keys()

    for key in keys:
        if isinstance(reports[0][key], dict):
            avg_report[key] = {}
            for metric in reports[0][key]:
                values = [r[key].get(metric, 0.0) for r in reports if key in r]
                avg_report[key][metric] = np.mean(values)
        else:  # accuracy
            values = [r.get(key, 0.0) for r in reports]
            avg_report[key] = np.mean(values)

    return avg_report

def collect_model_accuracies(folder_path):
    decoding_files = glob(os.path.join(folder_path, "*-decoding-[1-3].csv"))
    model_groups = defaultdict(list)

    for file_path in decoding_files:
        filename = os.path.basename(file_path)
        model_prefix = filename.rsplit("-decoding-", 1)[0]
        model_groups[model_prefix].append(file_path)

    for model_prefix, file_list in model_groups.items():
        print(f"\n🔍 Modello: {model_prefix}")
        reports = []
        for file in sorted(file_list):
            report = calculate_metrics(file, print_confusion=False)
            reports.append(report)
        avg = average_reports(reports)
        print_formatted_report(avg)

# Directory corrente
folder_path = "."
collect_model_accuracies(folder_path)