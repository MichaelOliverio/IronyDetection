import random
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Carica il dataset
df = pd.read_csv("baseline-1.csv")

# Le 8 etichette
labels = [
    'ANALOGY', 'HYPERBOLE', 'EUPHEMISM', 'RHETORICAL QUESTION',
    'EX:CONTEXT SHIFT', 'IM:FALSE ASSERTION', 'EX:OXYMORON PARADOX', 'OTHER'
]

# Funzione per estrarre le metriche desiderate
def get_metrics(true, pred):
    report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    results = {}

    # Estrai precision, recall, f1 per ogni label
    for label in labels:
        results[f"{label}_precision"] = report[label]['precision']
        results[f"{label}_recall"] = report[label]['recall']
        results[f"{label}_f1"] = report[label]['f1-score']

    # Macro avg
    results["macro_precision"] = report["macro avg"]["precision"]
    results["macro_recall"] = report["macro avg"]["recall"]
    results["macro_f1"] = report["macro avg"]["f1-score"]

    # Weighted avg
    results["weighted_precision"] = report["weighted avg"]["precision"]
    results["weighted_recall"] = report["weighted avg"]["recall"]
    results["weighted_f1"] = report["weighted avg"]["f1-score"]

    # Accuracy
    results["accuracy"] = accuracy_score(true, pred)

    return results

# Esegui 3 run e raccogli i risultati
all_results = []

for i in range(3):
    print(f"\n--- Run {i+1} ---")
    random_preds = [random.choice(labels) for _ in range(len(df))]
    metrics = get_metrics(df['actual'], random_preds)
    all_results.append(metrics)

    # Stampa il report completo di sklearn
    print(classification_report(df['actual'], random_preds, zero_division=0))

# Calcola la media
avg_results = {key: np.mean([r[key] for r in all_results]) for key in all_results[0]}

# Stampa media
print("\n=== Media dei risultati su 3 run (random baseline) ===")
for key, val in avg_results.items():
    print(f"{key}: {val:.3f}")