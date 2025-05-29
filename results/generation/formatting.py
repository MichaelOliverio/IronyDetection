import pandas as pd
import re

# Percorso del file CSV di input
csv_file = "fine-tuned-Ministral-8B-Instruct-2410-decoding-1.csv"

# Percorso del file Excel di output
excel_file = "generation.xlsx"

# Legge il file CSV
df = pd.read_csv(csv_file)

# Salva il dataframe come file Excel
df.to_excel(excel_file, index=False)

print(f"File Excel salvato come: {excel_file}")

# 3. Conta il numero di frasi
num_frasi = len(df)
print(f"\n📝 Numero totale di frasi: {num_frasi}")