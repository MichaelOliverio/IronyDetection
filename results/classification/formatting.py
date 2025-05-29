import pandas as pd
import re

# Percorso del file CSV di input
csv_file = "fine-tuned-Ministral-8B-Instruct-2410-decoding-1.csv"

# Percorso del file Excel di output
excel_file = "classification.xlsx"

# Legge il file CSV
df = pd.read_csv(csv_file)

# Salva il dataframe come file Excel
df.to_excel(excel_file, index=False)

print(f"File Excel salvato come: {excel_file}")

# 3. Conta il numero di frasi
num_frasi = len(df)
print(f"\n📝 Numero totale di frasi: {num_frasi}")

# 4. Estrai la figura retorica dalla colonna 'prediction'
def estrai_figura(pred):
    match = re.search(r"That's why is an example of ([A-Z: ]+)", str(pred))
    return match.group(1).strip() if match else "LABEL_NOT_FOUND"

df['rhetorical_figure'] = df['prediction'].apply(estrai_figura)

# 5. Statistica sulla distribuzione
distribuzione = df['rhetorical_figure'].value_counts()

print("\n📊 Distribuzione delle figure retoriche:")
print(distribuzione)

# (Opzionale) Salva anche le statistiche in un secondo foglio Excel
with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
    distribuzione.to_frame(name='count').to_excel(writer, sheet_name='DistribuzioneFigure')