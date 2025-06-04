import pandas as pd

# Carica il file Excel
df = pd.read_excel('baseline--ministral1.xlsx')  # o 'nomefile.xls'

# Salva in formato CSV
df.to_csv('baseline--ministral-refined.csv', index=False)