import pandas as pd
import os
import math

# Ispravna putanja
input_file = r"C:\Users\Nevena Perišić\Desktop\ProjekatCompFin\podaci\0DTE.xlsx"
output_folder = "trenazniPodaci"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_excel(input_file)

n = 10
chunk_size = math.ceil(len(df) / n)

for i in range(n):
    start = i * chunk_size
    end = start + chunk_size
    chunk = df.iloc[start:end]
    chunk.to_excel(os.path.join(output_folder, f"deo0DTE_{i+1}.xlsx"), index=False)

print("Uspešno podeljeno!")
