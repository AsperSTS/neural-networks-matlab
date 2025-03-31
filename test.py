import pandas as pd

try:
    df = pd.read_csv('dataset4_Cancer.csv')
    
    if len(df.columns) > 1: # Asegurarse de que haya al menos 3 columnas
        unique_values = df.iloc[:, 56].unique() # Usar iloc para acceder por índice
        print(sorted(unique_values)) # Imprimir los valores únicos ordenados
        print(len(unique_values))
    else:
        print("El DataFrame no tiene suficientes columnas (al menos 2).")
        
except FileNotFoundError:
    print("El archivo  no fue encontrado.")
except Exception as e:
    print(f"Ocurrió un error: {e}")