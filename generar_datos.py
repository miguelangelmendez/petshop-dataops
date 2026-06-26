import pandas as pd
import numpy as np

np.random.seed(42)  # Para que los datos sean reproducibles

n = 200  # Número de registros

productos = ['alimento_perro', 'alimento_gato', 'accesorio', 'medicina', 'juguete']
np.random.seed(42)

data = {
    'id_venta': range(1, n + 1),
    'producto': np.random.choice(productos, n),
    'cantidad': np.random.randint(1, 10, n),
    'precio': np.round(np.random.uniform(1000, 25000, n), 0),
    'cliente_rut': [f'{np.random.randint(10000000, 25000000)}-{np.random.randint(0,9)}' for _ in range(n)]
}

df = pd.DataFrame(data)
df['total_venta'] = df['cantidad'] * df['precio']

# Variable objetivo: 1 si total_venta > mediana, 0 si no
mediana = df['total_venta'].median()
df['categoria_venta'] = (df['total_venta'] > mediana).astype(int)

# Guardamos DOS versiones:
# 1. Con RUT (datos crudos - para mostrar el problema de seguridad)
df.to_csv('ventas_raw.csv', index=False)

# 2. Sin RUT (datos anonimizados - para entrenar el modelo)
df_anonimizado = df.drop(columns=['cliente_rut'])
df_anonimizado.to_csv('ventas_procesadas.csv', index=False)

print(f"Dataset generado: {n} registros")
print(f"Ventas altas (1): {df['categoria_venta'].sum()}")
print(f"Ventas bajas (0): {n - df['categoria_venta'].sum()}")
print(f"Mediana total_venta: ${mediana:,.0f}")
print("\nPrimeras 5 filas:")
print(df.head())
