import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para que funcione en Colab sin pantalla

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("  MODELO IA - PETSHOP DATAOPS")
print("=" * 50)

# ── 1. CARGAR DATOS ──────────────────────────────────
# Usamos ventas_procesadas.csv (sin RUT - ya anonimizado)
df = pd.read_csv('ventas_procesadas.csv')
print(f"\n✓ Datos cargados: {len(df)} registros")
print(f"  Columnas: {list(df.columns)}")

# ── 2. ANÁLISIS DE CALIDAD ───────────────────────────
print("\n--- ANÁLISIS DE CALIDAD ---")
print(f"Nulos por columna:\n{df.isnull().sum()}")
print(f"\nEstadísticas básicas:")
print(df[['cantidad', 'precio', 'total_venta']].describe().round(2))

# ── 3. PREPROCESAMIENTO ──────────────────────────────
# Convertir columna 'producto' (texto) a números
# Los modelos solo entienden números, no palabras
le = LabelEncoder()
df['producto_cod'] = le.fit_transform(df['producto'])

print(f"\n✓ Productos codificados: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 4. DEFINIR VARIABLES ─────────────────────────────
# X = variables que el modelo usa para aprender (features)
# y = lo que queremos predecir (target)
X = df[['producto_cod', 'cantidad', 'precio']]
y = df['categoria_venta']

print(f"\n✓ Variables de entrada (X): {list(X.columns)}")
print(f"✓ Variable objetivo (y): categoria_venta")
print(f"  Distribución: {y.value_counts().to_dict()}")

# ── 5. PARTICIÓN TRAIN / TEST ────────────────────────
# 80% para entrenar, 20% para evaluar
# Esto evita que el modelo "haga trampa" memorizando los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✓ Partición de datos:")
print(f"  Entrenamiento: {len(X_train)} registros (80%)")
print(f"  Prueba:        {len(X_test)} registros (20%)")

# ── 6. ENTRENAR EL MODELO ────────────────────────────
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
print(f"\n✓ Modelo entrenado con 100 árboles de decisión")

# ── 7. PREDICCIONES ──────────────────────────────────
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidades para ROC

# ── 8. MÉTRICAS ──────────────────────────────────────
print("\n" + "=" * 50)
print("  MÉTRICAS DEL MODELO")
print("=" * 50)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy:  {acc:.4f}  ({acc*100:.1f}%)")

print("\nReporte completo:")
print(classification_report(y_test, y_pred,
      target_names=['Venta Baja (0)', 'Venta Alta (1)']))

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Curva ROC y Gini
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
gini = 2 * roc_auc - 1
print(f"\nAUC-ROC:  {roc_auc:.4f}")
print(f"Gini:     {gini:.4f}  ({gini*100:.1f}%)")

# ── 9. GUARDAR MÉTRICAS EN CSV ───────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
metricas = {
    'accuracy': acc,
    'precision_0': report['0']['precision'],
    'recall_0': report['0']['recall'],
    'f1_0': report['0']['f1-score'],
    'precision_1': report['1']['precision'],
    'recall_1': report['1']['recall'],
    'f1_1': report['1']['f1-score'],
    'auc_roc': roc_auc,
    'gini': gini
}
pd.DataFrame([metricas]).to_csv('metricas_modelo.csv', index=False)
print("\n✓ Métricas guardadas en metricas_modelo.csv")

# ── 10. GRÁFICOS ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico 1: Matriz de confusión
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Matriz de Confusión')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Baja (0)', 'Alta (1)'])
axes[0].set_yticklabels(['Baja (0)', 'Alta (1)'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=16, fontweight='bold')

# Gráfico 2: Curva ROC
axes[1].plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC (AUC = {roc_auc:.2f}, Gini = {gini:.2f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Clasificador aleatorio')
axes[1].set_xlabel('Tasa de Falsos Positivos')
axes[1].set_ylabel('Tasa de Verdaderos Positivos')
axes[1].set_title('Curva ROC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos_modelo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Gráficos guardados en graficos_modelo.png")
print("\n¡Modelo listo para la presentación!")