# Insurance Charges — Regresión Log-Lineal

Predicción de costos médicos (`charges`) a partir de variables demográficas y de salud, usando **regresión lineal sobre el logaritmo de la variable objetivo**.

## 📁 Estructura del proyecto

```
Insurance/
├── 01_insurance_log_linear_regression.ipynb
├── insurance.csv
├── venv/
└── README.md
```

## 📊 Dataset

`insurance.csv` — 1338 registros, 7 columnas:

| Columna    | Tipo    | Descripción                          |
|------------|---------|---------------------------------------|
| age        | int     | Edad del asegurado                    |
| sex        | str     | Sexo (male/female)                    |
| bmi        | float   | Índice de masa corporal               |
| children   | int     | Número de hijos/dependientes          |
| smoker     | str     | Fumador (yes/no)                      |
| region     | str     | Región (northeast, southeast, ...)    |
| charges    | float   | Costo médico facturado (target)       |

Sin valores nulos.

## 🔧 Requisitos

```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 🚀 Uso

```bash
jupyter notebook 01_insurance_log_linear_regression.ipynb
```

## 🔬 Metodología

1. **EDA inicial**: carga de datos, inspección con `df.info()` y `df.head()`.

2. **Análisis de la variable objetivo**: `charges` presenta fuerte asimetría positiva (skew ≈ 1.52), con la mayoría de los costos entre \$2,000–\$10,000 y una minoría de valores muy altos.

3. **Transformación logarítmica** (`log_charges = np.log(charges)`), justificada por:
   - Reducir la asimetría y acercar la distribución de errores a una normal.
   - Estabilizar la varianza y disminuir el impacto de outliers.
   - Permitir interpretar los coeficientes como cambios porcentuales en el costo.

   Tras la transformación, el sesgo baja significativamente (verificado con `.skew()` antes/después).

4. **Preprocesamiento**: codificación one-hot de `sex`, `smoker` y `region` (`drop_first=True`).

5. **Split y entrenamiento**: 80/20 train-test (`random_state=42`), `LinearRegression` de scikit-learn entrenada sobre `log_charges`.

6. **Evaluación**:
   - Predicciones revertidas a escala original con `np.exp()`.
   - Métricas: **R² ≈ 0.61** y **RMSE ≈ \$7,815** en dólares.
   - Visualizaciones: predicción vs. real en escala log y en escala dólar.

## 📌 Conclusión

En escala logarítmica el modelo se ajusta razonablemente bien, pero al retransformar a dólares se observa **subestimación sistemática en los costos altos**, atribuible al **sesgo de retransformación** (Jensen's inequality: `E[exp(x)] ≠ exp(E[x])`).
