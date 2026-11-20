# Predictive Maintenance

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook o JupyterLab

Instala las dependencias desde una terminal ubicada en la raíz del proyecto:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost pyarrow jupyter torch optuna
```

`pyarrow` permite guardar los conjuntos separados en formato Parquet. Si no está
instalado, el notebook utiliza archivos `.csv.gz` como alternativa.

## Organización de los datos

Los archivos CSV originales deben estar dentro de la carpeta `archive/`:

```text
archive/
├── PdM_telemetry.csv
├── PdM_errors.csv
├── PdM_maint.csv
├── PdM_failures.csv
└── PdM_machines.csv
```

El notebook espera exactamente esos nombres y rutas relativas. Debe ejecutarse
desde la raíz del proyecto, donde también se encuentran `archive/` y `data/`.

## Cómo ejecutar

1. Abre una terminal en la carpeta del proyecto.
2. Inicia Jupyter:

   ```bash
   jupyter notebook
   ```

3. Abre `week1_predictive_maintenance.ipynb`.
4. Ejecuta las celdas en orden, preferiblemente con **Kernel > Restart Kernel and
   Run All**.

El notebook carga los datos, realiza el análisis exploratorio, construye las
etiquetas para un horizonte de 24 horas, separa los datos crudos temporalmente,
genera las variables históricas dentro de cada split y entrena los modelos de
regresión logística y XGBoost.

## Separación temporal

El pipeline utiliza las siguientes ventanas:

```text
Entrenamiento:  2015-01-01 hasta antes de 2015-08-31
Validación:     2015-09-01 hasta antes de 2015-10-31
Prueba:         2015-11-01 hasta 2016-01-01
```

Se deja un margen de 24 horas antes de las fronteras para evitar que las
etiquetas dependan de fallas posteriores a cada periodo. Las variables móviles,
errores y mantenimientos utilizan únicamente información disponible hasta el
instante de cada registro.

La división no depende del feature engineering. Se hace en dos pasos:

1. `split_raw_data`: cada fila cruda de telemetría se asigna a `train`,
   `validation` o `test` solo por su timestamp, y se construyen las etiquetas.
   En este punto todavía no existe ninguna variable derivada.
2. `build_split_features`: `build_features` se ejecuta por separado para cada
   split. Cada split recibe telemetría y errores en `[inicio - 24 h, fin)` (las
   24 h previas solo sirven de contexto para las ventanas móviles y nunca se usan
   como muestras) y los mantenimientos estrictamente anteriores a `fin`.

El notebook verifica que la pertenencia a cada split se fija antes de calcular
variables y que las variables de cada split no cambian si se borran todos los
datos crudos posteriores a ese split.

## Archivos generados

El pipeline crea automáticamente `data/processed/` y guarda:

```text
data/processed/
├── train.parquet
├── validation.parquet
├── test.parquet
├── baseline_binary_validation.csv
├── baseline_multiclass_validation.csv
├── imbalance_table.csv
└── ablation_validation.csv
```

Los archivos `train`, `validation` y `test` contienen las variables originales,
las variables derivadas, las etiquetas y la información temporal de cada
registro. El conjunto de prueba se genera y se guarda para una evaluación
posterior; no se utiliza para entrenar modelos ni para seleccionar umbrales en
este notebook.

Si `pyarrow` no está disponible, los tres conjuntos se guardan como:

```text
data/processed/train.csv.gz
data/processed/validation.csv.gz
data/processed/test.csv.gz
```

Los archivos `.csv` están excluidos por `.gitignore` para evitar subir datos al
repositorio.

## Semana 2: modelos secuenciales de deep learning

Ejecuta primero `week1_predictive_maintenance.ipynb` (genera `data/processed/*.parquet`) y después
`week2_deep_learning.ipynb`. El código reutilizable está en `pm_dl.py`:

- `build_hourly_channels`, `SequenceStore`, `SplitData`: tensor de entrada con ventanas de `L` horas
  (24, 72, 168) y 13 canales (4 sensores, 5 indicadores de error, 4 `log(1 + horas desde reemplazo)`),
  normalizado con estadísticas de entrenamiento. Las ventanas se cortan por índice, sin materializarlas.
- `RNNNet` (LSTM/GRU), `TCNNet` y `CNNLSTMNet`, todas con salida binaria + multiclase.
- `Strategy`: pérdida ponderada, focal loss, submuestreo y sobremuestreo.
- `train_model`: AdamW, dropout, early stopping sobre PR-AUC de validación.
- `evaluate`: PR-AUC, recall con precisión >= 0.80, F1 por componente, calibración y tiempo de anticipación
  (primera alerta sostenida de 3 horas).

En CPU (2 núcleos) la corrida completa tarda alrededor de 1.5 horas, la mayor parte en la malla
arquitectura × ventana y en Optuna (30 ensayos o 40 minutos). Cada corrida se guarda en `runs/week2/`,
así que volver a ejecutar el notebook no reentrena lo que ya existe; borra esa carpeta para empezar de cero.
Para una prueba rápida: `PM_QUICK=1 jupyter nbconvert --execute week2_deep_learning.ipynb`.

Salidas: `data/processed/week2_*.csv`, figuras en `figures/week2/` y el mejor modelo en `models/week2_best.pt`
(los CSV, `runs/` y `models/` no se suben al repositorio). El reporte y las slides de la entrega están en `reportes/`.
