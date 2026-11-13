# Predictive Maintenance

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook o JupyterLab

Instala las dependencias desde una terminal ubicada en la raíz del proyecto:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost pyarrow jupyter
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
etiquetas para un horizonte de 24 horas, genera variables históricas, separa los
datos temporalmente y entrena los modelos de regresión logística y XGBoost.

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
